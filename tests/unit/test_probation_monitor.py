"""Unit tests for the Stage D3 deterministic probation monitor.

The monitor closes Stage D's accept line: continuous monitoring wired to
probation. These tests pin:

- flag determinism: every flag is arithmetic over stored records;
- status tiers: critical vs warn vs ok;
- change-only emission on the probation.monitor bus event;
- the observe-only constitution: the monitor NEVER mutates probation state
  (exploding-store test + source scan);
- the read-only tool + registry stays consistent (49 tools).
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, List

import pytest

from forex_ai_agent.autonomy import probation_monitor as pm
from forex_ai_agent.autonomy.probation import ProbationStatus
from forex_ai_agent.autonomy.probation_monitor import (
    PROBATION_MONITOR_EVENT,
    ProbationMonitor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prob(**overrides) -> ProbationStatus:
    """A healthy mid-probation row by default (10/20 trades, 7/14 days)."""
    started = datetime.utcnow() - timedelta(days=7)
    defaults = dict(
        version_id="v-mon-1",
        strategy_name="MonStrat",
        pair="EUR/USD",
        status="active",
        observed_trades=10,
        required_trades=20,
        observed_days=7,
        required_days=14,
        days_elapsed=7,
        win_rate=0.5,
        profit_factor=1.4,
        worst_daily_loss_pct=0.01,
        last_metrics={
            "_accrual_mode": "event",
            "_event_through": (datetime.utcnow() - timedelta(minutes=10)).isoformat(),
        },
        eligible_for_approval=False,
        notes="",
        started_at=started.isoformat(),
        completed_at=None,
        min_win_rate=0.30,
        min_profit_factor=0.95,
        max_daily_loss_pct=0.03,
    )
    defaults.update(overrides)
    return ProbationStatus(**defaults)


def _seed_flags(monkeypatch, source: str) -> None:
    monkeypatch.setattr(pm, "get_feed_chain", lambda: SimpleNamespace(active_source=source))


class _RecordingBus:
    def __init__(self):
        self.published = []

    async def publish(self, event):
        self.published.append(event)

    async def subscribe(self, *_a, **_k):
        return None

    async def unsubscribe(self, *_a, **_k):
        return None


class _FakeStore:
    """Read-only fake; mutating methods explode if ever called."""

    def __init__(self, rows):
        self._rows = rows

    async def list(self, status=None):
        if status == "active":
            return [r for r in self._rows if r.status == "active"]
        return list(self._rows)

    async def get(self, version_id):
        for r in self._rows:
            if r.version_id == version_id:
                return r
        return None

    async def update(self, *a, **k):
        raise AssertionError("monitor must NEVER update probation state")

    async def approve(self, *a, **k):
        raise AssertionError("monitor must NEVER approve")

    async def reject(self, *a, **k):
        raise AssertionError("monitor must NEVER reject")


# ---------------------------------------------------------------------------
# Flag determinism
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_healthy_row_has_no_flags_and_ok_status():
    m = ProbationMonitor(store=_FakeStore([_prob()]))
    reports = await m.monitor_once(publish=False)
    r = reports[0]
    assert r["monitor_status"] == "ok"
    assert r["flags"] == []
    assert r["accrual_rate_per_day"] == pytest.approx(round(10 / 7, 3))
    assert r["projected_trades"] == pytest.approx(round((10 / 7) * 14, 1))
    assert "10/20 trades" in r["notes"]


@pytest.mark.asyncio
async def test_quota_at_risk_when_projection_below_fraction():
    # 4 trades over 10 days → projected 5.6 < 20*0.85 → warn flag
    started = datetime.utcnow() - timedelta(days=10)
    prob = _prob(
        version_id="v-slow",
        observed_trades=4,
        days_elapsed=10,
        started_at=started.isoformat(),
        last_metrics={"_accrual_mode": "replay", "_replay_at": datetime.utcnow().isoformat()},
    )
    m = ProbationMonitor(store=_FakeStore([prob]))
    reports = await m.monitor_once(publish=False)
    assert "quota_at_risk" in reports[0]["flags"]
    assert reports[0]["monitor_status"] == "warn"


@pytest.mark.asyncio
async def test_window_ending_flag():
    # 13/14 days elapsed with 10/20 trades → window ending, quota at risk
    started = datetime.utcnow() - timedelta(days=13)
    prob = _prob(
        version_id="v-late",
        observed_trades=10,
        days_elapsed=13,
        started_at=started.isoformat(),
        last_metrics={"_accrual_mode": "replay", "_replay_at": datetime.utcnow().isoformat()},
    )
    m = ProbationMonitor(store=_FakeStore([prob]))
    reports = await m.monitor_once(publish=False)
    flags = reports[0]["flags"]
    assert "window_ending" in flags
    assert "quota_at_risk" in flags


@pytest.mark.asyncio
async def test_gate_floor_flags():
    started = datetime.utcnow() - timedelta(days=5)
    prob = _prob(
        version_id="v-bad",
        observed_trades=10,
        days_elapsed=5,
        started_at=started.isoformat(),
        win_rate=0.1,      # < 0.30 floor
        profit_factor=0.5,  # < 0.95 floor
        worst_daily_loss_pct=0.05,  # > 0.03 ceiling
        last_metrics={"_accrual_mode": "replay", "_replay_at": datetime.utcnow().isoformat()},
    )
    m = ProbationMonitor(store=_FakeStore([prob]))
    r = (await m.monitor_once(publish=False))[0]
    assert "win_rate_below_floor" in r["flags"]
    assert "profit_factor_below_floor" in r["flags"]
    assert "worst_daily_loss_breach" in r["flags"]
    assert r["monitor_status"] == "critical"


@pytest.mark.asyncio
async def test_rate_floors_skipped_below_min_sample():
    started = datetime.utcnow() - timedelta(days=2)
    prob = _prob(
        version_id="v-thin",
        observed_trades=3,
        days_elapsed=2,
        started_at=started.isoformat(),
        win_rate=0.0,
        profit_factor=0.0,
        last_metrics={"_accrual_mode": "replay", "_replay_at": datetime.utcnow().isoformat()},
    )
    m = ProbationMonitor(store=_FakeStore([prob]))
    r = (await m.monitor_once(publish=False))[0]
    assert "win_rate_below_floor" not in r["flags"]
    assert "profit_factor_below_floor" not in r["flags"]


@pytest.mark.asyncio
async def test_infrastructure_flags_event_and_replay(monkeypatch):
    _seed_flags(monkeypatch, "mt5")
    # Stale event stream (> 45 min since _event_through)
    stale = _prob(
        version_id="v-stale",
        last_metrics={
            "_accrual_mode": "event",
            "_event_through": (datetime.utcnow() - timedelta(hours=3)).isoformat(),
        },
    )
    # Fresh event stream
    fresh = _prob(version_id="v-fresh")
    # Stale replay (> 45 min since _replay_at)
    replay_stale = _prob(
        version_id="v-replay-stale",
        last_metrics={
            "_accrual_mode": "replay",
            "_replay_at": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
        },
    )
    # No accrual stamp and old row
    old_no_stamp = _prob(
        version_id="v-nostamp",
        started_at=(datetime.utcnow() - timedelta(hours=5)).isoformat(),
        last_metrics={},
    )
    m = ProbationMonitor(store=_FakeStore([stale, fresh, replay_stale, old_no_stamp]))
    reports = {r["version_id"]: r for r in await m.monitor_once(publish=False)}
    assert "stream_stale" in reports["v-stale"]["flags"]
    assert "stream_stale" not in reports["v-fresh"]["flags"]
    assert "accrual_stale" in reports["v-replay-stale"]["flags"]
    assert "no_accrual_yet" in reports["v-nostamp"]["flags"]


@pytest.mark.asyncio
async def test_mock_feed_flag_is_critical(monkeypatch):
    _seed_flags(monkeypatch, "mock")
    m = ProbationMonitor(store=_FakeStore([_prob()]))
    r = (await m.monitor_once(publish=False))[0]
    assert "mock_feed" in r["flags"]
    assert r["monitor_status"] == "critical"


# ---------------------------------------------------------------------------
# Emission: only on change
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emits_on_change_only(monkeypatch):
    _seed_flags(monkeypatch, "mt5")
    bus = _RecordingBus()
    m = ProbationMonitor(store=_FakeStore([_prob()]), event_bus=bus)

    await m.monitor_once()  # first pass → emit
    assert len(bus.published) == 1
    ev = bus.published[0]
    assert ev.event_type == PROBATION_MONITOR_EVENT
    assert ev.aggregate_id == "v-mon-1"
    assert ev.payload["monitor_status"] == "ok"

    await m.monitor_once()  # unchanged flags → no event
    await m.monitor_once()
    assert len(bus.published) == 1

    # Flags change → exactly one more event
    prob = _prob(last_metrics={"_accrual_mode": "event", "_event_through": (datetime.utcnow() - timedelta(hours=2)).isoformat()})
    m.store = _FakeStore([prob])
    await m.monitor_once()
    assert len(bus.published) == 2
    assert bus.published[-1].payload["flags"] == ["stream_stale"]


@pytest.mark.asyncio
async def test_publish_false_never_emits(monkeypatch):
    _seed_flags(monkeypatch, "mt5")
    bus = _RecordingBus()
    m = ProbationMonitor(store=_FakeStore([_prob()]), event_bus=bus)
    await m.monitor_once(publish=False)
    assert bus.published == []


# ---------------------------------------------------------------------------
# Observe-only constitution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monitor_never_mutates_state(monkeypatch):
    _seed_flags(monkeypatch, "mt5")
    store = _FakeStore([_prob()])
    m = ProbationMonitor(store=store, event_bus=_RecordingBus())
    await m.monitor_once()  # would explode on update/approve/reject
    await m.monitor_once(publish=False)


def test_monitor_source_scan_no_mutation_calls():
    """Source scan: the monitor module must never touch mutating store APIs."""
    src = inspect.getsource(pm)
    for banned in ("place_order", "execute_trade", "risk_manager", ".approve(", ".reject(", ".update("):
        assert banned not in src, f"banned token in monitor source: {banned}"


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monitor_tool_snapshot(monkeypatch):
    _seed_flags(monkeypatch, "mt5")
    from forex_ai_agent.agent import tools as tools_mod

    async def _rows(status=None):
        rows = [_prob(), _prob(version_id="v-mon-2", observed_trades=2)]
        if status == "active":
            return [r for r in rows if r.status == "active"]
        return rows

    async def _explode(*_a, **_k):
        raise AssertionError("tool must not mutate")

    fake_store = SimpleNamespace(list=_rows, update=_explode, approve=_explode, reject=_explode)
    monkeypatch.setattr(pm, "ProbationStore", lambda: fake_store)

    from forex_ai_agent.agent.tools import ProbationMonitorArgs, probation_monitor_tool

    result = await probation_monitor_tool(ProbationMonitorArgs())
    assert result.summary["active"] == 2
    assert len(result.probations) == 2
    assert "v-mon-1" in result.rendered
    assert result.summary["ok"] + result.summary["warn"] + result.summary["critical"] == 2


def test_registry_has_monitor_tool_at_level_0():
    from forex_ai_agent.agent.tools import get_tool_registry

    reg = get_tool_registry(0)
    assert "monitor_probation_health" in reg._tools
    tool = reg._tools["monitor_probation_health"]
    assert tool.min_permission_level == 0
    assert tool.name in reg.get_schemas_for_prompt(min_level=0)


# ---------------------------------------------------------------------------
# Renderer + summary
# ---------------------------------------------------------------------------


def test_render_reports_and_summary():
    reports = [
        {"version_id": "a", "strategy_name": "S1", "pair": "EUR/USD", "monitor_status": "ok", "notes": "fine"},
        {"version_id": "b", "strategy_name": "S2", "pair": "GBP/USD", "monitor_status": "critical", "notes": "bad"},
    ]
    text = ProbationMonitor.render_reports(reports)
    assert "a (S1, EUR/USD): OK — fine" in text
    assert "b (S2, GBP/USD): CRITICAL — bad" in text
    summary = ProbationMonitor.summarize(reports)
    assert summary == {"active": 2, "ok": 1, "warn": 0, "critical": 1}
