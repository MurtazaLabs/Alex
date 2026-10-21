"""Stage E4 window tests: pinning, fail-closed eligibility, runner gates."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from forex_ai_agent.backtest.engine import BacktestEngine, BarData
from forex_ai_agent.brokers.base import OrderAck
from forex_ai_agent.execution.governor import ExecutionGovernor, GovernorLimits
from forex_ai_agent.execution.live.risk_state import iso_week_key
from forex_ai_agent.execution.live.signoff import (
    SCOPE_WINDOW,
    LiveSignoff,
    SignoffStore,
)
from forex_ai_agent.execution.live.transmitter import (
    TRANSMIT_FILLED,
    LiveTransmitter,
)
from forex_ai_agent.execution.live.window import (
    WindowStore,
    check_eligibility,
    start_window,
    stop_window,
)
from forex_ai_agent.execution.live.window_runner import (
    LiveWindowRunner,
    indicator_snapshot,
)
from forex_ai_agent.execution.policy import ExecutionPolicy, PolicyCaps
from forex_ai_agent.execution.tokens import ApprovalTokenStore
from forex_ai_agent.strategy.versioning import (
    StrategyVersionManager,
    spec_hash,
)

MONDAY_10 = datetime(2026, 10, 19, 10, 0)
VERSION_ID = "3f425957"
PAIR = "EUR/USD"

RULES = {
    "name": "MeanRevBB Short EUR/USD",
    "description": "test spec",
    "direction": "short",
    "timeframe": "H1",
    "entry_conditions": [
        {"left": "price", "verb": "above", "right": "bb_upper"},
        {"left": "rsi", "verb": "above", "right": "70"},
        {"left": "atr", "verb": "above", "right": "0.0002"},
    ],
    "atr_sl_mult": 1.5,
    "atr_tp_mult": 2.0,
    "min_score_pct": 0.6,
}


def _write_version(versions_dir: Path, *, approved: bool = True) -> Path:
    rules = json.loads(json.dumps(RULES))
    payload = {
        "id": VERSION_ID,
        "version": "0.1.0",
        "version_type": "minor",
        "parent_id": None,
        "name": RULES["name"],
        "description": RULES["description"],
        "rules": rules,
        "parameters": {"atr_sl_mult": 1.5, "atr_tp_mult": 2.0},
        "children_ids": [],
        "tags": [],
        "created_at": "2026-10-17T08:46:45.609494",
        "updated_at": "2026-10-17T08:46:45.609494",
        "created_by": "system",
        "status": "active" if approved else "draft",
        "backtest_results": {},
        "live_results": {},
        "approval": (
            {
                "approved_by": "murtaza",
                "approved_at": "2026-10-17T10:22:06",
                "spec_hash": spec_hash(rules, {"atr_sl_mult": 1.5, "atr_tp_mult": 2.0}),
                "prior_status": "draft",
            }
            if approved
            else {}
        ),
    }
    versions_dir.mkdir(parents=True, exist_ok=True)
    path = versions_dir / f"{VERSION_ID}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_drill(drill_dir: Path, *, phase: str = "executed", result: str = "PASS"):
    drill_dir.mkdir(parents=True, exist_ok=True)
    (drill_dir / "e4_live_verify_20261019_100000.json").write_text(
        json.dumps({
            "script": "e4_live_verify", "result": result,
            "order_phase": phase, "wire_attempts": 1 if phase == "executed" else 0,
        }),
        encoding="utf-8",
    )


def _write_signoff(signoff_path: Path, *, now: datetime, strategy_id=VERSION_ID):
    signoff_path.parent.mkdir(parents=True, exist_ok=True)
    SignoffStore(signoff_path).record(LiveSignoff(
        approver="murtaza",
        scope=SCOPE_WINDOW,
        pair=PAIR,
        strategy_id=strategy_id,
        declaration="test declaration",
        granted_at=now - timedelta(hours=1),
        expires_at=now + timedelta(days=90),
    ))


class _ProbationStub:
    def __init__(self, status="approved"):
        self.status = status

    async def get(self, version_id):
        return SimpleNamespace(
            version_id=version_id, status=self.status,
            observed_trades=5, required_trades=5,
            observed_days=24, required_days=24,
        )


def _env(tmp_path: Path, *, now: datetime = MONDAY_10):
    versions = tmp_path / "versions"
    drill = tmp_path / "drill"
    signoff = tmp_path / "signoffs.jsonl"
    state = tmp_path / "window"
    return versions, drill, signoff, state


async def _eligible_setup(tmp_path: Path, *, now: datetime = MONDAY_10):
    versions, drill, signoff, state = _env(tmp_path, now=now)
    _write_version(versions)
    _write_drill(drill)
    _write_signoff(signoff, now=now)
    return versions, drill, signoff, state


class TestSpecHash:
    def test_stable_and_sensitive(self):
        a = spec_hash({"x": 1, "y": [1, 2]}, {"p": 0.5})
        b = spec_hash({"y": [1, 2], "x": 1}, {"p": 0.5})
        c = spec_hash({"x": 1, "y": [1, 3]}, {"p": 0.5})
        assert a == b
        assert a != c

    def test_manager_approve_pins_and_tamper_breaks(self, tmp_path):
        async def scenario():
            manager = StrategyVersionManager(str(tmp_path / "versions"))
            version = manager.create_version(
                "T", rules=json.loads(json.dumps(RULES)), parameters={"p": 1.0},
            )
            await asyncio.sleep(0)
            assert manager.verify_pin(version.id) is False
            manager.approve(version.id, approver="murtaza")
            await asyncio.sleep(0)
            assert manager.verify_pin(version.id) is True
            version.rules["atr_sl_mult"] = 9.9
            manager.update_version(version)
            assert manager.verify_pin(version.id) is False

        asyncio.run(scenario())


class TestEligibility:
    def test_all_checks_fail_on_empty_state(self, tmp_path):
        async def scenario():
            versions, drill, signoff, state = _env(tmp_path)
            eligibility = await check_eligibility(
                version_id=VERSION_ID, pair=PAIR,
                versions_dir=versions, signoff_path=signoff, drill_dir=drill,
                probation_store=_ProbationStub(status="active"), now=MONDAY_10,
            )
            assert eligibility.all_met is False
            assert {c.name for c in eligibility.checks} == {
                "version_pinned", "probation_approved",
                "drill_verified", "window_signoff",
            }
            assert all(not c.ok for c in eligibility.checks)

        asyncio.run(scenario())

    def test_full_pass(self, tmp_path):
        async def scenario():
            versions, drill, signoff, _state = await _eligible_setup(tmp_path)
            eligibility = await check_eligibility(
                version_id=VERSION_ID, pair=PAIR,
                versions_dir=versions, signoff_path=signoff, drill_dir=drill,
                probation_store=_ProbationStub(), now=MONDAY_10,
            )
            assert eligibility.all_met, eligibility.to_dict()

        asyncio.run(scenario())

    def test_probation_passed_is_not_enough(self, tmp_path):
        async def scenario():
            versions, drill, signoff, _state = await _eligible_setup(tmp_path)
            eligibility = await check_eligibility(
                version_id=VERSION_ID, pair=PAIR,
                versions_dir=versions, signoff_path=signoff, drill_dir=drill,
                probation_store=_ProbationStub(status="passed"), now=MONDAY_10,
            )
            check = next(c for c in eligibility.checks
                         if c.name == "probation_approved")
            assert check.ok is False
            assert "awaiting human approval" in check.detail

        asyncio.run(scenario())

    def test_deferred_drill_blocks(self, tmp_path):
        async def scenario():
            versions, drill, signoff, _state = await _eligible_setup(tmp_path)
            _write_drill(drill, phase="deferred")
            eligibility = await check_eligibility(
                version_id=VERSION_ID, pair=PAIR,
                versions_dir=versions, signoff_path=signoff, drill_dir=drill,
                probation_store=_ProbationStub(), now=MONDAY_10,
            )
            assert eligibility.all_met is False

        asyncio.run(scenario())

    def test_signoff_for_other_version_blocks(self, tmp_path):
        async def scenario():
            versions, drill, signoff, _state = _env(tmp_path)
            _write_version(versions)
            _write_drill(drill)
            _write_signoff(signoff, now=MONDAY_10, strategy_id="deadbeef")
            eligibility = await check_eligibility(
                version_id=VERSION_ID, pair=PAIR,
                versions_dir=versions, signoff_path=signoff, drill_dir=drill,
                probation_store=_ProbationStub(), now=MONDAY_10,
            )
            assert eligibility.all_met is False

        asyncio.run(scenario())


class TestWindowLifecycle:
    def test_start_refused_writes_nothing(self, tmp_path):
        async def scenario():
            versions, drill, signoff, state = _env(tmp_path)
            ok, payload = await start_window(
                version_id=VERSION_ID, pair=PAIR, state_dir=state,
                versions_dir=versions, signoff_path=signoff, drill_dir=drill,
                probation_store=_ProbationStub(), now=MONDAY_10,
            )
            assert ok is False
            assert payload.all_met is False
            assert WindowStore(state).load_state() is None
            assert not (state / "window_events.jsonl").exists()

        asyncio.run(scenario())

    def test_start_success_then_expiry_and_double_start(self, tmp_path):
        async def scenario():
            versions, drill, signoff, state = await _eligible_setup(tmp_path)
            ok, window = await start_window(
                version_id=VERSION_ID, pair=PAIR, state_dir=state,
                versions_dir=versions, signoff_path=signoff, drill_dir=drill,
                probation_store=_ProbationStub(), now=MONDAY_10,
            )
            assert ok is True
            assert window.is_active(MONDAY_10)
            assert window.is_active(MONDAY_10 + timedelta(days=29))
            assert not window.is_active(MONDAY_10 + timedelta(days=30))
            stored = WindowStore(state).load_state()
            assert stored is not None and stored.window_id == window.window_id
            events = WindowStore(state).events()
            assert events and events[-1]["event"] == "window_started"

            ok2, payload2 = await start_window(
                version_id=VERSION_ID, pair=PAIR, state_dir=state,
                versions_dir=versions, signoff_path=signoff, drill_dir=drill,
                probation_store=_ProbationStub(), now=MONDAY_10,
            )
            assert ok2 is False
            assert any(c.name == "no_active_window" for c in payload2.checks)

        asyncio.run(scenario())

    def test_stop_is_recorded_and_idempotent(self, tmp_path):
        async def scenario():
            versions, drill, signoff, state = await _eligible_setup(tmp_path)
            await start_window(
                version_id=VERSION_ID, pair=PAIR, state_dir=state,
                versions_dir=versions, signoff_path=signoff, drill_dir=drill,
                probation_store=_ProbationStub(), now=MONDAY_10,
            )
            stopped = stop_window(state_dir=state, reason="operator stop",
                                  now=MONDAY_10 + timedelta(hours=1))
            assert stopped is not None and stopped.status == "stopped"
            again = stop_window(state_dir=state, reason="again", now=MONDAY_10)
            assert again.status == "stopped"
            events = WindowStore(state).events()
            assert [e["event"] for e in events][-1] == "window_stopped"

        asyncio.run(scenario())


class TestIndicatorParity:
    def test_snapshot_matches_backtest_engine(self):
        import numpy as np
        rng = np.random.default_rng(7)
        n = 250
        steps = rng.normal(0, 0.0006, n)
        closes = 1.10 + np.cumsum(steps)
        highs = closes + 0.0002
        lows = closes - 0.0002
        import pandas as pd
        df = pd.DataFrame({
            "open": closes, "high": highs, "low": lows,
            "close": closes, "volume": 1.0,
        })

        engine = BacktestEngine.__new__(BacktestEngine)
        engine.data_feed = SimpleNamespace(
            data={"EUR/USD": df}, _current_idx={"EUR/USD": len(df) - 1},
        )
        engine.current_bars = {"EUR/USD": BarData(
            timestamp=MONDAY_10, open=float(closes[-1]), high=float(highs[-1]),
            low=float(lows[-1]), close=float(closes[-1]),
        )}
        engine.indicators = {}
        engine._calculate_indicators()

        live = indicator_snapshot(df)
        assert live is not None
        for key, value in engine.indicators["EUR/USD"].items():
            assert live[key] == pytest.approx(float(value), rel=1e-12), key


def _trend_bars() -> list:
    bars = []
    t = MONDAY_10 - timedelta(hours=120)
    price = 1.10000
    for i in range(100):
        bars.append(BarData(timestamp=t + timedelta(hours=i), open=price,
                            high=price + 0.0002, low=price - 0.0002,
                            close=price))
    for i in range(19):
        price += 0.0005
        bars.append(BarData(timestamp=t + timedelta(hours=100 + i), open=price - 0.0005,
                            high=price + 0.0002, low=price - 0.0007, close=price))
    last = bars[-1].close + 0.0030
    bars.append(BarData(timestamp=t + timedelta(hours=119), open=bars[-1].close,
                        high=last + 0.0002, low=bars[-1].close - 0.0002,
                        close=last))
    return bars


class _StubAdapter:
    def __init__(self, equity: float = 10000.0):
        self.permission_level = 4
        self.shadow_mode = False
        self.equity = float(equity)
        self.requests = []

    def account(self):
        return SimpleNamespace(equity=self.equity, balance=self.equity,
                               login=1, server="test", currency="USD")

    def positions(self):
        return []

    def submit_order(self, request):
        self.requests.append(request)
        return OrderAck(sent=True, status="FILLED", retcode=10009, ticket=777)


class _StubFeed:
    def __init__(self, bars):
        self.bars = bars

    async def fetch_candles(self, pair, timeframe, start, end=None, limit=0):
        return self.bars


def _runner(tmp_path, *, now=MONDAY_10, start=True):
    versions, drill, signoff, state = _env(tmp_path, now=now)
    return versions, drill, signoff, state


async def _running_runner(tmp_path, adapter, prestate=None):
    """Eligible window + started + a runner primed for one live cycle.

    ``prestate`` (optional dict) is written to risk_baselines.json BEFORE
    the runner constructs its RiskBaselineStore, so the store loads it.
    """
    from forex_ai_agent.execution.live.tracker import PositionTracker
    versions, drill, signoff, state = await _eligible_setup(tmp_path)
    if prestate is not None:
        state.mkdir(parents=True, exist_ok=True)
        (state / "risk_baselines.json").write_text(
            json.dumps(prestate), encoding="utf-8",
        )
    governor = ExecutionGovernor(GovernorLimits(
        daily_loss_limit=100.0, max_positions=3,
    ))
    governor.arm()
    runner = LiveWindowRunner(
        adapter=adapter, feed=_StubFeed(_trend_bars()),
        tracker=PositionTracker(symbol=PAIR),
        policy=ExecutionPolicy(PolicyCaps(
            min_lot_size=0.01, max_lot_size=0.01,
        )),
        governor=governor,
        tokens=ApprovalTokenStore(), version_id=VERSION_ID,
        state_dir=state, versions_dir=versions,
        signoff_path=signoff, drill_dir=drill,
    )
    ok, window = await start_window(
        version_id=VERSION_ID, pair=PAIR, state_dir=state,
        versions_dir=versions, signoff_path=signoff, drill_dir=drill,
        probation_store=_ProbationStub(), now=MONDAY_10,
    )
    assert ok, window
    return runner, state


async def _prepare(versions, drill, signoff, state, adapter):
    _write_version(versions)
    _write_drill(drill)
    _write_signoff(signoff, now=MONDAY_10)
    ok, window = await start_window(
        version_id=VERSION_ID, pair=PAIR, state_dir=state,
        versions_dir=versions, signoff_path=signoff, drill_dir=drill,
        probation_store=_ProbationStub(), now=MONDAY_10,
    )
    assert ok, window
    governor = ExecutionGovernor(GovernorLimits(
        daily_loss_limit=15.0, max_positions=3,
    ))
    governor.arm()
    tokens = ApprovalTokenStore()
    runner = LiveWindowRunner(
        adapter=adapter, feed=_StubFeed(_trend_bars()), tracker=None,
        policy=ExecutionPolicy(PolicyCaps()), governor=governor,
        tokens=tokens, version_id=VERSION_ID,
        state_dir=state, versions_dir=versions, signoff_path=signoff,
        drill_dir=drill,
    )
    return runner, tokens, state


class TestRunner:
    def test_blocks_when_window_inactive(self, tmp_path):
        async def scenario():
            versions, drill, signoff, state = _env(tmp_path)
            adapter = _StubAdapter()
            governor = ExecutionGovernor(GovernorLimits())
            runner = LiveWindowRunner(
                adapter=adapter, feed=_StubFeed(_trend_bars()), tracker=None,
                policy=ExecutionPolicy(PolicyCaps()), governor=governor,
                tokens=ApprovalTokenStore(), version_id=VERSION_ID,
                state_dir=state, versions_dir=versions,
                signoff_path=signoff, drill_dir=drill,
            )
            summary = await runner.run_once(now=MONDAY_10)
            assert summary["status"] == "blocked"
            assert adapter.requests == []

        asyncio.run(scenario())

    def test_produces_intent_but_never_sends_without_approval(self, tmp_path):
        async def scenario():
            versions, drill, signoff, state = _env(tmp_path)
            adapter = _StubAdapter()
            from forex_ai_agent.execution.live.tracker import PositionTracker
            governor = ExecutionGovernor(GovernorLimits(
                daily_loss_limit=15.0, max_positions=3,
            ))
            governor.arm()
            runner = LiveWindowRunner(
                adapter=adapter, feed=_StubFeed(_trend_bars()),
                tracker=PositionTracker(symbol=PAIR),
                policy=ExecutionPolicy(PolicyCaps(
                    min_lot_size=0.01, max_lot_size=0.01,
                )),
                governor=governor,
                tokens=ApprovalTokenStore(), version_id=VERSION_ID,
                state_dir=state, versions_dir=versions,
                signoff_path=signoff, drill_dir=drill,
            )
            await _eligible_setup(tmp_path)
            ok, _ = await start_window(
                version_id=VERSION_ID, pair=PAIR, state_dir=state,
                versions_dir=versions, signoff_path=signoff, drill_dir=drill,
                probation_store=_ProbationStub(), now=MONDAY_10,
            )
            assert ok
            summary = await runner.run_once(now=MONDAY_10)
            assert summary["status"] == "ok", summary
            assert summary["signals"] >= 1
            assert summary["intents_valid"] == 1, summary
            assert adapter.requests == []

            records = [
                json.loads(line) for line in
                (state / "intents.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert len(records) == 1
            assert records[0]["side"] == "SELL"
            assert records[0]["status"] == "VALID"

        asyncio.run(scenario())

    def test_full_dispatch_after_human_approval(self, tmp_path):
        async def scenario():
            _write_version(tmp_path / "versions")
            _write_drill(tmp_path / "drill")
            _write_signoff(tmp_path / "signoffs.jsonl", now=MONDAY_10)
            adapter = _StubAdapter()
            from forex_ai_agent.execution.live.tracker import PositionTracker
            governor = ExecutionGovernor(GovernorLimits(
                daily_loss_limit=15.0, max_positions=3,
            ))
            governor.arm()
            runner = LiveWindowRunner(
                adapter=adapter, feed=_StubFeed(_trend_bars()),
                tracker=PositionTracker(symbol=PAIR),
                policy=ExecutionPolicy(PolicyCaps(
                    min_lot_size=0.01, max_lot_size=0.01,
                )),
                governor=governor,
                tokens=ApprovalTokenStore(), version_id=VERSION_ID,
                state_dir=tmp_path / "window",
                versions_dir=tmp_path / "versions",
                signoff_path=tmp_path / "signoffs.jsonl",
                drill_dir=tmp_path / "drill",
            )
            ok, _ = await start_window(
                version_id=VERSION_ID, pair=PAIR,
                state_dir=tmp_path / "window",
                versions_dir=tmp_path / "versions",
                signoff_path=tmp_path / "signoffs.jsonl",
                drill_dir=tmp_path / "drill",
                probation_store=_ProbationStub(), now=MONDAY_10,
            )
            assert ok
            first = await runner.run_once(now=MONDAY_10)
            assert first["intents_valid"] == 1, first
            assert adapter.requests == []

            intent_key = json.loads(
                (tmp_path / "window" / "intents.jsonl")
                .read_text(encoding="utf-8").splitlines()[0]
            )["key"]
            with (tmp_path / "window" / "approvals.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps({
                    "approval_id": "appr1", "intent_key": intent_key,
                    "approver": "murtaza", "approved_at": MONDAY_10.isoformat(),
                }) + "\n")

            second = await runner.run_once(now=MONDAY_10 + timedelta(seconds=30))
            assert second["dispatched"] == 1, second
            assert len(adapter.requests) == 1
            wire = adapter.requests[0]
            assert wire.side == "SELL" and wire.symbol == PAIR
            transmissions = (tmp_path / "window" / "transmissions.jsonl") \
                .read_text(encoding="utf-8").splitlines()
            assert json.loads(transmissions[0])["status"] == TRANSMIT_FILLED

            third = await runner.run_once(now=MONDAY_10 + timedelta(seconds=60))
            assert third["dispatched"] == 0
            assert len(adapter.requests) == 1

        asyncio.run(scenario())

    def test_policy_sizing_outside_window_lots_is_refused(self, tmp_path):
        async def scenario():
            versions, drill, signoff, state = await _eligible_setup(tmp_path)
            adapter = _StubAdapter()
            from forex_ai_agent.execution.live.tracker import PositionTracker
            governor = ExecutionGovernor(GovernorLimits(
                daily_loss_limit=15.0, max_positions=3,
            ))
            governor.arm()
            runner = LiveWindowRunner(
                adapter=adapter, feed=_StubFeed(_trend_bars()),
                tracker=PositionTracker(symbol=PAIR),
                policy=ExecutionPolicy(PolicyCaps()),
                governor=governor,
                tokens=ApprovalTokenStore(), version_id=VERSION_ID,
                state_dir=state, versions_dir=versions,
                signoff_path=signoff, drill_dir=drill,
            )
            ok, _ = await start_window(
                version_id=VERSION_ID, pair=PAIR, state_dir=state,
                versions_dir=versions, signoff_path=signoff, drill_dir=drill,
                probation_store=_ProbationStub(), now=MONDAY_10,
            )
            assert ok
            summary = await runner.run_once(now=MONDAY_10)
            assert summary["intents_valid"] == 0
            assert summary["intents_refused"] >= 1
            records = [
                json.loads(line) for line in
                (state / "intents.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert records[0]["status"] == "REFUSED_WINDOW_LOT_CAP"
            assert adapter.requests == []

        asyncio.run(scenario())

    def test_pin_tamper_blocks_the_cycle(self, tmp_path):
        async def scenario():
            _write_version(tmp_path / "versions")
            _write_drill(tmp_path / "drill")
            _write_signoff(tmp_path / "signoffs.jsonl", now=MONDAY_10)
            adapter = _StubAdapter()
            from forex_ai_agent.execution.live.tracker import PositionTracker
            governor = ExecutionGovernor(GovernorLimits(
                daily_loss_limit=15.0, max_positions=3,
            ))
            governor.arm()
            runner = LiveWindowRunner(
                adapter=adapter, feed=_StubFeed(_trend_bars()),
                tracker=PositionTracker(symbol=PAIR),
                policy=ExecutionPolicy(PolicyCaps()), governor=governor,
                tokens=ApprovalTokenStore(), version_id=VERSION_ID,
                state_dir=tmp_path / "window",
                versions_dir=tmp_path / "versions",
                signoff_path=tmp_path / "signoffs.jsonl",
                drill_dir=tmp_path / "drill",
            )
            await start_window(
                version_id=VERSION_ID, pair=PAIR,
                state_dir=tmp_path / "window",
                versions_dir=tmp_path / "versions",
                signoff_path=tmp_path / "signoffs.jsonl",
                drill_dir=tmp_path / "drill",
                probation_store=_ProbationStub(), now=MONDAY_10,
            )
            payload = json.loads(
                (tmp_path / "versions" / f"{VERSION_ID}.json")
                .read_text(encoding="utf-8")
            )
            payload["rules"]["atr_sl_mult"] = 9.9
            (tmp_path / "versions" / f"{VERSION_ID}.json").write_text(
                json.dumps(payload), encoding="utf-8",
            )
            summary = await runner.run_once(now=MONDAY_10)
            assert summary["status"] == "blocked"
            assert "pin" in summary["reason"]

        asyncio.run(scenario())


class TestMandateRiskGate:
    def test_daily_loss_trip_halts_the_cycle(self, tmp_path):
        async def scenario():
            adapter = _StubAdapter(equity=9900.0)
            runner, state = await _running_runner(tmp_path, adapter)
            # Baseline the day at $10000; the cycle then sees -1% ($9900)
            runner.risk_store.sync(10000.0, MONDAY_10 - timedelta(minutes=5))

            summary = await runner.run_once(now=MONDAY_10)
            assert summary["status"] == "halted", summary
            assert "daily" in summary["trips"]
            assert adapter.requests == []
            assert not (state / "intents.jsonl").exists()
            events = WindowStore(state).events()
            assert events[-1]["event"] == "cycle_halted"

            # Trip is persisted: the very next cycle halts without re-eval
            again = await runner.run_once(now=MONDAY_10 + timedelta(minutes=1))
            assert again["status"] == "halted"
            assert runner.risk_store.snapshot().tripped("daily")

        asyncio.run(scenario())

    def test_drawdown_trip_needs_manual_clear(self, tmp_path):
        async def scenario():
            adapter = _StubAdapter(equity=9400.0)
            # Baselines loaded by the store: day/week rooted at the current
            # equity, HWM at $10000 -> -6% drawdown, no daily/weekly loss.
            runner, _state = await _running_runner(tmp_path, adapter, prestate={
                "day_key": MONDAY_10.strftime("%Y-%m-%d"),
                "week_key": iso_week_key(MONDAY_10),
                "day_start_equity": 9400.0,
                "week_start_equity": 9400.0,
                "equity_hwm": 10000.0,
                "trips": {},
            })

            summary = await runner.run_once(now=MONDAY_10)
            assert summary["status"] == "halted", summary
            assert summary["trips"] == ["drawdown"]
            assert adapter.requests == []

            # No auto-expiry for drawdown: still halted later the same day
            again = await runner.run_once(now=MONDAY_10 + timedelta(hours=2))
            assert again["status"] == "halted"

            # Human clear rebases the HWM -> the very next cycle runs
            runner.risk_store.clear_trip("drawdown", reason="test review")
            fresh = await runner.run_once(now=MONDAY_10 + timedelta(hours=3))
            assert fresh["status"] == "ok", fresh

        asyncio.run(scenario())

    def test_daily_trip_auto_clears_next_day(self, tmp_path):
        async def scenario():
            adapter = _StubAdapter(equity=9900.0)
            runner, _state = await _running_runner(tmp_path, adapter)
            runner.risk_store.sync(10000.0, MONDAY_10 - timedelta(minutes=5))
            halted = await runner.run_once(now=MONDAY_10)
            assert halted["status"] == "halted"

            # Next UTC day: day baseline rebases at the fresh equity
            tomorrow = MONDAY_10 + timedelta(days=1)
            adapter.equity = 9900.0
            summary = await runner.run_once(now=tomorrow)
            assert summary["status"] == "ok", summary
            assert runner.risk_store.snapshot().tripped("daily") is False

        asyncio.run(scenario())
