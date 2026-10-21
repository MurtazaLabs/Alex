"""Mandate v1.0 enforcement tests (ALEX_TRADING_MANDATE.md, approved 2026-10-18).

Pins, end to end:

- the machine-readable mandate (risk_mandate.py) is the single source of
  truth, and PolicyCaps/LiveCaps mirror it;
- the policy enforces every limit at its EXACT boundary (0.25% per trade,
  0.75% aggregate stop risk, 1.0% daily, 2.0% weekly, 5.0% drawdown from
  the equity high-water mark) and fails closed on unverifiable open risk;
- the min-lot budget floor refuses instead of clamping size up;
- the governor gained an optional weekly breaker (legacy behavior intact);
- stop risk is computed from real position rows and returns None (fail
  closed) when any stop loss is missing;
- the runner-style sync (daily_pnl=0.0) does NOT double-count realized
  PnL against the mark-to-market equity, while the legacy call still does;
- RiskBaselineStore persists day/week/HWM baselines, auto-expires
  daily/weekly trips on key rollover, and only a human clear-trip ends a
  drawdown halt (rebasing the ruler).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from forex_ai_agent.execution.governor import (
    TRIP_DAILY_LOSS,
    TRIP_WEEKLY_LOSS,
    ExecutionGovernor,
    GovernorLimits,
)
from forex_ai_agent.execution.live.caps import DEFAULT_LIVE_CAPS
from forex_ai_agent.execution.live.risk_state import (
    RISK_TRIP_NAMES,
    RiskBaselineStore,
    iso_week_key,
)
from forex_ai_agent.execution.live.tracker import (
    PositionTracker,
    stop_risk_usd,
    sync_account_into,
)
from forex_ai_agent.execution.policy import (
    INTENT_REFUSED_AGGREGATE_RISK,
    INTENT_REFUSED_DAILY_LOSS,
    INTENT_REFUSED_DRAWDOWN,
    INTENT_REFUSED_RISK_BUDGET,
    INTENT_REFUSED_UNVERIFIED_RISK,
    INTENT_REFUSED_WEEKLY_LOSS,
    INTENT_VALID,
    ExecutionPolicy,
    PolicyCaps,
)
from forex_ai_agent.execution.risk_mandate import ACTIVE_MANDATE, MANDATE_V1

MONDAY_10 = datetime(2026, 10, 19, 10, 0)


def _intent(policy, equity=10000.0, entry=1.1000, sl=1.0980, tp=1.1040):
    return policy.create_intent(
        symbol="EUR/USD", side="BUY", entry=entry,
        stop_loss=sl, take_profit=tp, now=MONDAY_10,
    )


# ---------------------------------------------------------------------------
# Mandate source of truth
# ---------------------------------------------------------------------------

class TestMandateSourceOfTruth:
    def test_mandate_v1_values(self):
        assert MANDATE_V1.version == "1.0"
        assert MANDATE_V1.max_risk_per_trade_pct == 0.25
        assert MANDATE_V1.max_aggregate_stop_risk_pct == 0.75
        assert MANDATE_V1.max_daily_loss_pct == 1.0
        assert MANDATE_V1.max_weekly_loss_pct == 2.0
        assert MANDATE_V1.max_drawdown_pct == 5.0
        assert MANDATE_V1.max_concurrent_positions == 3

    def test_active_mandate_is_mandate_v1(self):
        assert ACTIVE_MANDATE.to_dict() == MANDATE_V1.to_dict()

    def test_policy_caps_mirror_mandate(self):
        caps = PolicyCaps()
        assert caps.max_risk_per_trade_pct == MANDATE_V1.max_risk_per_trade_pct
        assert (caps.max_aggregate_stop_risk_pct
                == MANDATE_V1.max_aggregate_stop_risk_pct)
        assert caps.max_daily_loss_pct == MANDATE_V1.max_daily_loss_pct
        assert caps.max_weekly_loss_pct == MANDATE_V1.max_weekly_loss_pct
        assert caps.max_drawdown_pct == MANDATE_V1.max_drawdown_pct
        assert (caps.max_concurrent_positions
                == MANDATE_V1.max_concurrent_positions)

    def test_live_caps_mirror_mandate(self):
        assert (DEFAULT_LIVE_CAPS.max_risk_per_trade_pct
                == MANDATE_V1.max_risk_per_trade_pct)
        assert (DEFAULT_LIVE_CAPS.max_aggregate_stop_risk_pct
                == MANDATE_V1.max_aggregate_stop_risk_pct)
        assert DEFAULT_LIVE_CAPS.max_daily_loss_pct == MANDATE_V1.max_daily_loss_pct
        assert (DEFAULT_LIVE_CAPS.max_weekly_loss_pct
                == MANDATE_V1.max_weekly_loss_pct)
        assert DEFAULT_LIVE_CAPS.max_drawdown_pct == MANDATE_V1.max_drawdown_pct

    def test_mandate_is_frozen(self):
        with pytest.raises(Exception):
            MANDATE_V1.max_risk_per_trade_pct = 9.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Policy boundaries (exact)
# ---------------------------------------------------------------------------

class TestPolicyBoundaries:
    def _policy(self, equity, **state):
        p = ExecutionPolicy(PolicyCaps())
        p.update_account_state(
            equity=equity, open_positions=state.pop("positions", 0),
            now=MONDAY_10, **state,
        )
        return p

    def test_budget_exactly_min_lot_is_valid(self):
        # equity 800: 0.25% = $2.00 budget; 20-pip SL needs $2.00 -> raw
        # is exactly 0.01 -> the boundary case that must PASS.
        p = self._policy(800.0)
        intent = _intent(p)
        assert intent.status == INTENT_VALID
        assert intent.volume == pytest.approx(0.01)
        assert intent.risk_usd == pytest.approx(2.00, abs=0.01)

    def test_budget_just_below_min_lot_refuses(self):
        p = self._policy(799.0)
        intent = _intent(p)
        assert intent.status == INTENT_REFUSED_RISK_BUDGET
        assert intent.volume == 0.0

    def test_aggregate_boundary_pass_and_refuse(self):
        # 0.75% of $10000 = $75 limit; projection = open + $25 budget
        p = self._policy(10000.0, positions=1, open_stop_risk_usd=49.99)
        assert _intent(p).status == INTENT_VALID
        p = self._policy(10000.0, positions=1, open_stop_risk_usd=50.01)
        assert _intent(p).status == INTENT_REFUSED_AGGREGATE_RISK

    def test_unverified_open_risk_fails_closed(self):
        p = self._policy(10000.0, positions=1)
        assert _intent(p).status == INTENT_REFUSED_UNVERIFIED_RISK

    def test_weekly_boundary_pass_and_refuse(self):
        # week started at $10000; 2% weekly halt
        p = self._policy(9801.0, week_start_equity=10000.0)
        assert _intent(p).status == INTENT_VALID
        p = self._policy(9800.0, week_start_equity=10000.0)
        assert _intent(p).status == INTENT_REFUSED_WEEKLY_LOSS

    def test_drawdown_boundary_pass_and_refuse(self):
        # HWM $10000; 5% drawdown halt
        p = self._policy(9501.0, equity_hwm=10000.0, day_start_equity=9501.0)
        assert _intent(p).status == INTENT_VALID
        p = self._policy(9500.0, equity_hwm=10000.0, day_start_equity=9500.0)
        assert _intent(p).status == INTENT_REFUSED_DRAWDOWN

    def test_drawdown_precedes_daily(self):
        # $9500 equity: -5% from HWM and -5% on the day -> DRAWDOWN first
        p = self._policy(
            9500.0, equity_hwm=10000.0, day_start_equity=10000.0,
        )
        assert _intent(p).status == INTENT_REFUSED_DRAWDOWN

    def test_daily_still_precedes_budget(self):
        # A $100 account: -$1.00 is 1% daily loss; the daily refusal comes
        # before the (also-failing) budget check.
        p = self._policy(100.0, day_start_equity=100.0)
        p.update_account_state(
            equity=100.0, open_positions=0, daily_pnl=-1.00, now=MONDAY_10,
        )
        assert _intent(p).status == INTENT_REFUSED_DAILY_LOSS


# ---------------------------------------------------------------------------
# Governor weekly breaker
# ---------------------------------------------------------------------------

class TestGovernorWeeklyBreaker:
    def _armed(self, **limits):
        g = ExecutionGovernor(GovernorLimits(**limits))
        g.arm()
        return g

    def test_weekly_branch_trips(self):
        g = self._armed(
            daily_loss_limit=500.0, max_drawdown=1000.0,
            weekly_loss_limit=200.0, max_positions=3,
        )
        g.check_breakers(
            equity=9800.0, day_start_equity=9800.0, peak_equity=10000.0,
            open_positions=0, now=MONDAY_10, week_start_equity=10000.0,
        )
        assert g.state.tripped
        assert g.trip_reason == TRIP_WEEKLY_LOSS

    def test_weekly_disabled_by_default(self):
        g = self._armed(daily_loss_limit=500.0, max_drawdown=1000.0)
        assert g.limits.weekly_loss_limit is None
        g.check_breakers(
            equity=9000.0, day_start_equity=9000.0, peak_equity=9000.0,
            open_positions=0, now=MONDAY_10, week_start_equity=10000.0,
        )
        assert g.state.tripped is False

    def test_daily_precedes_weekly(self):
        g = self._armed(
            daily_loss_limit=100.0, max_drawdown=1000.0,
            weekly_loss_limit=200.0, max_positions=3,
        )
        g.check_breakers(
            equity=9800.0, day_start_equity=10000.0, peak_equity=10000.0,
            open_positions=0, now=MONDAY_10, week_start_equity=10000.0,
        )
        assert g.trip_reason == TRIP_DAILY_LOSS


# ---------------------------------------------------------------------------
# stop_risk_usd
# ---------------------------------------------------------------------------

class TestStopRisk:
    def test_sums_pip_risk_across_rows(self):
        positions = [
            {"symbol": "EUR/USD", "price_open": 1.1000, "sl": 1.0950,
             "volume": 0.01},  # 50 pips x $10/pip x 0.01 = $5.00
            SimpleNamespace(symbol="EUR/USD", price_open=1.1000, sl=1.0980,
                            volume=0.02),  # 20 pips x $10 x 0.02 = $4.00
        ]
        assert stop_risk_usd(positions, symbol="EUR/USD") == pytest.approx(9.00)

    def test_missing_sl_returns_none(self):
        positions = [{"symbol": "EUR/USD", "price_open": 1.1000,
                      "sl": None, "volume": 0.01}]
        assert stop_risk_usd(positions, symbol="EUR/USD") is None

    def test_missing_entry_returns_none(self):
        positions = [{"symbol": "EUR/USD", "sl": 1.0980, "volume": 0.01}]
        assert stop_risk_usd(positions, symbol="EUR/USD") is None

    def test_empty_positions_is_zero(self):
        assert stop_risk_usd([], symbol="EUR/USD") == 0.0


# ---------------------------------------------------------------------------
# sync_account_into passthrough
# ---------------------------------------------------------------------------

class TestSyncPassthrough:
    def _account(self, equity=10000.0):
        return SimpleNamespace(equity=equity, balance=equity)

    def _tracker_with_loss(self, loss=-150.0):
        tracker = PositionTracker(symbol="EUR/USD")
        tracker.record_deal(profit=loss, ts=MONDAY_10, ticket=1,
                            symbol="EUR/USD", volume=0.01)
        return tracker

    def test_runner_sync_does_not_double_count_realized(self):
        # Equity already marks the $150 loss; the runner passes daily_pnl=0
        # and the persisted day-start, so the daily breaker sees 0%.
        policy = ExecutionPolicy()
        tracker = self._tracker_with_loss(-150.0)
        sync_account_into(
            tracker, self._account(equity=9850.0), policy=policy,
            daily_pnl=0.0, day_start_equity=9850.0,
            week_start_equity=9850.0, equity_hwm=10000.0,
            open_stop_risk_usd=0.0, now=MONDAY_10,
        )
        assert _intent(policy).status == INTENT_VALID

    def test_legacy_sync_still_counts_realized(self):
        # Legacy callers get realized_pnl_today as before: -$150 on a
        # $10000 day-start is 1.5% -> daily refusal.
        policy = ExecutionPolicy()
        tracker = self._tracker_with_loss(-150.0)
        sync_account_into(
            tracker, self._account(equity=9850.0), policy=policy,
            now=MONDAY_10,
        )
        assert _intent(policy).status == INTENT_REFUSED_DAILY_LOSS

    def test_weekly_baseline_reaches_policy(self):
        policy = ExecutionPolicy()
        tracker = PositionTracker(symbol="EUR/USD")
        sync_account_into(
            tracker, self._account(equity=9800.0), policy=policy,
            daily_pnl=0.0, day_start_equity=9800.0,
            week_start_equity=10000.0, equity_hwm=10000.0,
            open_stop_risk_usd=0.0, now=MONDAY_10,
        )
        assert _intent(policy).status == INTENT_REFUSED_WEEKLY_LOSS


# ---------------------------------------------------------------------------
# RiskBaselineStore
# ---------------------------------------------------------------------------

class TestRiskBaselineStore:
    def _store(self, tmp_path):
        return RiskBaselineStore(tmp_path / "risk_baselines.json")

    def test_init_and_persistence_roundtrip(self, tmp_path):
        store = self._store(tmp_path)
        snap = store.snapshot()
        assert snap.day_start_equity == 0.0
        assert snap.equity_hwm == 0.0
        store.sync(10000.0, MONDAY_10)
        reloaded = self._store(tmp_path)
        snap = reloaded.snapshot()
        assert snap.day_start_equity == 10000.0
        assert snap.week_start_equity == 10000.0
        assert snap.equity_hwm == 10000.0
        assert snap.day_key == "2026-10-19"

    def test_hwm_only_tracks_max(self, tmp_path):
        store = self._store(tmp_path)
        store.sync(10000.0, MONDAY_10)
        store.sync(9900.0, MONDAY_10 + timedelta(hours=1))
        snap = store.snapshot()
        assert snap.day_start_equity == 10000.0
        assert snap.equity_hwm == 10000.0
        store.sync(10100.0, MONDAY_10 + timedelta(hours=2))
        assert store.snapshot().equity_hwm == 10100.0

    def test_next_day_rebases_and_clears_daily_trip(self, tmp_path):
        store = self._store(tmp_path)
        store.sync(10000.0, MONDAY_10)
        store.trip("daily", now=MONDAY_10, detail={"loss_pct": 1.2})
        assert store.snapshot().tripped("daily")
        snap = store.sync(9900.0, MONDAY_10 + timedelta(days=1))
        assert snap.tripped("daily") is False
        assert snap.day_start_equity == 9900.0
        # weekly baseline survives the day rollover
        assert snap.week_start_equity == 10000.0

    def test_week_rollover_clears_weekly_trip(self, tmp_path):
        store = self._store(tmp_path)
        store.sync(10000.0, MONDAY_10)
        store.trip("weekly", now=MONDAY_10)
        next_monday = MONDAY_10 + timedelta(days=7)
        snap = store.sync(10200.0, next_monday)
        assert snap.tripped("weekly") is False
        assert snap.week_start_equity == 10200.0

    def test_drawdown_trip_persists_until_cleared(self, tmp_path):
        store = self._store(tmp_path)
        store.sync(10000.0, MONDAY_10)
        store.trip("drawdown", now=MONDAY_10, detail={"drawdown_pct": 5.5})
        snap = store.sync(9600.0, MONDAY_10 + timedelta(days=3))
        assert snap.tripped("drawdown")
        # a reload (process restart) keeps the halt
        reloaded = self._store(tmp_path)
        assert reloaded.snapshot().tripped("drawdown")

    def test_clear_trip_rebases_and_logs_events(self, tmp_path):
        store = self._store(tmp_path)
        store.sync(10000.0, MONDAY_10)
        store.trip("drawdown", now=MONDAY_10)
        snap = store.clear_trip(
            "drawdown", now=MONDAY_10 + timedelta(hours=1),
            by="operator", reason="postmortem complete",
        )
        assert snap.tripped("drawdown") is False
        # HWM rebased: the next sync re-roots it at the current equity
        snap = store.sync(9400.0, MONDAY_10 + timedelta(hours=2))
        assert snap.equity_hwm == 9400.0
        events = [
            json.loads(line) for line in
            store.events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        kinds = [e["event"] for e in events]
        assert "risk_tripped" in kinds
        assert "trip_cleared" in kinds
        cleared = [e for e in events if e["event"] == "trip_cleared"][0]
        assert cleared["trip"] == "drawdown"
        assert cleared["reason"] == "postmortem complete"
        assert cleared["by"] == "operator"

    def test_clear_daily_rebases_day_start(self, tmp_path):
        store = self._store(tmp_path)
        store.sync(10000.0, MONDAY_10)
        store.trip("daily", now=MONDAY_10)
        store.clear_trip("daily", now=MONDAY_10, reason="manual")
        snap = store.sync(9800.0, MONDAY_10 + timedelta(minutes=5))
        assert snap.tripped("daily") is False
        assert snap.day_start_equity == 9800.0

    def test_trip_is_idempotent(self, tmp_path):
        store = self._store(tmp_path)
        store.sync(10000.0, MONDAY_10)
        store.trip("weekly", now=MONDAY_10, detail={"first": True})
        first_at = store.snapshot().trips["weekly"]["tripped_at"]
        store.trip("weekly", now=MONDAY_10 + timedelta(hours=5),
                   detail={"second": True})
        again = store.snapshot().trips["weekly"]
        assert again["tripped_at"] == first_at
        assert "first" in again["detail"]

    def test_unknown_trip_name_raises(self, tmp_path):
        store = self._store(tmp_path)
        with pytest.raises(ValueError):
            store.trip("bogus", now=MONDAY_10)
        with pytest.raises(ValueError):
            store.clear_trip("bogus", now=MONDAY_10, reason="x")

    def test_corrupt_file_raises(self, tmp_path):
        path = tmp_path / "risk_baselines.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(RuntimeError):
            RiskBaselineStore(path)

    def test_iso_week_key_is_monday_based(self):
        assert iso_week_key(MONDAY_10) == "2026-W43"
        assert iso_week_key(datetime(2026, 10, 18, 12, 0)) == "2026-W42"
        assert iso_week_key(datetime(2026, 10, 25, 12, 0)) == "2026-W43"

    def test_trip_names_are_the_mandate_trio(self):
        assert RISK_TRIP_NAMES == ("daily", "weekly", "drawdown")
