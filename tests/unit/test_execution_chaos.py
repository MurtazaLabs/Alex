"""Unit tests for the Stage E1 execution governor + chaos/failure suite.

Every fault scenario pins the expected POST-STATE:

- kill switch: trip on EVERY breaker (daily loss, drawdown, position count,
  manual) ⇒ HALTED, pending intents cancelled, human-only re-arm;
- feed faults: stale data and clock regression refuse intents (no orders on
  old prices, no rewound clocks);
- duplicate intents: suppressed by idempotency key — one record, no
  double-fill;
- crash recovery: write-ahead journal snapshot/restore reconciles PENDING
  intents on restart (adopt-or-review, never blind re-send);
- broker fault table: every MT5 failure mode maps to its declared handler;
- governor in the simulator: trip mid-run cancels deferred intents and
  refuses new ones; governed clean-params runs stay fill-identical to the
  ungoverned clean run when nothing trips;
- constitution: pure state transitions — no network, no broker calls.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from forex_ai_agent.autonomy.validator import SpecSignalGenerator
from forex_ai_agent.backtest.engine import BacktestConfig, run_backtest
from forex_ai_agent.execution.governor import (
    BROKER_FAULT_TABLE,
    TRIP_DAILY_LOSS,
    TRIP_DRAWDOWN,
    TRIP_MANUAL,
    TRIP_POSITION_COUNT,
    TRIP_POSITION_FLATTEN,
    TRIP_POSITION_HOLD,
    TRIP_POSITION_POLICIES,
    ExecutionGovernor,
    GovernorLimits,
)
from forex_ai_agent.execution.journal import (
    INTENT_CANCELLED,
    INTENT_EXECUTED,
    INTENT_PENDING,
    IntentJournal,
)
from forex_ai_agent.execution.simulator import ExecutionParams, run_simulation
from forex_ai_agent.strategy.spec import fallback_spec
from forex_ai_agent.utils.math_helpers import pip_size_for_symbol

SYMBOL = "EUR/USD"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trend_bars(n: int = 220, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = np.full(n, 0.00018) + rng.normal(0, 0.0004, n)
    close = 1.10 + np.cumsum(steps)
    idx = pd.date_range(datetime(2026, 8, 1), periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close + np.abs(rng.normal(0, 0.0003, n)),
            "low": close - np.abs(rng.normal(0, 0.0003, n)),
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )


def _config(df: pd.DataFrame, symbol: str = SYMBOL) -> BacktestConfig:
    return BacktestConfig(
        start_date=df.index.min().to_pydatetime(),
        end_date=df.index.max().to_pydatetime(),
        initial_balance=10_000.0,
        symbols=[symbol],
        timeframe="H1",
        warmup_bars=12,
    )


def _fresh_gen(symbol: str = SYMBOL):
    return SpecSignalGenerator(
        fallback_spec(symbol), pip_size=pip_size_for_symbol(symbol)
    )


def _trade_sig(t: Dict[str, Any]):
    return (
        t["close_time"],
        round(float(t["profit"]), 10),
        t["open_price"],
        t["close_price"],
        t["volume"],
        t["commission"],
    )


# ---------------------------------------------------------------------------
# Kill switch: trip on EVERY breaker, human-only re-arm
# ---------------------------------------------------------------------------


def _assert_tripped(gov: ExecutionGovernor, reason: str) -> None:
    assert gov.state.tripped
    assert gov.state.state == "HALTED"
    assert gov.trip_reason == reason
    kinds = [e["kind"] for e in gov.events]
    assert "kill_switch_tripped" in kinds


def test_trip_manual_and_human_rearm():
    gov = ExecutionGovernor(GovernorLimits())
    gov.arm()
    gov.trip(TRIP_MANUAL)
    _assert_tripped(gov, TRIP_MANUAL)
    # Re-arm is the ONLY path out of HALTED (human action).
    gov.arm()
    assert gov.state.state == "ARMED"
    assert gov.trip_reason == TRIP_MANUAL  # audit trail keeps the reason


def test_trip_daily_loss_cancels_pending_intents():
    gov = ExecutionGovernor(GovernorLimits(daily_loss_limit=100.0))
    gov.arm()
    now = datetime(2026, 8, 1)
    rec = gov.accept_intent("k1", symbol=SYMBOL, side="BUY", volume=0.1,
                            sl=None, tp=None, now=now)
    assert rec is not None
    gov.check_breakers(equity=9_890.0, day_start_equity=10_000.0,
                       peak_equity=10_000.0, open_positions=1, now=now)
    _assert_tripped(gov, TRIP_DAILY_LOSS)
    assert gov.journal.get("k1")["state"] == INTENT_CANCELLED
    assert gov.pending_intents() == []
    # Tripped governor refuses everything.
    assert gov.accept_intent("k2", symbol=SYMBOL, side="BUY", volume=0.1,
                             sl=None, tp=None, now=now) is None


def test_trip_drawdown_and_position_count():
    gov = ExecutionGovernor(GovernorLimits(max_drawdown=50.0, max_positions=2))
    gov.arm()
    now = datetime(2026, 8, 1)
    gov.check_breakers(equity=9_940.0, day_start_equity=10_000.0,
                       peak_equity=10_000.0, open_positions=1, now=now)
    _assert_tripped(gov, TRIP_DRAWDOWN)

    gov2 = ExecutionGovernor(GovernorLimits(max_positions=2))
    gov2.arm()
    gov2.check_breakers(equity=10_000.0, day_start_equity=10_000.0,
                        peak_equity=10_000.0, open_positions=3, now=now)
    _assert_tripped(gov2, TRIP_POSITION_COUNT)


def test_trip_is_idempotent_first_reason_wins():
    gov = ExecutionGovernor(GovernorLimits())
    gov.arm()
    now = datetime(2026, 8, 1)
    gov.trip(TRIP_DAILY_LOSS, ts=now)
    gov.trip(TRIP_DRAWDOWN, ts=now + timedelta(minutes=1))
    _assert_tripped(gov, TRIP_DAILY_LOSS)
    trips = [e for e in gov.events if e["kind"] == "kill_switch_tripped"]
    assert len(trips) == 1


# ---------------------------------------------------------------------------
# Feed guards: staleness + clock regression
# ---------------------------------------------------------------------------


def test_stale_data_refuses_intents():
    gov = ExecutionGovernor(GovernorLimits(max_staleness_bars=3))
    gov.arm()
    t0 = datetime(2026, 8, 1, 12, 0)
    gov.record_bar(SYMBOL, t0)
    # Fresh bar → accepted.
    assert gov.accept_intent("k1", symbol=SYMBOL, side="BUY", volume=0.1,
                             sl=None, tp=None, now=t0 + timedelta(minutes=59)) is not None
    # Stale (4+ hours old data) → refused.
    late = t0 + timedelta(hours=4, minutes=1)
    rec = gov.accept_intent("k2", symbol=SYMBOL, side="BUY", volume=0.1,
                            sl=None, tp=None, now=late)
    assert rec is None
    refusals = [e for e in gov.events if e["kind"] == "intent_refused"]
    assert any(e["reason"] == "stale_data" for e in refusals)


def test_clock_regression_refuses_intents():
    gov = ExecutionGovernor()
    gov.arm()
    t0 = datetime(2026, 8, 1, 12, 0)
    gov.record_bar(SYMBOL, t0)
    back = t0 - timedelta(minutes=5)
    rec = gov.accept_intent("k1", symbol=SYMBOL, side="BUY", volume=0.1,
                            sl=None, tp=None, now=back)
    assert rec is None
    refusals = [e for e in gov.events if e["kind"] == "intent_refused"]
    assert any(e["reason"] == "clock_regression" for e in refusals)


def test_not_armed_refuses_everything():
    gov = ExecutionGovernor()
    now = datetime(2026, 8, 1)
    assert gov.accept_intent("k1", symbol=SYMBOL, side="BUY", volume=0.1,
                             sl=None, tp=None, now=now) is None
    assert [e for e in gov.events if e["kind"] == "intent_refused"]


# ---------------------------------------------------------------------------
# Duplicate intents: idempotency
# ---------------------------------------------------------------------------


def test_duplicate_intent_suppressed_by_key():
    gov = ExecutionGovernor()
    gov.arm()
    now = datetime(2026, 8, 1)
    r1 = gov.accept_intent("dup-key", symbol=SYMBOL, side="BUY", volume=0.1,
                           sl=None, tp=None, now=now)
    r2 = gov.accept_intent("dup-key", symbol=SYMBOL, side="BUY", volume=0.1,
                           sl=None, tp=None, now=now + timedelta(minutes=1))
    assert r1 is not None and r2 is None       # duplicate never re-accepted
    assert len(gov.journal) == 1               # exactly ONE record
    assert gov.journal.get("dup-key")["state"] == INTENT_PENDING
    dup_events = [e for e in gov.events if e["kind"] == "intent_duplicate_suppressed"]
    assert len(dup_events) == 1


# ---------------------------------------------------------------------------
# Crash recovery: write-ahead journal + reconciliation
# ---------------------------------------------------------------------------


def test_write_ahead_journal_snapshot_roundtrip():
    gov = ExecutionGovernor()
    gov.arm()
    now = datetime(2026, 8, 1)
    gov.accept_intent("k1", symbol=SYMBOL, side="BUY", volume=0.1,
                      sl=None, tp=None, now=now)
    gov.accept_intent("k2", symbol=SYMBOL, side="SELL", volume=0.2,
                      sl=None, tp=None, now=now)
    gov.record_fill("k1", {"price": 1.1, "volume": 0.1})

    snap = gov.journal.save_snapshot()

    # "Crash": a brand-new governor/journal restores the byte-exact image.
    gov2 = ExecutionGovernor(journal=IntentJournal())
    gov2.journal.load_snapshot(snap)
    assert gov2.journal.get("k1")["state"] == INTENT_EXECUTED
    assert gov2.journal.get("k2")["state"] == INTENT_PENDING
    assert gov2.journal.get("k2")["volume"] == 0.2


def test_reconciliation_adopt_review_never_resend():
    """Restart reconciliation: PENDING intents are classified — adoption of
    broker-confirmed positions vs human review for unacknowledged intents.
    Nothing is auto-re-sent."""
    gov = ExecutionGovernor()
    gov.arm()
    now = datetime(2026, 8, 1)
    gov.accept_intent("sent-not-confirmed", symbol=SYMBOL, side="BUY",
                      volume=0.1, sl=None, tp=None, now=now)
    gov.accept_intent("never-sent", symbol=SYMBOL, side="SELL", volume=0.1,
                      sl=None, tp=None, now=now)
    snapshot = gov.journal.save_snapshot()

    # ---- crash: new process, no in-memory state but the journal ----
    recovered = ExecutionGovernor(journal=IntentJournal())
    recovered.journal.load_snapshot(snapshot)

    # Broker-confirmed position exists for the first intent (the reconciliation
    # input); nothing for the second.
    broker_positions = {
        "EUR/USD": {"side": "BUY", "volume": 0.1},  # matches sent-not-confirmed
    }
    adopted, review = _reconcile(recovered, broker_positions)
    assert adopted == ["sent-not-confirmed"]     # broker position ADOPTED
    assert review == ["never-sent"]              # human review, NOT re-sent


def _reconcile(gov: ExecutionGovernor, broker_positions: Dict[str, Dict[str, Any]]):
    """Deterministic restart reconciliation (the E2 adapter will reuse this).

    For each PENDING journaled intent: if a broker position matching
    (symbol, side, volume) exists, the intent is ADOPTED (marked executed
    with the broker position as its fill); otherwise it goes to the human
    review list. NEVER re-sent.
    """
    adopted: List[str] = []
    review: List[str] = []
    remaining = dict(broker_positions)
    for intent in gov.pending_intents():
        pos = remaining.get(intent["symbol"])
        if pos and pos.get("side") == intent["side"] and abs(
            float(pos.get("volume", 0)) - float(intent["volume"])
        ) < 1e-9:
            gov.record_fill(intent["key"], {"price": None, "source": "adopted_position"})
            adopted.append(intent["key"])
            remaining.pop(intent["symbol"], None)
        else:
            review.append(intent["key"])
    return adopted, review


# ---------------------------------------------------------------------------
# Broker fault table
# ---------------------------------------------------------------------------


def test_broker_fault_table_complete_and_deterministic():
    """Every declared MT5 failure mode maps to its declared handler."""
    expected = {
        "TRADE_RETCODE_REQUOTE": "REPRICE_THEN_FILL",
        "TRADE_RETCODE_PRICE_OFF": "REJECT",
        "TRADE_RETCODE_INVALID_STOPS": "REJECT",
        "TRADE_RETCODE_MARKET_CLOSED": "CANCEL_INTENT",
        "TRADE_RETCODE_TOO_MANY_REQUESTS": "BOUNDED_RETRY_THEN_PENDING",
        "TRADE_RETCODE_SERVER_BUSY": "BOUNDED_RETRY_THEN_PENDING",
        "TRADE_RETCODE_CONNECTION_LOST": "HALT_AND_RECONCILE",
        "TRADE_RETCODE_DONE": "FILL",
        "TRADE_RETCODE_PLACED": "FILL",
    }
    assert BROKER_FAULT_TABLE == expected  # pinned, no silent edits


# ---------------------------------------------------------------------------
# Governor inside the simulator (chaos scenarios with pinned post-states)
# ---------------------------------------------------------------------------


def test_simulator_trip_midrun_cancels_deferred_and_refuses_new():
    """Daily-loss breaker trips mid-run. Pinned post-state: NOTHING fills
    after the trip bar, every later signal is refused, and no PENDING
    journal intent survives the run."""
    df = _trend_bars(260, seed=5)
    config = _config(df)
    # Tight daily loss so an early SL closes the day's book: breaker fires.
    gov = ExecutionGovernor(GovernorLimits(daily_loss_limit=50.0))
    gov.arm()
    params = ExecutionParams(latency_bars_min=2, latency_bars_max=2, seed=11,
                             record_intents=True)

    result = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=params,
                            governor=gov)
    assert gov.state.tripped
    trips = [e for e in gov.events if e["kind"] == "kill_switch_tripped"]
    assert trips and trips[0]["reason"] == TRIP_DAILY_LOSS

    # tripped_at is BAR time (check_breakers passes the bar timestamp), so
    # the trip bar index is recoverable deterministically.
    trip_ts = pd.Timestamp(gov.tripped_at)
    trip_bar_index = df.index.get_loc(trip_ts) + 1  # engine bars are 1-based

    # Nothing fills after the trip bar.
    post_fills = [f for f in result.fills if f["executed_bar"] > trip_bar_index]
    assert post_fills == []
    # Every signal after the trip is refused (fail-closed).
    post_refusals = [
        r for r in result.rejected_intents
        if r["created_bar"] is not None and r["created_bar"] >= trip_bar_index
        and r["reason"] in ("governor_refused", "not_armed:HALTED")
    ]
    assert post_refusals, "post-trip signals must appear as refusals"
    # Journal post-state: no PENDING intents survive a tripped run.
    assert gov.pending_intents() == []


def test_cancel_deferred_on_trip_mechanism():
    """Unit pin: an in-flight deferred intent at trip time is cancelled with
    the kill-switch reason, its order dies, and the book empties."""
    from forex_ai_agent.backtest.engine import DataFeed
    from forex_ai_agent.backtest.execution import OrderSide, OrderType
    from forex_ai_agent.execution.simulator import ExecutionSimulator

    df = _trend_bars(60, seed=3)
    config = _config(df)
    gov = ExecutionGovernor()
    gov.arm()
    gov.trip(TRIP_MANUAL, ts=datetime(2026, 8, 1, 10, 0))  # first: the trip
    sim = ExecutionSimulator(config, DataFeed({SYMBOL: df}), _fresh_gen(),
                             params=ExecutionParams(seed=1, record_intents=True),
                             governor=gov)
    order = sim.order_manager.create_order(
        symbol=SYMBOL, side=OrderSide.BUY, order_type=OrderType.MARKET,
        volume=0.5, sl=None, tp=None, comment="in-flight",
    )
    sim._intents[order.id] = {
        "key": order.id,
        "order": order,
        "created_bar": 10,
        "latency_bars": 2,
        "due_bar": 12,
        "journal_key": None,
    }
    sim.current_time = datetime(2026, 8, 1, 10, 0)
    sim._enforce_trip_policy()
    assert sim._intents == {}                          # book emptied
    assert order.status.value == "CANCELLED"          # order dead
    assert len(sim.rejected_intents) == 1
    assert sim.rejected_intents[0]["reason"].startswith("KILL_SWITCH:")
    assert sim.rejected_intents[0]["executed_bar"] is None


def test_governed_clean_run_fill_identical_when_no_trip():
    """Clean microstructure under a governor (no trip) = the ungoverned
    clean run, fill-for-fill — governance must not change the numbers."""
    df = _trend_bars(220, seed=3)
    config = _config(df)
    gov = ExecutionGovernor(GovernorLimits(daily_loss_limit=1e9, max_drawdown=1e9,
                                           max_positions=1000))
    gov.arm()
    params = ExecutionParams(record_intents=True)

    ungoverned = run_simulation({SYMBOL: df}, _fresh_gen(), config,
                                params=ExecutionParams(record_intents=True))
    governed = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=params,
                              governor=gov)
    assert not gov.state.tripped
    assert [_trade_sig(t) for t in governed.trades] == [_trade_sig(t) for t in ungoverned.trades]
    assert governed.fills == ungoverned.fills
    # Journal reflects every fill as EXECUTED (write-ahead worked).
    executed = [i for i in gov.journal.all() if i["state"] == INTENT_EXECUTED]
    assert len(executed) == len(governed.trades)


def test_disarmed_governor_blocks_all_fills():
    """A governor that was never armed refuses every intent — fail-closed."""
    df = _trend_bars(220, seed=3)
    config = _config(df)
    gov = ExecutionGovernor()  # DISARMED
    result = run_simulation({SYMBOL: df}, _fresh_gen(), config,
                            params=ExecutionParams(record_intents=True),
                            governor=gov)
    assert result.fills == []
    assert result.trades == []
    assert all(
        e["reason"] == "not_armed:DISARMED"
        for e in gov.events if e["kind"] == "intent_refused"
    )


# ---------------------------------------------------------------------------
# Trip position policy: flatten-or-hold (human-declared constant)
# ---------------------------------------------------------------------------


class _TripAfterNBarsGovernor(ExecutionGovernor):
    """Deterministic chaos: trips MANUAL on the Nth observed bar."""

    def __init__(self, trip_on_bar: int, *a, **k):
        super().__init__(*a, **k)
        self._trip_on_bar = trip_on_bar
        self._bars_seen = 0

    def record_bar(self, symbol: str, ts: datetime) -> None:
        super().record_bar(symbol, ts)
        self._bars_seen += 1
        if self._bars_seen == self._trip_on_bar and not self.state.tripped:
            self.trip(TRIP_MANUAL, ts=ts)


def _first_straddling_trade(df: pd.DataFrame, trades: List[Dict[str, Any]],
                            min_span: int = 4):
    """Pick a baseline trade whose position stays open several bars after
    entry; return (trip_bar_1based, trade) with the trip bar strictly
    inside its hold window (no SL/TP can fire there — the recorded close
    bar is the FIRST hit by construction)."""
    for t in trades:
        oi = df.index.get_loc(pd.Timestamp(t["open_time"]))
        ci = df.index.get_loc(pd.Timestamp(t["close_time"]))
        if ci - oi >= min_span:
            return oi + 3, t
    pytest.fail("baseline has no trade holding >= min_span bars — drill impossible")


def _run_trip_policy_drill(policy: str):
    """Ungoverned baseline (latency 2) → MANUAL trip inside the first
    long-held position's window → governed run under ``policy``."""
    df = _trend_bars(260, seed=5)
    config = _config(df)
    params = ExecutionParams(latency_bars_min=2, latency_bars_max=2, seed=11,
                             record_intents=True)
    baseline = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=params)
    trip_bar, held = _first_straddling_trade(df, baseline.trades)
    gov = _TripAfterNBarsGovernor(
        trip_bar, GovernorLimits(trip_position_policy=policy)
    )
    gov.arm()
    result = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=params,
                            governor=gov)
    return df, config, baseline, trip_bar, held, gov, result


def test_trip_position_policy_constants_and_default():
    """The flatten-or-hold choice is a DECLARED constant, never a trip-time
    decision: exactly two policies exist, HOLD is the default, and an
    unknown policy is a construction error (fail-closed)."""
    assert TRIP_POSITION_POLICIES == (TRIP_POSITION_HOLD, TRIP_POSITION_FLATTEN)
    assert TRIP_POSITION_HOLD != TRIP_POSITION_FLATTEN
    assert GovernorLimits().trip_position_policy == TRIP_POSITION_HOLD
    assert (
        GovernorLimits(trip_position_policy=TRIP_POSITION_FLATTEN).trip_position_policy
        == TRIP_POSITION_FLATTEN
    )
    with pytest.raises(ValueError):
        GovernorLimits(trip_position_policy="FLATTEN")
    with pytest.raises(ValueError):
        GovernorLimits(trip_position_policy="hold")


def test_trip_event_declares_policy_and_flatten_is_audited():
    """kill_switch_tripped carries the declared policy; every flatten close
    is reported back into the governor's audit trail."""
    for policy in TRIP_POSITION_POLICIES:
        gov = ExecutionGovernor(GovernorLimits(trip_position_policy=policy))
        gov.arm()
        gov.trip(TRIP_MANUAL, ts=datetime(2026, 8, 1))
        trips = [e for e in gov.events if e["kind"] == "kill_switch_tripped"]
        assert trips and trips[0]["position_policy"] == policy
    assert ExecutionGovernor(GovernorLimits()).flattened == []  # HOLD default
    gov = ExecutionGovernor(
        GovernorLimits(trip_position_policy=TRIP_POSITION_FLATTEN)
    )
    gov.arm()
    gov.trip(TRIP_MANUAL)
    gov.record_flattened({"position_id": "p1", "symbol": SYMBOL, "profit": -1.5})
    assert len(gov.flattened) == 1
    flats = [e for e in gov.events if e["kind"] == "position_flattened"]
    assert flats and flats[0]["position_id"] == "p1"


def test_hold_policy_keeps_open_position_lifecycle_untouched():
    """Pinned post-state (HOLD — the default): the trip gates NEW risk only.
    The position open at the trip keeps the engine's own lifecycle — it
    closes LATER by its own SL/TP at exactly the ungoverned close price,
    time and profit; the governor's flatten audit stays empty."""
    df, config, baseline, trip_bar, held, gov, result = _run_trip_policy_drill(
        TRIP_POSITION_HOLD)
    trip_ts = pd.Timestamp(gov.tripped_at)
    assert gov.state.tripped and gov.trip_reason == TRIP_MANUAL
    assert trip_ts == df.index[trip_bar - 1]      # trip bar is deterministic
    assert gov.flattened == []                    # nothing was force-closed
    assert not [e for e in gov.events if e["kind"] == "position_flattened"]
    assert not [t for t in result.trades
                if str(t["reason"]).startswith("KILL_SWITCH")]
    # The held position's lifecycle is EXACTLY the ungoverned one.
    match = [t for t in result.trades
             if t["open_time"] == held["open_time"] and t["volume"] == held["volume"]]
    assert len(match) == 1
    g = match[0]
    assert g["close_time"] == held["close_time"]      # closes at the same bar
    assert g["close_price"] == held["close_price"]    # at the same price
    assert g["reason"] == held["reason"]              # via its own SL/TP
    assert g["profit"] == held["profit"]              # same realized number
    # Pre-trip fills are identical to the baseline (prefix determinism);
    # post-trip fills are refused (nothing new fills under HALTED).
    pre = [f for f in baseline.fills if f["executed_bar"] <= trip_bar]
    assert result.fills == pre


def test_flatten_policy_closes_open_positions_at_market_on_trip_bar():
    """Pinned post-state (FLATTEN): every position open at the trip bar is
    closed AT MARKET on the trip bar itself through the engine's own
    close path — reason KILL_SWITCH:*, close price == trip bar close,
    profit == the engine's own close_position formula — and afterwards the
    account holds zero open risk (equity == balance on every later bar)."""
    df, config, baseline, trip_bar, held, gov, result = _run_trip_policy_drill(
        TRIP_POSITION_FLATTEN)
    trip_ts = pd.Timestamp(gov.tripped_at)
    assert gov.state.tripped and trip_ts == df.index[trip_bar - 1]
    # Exactly the positions open at the trip were flattened (audited).
    straddles = [
        t for t in baseline.trades
        if df.index.get_loc(pd.Timestamp(t["open_time"])) <= trip_bar - 1
        and df.index.get_loc(pd.Timestamp(t["close_time"])) >= trip_bar
    ]
    assert straddles and len(gov.flattened) == len(straddles)
    assert all(rec["trip_reason"] == TRIP_MANUAL for rec in gov.flattened)
    bar_close = float(df.iloc[trip_bar - 1]["close"])
    by_id = {t["id"]: t for t in result.trades}
    for rec in gov.flattened:
        t = by_id[rec["position_id"]]                 # the close is a REAL trade
        assert str(t["reason"]).startswith("KILL_SWITCH")
        assert t["close_price"] == rec["close_price"] == bar_close
        assert t["close_time"] == rec["close_time"] == trip_ts
        # The engine's own close_position formula, reproduced exactly:
        sign = 1.0 if t["side"] == "BUY" else -1.0
        raw = (t["close_price"] - t["open_price"]) * t["volume"] * 100000 * sign
        assert t["profit"] == raw - t["commission"] + t["swap"]
        # It is the baseline straddle, force-closed early (same entry).
        assert round(t["open_price"], 10) == round(rec["open_price"], 10)
    # Nothing closes after the trip bar: no open risk survives HALTED.
    assert max(pd.Timestamp(t["close_time"]) for t in result.trades) == trip_ts
    # Zero open risk afterwards → equity == balance on every later bar.
    expected_equity = config.initial_balance + sum(
        t["profit"] for t in result.trades
    )
    post = [eq for ts, eq in result.equity_curve if ts > trip_ts]
    assert post and all(
        eq == pytest.approx(expected_equity, abs=1e-6) for eq in post
    )
    # The position policy never touches fills (identical prefix).
    pre = [f for f in baseline.fills if f["executed_bar"] <= trip_bar]
    assert result.fills == pre


def test_flatten_policy_is_noop_when_nothing_is_open():
    """Trip before any fill: FLATTEN has nothing to close — no flatten
    events, no trades, no fills, run completes deterministically."""
    df = _trend_bars(260, seed=5)
    config = _config(df)
    params = ExecutionParams(latency_bars_min=2, latency_bars_max=2, seed=11,
                             record_intents=True)
    gov = _TripAfterNBarsGovernor(
        3, GovernorLimits(trip_position_policy=TRIP_POSITION_FLATTEN)
    )
    gov.arm()
    result = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=params,
                            governor=gov)
    assert gov.state.tripped
    assert gov.flattened == []
    assert not [e for e in gov.events if e["kind"] == "position_flattened"]
    assert result.trades == [] and result.fills == []


# ---------------------------------------------------------------------------
# Constitution
# ---------------------------------------------------------------------------


def test_governor_source_scan_no_send_paths():
    """Source scan: governor + journal are pure state machines — no network,
    no broker calls, no risk delegation, no permission self-promotion."""
    import forex_ai_agent.execution.governor as gov_mod
    import forex_ai_agent.execution.journal as j_mod

    for mod in (gov_mod, j_mod):
        src = inspect.getsource(mod)
        for banned in (
            "place_order", "execute_trade", "risk_manager",
            "requests.", "urllib", "socket", "mt5", "MetaTrader5",
            "upgrade_level", "MAX_PERMISSION_LEVEL = 4",
        ):
            assert banned not in src, f"banned token {banned!r} in {mod.__name__}"
