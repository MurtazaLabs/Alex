"""Stage E1 live verification: chaos/recovery on REAL market bars.

1. Fetch REAL recent H1 bars for EUR/USD from the resilient feed chain.
2. Governed baseline: a governed clean run (governor armed, generous
   breakers) is fill-identical to the ungoverned run — governance does not
   change the numbers.
3. Chaos drill A — mid-run kill switch: re-run with a MANUAL trip injected
   after a third of the bars (deterministic); pinned post-state: nothing
   fills after the trip bar, later signals are refused, journal shows no
   surviving PENDING intents.
4. Chaos drill B — crash/restart reconciliation: take a governed run that
   filled trades, snapshot the journal mid-run ("crash"), restore it into a
   fresh process-state, reconcile against a declared broker-position book:
   matching positions are ADOPTED, unmatched intents go to human review,
   NOTHING is re-sent.
5. Chaos drill C — trip position policy: a MANUAL trip fires while a real
   position is open (derived from the baseline). Under HOLD (default) the
   position rides its own SL/TP to EXACTLY the baseline close; under
   FLATTEN it is closed at market ON the trip bar (engine's own close
   path, reason KILL_SWITCH:*, equity == balance afterwards). Fills are
   identical in both runs — the policy never touches fills.
6. Write nothing to the DB — pure computation.

Run:  .venv311/Scripts/python.exe scripts/e1_live_verify.py
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta

sys.path.insert(0, "src")

import pandas as pd

from forex_ai_agent.autonomy.validator import SpecSignalGenerator
from forex_ai_agent.backtest.engine import BacktestConfig
from forex_ai_agent.data.providers.factory import get_feed_chain
from forex_ai_agent.execution.governor import (
    ExecutionGovernor,
    GovernorLimits,
    TRIP_MANUAL,
    TRIP_POSITION_FLATTEN,
)
from forex_ai_agent.execution.journal import (
    INTENT_CANCELLED,
    INTENT_EXECUTED,
    INTENT_PENDING,
    IntentJournal,
)
from forex_ai_agent.execution.simulator import ExecutionParams, run_simulation
from forex_ai_agent.strategy.spec import parse_spec
from forex_ai_agent.utils.math_helpers import pip_size_for_symbol

SYMBOL = "EUR/USD"


def _trade_sig(t: dict):
    return (
        t["open_price"],
        t["close_price"],
        round(float(t["profit"]), 10),
        round(float(t["commission"]), 10),
        t["volume"],
        str(t["open_time"]),
        str(t["close_time"]),
    )


class _TripAfterNBarsGovernor(ExecutionGovernor):
    """Deterministic chaos: trips MANUALLY on the Nth observed bar."""

    def __init__(self, trip_on_bar: int, *a, **k):
        super().__init__(*a, **k)
        self._trip_on_bar = trip_on_bar
        self._bars_seen = 0

    def record_bar(self, symbol: str, ts: datetime) -> None:
        super().record_bar(symbol, ts)
        self._bars_seen += 1
        if self._bars_seen == self._trip_on_bar and not self.state.tripped:
            self.trip(TRIP_MANUAL, ts=ts)


def _reconcile(gov: ExecutionGovernor, broker_positions: dict):
    """Same deterministic reconciliation the chaos test pins (E2 reuses)."""
    adopted, review = [], []
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


async def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("forex_ai_agent").setLevel(logging.ERROR)

    chain = get_feed_chain()
    print("feed chain active source:", chain.active_source)
    if chain.active_source == "mock":
        print("SKIP: feed degraded to mock — no real data to verify against")
        return 2

    end = datetime.utcnow()
    start = end - timedelta(days=30)
    bars = await chain.fetch_candles(SYMBOL, "H1", start, end=end, limit=800)
    if not bars or len(bars) < 220:
        print(f"FAIL: only {0 if not bars else len(bars)} live bars returned")
        return 1
    print(
        f"fetched {len(bars)} real H1 bars "
        f"({bars[0].timestamp} → {bars[-1].timestamp})"
    )

    df = pd.DataFrame(
        {
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [float(getattr(b, "volume", 0) or 0) for b in bars],
        },
        index=[b.timestamp for b in bars],
    )
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    rules = {
        "name": "E1 Chaos Probe",
        "direction": "both",
        "timeframe": "H1",
        "entry_conditions": [
            {"left": "sma", "verb": "above", "right": "sma_slow"},
            {"left": "price", "verb": "above", "right": "sma"},
        ],
    }
    spec, errs = parse_spec(rules)
    if spec is None:
        print("FAIL: probe spec invalid:", errs)
        return 1

    def fresh_gen():
        return SpecSignalGenerator(spec, pip_size=pip_size_for_symbol(SYMBOL))

    config = BacktestConfig(
        start_date=df.index.min().to_pydatetime(),
        end_date=df.index.max().to_pydatetime(),
        initial_balance=10_000.0,
        symbols=[SYMBOL],
        timeframe="H1",
        warmup_bars=12,
    )
    params = ExecutionParams(record_intents=True)
    limits = GovernorLimits(daily_loss_limit=1e9, max_drawdown=1e9, max_positions=1000)

    # ── Check 1: governed baseline == ungoverned baseline ────────────────
    base = run_simulation({SYMBOL: df}, fresh_gen(), config, params=params)
    gov_base = ExecutionGovernor(limits)
    gov_base.arm()
    governed = run_simulation({SYMBOL: df}, fresh_gen(), config, params=params,
                              governor=gov_base)
    check1 = (
        [_trade_sig(t) for t in governed.trades] == [_trade_sig(t) for t in base.trades]
        and governed.fills == base.fills
        and not gov_base.state.tripped
    )
    print(f"\nCheck 1 — governance does not change numbers: {check1} "
          f"({len(base.trades)} trade(s), {len(base.fills)} fill(s))")
    executed = [i for i in gov_base.journal.all() if i["state"] == INTENT_EXECUTED]
    print(f"  journal: {len(gov_base.journal)} intent(s), {len(executed)} executed "
          f"(write-ahead holds: {len(executed) == len(governed.trades)})")

    # ── Check 2: mid-run MANUAL trip ─────────────────────────────────────
    # 2-bar latency so an intent is genuinely IN FLIGHT when the trip fires
    # (the cancellation path is exercised, not just the refusal path).
    chaos_params = ExecutionParams(latency_bars_min=2, latency_bars_max=2,
                                   seed=11, record_intents=True)
    # Deterministic probe run (no trip) locates the FIRST journaled signal
    # bar; the chaos run below consumes the same seed, so its pre-trip
    # prefix is identical and tripping at first_signal_bar + 1 catches that
    # intent genuinely in flight (created < trip_bar <= due_bar - 1).
    probe_gov = ExecutionGovernor(limits)
    probe_gov.arm()
    run_simulation({SYMBOL: df}, fresh_gen(), config, params=chaos_params,
                   governor=probe_gov)
    probe_signal_bars = sorted(
        i["created_bar"] for i in probe_gov.journal.all()
        if i.get("created_bar") is not None
    )
    if not probe_signal_bars:
        print("FAIL: probe run produced no journaled intents — nothing to trip on")
        return 1
    trip_on = probe_signal_bars[0] + 1
    gov_chaos = _TripAfterNBarsGovernor(trip_on, limits)
    gov_chaos.arm()
    chaos = run_simulation({SYMBOL: df}, fresh_gen(), config, params=chaos_params,
                           governor=gov_chaos)
    tripped = gov_chaos.state.tripped and gov_chaos.trip_reason == TRIP_MANUAL
    trip_ts = pd.Timestamp(gov_chaos.tripped_at)
    trip_bar = df.index.get_loc(trip_ts) + 1
    post_fills = [f for f in chaos.fills if f["executed_bar"] > trip_bar]
    survivors = gov_chaos.pending_intents()
    check2 = tripped and post_fills == [] and survivors == []
    print(f"Check 2 — mid-run kill switch (bar {trip_bar}/{len(df)}): "
          f"tripped={tripped}, fills after trip={len(post_fills)} "
          f"(must be 0), surviving PENDING={len(survivors)} (must be 0)")
    cancelled = [i for i in gov_chaos.journal.all() if i["state"] == INTENT_CANCELLED]
    # The governor writes the lowercase 'kill_switch:<reason>' prefix when
    # IT cancels an in-flight intent (post-trip refusals are never journaled
    # and the simulator's uppercase reason lives only in rejected_intents).
    in_flight_cancelled = any(
        "kill_switch" in str(i.get("cancel_reason") or "").lower()
        for i in cancelled
    )
    check2 = check2 and in_flight_cancelled
    print(f"  journal: {len(gov_chaos.journal)} intent(s), {len(cancelled)} cancelled "
          f"({'in-flight at trip cancelled: yes' if in_flight_cancelled else 'refusal-path only'})")

    # ── Check 3: crash / restart reconciliation ──────────────────────────
    # Crash between send and confirm: cut the journal image at the LAST
    # journaled signal bar. In the baseline that intent's fill confirmed at
    # created+1, so it reverts to PENDING in the crash image — exactly the
    # unconfirmed state a restart must reconcile. Everything confirmed at or
    # before the crash stays EXECUTED; nothing after it is visible.
    full_journal = gov_base.journal.save_snapshot()
    created_bars = [i["created_bar"] for i in gov_base.journal.all()
                    if i.get("created_bar") is not None]
    if not created_bars:
        print("FAIL: baseline journal has no intents — no crash image to cut")
        return 1
    crash_bar = max(created_bars)
    cut_intents = {}
    for k, v in full_journal["intents"].items():
        if (v.get("created_bar") or 0) > crash_bar:
            continue  # written after the crash: the new process never saw it
        fill = v.get("fill")
        if fill and (fill.get("executed_bar") or 0) > crash_bar:
            v = {**v, "state": INTENT_PENDING, "fill": None}  # unconfirmed
        cut_intents[k] = v
    cut = {"intents": cut_intents}
    recovered = [ExecutionGovernor(journal=IntentJournal()) for _ in range(2)]
    for g in recovered:
        g.journal.load_snapshot(cut)
    pend_before = [i["key"] for i in recovered[0].pending_intents()]
    if not pend_before:
        print("FAIL: crash image recovered 0 PENDING intents — drill is vacuous")
        return 1

    # Drill A — the broker reports the position belonging to the FIRST
    # recovered intent: that one is ADOPTED, the rest go to human review.
    first = recovered[0].journal.get(pend_before[0])
    broker_book = {first["symbol"]: {"side": first["side"], "volume": first["volume"]}}
    adopted, review_a = _reconcile(recovered[0], broker_book)

    # Drill B — the broker reports NOTHING: every intent goes to human
    # review. Nothing is ever re-sent: reconciliation must leave the
    # journal byte-count intact (no new keys recorded, none removed).
    journal_size = len(recovered[1].journal)
    adopted_b, review_b = _reconcile(recovered[1], {})

    check3 = (
        adopted == pend_before[:1]
        and sorted(review_a) == sorted(pend_before[1:])
        and adopted_b == []
        and sorted(review_b) == sorted(pend_before)
        and len(recovered[1].journal) == journal_size
    )
    print(f"Check 3 — crash/restart reconcile (crash bar {crash_bar}/{len(df)}): "
          f"recovered {len(pend_before)} PENDING intent(s); "
          f"drill A adopted={len(adopted)} review={len(review_a)}; "
          f"drill B (empty book) review={len(review_b)}; "
          f"re-send count=0 (never re-sent): {check3}")

    # ── Check 4: trip position policy — HOLD vs FLATTEN on real bars ────
    # The drill trips MANUALLY while a real baseline position is open
    # (open ≥ 3 bars before its own SL/TP close, so nothing natural can
    # close it at the trip bar). The ONLY allowed difference between the
    # two governed runs is what happens to that open position.
    straddles, trip_bar_4 = [], None
    for t in base.trades:
        oi = df.index.get_loc(pd.Timestamp(t["open_time"]))
        ci = df.index.get_loc(pd.Timestamp(t["close_time"]))
        if ci - oi >= 4:
            trip_bar_4 = oi + 3
            break
    if trip_bar_4 is None:
        print("FAIL: no baseline trade holds >= 4 bars — flatten drill impossible")
        return 1
    straddles = [
        t for t in base.trades
        if df.index.get_loc(pd.Timestamp(t["open_time"])) <= trip_bar_4 - 1
        and df.index.get_loc(pd.Timestamp(t["close_time"])) >= trip_bar_4
    ]

    def _trip_gov(policy):
        g = _TripAfterNBarsGovernor(
            trip_bar_4,
            GovernorLimits(daily_loss_limit=1e9, max_drawdown=1e9,
                           max_positions=1000, trip_position_policy=policy),
        )
        g.arm()
        return g

    gov_hold = _trip_gov("POSITION_HOLD")
    hold_run = run_simulation({SYMBOL: df}, fresh_gen(), config, params=params,
                              governor=gov_hold)
    gov_flat = _trip_gov(TRIP_POSITION_FLATTEN)
    flat_run = run_simulation({SYMBOL: df}, fresh_gen(), config, params=params,
                              governor=gov_flat)
    trip4_ts = pd.Timestamp(gov_flat.tripped_at)
    bar_close_4 = float(df.iloc[trip_bar_4 - 1]["close"])
    pre_fills = [f for f in base.fills if f["executed_bar"] <= trip_bar_4]

    shared_ok = (
        gov_hold.state.tripped and gov_flat.state.tripped
        and pd.Timestamp(gov_hold.tripped_at) == trip4_ts == df.index[trip_bar_4 - 1]
        and hold_run.fills == pre_fills and flat_run.fills == pre_fills
    )

    def _match_close(trades, ref):
        m = [t for t in trades
             if t["open_time"] == ref["open_time"] and t["volume"] == ref["volume"]]
        return m[0] if len(m) == 1 else None

    # HOLD pin: the open position rides its own SL/TP to EXACTLY the
    # baseline close (same bar, price, reason, profit).
    hold_ok = gov_hold.flattened == [] and not any(
        str(t["reason"]).startswith("KILL_SWITCH") for t in hold_run.trades
    )
    for s in straddles:
        h = _match_close(hold_run.trades, s)
        hold_ok = hold_ok and h is not None and (
            h["close_time"] == s["close_time"]
            and h["close_price"] == s["close_price"]
            and h["reason"] == s["reason"]
            and h["profit"] == s["profit"]
        )

    # FLATTEN pin: every open position is closed at market ON the trip bar
    # through the engine's own close path (engine formula on the numbers).
    flat_trades = {t["id"]: t for t in flat_run.trades
                   if str(t["reason"]).startswith("KILL_SWITCH")}
    flat_ok = len(gov_flat.flattened) == len(straddles) == len(flat_trades)
    if flat_ok:
        for rec in gov_flat.flattened:
            ft = flat_trades.get(rec["position_id"])
            raw = (1.0 if ft["side"] == "BUY" else -1.0) * (
                ft["close_price"] - ft["open_price"]) * ft["volume"] * 100000
            flat_ok = flat_ok and (
                ft["close_price"] == rec["close_price"] == bar_close_4
                and ft["close_time"] == rec["close_time"] == trip4_ts
                and rec["profit"] == ft["profit"]
                and abs(ft["profit"] - (raw - ft["commission"] + ft["swap"])) < 1e-9
            )
        flat_ok = flat_ok and max(
            pd.Timestamp(t["close_time"]) for t in flat_run.trades
        ) == trip4_ts
        expected_eq = config.initial_balance + sum(
            t["profit"] for t in flat_run.trades
        )
        post_eq = [eq for ts, eq in flat_run.equity_curve if ts > trip4_ts]
        flat_ok = flat_ok and bool(post_eq) and all(
            abs(eq - expected_eq) < 1e-6 for eq in post_eq
        )

    check4 = shared_ok and hold_ok and flat_ok
    print(f"Check 4 — trip position policy (trip bar {trip_bar_4}/{len(df)}, "
          f"{len(straddles)} position(s) open): HOLD kept its own SL/TP "
          f"close: {hold_ok}; FLATTEN closed at market on the trip bar, "
          f"zero open risk after: {flat_ok}: {check4}")

    ok = check1 and check2 and check3 and check4
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
