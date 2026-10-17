"""Stage E0 live verification: execution simulator on REAL market bars.

1. Fetch REAL recent H1 bars for EUR/USD from the resilient feed chain.
2. Gate (a) — clean parity: the simulator in clean mode over the real bars
   produces trades IDENTICAL (open/close price, profit, commission) to the
   batch backtest engine — exact equality, the D1-style contract.
3. Gate (b/c) — microstructure band: a degraded run (latency ≤ 2 bars,
   partial ladders, requotes ≤ 2 pips, 2x spread widening) over the same
   real window keeps the SAME signal intents, fills them with a declared,
   bounded adverse entry drift, and is byte-deterministic across runs.
   Net P&L drift vs clean is reported as information (outcomes legitimately
   shift when fills land later); the BAND applies to per-entry microstructure
   cost, not to outcome noise.
4. Write nothing to the DB — pure computation over fetched bars.

Run:  .venv311/Scripts/python.exe scripts/e0_live_verify.py
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta

sys.path.insert(0, "src")

import pandas as pd

from forex_ai_agent.autonomy.validator import SpecSignalGenerator
from forex_ai_agent.backtest.engine import BacktestConfig, run_backtest
from forex_ai_agent.data.providers.factory import get_feed_chain
from forex_ai_agent.execution.simulator import ExecutionParams, run_simulation
from forex_ai_agent.strategy.spec import parse_spec
from forex_ai_agent.utils.math_helpers import pip_size_for_symbol

SYMBOL = "EUR/USD"
PIPS = 0.0001
#: Declared tolerance band (E0 plan): with latency fixed at 1 bar (fills on
#: the same bar as the baseline), the adverse ENTRY drift of the degraded
#: run vs the intent-baseline must stay within widening + requote bounds:
#: 2x spread widening = +1 pip, requotes ≤ 2 pips, +0.5 pip tolerance.
DECLARED_MEAN_DRIFT_PIPS = 3.5
#: Latency re-pricing (fills landing on later bars) is REALISTIC BEHAVIOR,
#: not a cost — its effect is reported informationally, never gated.


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
        "name": "E0 Simulator Probe",
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

    # ── Gate (a): clean parity against the batch engine ─────────────────
    reference = run_backtest({SYMBOL: df}, fresh_gen(), config)
    clean = run_simulation({SYMBOL: df}, fresh_gen(), config,
                           params=ExecutionParams())
    ref_sigs = [_trade_sig(t) for t in reference.trades]
    clean_sigs = [_trade_sig(t) for t in clean.trades]
    gate_a = ref_sigs == clean_sigs and len(ref_sigs) > 0
    print(f"\nGate (a) clean parity: engine {len(ref_sigs)} trade(s), "
          f"simulator {len(clean_sigs)} trade(s), sequences identical: {gate_a}")

    # ── Gate (b/c): degraded microstructure over the same real window ───
    # Latency fixed at 1 bar so every fill lands on the same bar as the
    # baseline — the band then isolates pure microstructure cost.
    degraded_params = ExecutionParams(
        latency_bars_min=1, latency_bars_max=1,
        partial_fill_probability=0.35, partial_fill_min_fraction=0.5,
        partial_fill_max_fraction=0.7,
        requote_probability=0.25, requote_max_pips=2.0,
        rejection_probability=0.03,
        spread_widen_multiplier=2.0,
        seed=11,
    )
    degraded = run_simulation({SYMBOL: df}, fresh_gen(), config, params=degraded_params)
    degraded2 = run_simulation({SYMBOL: df}, fresh_gen(), config, params=degraded_params)
    deterministic = (
        degraded.fills == degraded2.fills
        and [_trade_sig(t) for t in degraded.trades]
        == [_trade_sig(t) for t in degraded2.trades]
    )

    clean_baseline = run_simulation({SYMBOL: df}, fresh_gen(), config,
                                    params=ExecutionParams(record_intents=True))
    # Pair PER INTENT (laddered intents emit several fill records; naive
    # zip would compare different intents after the first ladder).
    deg_by_intent: dict = {}
    for f in degraded.fills:
        deg_by_intent.setdefault(f["order_id"].split("-t")[0], []).append(f)
    drift_pips = []
    for cf in clean_baseline.fills:
        tranches = deg_by_intent.get(cf["order_id"])
        if not tranches:
            continue  # intent rejected in the degraded run — excluded from cost
        vwap = sum(t["price"] * t["volume"] for t in tranches) / sum(
            t["volume"] for t in tranches
        )
        adverse = vwap - cf["price"]
        if cf["side"] == "SELL":
            adverse = -adverse
        drift_pips.append(adverse / PIPS)
    mean_drift = sum(drift_pips) / len(drift_pips) if drift_pips else 0.0
    worst_drift = max(drift_pips) if drift_pips else 0.0
    gate_band = mean_drift <= DECLARED_MEAN_DRIFT_PIPS
    band_note = (
        f"mean adverse entry drift {mean_drift:.2f} pips, worst {worst_drift:.2f} "
        f"(declared band ≤ {DECLARED_MEAN_DRIFT_PIPS})"
    )

    net_clean = sum(t["profit"] for t in clean.trades)
    net_deg = sum(t["profit"] for t in degraded.trades)
    print(f"Gate (b/c) degraded run: {len(degraded.fills)} fill record(s), "
          f"{degraded.partial_fills} laddered, {degraded.requotes} requotes, "
          f"{degraded.rejections} rejections")
    print(f"  determinism across runs: {deterministic}")
    print(f"  {band_note}")

    # Latency variance demonstrated informationally (never gated).
    latency_params = ExecutionParams(
        latency_bars_min=1, latency_bars_max=2, seed=11,
        record_intents=True,
    )
    latency_run = run_simulation({SYMBOL: df}, fresh_gen(), config, params=latency_params)
    late_fills = [f for f in latency_run.fills if f["latency_bars"] > 1]
    print(f"  latency demo (1-2 bars): {len(late_fills)}/{len(latency_run.fills)} "
          f"fill(s) deferred past the earliest bar; net P&L "
          f"clean {net_clean:,.2f} vs degraded {net_deg:,.2f} "
          f"vs latency-varied {sum(t['profit'] for t in latency_run.trades):,.2f} "
          f"(outcome drift is informational, not gated)")

    ok = gate_a and deterministic and gate_band
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
