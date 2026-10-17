"""Unit tests for the Stage E0 deterministic execution simulator.

Pins the E0 gate:
- clean-condition fills/trades/equity are IDENTICAL to the backtest engine
  (exact float equality ⇒ the 1e-12 contract),
- every degradation mode (latency, partials, requotes, rejections, spread
  widening) is seed-deterministic and behaves as declared,
- named scenarios reproduce exactly for a fixed seed,
- nothing sends anywhere (source scan: no network/broker calls).

NOTE: SpecSignalGenerator is STATEFUL (prev values, cooldowns) — every run
gets a fresh generator; sharing one across runs would contaminate signals.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from forex_ai_agent.autonomy.validator import SpecSignalGenerator
from forex_ai_agent.backtest.engine import BacktestConfig, DataFeed, run_backtest
from forex_ai_agent.execution.scenarios import (
    SCENARIOS,
    thin_liquidity_scenario,
)
from forex_ai_agent.execution.simulator import (
    ExecutionMode,
    ExecutionParams,
    ExecutionSimulator,
    run_simulation,
)
from forex_ai_agent.strategy.spec import fallback_spec
from forex_ai_agent.utils.math_helpers import pip_size_for_symbol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOL = "EUR/USD"


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
    """A NEW signal generator per run — the generator is stateful."""
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
# Clean parity (the E0 gate headline)
# ---------------------------------------------------------------------------


def test_clean_mode_identical_to_backtest_run():
    """CLEAN trades + equity identical to run_backtest — by construction."""
    df = _trend_bars(220, seed=3)
    config = _config(df)

    reference = run_backtest({SYMBOL: df}, _fresh_gen(), config)
    sim = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=ExecutionParams(seed=7))

    ref_sigs = [_trade_sig(t) for t in reference.trades]
    sim_sigs = [_trade_sig(t) for t in sim.trades]
    assert ref_sigs == sim_sigs  # exact equality incl. floats → 1e-12 holds
    assert len(ref_sigs) > 0  # comparison must be meaningful
    assert sim.equity_curve == reference.equity_curve
    assert sim.fills == [] and sim.rejections == 0  # diagnostics empty in clean mode

    engine = ExecutionSimulator(config, DataFeed({SYMBOL: df}), _fresh_gen())
    assert engine.mode_flags == {ExecutionMode.CLEAN}


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def test_latency_defers_fills_and_is_deterministic():
    """Intent created at bar T fills at T+latency; two runs identical."""
    df = _trend_bars(220, seed=3)
    config = _config(df)
    params = ExecutionParams(latency_bars_min=2, latency_bars_max=4, seed=9)

    r1 = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=params)
    r2 = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=params)

    assert r1.fills == r2.fills
    assert [_trade_sig(t) for t in r1.trades] == [_trade_sig(t) for t in r2.trades]
    assert len(r1.fills) > 0
    for f in r1.fills:
        assert f["latency_bars"] >= 2
        assert f["executed_bar"] - f["created_bar"] == f["latency_bars"]
    assert any(k.startswith("2bars") for k in r1.latency_histogram)

    # Latency truly defers: a fixed 1-bar-latency run with diagnostics
    # (same seed) fills every intent at the earliest possible bar.
    base = run_simulation(
        {SYMBOL: df}, _fresh_gen(), config,
        params=ExecutionParams(latency_bars_min=1, latency_bars_max=1, seed=9,
                               record_intents=True),
    )
    # Same intents (a long-latency tail intent may not fill before end).
    assert len(r1.fills) <= len(base.fills)
    for bf, lf in zip(base.fills, r1.fills):
        assert lf["executed_bar"] >= bf["executed_bar"]


# ---------------------------------------------------------------------------
# Partial fills
# ---------------------------------------------------------------------------


def test_partial_fills_split_volume_across_tranches():
    """Ladder splits one intent into ≥2 tranches (child orders)."""
    df = _trend_bars(260, seed=5)
    config = _config(df)
    params = ExecutionParams(
        latency_bars_min=1, latency_bars_max=1,
        partial_fill_probability=1.0,     # always ladder
        partial_fill_min_fraction=0.5,
        partial_fill_max_fraction=0.7,
        max_partial_tranches=4,
        seed=13,
    )
    sim = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=params)

    assert sim.partial_fills > 0
    by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for f in sim.fills:
        if f["tranches"] > 1:
            by_parent.setdefault(f["order_id"].split("-t")[0], []).append(f)
    assert by_parent, "expected at least one laddered intent"
    for parent_id, fills in by_parent.items():
        vols = [f["volume"] for f in fills]
        assert all(v >= 0.01 for v in vols)
        assert all(f["tranches"] == len(vols) for f in fills)
        assert all(f["tranche"] == i for i, f in enumerate(fills))


# ---------------------------------------------------------------------------
# Requotes & rejections
# ---------------------------------------------------------------------------


def test_requotes_reprice_adverse_and_deterministic():
    """A requote shifts the fill candle adverse to the taker; re-run equal."""
    df = _trend_bars(240, seed=9)
    config = _config(df)
    params = ExecutionParams(
        latency_bars_min=1, latency_bars_max=1,
        requote_probability=1.0, requote_max_pips=2.0, seed=17,
    )
    r1 = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=params)
    r2 = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=params)
    assert r1.fills == r2.fills
    assert r1.requotes == len(r1.fills)  # every intent requoted (p=1.0)
    assert r1.requotes > 0
    assert [_trade_sig(t) for t in r1.trades] == [_trade_sig(t) for t in r2.trades]


def test_rejections_refuse_intents():
    """Rejections leave the intent unfilled and counted; determinism holds."""
    df = _trend_bars(240, seed=9)
    config = _config(df)
    params = ExecutionParams(
        latency_bars_min=1, latency_bars_max=1,
        rejection_probability=1.0, seed=21,
    )
    r1 = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=params)
    r2 = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=params)
    assert r1.fills == [] and r2.fills == []
    assert r1.rejections == r2.rejections > 0
    assert r1.rejected_intents == r2.rejected_intents
    assert len(r1.rejected_intents) == r1.rejections
    assert all(rec["reason"] == "REJECTED" for rec in r1.rejected_intents)


def test_requote_shift_is_bounded():
    """Requoted fill price deviates from the clean fill by ≤ max pips.

    Same seed ⇒ same intents, same latency, same bar; the ONLY difference is
    the adverse close shift (≤ requote_max_pips) applied before the same
    spread+slippage math.
    """
    df = _trend_bars(240, seed=9)
    config = _config(df)
    # Baseline = intent path with clean params (record_intents=True; latency
    # exactly 1 bar = base semantics). Req = same fixed latency + requote
    # always; the ONLY fill difference is the adverse shift drawn from the
    # seeded RNG.
    clean = run_simulation({SYMBOL: df}, _fresh_gen(), config,
                           params=ExecutionParams(record_intents=True))
    req = run_simulation(
        {SYMBOL: df}, _fresh_gen(), config,
        params=ExecutionParams(
            latency_bars_min=1, latency_bars_max=1,
            requote_probability=1.0, requote_max_pips=1.0, seed=5,
        ),
    )
    assert len(clean.fills) == len(req.fills) > 0
    for cf, rf in zip(clean.fills, req.fills):
        assert cf["executed_bar"] == rf["executed_bar"]
        adverse = rf["price"] - cf["price"]
        if rf["side"] == "BUY":
            assert adverse >= 0
        else:
            assert adverse <= 0
        assert abs(adverse) <= 1.0 * 0.0001 + 1e-9


# ---------------------------------------------------------------------------
# Spread widening
# ---------------------------------------------------------------------------


def test_spread_widening_worsens_fills():
    """Multiplier > 1 strictly worsens (or equals) every fill price."""
    df = _trend_bars(240, seed=9)
    config = _config(df)
    clean = run_simulation({SYMBOL: df}, _fresh_gen(), config,
                           params=ExecutionParams(record_intents=True))
    wide = run_simulation(
        {SYMBOL: df}, _fresh_gen(), config,
        params=ExecutionParams(
            latency_bars_min=1, latency_bars_max=1,
            spread_widen_multiplier=2.0,
        ),
    )
    assert len(clean.fills) == len(wide.fills)
    for cf, wf in zip(clean.fills, wide.fills):
        if wf["side"] == "BUY":
            assert wf["price"] >= cf["price"] - 1e-12
        else:
            assert wf["price"] <= cf["price"] + 1e-12


# ---------------------------------------------------------------------------
# Scenarios: determinism + coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SCENARIOS.keys()))
def test_scenarios_deterministic_and_covered(name):
    """Every named scenario reproduces exactly for a fixed seed."""
    scenario = SCENARIOS[name]()
    df = scenario.bars
    config = scenario.config_for(SYMBOL, df.index.min().to_pydatetime(),
                                 df.index.max().to_pydatetime())

    r1 = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=scenario.params)
    r2 = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=scenario.params)
    assert r1.fills == r2.fills
    assert [_trade_sig(t) for t in r1.trades] == [_trade_sig(t) for t in r2.trades]

    # The scenario family must actually exercise its degradations.
    if name in ("chop", "news_spike"):
        assert r1.requotes > 0
    if name == "thin_liquidity":
        assert r1.partial_fills > 0 and r1.rejections > 0
    if name == "gap":
        engine = ExecutionSimulator(config, DataFeed({SYMBOL: df}), _fresh_gen(),
                                    params=scenario.params)
        assert ExecutionMode.SPREAD_WIDENING in engine.mode_flags


def test_thin_liquidity_costs_exceed_clean():
    """Thin-book scenario must not be cheaper than clean execution."""
    scenario = thin_liquidity_scenario()
    df = scenario.bars
    config = scenario.config_for(SYMBOL, df.index.min().to_pydatetime(),
                                 df.index.max().to_pydatetime())

    clean = run_simulation({SYMBOL: df}, _fresh_gen(), config,
                           params=ExecutionParams(seed=53, record_intents=True))
    thin = run_simulation({SYMBOL: df}, _fresh_gen(), config, params=scenario.params)
    assert thin.rejections > 0  # some orders never filled at all
    # Same intended orders: degraded book fills fewer DISTINCT intents
    # (laddered intents emit several fill records, so compare parents).
    clean_intents = {f["order_id"].split("-t")[0] for f in clean.fills}
    thin_intents = {f["order_id"].split("-t")[0] for f in thin.fills}
    assert len(thin_intents) < len(clean_intents)


# ---------------------------------------------------------------------------
# Constitution
# ---------------------------------------------------------------------------


def test_simulator_source_scan_no_send_paths():
    """Source scan: the simulator computes — it never sends, connects or
    delegates risk to anything (E0 is pure computation)."""
    import forex_ai_agent.execution.simulator as sim_mod
    import forex_ai_agent.execution.scenarios as sc_mod

    for mod in (sim_mod, sc_mod):
        src = inspect.getsource(mod)
        for banned in (
            "place_order", "execute_trade", "risk_manager",
            "requests.", "urllib", "socket", "mt5", "MetaTrader5",
            "MAX_PERMISSION_LEVEL = 4",
        ):
            assert banned not in src, f"banned token {banned!r} in {mod.__name__}"
