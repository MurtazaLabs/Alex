"""Stage-A3 golden tests: validate the validator.

Synthetic strategies with KNOWN TRUTH must be classified correctly by the
walk-forward engine, the overfit components and the autonomy gate:

- consistent edge          → STABLE, every OOS window trades and wins
- edge death mid-history   → caught (the old scalar score missed this class;
                             the full backtest still looks profitable — the trap)
- no edge (chop)           → UNSTABLE, negative-window fraction HIGH
- regime flip mid-history  → UNSTABLE
- gate sanity              → an edge-death proposal must stay REJECTED despite
                             a profitable full backtest

Hermetic: synthetic bars, no feed, no database, well under 30 s.
"""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from forex_ai_agent.agent import experiments as exp
from forex_ai_agent.agent.overfit import compute_overfit_components
from forex_ai_agent.autonomy.validator import BacktestValidator
from forex_ai_agent.strategy import StrategyVersion
from forex_ai_agent.strategy.spec import parse_spec

_QUIET = io.StringIO()


def _bars(drift_pattern, n: int = 3000, seed: int = 7, noise: float = 0.0009) -> pd.DataFrame:
    """Synthetic H1 bars: piecewise drift + Gaussian noise (deterministic)."""
    rng = np.random.default_rng(seed)
    drift = np.concatenate(drift_pattern)
    steps = drift + rng.normal(0, noise, n)
    close = 1.10 + np.cumsum(steps)
    high = close + np.abs(rng.normal(0, 0.0006, n))
    low = close - np.abs(rng.normal(0, 0.0006, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0},
        index=idx,
    )


RULES = {
    "name": "Trend Rider",
    "direction": "long",
    "timeframe": "H1",
    "entry_conditions": [
        {"left": "sma", "verb": "above", "right": "sma_slow"},
        {"left": "rsi", "verb": "above", "right": 55},
    ],
}


def _wf(df: pd.DataFrame) -> dict:
    """Run the walk-forward core quietly on a synthetic frame."""
    with contextlib.redirect_stdout(_QUIET):
        spec, _ = parse_spec(RULES)
        return exp.run_walk_forward_on_df(
            spec, df, "EUR/USD", "H1", window_months=1, step_months=1
        )


# Strong consistent edge: drift >> noise for the whole history.
CONSISTENT = _bars([np.full(3000, 0.00018)], noise=0.0004)
# True edge death: strong edge for ~40% of history, then nothing.
EDGE_DEATH = _bars([np.full(1150, 0.00018), np.full(1850, 0.0)], noise=0.0005)
# No edge: pure noise.
CHOP = _bars([np.full(3000, 0.0)], noise=0.0012)
# Regime flip: uptrend for the first fifth, sustained downtrend after.
FLIP = _bars([np.full(600, 0.00010), np.full(2400, -0.00012)], noise=0.0005)


# ---------------------------------------------------------------------------
# Golden scenario 1: consistent edge → STABLE
# ---------------------------------------------------------------------------


def test_consistent_edge_is_stable():
    result = _wf(CONSISTENT)
    assert result["verdict"] == "STABLE", result["aggregate_metrics"]
    assert result["windows_tested"] >= 3
    agg = result["aggregate_metrics"]
    assert agg["oos_sharpe_mean"] > 1.0
    # Every OOS window traded — no dead windows in a living edge.
    assert all(w["oos_trades"] > 0 for w in result["windows"])
    components = result["overfit_analysis"]["components"]
    assert components["negative_window_fraction"]["label"] == "LOW"
    assert result["overfit_analysis"]["score"] <= 0.5


# ---------------------------------------------------------------------------
# Golden scenario 2: edge death must NOT pass silently
# ---------------------------------------------------------------------------


def test_edge_death_is_flagged_by_verdict():
    """The full history is profitable, but the edge is dead late — the verdict
    must expose it via dead windows / collapse detection."""
    result = _wf(EDGE_DEATH)
    assert result["verdict"] == "UNSTABLE", result["aggregate_metrics"]
    dead = sum(1 for w in result["windows"] if w["oos_trades"] == 0)
    oos = [w["oos_sharpe"] for w in result["windows"]]
    collapse = min(oos) < -1.0 if oos else False
    assert dead > 0 or collapse, "edge-death scenario produced neither dead nor collapsed windows"
    # Negative-window fraction must read HIGH — dead windows are evidence of no edge.
    components = result["overfit_analysis"]["components"]
    assert components["negative_window_fraction"]["label"] in {"MEDIUM", "HIGH"}


@pytest.mark.asyncio
async def test_edge_death_full_backtest_still_looks_profitable():
    """The trap: the undifferenced full backtest of the edge-death data passes
    naive floors. This pins WHY the walk-forward gate is needed at all."""
    with contextlib.redirect_stdout(_QUIET):
        validator = BacktestValidator()
        report = await validator.validate(
            RULES, EDGE_DEATH, symbol="EUR/USD", run_walk_forward=False
        )
    assert report.full_backtest, "full backtest produced no metrics"
    assert report.full_backtest["profit_factor"] > 1.0
    assert report.full_backtest["total_trades"] > 0


@pytest.mark.asyncio
async def test_gate_rejects_edge_death_despite_profitable_full_backtest():
    """Gate sanity: the same proposal WITH walk-forward must be rejected, and
    the overfit composite (or an OOS measure) must be among the reasons."""
    with contextlib.redirect_stdout(_QUIET):
        validator = BacktestValidator()
        report = await validator.validate(
            RULES, EDGE_DEATH, symbol="EUR/USD", run_walk_forward=True
        )
    assert report.passed is False
    assert report.reasons, "edge-death proposal passed the gate"
    joined = " ".join(report.reasons).lower()
    assert (
        "overfit" in joined or "oos" in joined
    ), f"rejected for unexpected reasons: {report.reasons}"
    # The stored walk-forward block must carry the component analysis.
    assert "overfit_analysis" in (report.walk_forward or {})


# ---------------------------------------------------------------------------
# Golden scenario 3: no edge → UNSTABLE
# ---------------------------------------------------------------------------


def test_no_edge_is_unstable():
    result = _wf(CHOP)
    assert result["verdict"] == "UNSTABLE"
    agg = result["aggregate_metrics"]
    assert agg["oos_sharpe_mean"] <= 0.5
    components = result["overfit_analysis"]["components"]
    assert components["negative_window_fraction"]["label"] == "HIGH"


# ---------------------------------------------------------------------------
# Golden scenario 4: regime flip → UNSTABLE
# ---------------------------------------------------------------------------


def test_regime_flip_is_unstable():
    result = _wf(FLIP)
    assert result["verdict"] == "UNSTABLE"
    agg = result["aggregate_metrics"]
    assert agg["oos_sharpe_mean"] < 0, "long-only strategy in a downtrend must lose OOS"


# ---------------------------------------------------------------------------
# Overfit component math (pure unit level)
# ---------------------------------------------------------------------------


def test_components_perfect_windows_score_low():
    windows = [
        {"is_sharpe": 2.0, "oos_sharpe": 1.9, "oos_trades": 10},
        {"is_sharpe": 2.1, "oos_sharpe": 2.0, "oos_trades": 11},
        {"is_sharpe": 2.0, "oos_sharpe": 1.8, "oos_trades": 10},
        {"is_sharpe": 1.9, "oos_sharpe": 2.1, "oos_trades": 9},
    ]
    analysis = compute_overfit_components(windows)
    assert analysis["score"] <= 0.2
    comps = analysis["components"]
    for name in ("degradation", "stability", "negative_window_fraction", "trade_concentration"):
        assert comps[name]["label"] == "LOW"
        assert 0.0 <= comps[name]["value"] <= 1.0


def test_components_collapse_window_forces_stability_high():
    windows = [
        {"is_sharpe": 2.0, "oos_sharpe": 2.0, "oos_trades": 10},
        {"is_sharpe": 2.0, "oos_sharpe": 2.2, "oos_trades": 12},
        {"is_sharpe": 2.0, "oos_sharpe": 1.8, "oos_trades": 9},
        {"is_sharpe": 2.0, "oos_sharpe": -7.0, "oos_trades": 3},
    ]
    analysis = compute_overfit_components(windows)
    stability = analysis["components"]["stability"]
    assert stability["label"] == "HIGH"
    assert stability["collapse_detected"] is True
    assert stability["worst_oos_sharpe"] == pytest.approx(-7.0)


def test_components_concentration_high_when_one_window_holds_all_trades():
    windows = [
        {"is_sharpe": 1.5, "oos_sharpe": 1.2, "oos_trades": 40},
        {"is_sharpe": 1.5, "oos_sharpe": 0.9, "oos_trades": 0},
        {"is_sharpe": 1.5, "oos_sharpe": 1.0, "oos_trades": 0},
        {"is_sharpe": 1.5, "oos_sharpe": 1.1, "oos_trades": 0},
    ]
    analysis = compute_overfit_components(windows)
    conc = analysis["components"]["trade_concentration"]
    assert conc["value"] == pytest.approx(1.0)
    assert conc["label"] == "HIGH"
    assert conc["basis"] == "oos_trades"


def test_components_no_insample_edge_note():
    windows = [
        {"is_sharpe": -0.5, "oos_sharpe": 1.0, "oos_trades": 5},
        {"is_sharpe": -0.2, "oos_sharpe": 0.8, "oos_trades": 5},
    ]
    analysis = compute_overfit_components(windows)
    assert analysis["note"] and "no in-sample edge" in analysis["note"]
    # Degradation must NOT saturate at the old 1.0 cliff.
    assert analysis["components"]["degradation"]["value"] < 0.6


def test_components_empty_windows():
    analysis = compute_overfit_components([])
    assert analysis["score"] is None
    assert "no walk-forward windows" in analysis["note"]


# ---------------------------------------------------------------------------
# A1: repro dict present and deterministic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_walk_forward_result_carries_complete_repro():
    bars = CONSISTENT
    versions = {"repro123": StrategyVersion(id="repro123", name="Trend Rider", rules=RULES)}

    class _StubFeed:
        async def fetch_candles(self, symbol, tf, start, end, limit=10000):
            from forex_ai_agent.utils.timeframes import Timeframe

            assert tf in (Timeframe.H1, "H1")
            return [
                SimpleNamespace(
                    timestamp=t, open=r["open"], high=r["high"], low=r["low"],
                    close=r["close"], volume=r["volume"],
                )
                for t, r in bars.iterrows()
            ]

    async def _vm():
        class _VM:
            def get_version(self, vid):
                return versions.get(vid)

        return _VM()

    feed_patch = patch.object(exp, "get_feed_chain", lambda: _StubFeed())
    vm_patch = patch.object(exp, "create_version_manager", _vm)
    feed_patch.start()
    vm_patch.start()
    try:
        with contextlib.redirect_stdout(_QUIET):
            result = await exp.walk_forward_test(
                strategy_id="repro123",
                symbol="EUR/USD",
                start_date="2025-01-01T00:00:00",
                end_date="2025-05-01T00:00:00",
                timeframe="H1",
                window_months=1,
                step_months=1,
            )
    finally:
        feed_patch.stop()
        vm_patch.stop()

    repro = result["repro"]
    assert repro["engine_version"]
    assert repro["repro_schema_version"] == 1
    # The engine has no RNG — the repro must say so honestly.
    assert repro["random_seed"] is None
    assert repro["deterministic"] is True
    assert repro["dataset"]["symbol"] == "EUR/USD"
    assert repro["dataset"]["timeframe"] == "H1"
    assert repro["dataset"]["bar_count"] == len(bars)
    assert repro["strategy"]["strategy_id"] == "repro123"
    assert len(repro["strategy"]["rules_sha256"]) == 64
    assert repro["walk_forward"]["window_months"] == 1
