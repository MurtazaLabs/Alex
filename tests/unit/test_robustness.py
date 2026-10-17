"""Stage-B3 unit tests: cross-pair / regime robustness sweep.

Pins the roadmap acceptance: a strategy's walk-forward can be swept across its
intended pair plus the majors with every OOS window regime-tagged, the sweep
is stored on the version, and the health card shows all seven lights with
PAIR GENERALITY / REGIME ROBUSTNESS backed by per-pair, per-regime WF rows.

Hermetic: synthetic bars (golden-test style), fake feed, temp version store.
"""

from __future__ import annotations

import contextlib
import io
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from forex_ai_agent.agent import robustness as rb
from forex_ai_agent.agent.robustness import (
    classify_regime,
    run_robustness_sweep,
)

_QUIET = io.StringIO()

RULES = {
    "name": "Trend Rider",
    "direction": "long",
    "timeframe": "H1",
    "entry_conditions": [
        {"left": "sma", "verb": "above", "right": "sma_slow"},
        {"left": "rsi", "verb": "above", "right": 55},
    ],
}


def _bars(drift: float, n: int = 3000, seed: int = 7, noise: float = 0.0006) -> pd.DataFrame:
    """Synthetic H1 bars with constant drift (deterministic per seed)."""
    rng = np.random.default_rng(seed)
    steps = np.full(n, drift) + rng.normal(0, noise, n)
    close = 1.10 + np.cumsum(steps)
    high = close + np.abs(rng.normal(0, 0.0006, n))
    low = close - np.abs(rng.normal(0, 0.0006, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0},
        index=idx,
    )


class _Candle:
    def __init__(self, ts, o, h, l, c):
        self.timestamp = ts
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = 1000.0


class _FakeFeed:
    """Serves pre-built DataFrames as candles, per symbol."""

    def __init__(self, frames: dict):
        self.frames = frames

    async def fetch_candles(self, symbol, tf, start, end=None, limit=None):
        df = self.frames[symbol]
        return [
            _Candle(ts, r["open"], r["high"], r["low"], r["close"])
            for ts, r in df.iterrows()
        ]


@pytest.fixture()
def temp_vm(monkeypatch, tmp_path):
    """Version manager over a temp store, with one saved strategy."""
    from forex_ai_agent.strategy import StrategyVersion, StrategyVersionManager
    from forex_ai_agent.strategy.spec import parse_spec

    import forex_ai_agent.agent.robustness as rb_mod

    vm = StrategyVersionManager(storage_path=str(tmp_path / "versions"))
    spec, _ = parse_spec(RULES)
    version = vm.create_version(
        name="Trend Rider",
        description="test",
        rules=RULES,
        parameters={"symbol": "GBP/JPY", "timeframe": "H1"},
    )
    monkeypatch.setattr(
        rb_mod, "create_version_manager", lambda: _async_vm(vm), raising=False
    )

    class _VMProxy:
        def __getattr__(self, name):
            return getattr(vm, name)

    # run_robustness_sweep imports create_version_manager from strategy pkg
    # via a local import — patch there too for consistency.

    async def _async_vm():
        return vm

    yield vm, version


def _patch_core(monkeypatch, frames: dict):
    """Patch the feed and run the sweep quietly."""
    import forex_ai_agent.data.providers as providers_mod

    fake = _FakeFeed(frames)
    monkeypatch.setattr(providers_mod, "get_feed_chain", lambda: fake)
    return fake


# ---------------------------------------------------------------------------
# classify_regime — deterministic tagging
# ---------------------------------------------------------------------------


def test_classify_regime_trending():
    n = 300
    rng = np.random.default_rng(3)
    close = 1.10 + np.cumsum(np.full(n, 0.0002) + rng.normal(0, 0.0002, n))
    assert classify_regime(close, frame_vol=0.0004) == "trending"


def test_classify_regime_ranging():
    n = 300
    rng = np.random.default_rng(3)
    close = 1.10 + rng.normal(0, 0.0006, n).cumsum() * 0  # flat + tiny noise
    close = 1.10 + rng.normal(0, 0.0002, n)
    assert classify_regime(close, frame_vol=0.0006) == "ranging"


def test_classify_regime_volatile():
    n = 300
    rng = np.random.default_rng(5)
    calm = 0.0002
    wild = rng.normal(0, 0.004, n)  # huge noise, no drift
    close = 1.10 + np.cumsum(wild)
    assert classify_regime(close, frame_vol=calm) == "volatile"


def test_classify_regime_short_series_falls_back_to_ranging():
    assert classify_regime([1.0, 1.1]) == "ranging"
    assert classify_regime([1.0] * 50) == "ranging"  # zero vol


def test_base_pair_resolution_includes_gate_report():
    """Pair resolution ladder: parameters → rules → stored gate metrics."""
    from types import SimpleNamespace

    from forex_ai_agent.agent.robustness import _resolve_base_pair

    assert _resolve_base_pair(SimpleNamespace(parameters={"symbol": "GBP/JPY"}, rules={}, backtest_results={})) == "GBP/JPY"
    assert _resolve_base_pair(SimpleNamespace(parameters={}, rules={"symbol": "EUR/USD"}, backtest_results={})) == "EUR/USD"
    # f39b8d40 pattern: pair only lives in the stored gate report
    assert _resolve_base_pair(SimpleNamespace(parameters={}, rules={}, backtest_results={"symbol": "USD/JPY"})) == "USD/JPY"
    assert _resolve_base_pair(SimpleNamespace(parameters={}, rules={}, backtest_results={})) == ""


# ---------------------------------------------------------------------------
# run_robustness_sweep — cross-pair sweep over synthetic data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_runs_per_pair_tags_regimes_and_saves(monkeypatch, tmp_path):
    from forex_ai_agent.strategy import StrategyVersionManager

    vm = StrategyVersionManager(storage_path=str(tmp_path / "v"))
    version = vm.create_version(
        name="Trend Rider", rules=RULES, parameters={"symbol": "GBP/JPY", "timeframe": "H1"}
    )
    frames = {
        "GBP/JPY": _bars(0.00018, seed=7),
        "EUR/USD": _bars(0.00015, seed=11),
        "GBP/USD": _bars(0.0, seed=13, noise=0.0012),  # chop pair
    }
    import forex_ai_agent.data.providers as providers_mod
    import forex_ai_agent.strategy as strategy_pkg

    monkeypatch.setattr(providers_mod, "get_feed_chain", lambda: _FakeFeed(frames))

    async def fake_vm():
        return vm

    monkeypatch.setattr(strategy_pkg, "create_version_manager", fake_vm)

    with contextlib.redirect_stdout(_QUIET):
        result = await run_robustness_sweep(
            version.id,
            pairs=["GBP/JPY", "EUR/USD", "GBP/USD"],
            window_months=1,
            step_months=1,
        )

    # per-pair rows for every requested pair, base flagged
    by_symbol = {r["symbol"]: r for r in result["pairs"]}
    assert set(by_symbol) == {"GBP/JPY", "EUR/USD", "GBP/USD"}
    assert by_symbol["GBP/JPY"]["base"] is True
    assert by_symbol["GBP/JPY"]["status"] == "OK"
    assert by_symbol["GBP/JPY"]["windows_tested"] >= 3

    # every OOS window carries a regime tag
    for row in result["pairs"]:
        for w in row["windows"]:
            assert w["regime"] in ("trending", "ranging", "volatile")

    # a trending synthetic edge must be tagged trending somewhere
    regimes_seen = {w["regime"] for r in result["pairs"] for w in r["windows"]}
    assert "trending" in regimes_seen

    # per-regime aggregation present with real numbers
    assert result["per_regime"], "regime buckets missing"
    for name, r in result["per_regime"].items():
        assert r["windows"] > 0 and r["distinct_pairs"] >= 1
        assert isinstance(r["mean_oos_sharpe"], float)

    # verdict + budget accounting
    assert result["verdict"] in ("ROBUST", "MIXED", "FRAGILE", "NO_DATA")
    assert result["budget"]["backtests_run"] > 0
    assert result["budget"]["pairs_run"] == 3

    # stored on the version (save=True default) under backtest_results.robustness
    stored = vm.get_version(version.id)
    assert stored.backtest_results["robustness"]["evaluated_at"] == result["evaluated_at"]
    assert stored.backtest_results["robustness"]["verdict"] == result["verdict"]
    # repro carried through (strategy id lives under repro.strategy)
    assert result["repro"]["strategy"]["strategy_id"] == version.id


@pytest.mark.asyncio
async def test_sweep_default_pairs_include_base_and_majors(monkeypatch, tmp_path):
    from forex_ai_agent.strategy import StrategyVersionManager

    vm = StrategyVersionManager(storage_path=str(tmp_path / "v"))
    version = vm.create_version(
        name="Trend Rider", rules=RULES, parameters={"symbol": "AUD/NZD", "timeframe": "H1"}
    )
    frames = {s: _bars(0.0001, n=2600, seed=s.__hash__() % 100) for s in
              ("AUD/NZD", "EUR/USD", "GBP/USD", "USD/JPY", "EXTRA/PAIR")}
    import forex_ai_agent.data.providers as providers_mod
    import forex_ai_agent.strategy as strategy_pkg

    monkeypatch.setattr(providers_mod, "get_feed_chain", lambda: _FakeFeed(frames))

    async def fake_vm():
        return vm

    monkeypatch.setattr(strategy_pkg, "create_version_manager", fake_vm)

    with contextlib.redirect_stdout(_QUIET):
        result = await run_robustness_sweep(version.id, window_months=1, step_months=1)

    swept = [r["symbol"] for r in result["pairs"]]
    assert swept[0] == "AUD/NZD"  # base pair first
    assert "EXTRA/PAIR" not in swept  # cap enforced
    assert len(swept) <= 4
    # default universe pads with majors after the base pair
    assert "EUR/USD" in swept and "GBP/USD" in swept


@pytest.mark.asyncio
async def test_sweep_survives_a_failing_pair(monkeypatch, tmp_path):
    from forex_ai_agent.strategy import StrategyVersionManager

    vm = StrategyVersionManager(storage_path=str(tmp_path / "v"))
    version = vm.create_version(
        name="Trend Rider", rules=RULES, parameters={"symbol": "GBP/JPY", "timeframe": "H1"}
    )
    frames = {"GBP/JPY": _bars(0.00018, seed=7)}

    class _ExplodingFeed(_FakeFeed):
        async def fetch_candles(self, symbol, tf, start, end=None, limit=None):
            if symbol != "GBP/JPY":
                raise RuntimeError("feed boom")
            return await super().fetch_candles(symbol, tf, start, end, limit)

    import forex_ai_agent.data.providers as providers_mod
    import forex_ai_agent.strategy as strategy_pkg

    monkeypatch.setattr(providers_mod, "get_feed_chain", lambda: _ExplodingFeed(frames))

    async def fake_vm():
        return vm

    monkeypatch.setattr(strategy_pkg, "create_version_manager", fake_vm)

    with contextlib.redirect_stdout(_QUIET):
        result = await run_robustness_sweep(
            version.id, pairs=["GBP/JPY", "EUR/USD"], window_months=1, step_months=1
        )

    by_symbol = {r["symbol"]: r for r in result["pairs"]}
    assert by_symbol["GBP/JPY"]["status"] == "OK"
    assert by_symbol["EUR/USD"]["status"] == "ERROR"
    assert "feed boom" in by_symbol["EUR/USD"]["note"]
    # the good pair still produced evidence
    assert by_symbol["GBP/JPY"]["windows_tested"] >= 3
    assert result["pairs_summary"]["usable"] == 1


@pytest.mark.asyncio
async def test_sweep_skips_pairs_with_thin_history(monkeypatch, tmp_path):
    from forex_ai_agent.strategy import StrategyVersionManager

    vm = StrategyVersionManager(storage_path=str(tmp_path / "v"))
    version = vm.create_version(
        name="Trend Rider", rules=RULES, parameters={"symbol": "GBP/JPY", "timeframe": "H1"}
    )
    frames = {"GBP/JPY": _bars(0.00018, seed=7), "EUR/USD": _bars(0.0001, n=100, seed=9)}

    import forex_ai_agent.data.providers as providers_mod
    import forex_ai_agent.strategy as strategy_pkg

    monkeypatch.setattr(providers_mod, "get_feed_chain", lambda: _FakeFeed(frames))

    async def fake_vm():
        return vm

    monkeypatch.setattr(strategy_pkg, "create_version_manager", fake_vm)

    with contextlib.redirect_stdout(_QUIET):
        result = await run_robustness_sweep(
            version.id, pairs=["GBP/JPY", "EUR/USD"], window_months=1, step_months=1
        )

    by_symbol = {r["symbol"]: r for r in result["pairs"]}
    assert by_symbol["EUR/USD"]["status"] == "SKIPPED"
    assert "need" in by_symbol["EUR/USD"]["note"]
    assert by_symbol["EUR/USD"]["windows_tested"] == 0


@pytest.mark.asyncio
async def test_sweep_unknown_strategy_raises():
    from forex_ai_agent.strategy import create_version_manager as real_cvm

    with pytest.raises(ValueError):
        await run_robustness_sweep("nonexist")


def test_render_robustness_text_deterministic():
    result = {
        "strategy_name": "Trend Rider",
        "strategy_id": "abc12345",
        "base_pair": "GBP/JPY",
        "timeframe": "H1",
        "verdict": "ROBUST",
        "pairs_summary": {
            "profitable": 3,
            "usable": 3,
            "profitable_pairs": ["GBP/JPY", "EUR/USD", "GBP/USD"],
        },
        "per_regime": {
            "trending": {"windows": 8, "positive_windows": 7, "mean_oos_sharpe": 0.9,
                          "worst_oos_sharpe": 0.2, "distinct_pairs": 3},
        },
        "budget": {"backtests_run": 24, "pairs_run": 3},
    }
    text = rb.render_robustness_text(result)
    assert text == rb.render_robustness_text(result)
    assert "ROBUST" in text and "Trend Rider" in text
    assert "3/3 OOS-profitable" in text
    assert "Regime trending" in text


# ---------------------------------------------------------------------------
# Health: the two new lights read the stored sweep
# ---------------------------------------------------------------------------


def test_health_seven_lights_from_sweep():
    from forex_ai_agent.agent.health import compute_strategy_health
    import sys

    sys.path.insert(0, "tests/unit")
    from test_health import _gate, _wf, _sweep, _NOW

    health = compute_strategy_health(_gate(_wf(), robustness=_sweep()), now=_NOW)
    assert [l["name"] for l in health["lights"]][-2:] == [
        "PAIR GENERALITY",
        "REGIME ROBUSTNESS",
    ]
    by_name = {l["name"]: l for l in health["lights"]}
    assert by_name["PAIR GENERALITY"]["status"] == "GREEN"
    assert by_name["REGIME ROBUSTNESS"]["status"] in ("GREEN", "AMBER")
    assert health["badge"] == "CANDIDATE"


def test_health_failing_pairs_make_generality_red():
    from forex_ai_agent.agent.health import compute_strategy_health
    import sys

    sys.path.insert(0, "tests/unit")
    from test_health import _gate, _wf, _sweep, _NOW

    sweep = _sweep(profitable=1, failing=2)
    health = compute_strategy_health(_gate(_wf(), robustness=sweep), now=_NOW)
    by_name = {l["name"]: l for l in health["lights"]}
    assert by_name["PAIR GENERALITY"]["status"] == "RED"
    assert health["badge"] == "RETIRE?"


def test_health_collapsed_regime_makes_regime_light_red():
    from forex_ai_agent.agent.health import compute_strategy_health
    import sys

    sys.path.insert(0, "tests/unit")
    from test_health import _gate, _wf, _sweep, _NOW

    sweep = _sweep(
        regimes={
            "trending": {"windows": 8, "trades": 40, "distinct_pairs": 3,
                         "mean_oos_sharpe": 0.9, "positive_windows": 7,
                         "worst_oos_sharpe": 0.2},
            "volatile": {"windows": 4, "trades": 10, "distinct_pairs": 2,
                         "mean_oos_sharpe": -1.4, "positive_windows": 0,
                         "worst_oos_sharpe": -2.2},  # collapse
        }
    )
    health = compute_strategy_health(_gate(_wf(), robustness=sweep), now=_NOW)
    by_name = {l["name"]: l for l in health["lights"]}
    assert by_name["REGIME ROBUSTNESS"]["status"] == "RED"
    assert "worst window Sharpe -2.20" in by_name["REGIME ROBUSTNESS"]["detail"]


def test_health_thin_regime_buckets_are_grey_not_red():
    from forex_ai_agent.agent.health import compute_strategy_health
    import sys

    sys.path.insert(0, "tests/unit")
    from test_health import _gate, _wf, _sweep, _NOW

    sweep = _sweep(
        regimes={
            "trending": {"windows": 2, "trades": 5, "distinct_pairs": 1,
                         "mean_oos_sharpe": -0.9, "positive_windows": 0,
                         "worst_oos_sharpe": -0.9},
        }
    )
    health = compute_strategy_health(_gate(_wf(), robustness=sweep), now=_NOW)
    by_name = {l["name"]: l for l in health["lights"]}
    # 2 windows < _REGIME_WINDOWS_MIN → no measured regimes → GREY (missing
    # evidence), never a RED condemnation from noise.
    assert by_name["REGIME ROBUSTNESS"]["status"] == "GREY"
