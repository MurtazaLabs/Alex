"""Unit tests for the real walk-forward implementation in agent.experiments.

Hermetic: the feed chain and version manager are stubbed, the backtest
engine runs for real on synthetic H1 bars — no network, no database.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from forex_ai_agent.agent import experiments as exp
from forex_ai_agent.strategy import StrategyVersion

RULES = {
    "name": "Trend Rider",
    "direction": "long",
    "timeframe": "H1",
    "entry_conditions": [
        {"left": "sma", "verb": "above", "right": "sma_slow"},
        {"left": "rsi", "verb": "above", "right": 55},
    ],
}

NEVER_RULES = {
    "name": "Never Trades",
    "direction": "long",
    "timeframe": "H1",
    # RSI above 99 essentially never fires → every OOS window is flat.
    "entry_conditions": [{"left": "rsi", "verb": "above", "right": 99}],
}


def _bars(n: int = 1500, seed: int = 7) -> list:
    """Synthetic H1 bars with trend legs — same recipe as test_autonomy."""
    rng = np.random.default_rng(seed)
    drift = np.concatenate(
        [
            np.full(n // 3, 0.00008),
            np.full(n // 3, -0.00006),
            np.full(n - 2 * (n // 3), 0.00012),
        ]
    )
    steps = drift + rng.normal(0, 0.0009, n)
    close = 1.10 + np.cumsum(steps)
    high = close + np.abs(rng.normal(0, 0.0006, n))
    low = close - np.abs(rng.normal(0, 0.0006, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    return [
        SimpleNamespace(timestamp=t, open=o, high=h, low=l, close=c, volume=1000.0)
        for t, o, h, l, c in zip(idx, open_, high, low, close)
    ]


class _StubFeed:
    def __init__(self, bars: list) -> None:
        self.bars = bars

    async def fetch_candles(self, symbol, tf, start, end, limit=10000):
        return self.bars[:limit]


class _StubVM:
    def __init__(self, versions: dict) -> None:
        self._v = versions

    def get_version(self, vid: str):
        return self._v.get(vid)


def _install(bars: list, versions: dict):
    feed_patch = patch.object(exp, "get_feed_chain", lambda: _StubFeed(bars))

    async def _vm():
        return _StubVM(versions)

    vm_patch = patch.object(exp, "create_version_manager", _vm)
    feed_patch.start()
    vm_patch.start()
    return feed_patch, vm_patch


@pytest.mark.asyncio
async def test_walk_forward_returns_real_windows():
    bars = _bars()
    versions = {"wf12345": StrategyVersion(id="wf12345", name="Trend Rider", rules=RULES)}
    feed_patch, vm_patch = _install(bars, versions)
    try:
        result = await exp.walk_forward_test(
            strategy_id="wf12345",
            symbol="EUR/USD",
            start_date="2025-01-01T00:00:00",
            end_date="2025-04-01T00:00:00",
            timeframe="H1",
            window_months=1,
            step_months=1,
        )
    finally:
        feed_patch.stop()
        vm_patch.stop()

    assert result["windows_tested"] >= 2
    assert result["strategy_name"] == "Trend Rider"
    row = result["windows"][0]
    for key in ("window", "is_sharpe", "oos_sharpe", "oos_pf", "oos_trades", "oos_max_dd"):
        assert key in row, f"window row missing {key}"
    agg = result["aggregate_metrics"]
    for key in (
        "is_sharpe_mean",
        "oos_sharpe_mean",
        "oos_sharpe_std",
        "oos_pf_mean",
        "oos_trades_total",
        "oos_max_drawdown",
        "degradation_overfitting",
    ):
        assert key in agg, f"aggregate missing {key}"
    assert 0.0 <= result["stability_score"] <= 1.0
    assert result["verdict"] in {"STABLE", "MIXED", "UNSTABLE"}


@pytest.mark.asyncio
async def test_walk_forward_flat_strategy_is_unstable():
    """A strategy that never trades must score 0 OOS sharpe → UNSTABLE verdict."""
    bars = _bars()
    versions = {"flat1234": StrategyVersion(id="flat1234", name="Never", rules=NEVER_RULES)}
    feed_patch, vm_patch = _install(bars, versions)
    try:
        result = await exp.walk_forward_test(
            strategy_id="flat1234",
            symbol="EUR/USD",
            start_date="2025-01-01T00:00:00",
            end_date="2025-03-01T00:00:00",
            timeframe="H1",
            window_months=1,
            step_months=1,
        )
    finally:
        feed_patch.stop()
        vm_patch.stop()

    assert result["windows_tested"] >= 1
    assert result["aggregate_metrics"]["oos_sharpe_mean"] == 0.0
    assert result["aggregate_metrics"]["oos_trades_total"] == 0
    assert result["verdict"] == "UNSTABLE"
    assert result["stability_score"] > 0.5


@pytest.mark.asyncio
async def test_walk_forward_rejects_short_history():
    feed_patch, vm_patch = _install(_bars(100), {"wf12345": StrategyVersion(id="wf12345", rules=RULES)})
    try:
        with pytest.raises(ValueError, match="≥250"):
            await exp.walk_forward_test(
                strategy_id="wf12345",
                symbol="EUR/USD",
                start_date="2025-01-01T00:00:00",
                end_date="2025-01-06T00:00:00",
            )
    finally:
        feed_patch.stop()
        vm_patch.stop()


@pytest.mark.asyncio
async def test_walk_forward_rejects_history_too_short_for_windows():
    """300 bars cannot host a 6-month IS + 2-month OOS window → explicit error."""
    feed_patch, vm_patch = _install(_bars(300), {"wf12345": StrategyVersion(id="wf12345", rules=RULES)})
    try:
        with pytest.raises(ValueError, match="Not enough history for walk-forward"):
            await exp.walk_forward_test(
                strategy_id="wf12345",
                symbol="EUR/USD",
                start_date="2025-01-01T00:00:00",
                end_date="2025-01-14T00:00:00",
                window_months=6,
                step_months=2,
            )
    finally:
        feed_patch.stop()
        vm_patch.stop()


@pytest.mark.asyncio
async def test_walk_forward_unknown_strategy():
    feed_patch, vm_patch = _install(_bars(), {})
    try:
        with pytest.raises(ValueError, match="not found"):
            await exp.walk_forward_test(
                strategy_id="missing0",
                symbol="EUR/USD",
                start_date="2025-01-01T00:00:00",
                end_date="2025-02-01T00:00:00",
            )
    finally:
        feed_patch.stop()
        vm_patch.stop()


@pytest.mark.asyncio
async def test_walk_forward_invalid_rules():
    feed_patch, vm_patch = _install(_bars(), {"badrules": StrategyVersion(id="badrules", rules={})})
    try:
        with pytest.raises(ValueError, match="not a valid spec"):
            await exp.walk_forward_test(
                strategy_id="badrules",
                symbol="EUR/USD",
                start_date="2025-01-01T00:00:00",
                end_date="2025-02-01T00:00:00",
            )
    finally:
        feed_patch.stop()
        vm_patch.stop()
