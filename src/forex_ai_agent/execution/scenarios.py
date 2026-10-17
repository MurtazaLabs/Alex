"""Stage E0: named, seeded execution scenarios.

Each scenario is a pure data descriptor: bars + params + (optionally) a
spec for the signal generator. Everything is deterministic given the seed —
the same tuple reproduces the exact same fills, rejects and trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestConfig
from .simulator import ExecutionParams

__all__ = ["Scenario", "build_bars", "trend_scenario", "chop_scenario",
           "gap_scenario", "news_spike_scenario", "thin_liquidity_scenario",
           "SCENARIOS"]


@dataclass
class Scenario:
    """One named execution scenario (data + microstructure parameters)."""

    name: str
    description: str
    params: ExecutionParams
    bars: Optional[pd.DataFrame] = None
    timeframe: str = "H1"
    warmup_bars: int = 12
    seed: int = 7

    def config_for(self, symbol: str, start: datetime, end: datetime,
                   initial_balance: float = 10_000.0) -> BacktestConfig:
        return BacktestConfig(
            start_date=start,
            end_date=end,
            initial_balance=initial_balance,
            symbols=[symbol],
            timeframe=self.timeframe,
            warmup_bars=self.warmup_bars,
        )


def build_bars(
    kind: str,
    n: int = 300,
    seed: int = 11,
    start: datetime = datetime(2026, 8, 1),
) -> pd.DataFrame:
    """Deterministic synthetic OHLC series for scenarios."""
    rng = np.random.default_rng(seed)
    if kind == "trend":
        steps = np.full(n, 0.00018) + rng.normal(0, 0.0004, n)
    elif kind == "chop":
        steps = rng.normal(0, 0.0006, n)
    elif kind == "gap":
        steps = rng.normal(0, 0.0004, n)
        for i in range(50, n, 60):  # a gap every ~60 bars
            steps[i] += 0.004
    elif kind == "news_spike":
        steps = rng.normal(0, 0.0003, n)
        for i in range(80, n, 90):  # violent spike + revert
            steps[i] += 0.010
            if i + 1 < n:
                steps[i + 1] -= 0.009
    elif kind == "thin":
        steps = rng.normal(0, 0.00015, n)  # small moves, wide spreads
    else:  # pragma: no cover
        raise ValueError(f"unknown bar kind: {kind}")
    close = 1.10 + np.cumsum(steps)
    spread_noise = {"thin": 0.0012, "news_spike": 0.0008}.get(kind, 0.0003)
    return pd.DataFrame(
        {
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close + np.abs(rng.normal(0, spread_noise, n)),
            "low": close - np.abs(rng.normal(0, spread_noise, n)),
            "close": close,
            "volume": 1000.0,
        },
        index=pd.date_range(start, periods=n, freq="1h"),
    )


def trend_scenario(seed: int = 11) -> Scenario:
    return Scenario(
        name="trend",
        description="Steady trend with realistic latency and occasional partials",
        params=ExecutionParams(
            latency_bars_min=1, latency_bars_max=2,
            partial_fill_probability=0.2, seed=seed,
        ),
        bars=build_bars("trend", seed=seed),
    )


def chop_scenario(seed: int = 23) -> Scenario:
    return Scenario(
        name="chop",
        description="Range chop with requotes and flat rejections",
        params=ExecutionParams(
            latency_bars_min=1, latency_bars_max=2,
            requote_probability=0.25, requote_max_pips=1.5,
            rejection_probability=0.05, seed=seed,
        ),
        bars=build_bars("chop", seed=seed),
    )


def gap_scenario(seed: int = 31) -> Scenario:
    return Scenario(
        name="gap",
        description="Weekend-style gaps with latency and wider spreads",
        params=ExecutionParams(
            latency_bars_min=1, latency_bars_max=3,
            spread_widen_multiplier=1.5, seed=seed,
        ),
        bars=build_bars("gap", seed=seed),
    )


def news_spike_scenario(seed: int = 41) -> Scenario:
    return Scenario(
        name="news_spike",
        description="News spikes: widened spreads, requotes, partial ladders",
        params=ExecutionParams(
            latency_bars_min=1, latency_bars_max=2,
            spread_widen_multiplier=2.0,
            requote_probability=0.35, requote_max_pips=3.0,
            partial_fill_probability=0.4, seed=seed,
        ),
        bars=build_bars("news_spike", seed=seed),
    )


def thin_liquidity_scenario(seed: int = 53) -> Scenario:
    return Scenario(
        name="thin_liquidity",
        description="Thin book: heavy partials and elevated rejections",
        params=ExecutionParams(
            latency_bars_min=2, latency_bars_max=3,
            partial_fill_probability=0.6,
            partial_fill_min_fraction=0.3, partial_fill_max_fraction=0.6,
            rejection_probability=0.10, spread_widen_multiplier=1.8,
            seed=seed,
        ),
        bars=build_bars("thin", seed=seed),
    )


SCENARIOS = {
    "trend": trend_scenario,
    "chop": chop_scenario,
    "gap": gap_scenario,
    "news_spike": news_spike_scenario,
    "thin_liquidity": thin_liquidity_scenario,
}
