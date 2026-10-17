"""Technical indicators package.

Phase 1 ships vectorized (NumPy) implementations of the core indicators; all
functions are pure and deterministic so they can be verified against known values.
"""

from .momentum import rsi, stochastic
from .price_action import (
    candle_direction,
    detect_swing_points,
    is_swing_high_broken,
    is_swing_low_broken,
)
from .trend import ema, macd, sma
from .volatility import atr, bollinger_bands, true_range

__all__ = [
    "sma",
    "ema",
    "macd",
    "rsi",
    "stochastic",
    "atr",
    "true_range",
    "bollinger_bands",
    "candle_direction",
    "detect_swing_points",
    "is_swing_high_broken",
    "is_swing_low_broken",
]
