"""
Price action helpers: swing detection and candle pattern classification.

These are deterministic building blocks for the market structure agent (Phase 2).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ._utils import as_float_ndarray


def detect_swing_points(
    highs, lows, left: int = 5, right: int = 5
) -> Tuple[List[int], List[int]]:
    """
    Detect swing highs and swing lows (fractal pivots).

    A bar at index ``i`` is a swing high when it is the strict maximum of the
    window ``[i-left, i+right]`` (and vice-versa for swing lows).

    Returns ``(swing_high_indices, swing_low_indices)``.
    """
    highs = as_float_ndarray(highs)
    lows = as_float_ndarray(lows)
    n = len(highs)
    swing_highs: List[int] = []
    swing_lows: List[int] = []

    for i in range(left, n - right):
        window_hi = highs[i - left : i + right + 1]
        window_lo = lows[i - left : i + right + 1]
        if highs[i] == np.max(window_hi):
            swing_highs.append(i)
        if lows[i] == np.min(window_lo):
            swing_lows.append(i)
    return swing_highs, swing_lows


def candle_direction(open_: float, close: float) -> str:
    """Return 'BULLISH', 'BEARISH' or 'NEUTRAL' for a candle."""
    if close > open_:
        return "BULLISH"
    if close < open_:
        return "BEARISH"
    return "NEUTRAL"


def is_swing_high_broken(highs, idx_a: int, idx_b: int) -> bool:
    """Whether the swing high at ``idx_a`` was exceeded by the bar at ``idx_b``."""
    highs = as_float_ndarray(highs)
    return bool(highs[idx_b] > highs[idx_a])


def is_swing_low_broken(lows, idx_a: int, idx_b: int) -> bool:
    """Whether the swing low at ``idx_a`` was undercut by the bar at ``idx_b``."""
    lows = as_float_ndarray(lows)
    return bool(lows[idx_b] < lows[idx_a])
