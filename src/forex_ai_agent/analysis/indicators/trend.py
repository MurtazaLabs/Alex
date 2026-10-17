"""Trend indicators: SMA, EMA, MACD."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ._utils import as_float_ndarray


def sma(values, period: int = 20) -> np.ndarray:
    """
    Simple moving average.

    Returns an array of the same length as ``values``; warm-up periods are NaN.
    """
    values = as_float_ndarray(values)
    if period < 1:
        raise ValueError("period must be >= 1")
    if len(values) < period:
        return np.full(len(values), np.nan)

    result = np.full(len(values), np.nan)
    kernel = np.ones(period, dtype=float) / period
    result[period - 1 :] = np.convolve(values, kernel, mode="valid")
    return result


def ema(values, period: int = 20) -> np.ndarray:
    """
    Exponential moving average (seed = SMA of the first ``period`` points).

    Returns an array of the same length as ``values``; warm-up periods are NaN.
    """
    values = as_float_ndarray(values)
    if period < 1:
        raise ValueError("period must be >= 1")

    alpha = 2.0 / (period + 1.0)
    result = np.full(len(values), np.nan)
    if len(values) < period:
        return result

    seed = np.mean(values[:period])
    result[period - 1] = seed
    for i in range(period, len(values)):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]
    return result


def macd(
    closes,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Dict[str, np.ndarray]:
    """
    MACD indicator.

    Returns ``{"macd": ..., "signal": ..., "histogram": ...}`` arrays aligned
    with the input length (NaN during warm-up).
    """
    closes = as_float_ndarray(closes)
    if fast >= slow:
        raise ValueError("fast period must be smaller than slow period")

    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)

    # Only emit where both commodity lines are defined
    valid = np.logical_and(~np.isnan(macd_line), ~np.isnan(signal_line))
    histogram = np.where(valid, macd_line - signal_line, np.nan)
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}
