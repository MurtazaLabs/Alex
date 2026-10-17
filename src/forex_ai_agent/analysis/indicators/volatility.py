"""Volatility indicators: ATR, Bollinger Bands."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ._utils import as_float_ndarray


def true_range(highs, lows, closes) -> np.ndarray:
    """True Range = max(high-low, |high-prev_close|, |low-prev_close|)."""
    highs = as_float_ndarray(highs)
    lows = as_float_ndarray(lows)
    closes = as_float_ndarray(closes)
    n = len(closes)
    if n < 2:
        raise ValueError("Need at least two bars to compute true range")

    prev_close = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)),
    )
    tr[0] = np.nan  # no previous close for the first bar
    return tr


def atr(highs, lows, closes, period: int = 14) -> np.ndarray:
    """
    Average True Range (SMA of True Range).

    Returns an array aligned with the input; warm-up entries are NaN.
    """
    tr = true_range(highs, lows, closes)
    n = len(tr)
    result = np.full(n, np.nan)
    if n < period + 1:
        return result
    # Drop the leading NaN (no previous close) before smoothing
    kernel = np.ones(period, dtype=float) / period
    sma_tr = np.convolve(tr[1:], kernel, mode="valid")
    # First valid value sits at index `period` (0-based): tr[1:period+1]
    result[period : period + len(sma_tr)] = sma_tr
    return result


def bollinger_bands(
    closes, period: int = 20, num_std: float = 2.0
) -> Dict[str, np.ndarray]:
    """
    Bollinger Bands centred on a simple moving average.

    Returns ``{"middle": ..., "upper": ..., "lower": ...}`` aligned with input.
    """
    closes = as_float_ndarray(closes)
    n = len(closes)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    if n < period:
        return {"middle": middle, "upper": upper, "lower": lower}

    kernel = np.ones(period, dtype=float) / period
    mean = np.convolve(closes, kernel, mode="valid")

    # Rolling standard deviation (population) over the identical windows
    std = np.empty(len(mean))
    for i in range(len(mean)):
        window = closes[i : i + period]
        std[i] = np.std(window)

    start = period - 1
    middle[start : start + len(mean)] = mean
    upper[start : start + len(mean)] = mean + num_std * std
    lower[start : start + len(mean)] = mean - num_std * std
    return {"middle": middle, "upper": upper, "lower": lower}
