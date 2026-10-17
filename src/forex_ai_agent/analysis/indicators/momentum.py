"""Momentum indicators: RSI, Stochastic oscillator."""

from __future__ import annotations

import numpy as np

from ._utils import as_float_ndarray


def rsi(closes, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index (Wilder / standard) with SMA-smoothed gains/losses.

    Returns an array aligned with the input; warm-up entries are NaN.
    """
    closes = as_float_ndarray(closes)
    if period < 1:
        raise ValueError("period must be >= 1")

    result = np.full(len(closes), np.nan)
    if len(closes) < period + 1:
        return result

    delta = np.diff(closes)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # SMA over period of the gain/loss series (length = n - period)
    kernel = np.ones(period, dtype=float) / period
    avg_gain = np.convolve(gains, kernel, mode="valid")
    avg_loss = np.convolve(losses, kernel, mode="valid")

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi_values = 100.0 - (100.0 / (1.0 + rs))
        # Edge cases: zero average loss
        rsi_values = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, rsi_values)
        rsi_values = np.where((avg_loss == 0) & (avg_gain == 0), 50.0, rsi_values)
        rsi_values = np.where(np.isinf(rs), 100.0, rsi_values)

    start = period  # first valid RSI index
    result[start : start + len(rsi_values)] = rsi_values
    return result


def stochastic(highs, lows, closes, period: int = 14) -> np.ndarray:
    """
    Stochastic oscillator (%K).

    Returns a value in [0, 100] aligned with input length (NaN warm-up).
    """
    highs = as_float_ndarray(highs)
    lows = as_float_ndarray(lows)
    closes = as_float_ndarray(closes)
    if period < 1:
        raise ValueError("period must be >= 1")
    n = len(closes)
    result = np.full(n, np.nan)
    if n < period:
        return result

    for i in range(period - 1, n):
        high_max = np.max(highs[i - period + 1 : i + 1])
        low_min = np.min(lows[i - period + 1 : i + 1])
        rng = high_max - low_min
        if rng == 0:
            result[i] = 50.0
        else:
            result[i] = 100.0 * (closes[i] - low_min) / rng
    return result
