"""Unit tests verifying indicator calculations against known values."""

from __future__ import annotations

import numpy as np
import pytest

from forex_ai_agent.analysis.indicators import (
    atr,
    bollinger_bands,
    candle_direction,
    detect_swing_points,
    ema,
    macd,
    rsi,
    sma,
    stochastic,
    true_range,
)


def test_sma_known_values():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = sma(values, period=3)
    assert np.isnan(result[0]) and np.isnan(result[1])
    assert result[2] == pytest.approx(2.0)
    assert result[3] == pytest.approx(3.0)
    assert result[4] == pytest.approx(4.0)


def test_ema_known_values():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = ema(values, period=3)  # alpha = 0.5, seed = 2.0
    assert np.isnan(result[0]) and np.isnan(result[1])
    assert result[2] == pytest.approx(2.0)
    assert result[3] == pytest.approx(3.0)
    assert result[4] == pytest.approx(4.0)


def test_rsi_monotonic_up_is_100():
    closes = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = rsi(closes, period=1)
    assert np.isnan(result[0])
    assert np.allclose(result[1:], 100.0)


def test_rsi_monotonic_down_is_0():
    closes = [5.0, 4.0, 3.0, 2.0, 1.0]
    result = rsi(closes, period=1)
    assert np.isnan(result[0])
    assert np.allclose(result[1:], 0.0)


def test_rsi_flat_gives_50():
    closes = [1.0, 1.0, 1.0, 1.0]
    result = rsi(closes, period=1)
    assert result[1] == pytest.approx(50.0)


def test_true_range_known():
    highs = [10, 11, 12]
    lows = [9, 9, 8]
    closes = [9.5, 10.0, 10.5]
    tr = true_range(highs, lows, closes)
    assert np.isnan(tr[0])
    assert tr[1] == pytest.approx(2.0)  # 11-9
    assert tr[2] == pytest.approx(4.0)  # 12-8


def test_atr_known_values():
    highs = [10, 11, 12, 13]
    lows = [9, 9, 8, 11]
    closes = [9.5, 10.0, 10.5, 11.0]
    result = atr(highs, lows, closes, period=2)
    assert np.isnan(result[0]) and np.isnan(result[1])
    # tr[2] = max(12-8, |12-10.0|, |8-10.0|) = 4.0
    # tr[1] = max(11-9, |11-9.5|, |9-9.5|) = 2.0 -> mean 3.0
    assert result[2] == pytest.approx(3.0)
    # tr[3] = max(13-11, |13-10.5|, |11-10.5|) = 2.5 -> mean(4.0, 2.5) = 3.25
    assert result[3] == pytest.approx(3.25)


def test_bollinger_known_values():
    closes = [1.0, 2.0, 3.0, 4.0, 5.0]
    bands = bollinger_bands(closes, period=3, num_std=2.0)
    assert np.isnan(bands["middle"][0])
    assert bands["middle"][2] == pytest.approx(2.0)
    assert bands["middle"][4] == pytest.approx(4.0)
    # population std of [1,2,3] = sqrt(2/3)
    expected_std = np.sqrt(2.0 / 3.0)
    assert bands["upper"][2] == pytest.approx(2.0 + 2.0 * expected_std)
    assert bands["lower"][2] == pytest.approx(2.0 - 2.0 * expected_std)


def test_macd_shape_and_sign():
    closes = np.linspace(1.0, 10.0, 200)
    out = macd(closes, fast=12, slow=26, signal=9)
    assert out["macd"].shape == closes.shape
    assert out["signal"].shape == closes.shape
    assert out["histogram"].shape == closes.shape
    # Strictly rising series: fast EMA always above slow EMA -> positive MACD
    assert np.nanmean(out["macd"]) > 0


def test_stochastic_bounds():
    highs = [10, 11, 12, 13, 14, 15]
    lows = [9, 9, 9, 9, 9, 9]
    closes = [9.5, 10.0, 11.0, 12.0, 13.0, 14.0]
    out = stochastic(highs, lows, closes, period=3)
    valid = out[~np.isnan(out)]
    assert (valid >= 0).all() and (valid <= 100).all()


def test_detect_swing_points():
    highs = [1, 2, 3, 2, 1, 4, 1, 2]
    lows = [1, 1, 0, 1, 1, 0, 1, 2]
    sh, sl = detect_swing_points(highs, lows, left=2, right=2)
    assert 2 in sh  # local max of the first window
    assert 5 in sh  # higher local max
    assert 2 in sl  # local min of lows
    assert 5 in sl


def test_candle_direction():
    assert candle_direction(1.0, 1.1) == "BULLISH"
    assert candle_direction(1.1, 1.0) == "BEARISH"
    assert candle_direction(1.0, 1.0) == "NEUTRAL"
