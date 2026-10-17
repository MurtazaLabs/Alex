"""Unit tests for the mean-reversion strategy (Bollinger + RSI).

Pins:
- Oversold condition generates LONG with correct SL/TP math
- Overbought condition generates SHORT with correct SL/TP math
- Neutral data generates no signal
- ATR filter blocks dead markets
- Spread filter blocks wide-spread conditions
- scan_history finds all setups
- R:R ratio matches ATR multiplier ratio
"""

from __future__ import annotations

import numpy as np
import pytest

from forex_ai_agent.strategy.mean_reversion import (
    MeanReversionStrategy,
    MeanReversionParams,
    MeanReversionSignal,
)


class TestMeanReversionParams:
    def test_defaults(self):
        p = MeanReversionParams()
        assert p.bb_period == 20
        assert p.bb_std == 2.0
        assert p.rsi_period == 14
        assert p.rsi_oversold == 30.0
        assert p.rsi_overbought == 70.0
        assert p.sl_atr_mult == 1.5
        assert p.tp_atr_mult == 2.0


class TestMeanReversionSignal:
    def _oversold_data(self, n=50):
        """Price stable then sharp drop → oversold."""
        closes = np.full(n, 1.10)
        closes[-3:] = [1.098, 1.097, 1.096]
        return closes

    def _overbought_data(self, n=50):
        """Price stable then sharp rise → overbought."""
        closes = np.full(n, 1.10)
        closes[-3:] = [1.102, 1.103, 1.104]
        return closes

    def _neutral_data(self, n=50):
        """Price oscillating around middle → no signal."""
        rng = np.random.default_rng(42)
        return 1.10 + rng.uniform(-0.001, 0.001, n)

    def test_oversold_generates_long(self):
        s = MeanReversionStrategy()
        sig = s.generate_signal(self._oversold_data())
        assert sig.direction == "LONG"
        assert sig.entry > 0
        assert sig.sl < sig.entry  # SL below entry for LONG
        assert sig.tp > sig.entry  # TP above entry for LONG

    def test_overbought_generates_short(self):
        s = MeanReversionStrategy()
        sig = s.generate_signal(self._overbought_data())
        assert sig.direction == "SHORT"
        assert sig.sl > sig.entry  # SL above entry for SHORT
        assert sig.tp < sig.entry  # TP below entry for SHORT

    def test_neutral_generates_no_signal(self):
        s = MeanReversionStrategy()
        sig = s.generate_signal(self._neutral_data())
        assert sig.direction is None
        assert "no_setup" in sig.reason

    def test_insufficient_data(self):
        s = MeanReversionStrategy()
        sig = s.generate_signal(np.array([1.10, 1.101]))
        assert sig.direction is None
        assert sig.reason == "insufficient_data"

    def test_atr_filter_blocks_dead_market(self):
        """Flat price → ATR ≈ 0 → no signal."""
        s = MeanReversionStrategy(MeanReversionParams(min_atr=0.001))
        closes = np.full(50, 1.10)  # completely flat
        sig = s.generate_signal(closes)
        assert sig.direction is None
        assert "atr_too_low" in sig.reason

    def test_spread_filter_blocks_wide_spread(self):
        s = MeanReversionStrategy(MeanReversionParams(max_spread=0.0001))
        sig = s.generate_signal(
            self._oversold_data(), spread=0.001,  # 10x max spread
        )
        assert sig.direction is None
        assert "spread_too_wide" in sig.reason

    def test_rr_ratio_matches_atr_mult(self):
        """TP/SL distance ratio should equal tp_mult/sl_mult = 2.0/1.5."""
        s = MeanReversionStrategy()
        sig = s.generate_signal(self._oversold_data())
        if sig.direction == "LONG":
            risk = sig.entry - sig.sl
            reward = sig.tp - sig.entry
            assert reward / risk == pytest.approx(2.0 / 1.5, rel=0.01)

    def test_sl_distance_is_1_5_atr(self):
        s = MeanReversionStrategy()
        sig = s.generate_signal(self._oversold_data())
        if sig.direction == "LONG":
            sl_distance = sig.entry - sig.sl
            assert sl_distance == pytest.approx(1.5 * sig.atr, rel=0.01)

    def test_scan_history_finds_signals(self):
        s = MeanReversionStrategy()
        closes = self._oversold_data()
        signals = s.scan_history(closes)
        assert len(signals) >= 1
        assert all(sig.direction in ("LONG", "SHORT") for sig in signals)

    def test_scan_history_neutral_few_signals(self):
        s = MeanReversionStrategy()
        signals = s.scan_history(self._neutral_data())
        # Random data around middle should produce few/no signals
        assert len(signals) <= 2

    def test_signal_to_dict(self):
        s = MeanReversionStrategy()
        sig = s.generate_signal(self._oversold_data())
        d = sig.to_dict()
        assert "direction" in d
        assert "entry" in d
        assert "sl" in d
        assert "tp" in d
        assert "rsi" in d
        assert "atr" in d

    def test_long_sl_is_atr_based(self):
        """SL = entry - 1.5 * ATR."""
        s = MeanReversionStrategy()
        sig = s.generate_signal(self._oversold_data())
        if sig.direction == "LONG":
            expected_sl = sig.entry - 1.5 * sig.atr
            assert sig.sl == pytest.approx(expected_sl, rel=0.01)

    def test_short_sl_is_atr_based(self):
        """SL = entry + 1.5 * ATR."""
        s = MeanReversionStrategy()
        sig = s.generate_signal(self._overbought_data())
        if sig.direction == "SHORT":
            expected_sl = sig.entry + 1.5 * sig.atr
            assert sig.sl == pytest.approx(expected_sl, rel=0.01)
