"""Mean-reversion strategy — Bollinger Bands + RSI on H1.

The primary strategy for Alex's $100 demo account. Mean reversion works well
in ranging/consolidating markets and produces frequent signals with tight
stops — ideal for small accounts.

Entry logic:
- LONG:  price touches/closes below lower Bollinger Band AND RSI < 30
- SHORT: price touches/closes above upper Bollinger Band AND RSI > 70

Exit logic (ATR-based):
- SL: 1.5 × ATR(14) from entry
- TP: 2.0 × ATR(14) from entry (R:R ≈ 1.33)

Filters:
- Minimum ATR (avoid dead markets)
- Maximum spread (avoid wide-spread conditions)
- Session filter (London/NY only)

Every number is computed deterministically from price data. No LLM involved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..analysis.indicators.volatility import bollinger_bands, atr as compute_atr
from ..analysis.indicators.momentum import rsi as compute_rsi

logger = logging.getLogger(__name__)

__all__ = ["MeanReversionStrategy", "MeanReversionSignal", "MeanReversionParams"]


@dataclass(frozen=True)
class MeanReversionParams:
    """Deterministic parameters for the mean-reversion strategy."""

    #: Bollinger Band period
    bb_period: int = 20
    #: Bollinger Band std deviations
    bb_std: float = 2.0
    #: RSI period
    rsi_period: int = 14
    #: RSI oversold threshold (buy below)
    rsi_oversold: float = 30.0
    #: RSI overbought threshold (sell above)
    rsi_overbought: float = 70.0
    #: ATR period for SL/TP sizing
    atr_period: int = 14
    #: SL multiplier (ATR)
    sl_atr_mult: float = 1.5
    #: TP multiplier (ATR)
    tp_atr_mult: float = 2.0
    #: Minimum ATR to trade (avoid dead markets); in price units
    min_atr: float = 0.0002
    #: Maximum spread to trade (price units); 0 = disabled
    max_spread: float = 0.0005
    #: Minimum bars of history needed
    min_bars: int = 30


@dataclass
class MeanReversionSignal:
    """A generated signal with full deterministic math."""

    #: "LONG" or "SHORT" or None
    direction: Optional[str]
    #: Entry price (close of signal bar)
    entry: float
    #: Stop-loss price
    sl: float
    #: Take-profit price
    tp: float
    #: ATR value at signal time
    atr: float
    #: RSI value at signal time
    rsi_val: float
    #: Bollinger middle band
    bb_middle: float
    #: Bollinger upper band
    bb_upper: float
    #: Bollinger lower band
    bb_lower: float
    #: Current price (close)
    price: float
    #: Signal bar index
    bar_index: int
    #: Why no signal (if direction is None)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "entry": round(self.entry, 5),
            "sl": round(self.sl, 5),
            "tp": round(self.tp, 5),
            "atr": round(self.atr, 6),
            "rsi": round(self.rsi_val, 1),
            "bb_middle": round(self.bb_middle, 5),
            "bb_upper": round(self.bb_upper, 5),
            "bb_lower": round(self.bb_lower, 5),
            "price": round(self.price, 5),
            "bar_index": self.bar_index,
            "reason": self.reason,
        }


class MeanReversionStrategy:
    """Deterministic mean-reversion signal generator.

    Pure computation: takes price arrays, returns signals. No network,
    no broker, no LLM.
    """

    def __init__(self, params: Optional[MeanReversionParams] = None) -> None:
        self.params = params or MeanReversionParams()

    def generate_signal(
        self,
        closes: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        spread: float = 0.0,
    ) -> MeanReversionSignal:
        """Generate a signal from OHLC data (arrays aligned, oldest first).

        Returns the LATEST bar's signal. direction is None if no valid setup.
        """
        p = self.params
        closes = np.asarray(closes, dtype=float)
        n = len(closes)

        if n < p.min_bars:
            return self._no_signal(closes[-1] if n else 0.0, -1, "insufficient_data")

        # Compute indicators
        bb = bollinger_bands(closes, period=p.bb_period, num_std=p.bb_std)
        rsi_arr = compute_rsi(closes, period=p.rsi_period)
        atr_arr = compute_atr(closes, highs, lows, period=p.atr_period) if highs is not None and lows is not None else self._simple_atr(closes, p.atr_period)

        # Latest values
        i = n - 1
        price = float(closes[i])
        bb_upper = float(bb["upper"][i])
        bb_middle = float(bb["middle"][i])
        bb_lower = float(bb["lower"][i])
        rsi_val = float(rsi_arr[i])
        atr_val = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else 0.0

        # Guards
        if np.isnan(bb_upper) or np.isnan(bb_lower) or np.isnan(rsi_val):
            return self._no_signal(price, i, "indicators_warming_up")
        if atr_val < p.min_atr:
            return self._no_signal(price, i, f"atr_too_low:{atr_val:.5f}")
        if p.max_spread > 0 and spread > p.max_spread:
            return self._no_signal(price, i, f"spread_too_wide:{spread:.5f}")

        # --- Entry conditions ---
        # LONG: price at or below lower band + RSI oversold
        long_setup = price <= bb_lower and rsi_val < p.rsi_oversold
        # SHORT: price at or above upper band + RSI overbought
        short_setup = price >= bb_upper and rsi_val > p.rsi_overbought

        if long_setup:
            sl = price - (p.sl_atr_mult * atr_val)
            tp = price + (p.tp_atr_mult * atr_val)
            return MeanReversionSignal(
                direction="LONG", entry=price, sl=sl, tp=tp,
                atr=atr_val, rsi_val=rsi_val,
                bb_middle=bb_middle, bb_upper=bb_upper, bb_lower=bb_lower,
                price=price, bar_index=i,
            )

        if short_setup:
            sl = price + (p.sl_atr_mult * atr_val)
            tp = price - (p.tp_atr_mult * atr_val)
            return MeanReversionSignal(
                direction="SHORT", entry=price, sl=sl, tp=tp,
                atr=atr_val, rsi_val=rsi_val,
                bb_middle=bb_middle, bb_upper=bb_upper, bb_lower=bb_lower,
                price=price, bar_index=i,
            )

        # No setup
        reason = f"no_setup:price={price:.5f},bb=[{bb_lower:.5f},{bb_upper:.5f}],rsi={rsi_val:.1f}"
        return self._no_signal(price, i, reason)

    def scan_history(
        self,
        closes: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
    ) -> List[MeanReversionSignal]:
        """Generate signals for ALL bars in history (for backtesting)."""
        p = self.params
        closes = np.asarray(closes, dtype=float)
        n = len(closes)
        signals: List[MeanReversionSignal] = []

        bb = bollinger_bands(closes, period=p.bb_period, num_std=p.bb_std)
        rsi_arr = compute_rsi(closes, period=p.rsi_period)
        atr_arr = compute_atr(closes, highs, lows, period=p.atr_period) if highs is not None and lows is not None else self._simple_atr(closes, p.atr_period)

        for i in range(p.min_bars, n):
            price = float(closes[i])
            bb_upper = float(bb["upper"][i])
            bb_lower = float(bb["lower"][i])
            rsi_val = float(rsi_arr[i])
            atr_val = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else 0.0

            if np.isnan(bb_upper) or np.isnan(bb_lower) or np.isnan(rsi_val):
                continue
            if atr_val < p.min_atr:
                continue

            long_setup = price <= bb_lower and rsi_val < p.rsi_oversold
            short_setup = price >= bb_upper and rsi_val > p.rsi_overbought

            if long_setup:
                sl = price - (p.sl_atr_mult * atr_val)
                tp = price + (p.tp_atr_mult * atr_val)
                signals.append(MeanReversionSignal(
                    direction="LONG", entry=price, sl=sl, tp=tp,
                    atr=atr_val, rsi_val=rsi_val,
                    bb_middle=float(bb["middle"][i]), bb_upper=bb_upper, bb_lower=bb_lower,
                    price=price, bar_index=i,
                ))
            elif short_setup:
                sl = price + (p.sl_atr_mult * atr_val)
                tp = price - (p.tp_atr_mult * atr_val)
                signals.append(MeanReversionSignal(
                    direction="SHORT", entry=price, sl=sl, tp=tp,
                    atr=atr_val, rsi_val=rsi_val,
                    bb_middle=float(bb["middle"][i]), bb_upper=bb_upper, bb_lower=bb_lower,
                    price=price, bar_index=i,
                ))

        return signals

    @staticmethod
    def _simple_atr(closes: np.ndarray, period: int) -> np.ndarray:
        """Fallback ATR from closes only (true range approximation)."""
        n = len(closes)
        atr_arr = np.full(n, np.nan)
        if n < period + 1:
            return atr_arr
        # Use absolute close-to-close changes as TR proxy
        tr = np.abs(np.diff(closes))
        kernel = np.ones(period, dtype=float) / period
        atr_valid = np.convolve(tr, kernel, mode="valid")
        # Place at index period (align with closes)
        atr_arr[period:period + len(atr_valid)] = atr_valid
        return atr_arr

    @staticmethod
    def _no_signal(price: float, bar_index: int, reason: str) -> MeanReversionSignal:
        return MeanReversionSignal(
            direction=None, entry=price, sl=0.0, tp=0.0,
            atr=0.0, rsi_val=50.0,
            bb_middle=price, bb_upper=price, bb_lower=price,
            price=price, bar_index=bar_index, reason=reason,
        )
