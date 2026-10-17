"""Consolidation, breakout and retest detection."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .models import Consolidation, Direction, Retest, StructureEvent


def detect_consolidation(
    highs,
    lows,
    closes,
    atr_pips: Optional[float] = None,
    pip_size: float = 0.0001,
    window: int = 20,
    compression_atr: float = 0.5,
) -> Optional[Consolidation]:
    """
    Detect a recent range contraction (consolidation).

    A consolidation exists when the high-low range of the trailing ``window``
    candles is narrower than ``compression_atr`` multiples of ATR.
    """
    highs_arr = np.asarray(highs, dtype=float)
    lows_arr = np.asarray(lows, dtype=float)
    n = len(highs_arr)
    if n < window or atr_pips is None or atr_pips <= 0:
        return None

    lo, hi = int(max(0, n - window)), n
    window_high = float(highs_arr[lo:hi].max())
    window_low = float(lows_arr[lo:hi].min())
    range_pips = (window_high - window_low) / pip_size

    if range_pips > compression_atr * atr_pips:
        return None

    return Consolidation(
        start_index=lo,
        end_index=n - 1,
        high=window_high,
        low=window_low,
    )


def detect_breakout(
    consolidation: Optional[Consolidation],
    closes,
) -> Optional[StructureEvent]:
    """Detect the breakout direction out of a consolidation range."""
    if consolidation is None:
        return None
    closes_arr = np.asarray(closes, dtype=float)
    recent_close = float(closes_arr[-1])
    buffer = (consolidation.high - consolidation.low) * 0.05  # 5% above range

    if recent_close > consolidation.high + buffer:
        return StructureEvent(
            kind="BREAKOUT",
            direction=Direction.LONG,
            index=len(closes_arr) - 1,
            level=consolidation.high,
            price=recent_close,
            confidence=0.8,
            details="breakout above consolidation high",
        )
    if recent_close < consolidation.low - buffer:
        return StructureEvent(
            kind="BREAKOUT",
            direction=Direction.SHORT,
            index=len(closes_arr) - 1,
            level=consolidation.low,
            price=recent_close,
            confidence=0.8,
            details="breakout below consolidation low",
        )
    return None


def detect_retest(
    levels,
    closes,
    direction: Direction,
    tolerance_pips: float = 5.0,
    pip_size: float = 0.0001,
) -> Optional[Retest]:
    """Detect a retest of a key level after a breakout."""
    closes_arr = np.asarray(closes, dtype=float)
    recent = float(closes_arr[-1])
    tolerance = tolerance_pips * pip_size
    candidates = []
    for level in levels:
        if direction == Direction.LONG and level.kind == "resistance":
            if abs(recent - level.price) <= tolerance:
                candidates.append(level)
        elif direction == Direction.SHORT and level.kind == "support":
            if abs(recent - level.price) <= tolerance:
                candidates.append(level)
    if not candidates:
        return None
    level = max(candidates, key=lambda lv: (lv.strength, lv.touches))
    return Retest(
        index=len(closes_arr) - 1,
        level=level.price,
        direction=direction,
        success=True,
        confidence=round(0.5 + 0.25 * level.strength, 2),
    )


__all__ = ["detect_consolidation", "detect_breakout", "detect_retest"]