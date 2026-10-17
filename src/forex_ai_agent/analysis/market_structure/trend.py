"""Trend classification from swing structure (HH/HL/LH/LL)."""
from __future__ import annotations

from typing import List, Optional

from .models import Direction, SwingPoint, Trend


def is_higher_high(swing_highs: List[SwingPoint]) -> bool:
    """True when the most recent swing high exceeds the previous one (HH)."""
    if len(swing_highs) < 2:
        return False
    return swing_highs[-1].price > swing_highs[-2].price


def is_higher_low(swing_lows: List[SwingPoint]) -> bool:
    """True when the most recent swing low is above the previous one (HL)."""
    if len(swing_lows) < 2:
        return False
    return swing_lows[-1].price > swing_lows[-2].price


def classify_trend(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
) -> Trend:
    """
    Classify market structure trend using the last two swing points of each type.

    - HH + HL            -> BULLISH
    - LH + LL            -> BEARISH
    - anything else      -> NEUTRAL (mixed / transitioning)
    - too few pivots     -> UNKNOWN

    This is intentionally simple and deterministic; it forms one *input* to the
    signal rule engine, not an oracle.
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return Trend.UNKNOWN

    hh = is_higher_high(swing_highs)
    hl = is_higher_low(swing_lows)

    if hh and hl:
        return Trend.BULLISH
    if not hh and not hl:
        return Trend.BEARISH
    return Trend.NEUTRAL


def trend_direction(trend: Trend) -> Direction:
    """Map a trend to the trading direction it supports."""
    if trend == Trend.BULLISH:
        return Direction.LONG
    if trend == Trend.BEARISH:
        return Direction.SHORT
    return Direction.NONE


def structure_sequence_label(swing_points: List[SwingPoint]) -> str:
    """Human-readable label of the most recent HH/HL/LH/LL sequence."""
    if len(swing_points) < 2:
        return "INSUFFICIENT"
    seq = []
    for a, b in zip(swing_points[:-1], swing_points[1:]):
        seq.append("H" if b.price > a.price else "L")
    return "".join(seq)


__all__ = [
    "is_higher_high",
    "is_higher_low",
    "classify_trend",
    "trend_direction",
    "structure_sequence_label",
]