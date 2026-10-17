"""Break of Structure (BOS) and Change of Character (CHoCH) detection."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .models import Direction, StructureEvent, SwingPoint, Trend


def detect_bos(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    closes,
    trend: Trend = Trend.UNKNOWN,
) -> Optional[StructureEvent]:
    """
    Detect the most recent Break of Structure.

    A BOS-LONG is confirmed when the last confirmed swing high has been
    exceeded by a subsequent close; BOS-SHORT mirrors on swing lows. When both
    are (rarely) active the structure emits only the most recent.
    """
    closes_arr = np.asarray(closes, dtype=float)

    long_candidates: list[StructureEvent] = []
    for sh in swing_highs:
        if sh.index + 1 >= len(closes_arr):
            continue
        crossed = np.flatnonzero(closes_arr[sh.index + 1 :] > sh.price)
        if crossed.size:
            idx = sh.index + 1 + int(crossed[0])
            confidence = _confidence(crossed=True, trend_aligns=trend in (Trend.BULLISH, Trend.UNKNOWN))
            long_candidates.append(
                StructureEvent(
                    kind="BOS",
                    direction=Direction.LONG,
                    index=idx,
                    level=sh.price,
                    price=float(closes_arr[idx]),
                    confidence=confidence,
                    details=f"broke swing high at {sh.price:.5f}",
                )
            )

    short_candidates: list[StructureEvent] = []
    for sl in swing_lows:
        if sl.index + 1 >= len(closes_arr):
            continue
        crossed = np.flatnonzero(closes_arr[sl.index + 1 :] < sl.price)
        if crossed.size:
            idx = sl.index + 1 + int(crossed[0])
            confidence = _confidence(crossed=True, trend_aligns=trend in (Trend.BEARISH, Trend.UNKNOWN))
            short_candidates.append(
                StructureEvent(
                    kind="BOS",
                    direction=Direction.SHORT,
                    index=idx,
                    level=sl.price,
                    price=float(closes_arr[idx]),
                    confidence=confidence,
                    details=f"broke swing low at {sl.price:.5f}",
                )
            )

    # Prefer the most recent event (largest index), all else equal.
    candidates = long_candidates + short_candidates
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.index)


def detect_choch(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
) -> Optional[StructureEvent]:
    """
    Detect a Change of Character.

    A CHoCH is a structural inversion:
    - after HH/HL (bullish) a break below the last swing low = SHORT CHoCH
    - after LH/LL (bearish) a break above the last swing high = LONG CHoCH
    """
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        bullish = (
            swing_highs[-1].price > swing_highs[-2].price
            and swing_lows[-1].price > swing_lows[-2].price
        )
        bearish = (
            swing_highs[-1].price < swing_highs[-2].price
            and swing_lows[-1].price < swing_lows[-2].price
        )
        if bullish:
            pivot = swing_lows[-1]
            return StructureEvent(
                kind="CHoCH",
                direction=Direction.SHORT,
                index=pivot.index,
                level=pivot.price,
                price=pivot.price,
                confidence=0.75,
                details="bullish HH/HL then break below last swing low",
            )
        if bearish:
            pivot = swing_highs[-1]
            return StructureEvent(
                kind="CHoCH",
                direction=Direction.LONG,
                index=pivot.index,
                level=pivot.price,
                price=pivot.price,
                confidence=0.75,
                details="bearish LH/LL then break above last swing high",
            )
    return None


def _confidence(crossed: bool, trend_aligns: bool) -> float:
    """Deterministic confidence blending crossing + trend agreement."""
    base = 0.65
    if crossed:
        base += 0.1
    if trend_aligns:
        base += 0.15
    return round(min(base, 1.0), 2)


__all__ = ["detect_bos", "detect_choch"]