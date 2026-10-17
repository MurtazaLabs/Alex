"""Structure snapshot builder: wires all detectors together.

``build_structure_snapshot`` takes raw OHLC arrays and returns a complete
``StructureSnapshot`` used by the MarketAgent and downstream rules.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

from .consolidation import detect_breakout, detect_consolidation, detect_retest
from .levels import identify_levels, key_levels
from .liquidity import detect_liquidity_sweeps
from .models import Direction, StructureEvent, StructureSnapshot, Trend
from .structure_events import detect_bos, detect_choch
from .swings import detect_swings
from .trend import classify_trend, trend_direction

DEFAULT_SWING_LEFT = 5
DEFAULT_SWING_RIGHT = 5
DEFAULT_SWING_STRENGTH = 0.15


def build_structure_snapshot(
    timeframe: str,
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    atr_pips: Optional[float] = None,
    pip_size: float = 0.0001,
    timestamps: Optional[Sequence[datetime]] = None,
    sw_left: int = DEFAULT_SWING_LEFT,
    sw_right: int = DEFAULT_SWING_RIGHT,
    min_strength: float = DEFAULT_SWING_STRENGTH,
) -> StructureSnapshot:
    """
    Compute a full market-structure snapshot from OHLC candles.

    Pipeline: swings -> trend -> BOS/CHoCH -> S/R levels -> liquidity sweeps
    -> consolidation / breakout / retest. All deterministic.
    """
    highs_list = list(highs)
    lows_list = list(lows)
    closes_list = list(closes)
    if not (len(highs_list) == len(lows_list) == len(closes_list)) or not closes_list:
        raise ValueError("highs/lows/closes must be non-empty and equal length")

    current_price = float(closes_list[-1])

    sw_highs, sw_lows = detect_swings(
        highs_list,
        lows_list,
        left=sw_left,
        right=sw_right,
        min_strength=min_strength,
        timestamps=timestamps,
    )

    trend = classify_trend(sw_highs, sw_lows)
    bos = detect_bos(sw_highs, sw_lows, closes_list, trend=trend)
    choch = detect_choch(sw_highs, sw_lows)
    levels = identify_levels(sw_highs, sw_lows)
    key_s, key_r = key_levels(levels, current_price)

    sweeps = (
        detect_liquidity_sweeps(sw_lows, sw_highs, atr_pips, pip_size)
        if atr_pips
        else []
    )

    consolidation = detect_consolidation(
        highs_list, lows_list, closes_list, atr_pips, pip_size
    )
    breakout = detect_breakout(consolidation, closes_list)
    retest = None
    if breakout is not None:
        retest = detect_retest(levels, closes_list, breakout.direction, pip_size=pip_size)
        if consolidation is not None:
            consolidation.broken = True
            consolidation.breakout_direction = breakout.direction
            consolidation.breakout_index = len(closes_list) - 1

    events: list[StructureEvent] = []
    for event in (bos, choch, breakout):
        if event is not None:
            events.append(event)
    events.extend(
        StructureEvent(
            kind="LIQUIDITY_SWEEP",
            direction=swp.direction,
            index=swp.index,
            level=swp.swept_price,
            price=swp.swept_price,
            confidence=swp.confidence,
            details=f"sweep depth {swp.depth_atr:.2f} ATR",
        )
        for swp in sweeps
    )

    primary = _primary_direction(bos, choch, trend)
    return StructureSnapshot(
        timeframe=timeframe,
        current_price=current_price,
        trend=trend,
        swing_highs=sw_highs,
        swing_lows=sw_lows,
        last_bos=bos,
        last_choch=choch,
        key_support=key_s,
        key_resistance=key_r,
        levels=levels,
        liquidity_sweeps=sweeps,
        consolidation=consolidation,
        retests=[retest] if retest else [],
        events=events,
    )


def _primary_direction(
    bos: Optional[StructureEvent],
    choch: Optional[StructureEvent],
    trend: Trend,
) -> Direction:
    """Direction the structure most strongly supports right now."""
    if trend in (Trend.BULLISH, Trend.BEARISH):
        return Direction.LONG if trend == Trend.BULLISH else Direction.SHORT
    # Mixed trend: defer to BOS then CHoCH
    for event in (bos, choch):
        if event is not None and event.direction in (Direction.LONG, Direction.SHORT):
            return event.direction
    return Direction.NONE


__all__ = ["build_structure_snapshot"]