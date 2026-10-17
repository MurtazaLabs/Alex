"""Market structure detection package (Phase 2).

Deterministic analysis: swing detection, trend classification, BOS/CHoCH,
support/resistance, liquidity sweeps, consolidation and breakouts. The
``MarketAgent`` uses ``build_structure_snapshot`` to compute snapshots from
raw OHLC data.
"""
from .consolidation import detect_breakout, detect_consolidation, detect_retest
from .engine import build_structure_snapshot
from .levels import identify_levels, key_levels
from .liquidity import detect_liquidity_sweeps
from .models import (
    Consolidation,
    Direction,
    Level,
    LiquiditySweep,
    Retest,
    StructureEvent,
    StructureSnapshot,
    SwingPoint,
    Trend,
)
from .structure_events import detect_bos, detect_choch
from .swings import detect_swings
from .trend import (
    classify_trend,
    is_higher_high,
    is_higher_low,
    structure_sequence_label,
    trend_direction,
)

__all__ = [
    "build_structure_snapshot",
    "detect_swings",
    "classify_trend",
    "is_higher_high",
    "is_higher_low",
    "structure_sequence_label",
    "trend_direction",
    "detect_bos",
    "detect_choch",
    "identify_levels",
    "key_levels",
    "detect_liquidity_sweeps",
    "detect_consolidation",
    "detect_breakout",
    "detect_retest",
    "Trend",
    "Direction",
    "SwingPoint",
    "StructureEvent",
    "Level",
    "LiquiditySweep",
    "Consolidation",
    "Retest",
    "StructureSnapshot",
]