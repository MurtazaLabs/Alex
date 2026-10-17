"""Market structure domain models (Phase 2).

These are deterministic, LLM-free data structures: the market structure agent
computes them from raw candles and downstream agents (signal / risk / strategy)
consume them.

See ``docs/Forex docmentation/EVENT_ARCHITECTURE.md`` for the event names these
models also get emitted as.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class Trend(str, Enum):
    """Higher-timeframe trend classification."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class Direction(str, Enum):
    """Trade / structure direction."""

    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class SwingPoint:
    """A confirmed swing high or swing low (fractal pivot)."""

    index: int
    price: float
    kind: str  # "high" | "low"
    strength: float = 1.0  # 0..1 dominance within its detection window
    timestamp: Optional[datetime] = None

    @property
    def is_high(self) -> bool:
        return self.kind == "high"


@dataclass
class StructureEvent:
    """A market-structure event (BOS, CHoCH, breakout, retest, sweep)."""

    kind: str  # BOS | CHoCH | BREAKOUT | LIQUIDITY_SWEEP | RETEST
    direction: Direction
    index: int
    level: float
    price: float
    confidence: float  # 0..1
    details: str = ""


@dataclass
class Level:
    """A support or resistance level."""

    price: float
    kind: str  # "support" | "resistance"
    touches: int = 1
    strength: float = 0.0  # 0..1 dominance, usually touches-based


@dataclass
class LiquiditySweep:
    """A liquidity sweep event: price took out an equal extreme and returned."""

    index: int
    direction: Direction  # LONG means buy-side liquidity swept above
    swept_price: float
    depth_atr: float  # how deep the sweep went in ATR units
    confidence: float


@dataclass
class Consolidation:
    """A compression range (range contraction)."""

    start_index: int
    end_index: int
    high: float
    low: float
    broken: bool = False
    breakout_direction: Optional[Direction] = None
    breakout_index: Optional[int] = None


@dataclass
class Retest:
    """A retest of a broken structural level."""

    index: int
    level: float
    direction: Direction
    success: bool
    confidence: float


@dataclass
class StructureSnapshot:
    """
    A computed market-structure snapshot for one timeframe.

    All values are derived deterministically from OHLC data by
    ``market_structure.engine.build_structure_snapshot``.
    """

    timeframe: str
    current_price: float
    trend: Trend = Trend.UNKNOWN
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    last_bos: Optional[StructureEvent] = None
    last_choch: Optional[StructureEvent] = None
    key_support: Optional[float] = None
    key_resistance: Optional[float] = None
    levels: List[Level] = field(default_factory=list)
    liquidity_sweeps: List[LiquiditySweep] = field(default_factory=list)
    consolidation: Optional[Consolidation] = None
    retests: List[Retest] = field(default_factory=list)
    events: List[StructureEvent] = field(default_factory=list)

    @property
    def last_swing_high(self) -> Optional[SwingPoint]:
        """Most recent confirmed swing high (alias matching docs naming)."""
        return self.swing_highs[-1] if self.swing_highs else None

    @property
    def last_swing_low(self) -> Optional[SwingPoint]:
        """Most recent confirmed swing low (alias matching docs naming)."""
        return self.swing_lows[-1] if self.swing_lows else None

    def to_dict(self) -> dict:
        """JSON-friendly serialization for events / signal evidence storage."""
        return {
            "timeframe": self.timeframe,
            "current_price": self.current_price,
            "trend": self.trend.value,
            "swing_high_count": len(self.swing_highs),
            "swing_low_count": len(self.swing_lows),
            "last_swing_high": (
                self.last_swing_high.price if self.last_swing_high else None
            ),
            "last_swing_low": (
                self.last_swing_low.price if self.last_swing_low else None
            ),
            "last_bos": (
                {
                    "direction": self.last_bos.direction.value,
                    "level": self.last_bos.level,
                    "confidence": self.last_bos.confidence,
                }
                if self.last_bos
                else None
            ),
            "last_choch": (
                {
                    "direction": self.last_choch.direction.value,
                    "level": self.last_choch.level,
                    "confidence": self.last_choch.confidence,
                }
                if self.last_choch
                else None
            ),
            "key_support": self.key_support,
            "key_resistance": self.key_resistance,
            "events": [
                {
                    "kind": e.kind,
                    "direction": e.direction.value,
                    "index": e.index,
                    "level": e.level,
                    "price": e.price,
                    "confidence": e.confidence,
                    "details": e.details,
                }
                for e in self.events
            ],
        }


__all__ = [
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