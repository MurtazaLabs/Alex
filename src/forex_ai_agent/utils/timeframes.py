"""Timeframe helpers for the Forex AI Suite.

Supported timeframes follow the MT5 convention used across the docs:
M1, M5, M15, M30, H1, H4, D1.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Sequence

TIMEFRAME_ORDER: List[str] = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

_MINUTES: dict = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}

_DISPLAY: dict = {
    "M1": "1 Minute",
    "M5": "5 Minutes",
    "M15": "15 Minutes",
    "M30": "30 Minutes",
    "H1": "1 Hour",
    "H4": "4 Hours",
    "D1": "1 Day",
}


class Timeframe(str, Enum):
    """Canonical timeframe identifiers."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"

    @property
    def minutes(self) -> int:
        """Duration of this timeframe in minutes."""
        return _MINUTES[self.value]

    @property
    def display_name(self) -> str:
        """Human-readable label for the timeframe."""
        return _DISPLAY[self.value]

    @property
    def position(self) -> int:
        """Zero-based index in the canonical timeframe ordering."""
        return TIMEFRAME_ORDER.index(self.value)


def is_valid_timeframe(value: str) -> bool:
    """Check whether the value is a supported timeframe."""
    return value in TIMEFRAME_ORDER


def timeframe_minutes(timeframe: str) -> int:
    """Return the duration in minutes for a timeframe string."""
    if not is_valid_timeframe(timeframe):
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    return _MINUTES[timeframe]


def higher_timeframes(
    timeframe: str, values: Optional[Sequence[str]] = None
) -> List[str]:
    """
    Return timeframes with *longer* duration than the given one.

    e.g. ``higher_timeframes("M15") == ["M30", "H1", "H4", "D1"]``.
    """
    if not is_valid_timeframe(timeframe):
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    order = list(values) if values else TIMEFRAME_ORDER
    candidates = order if values else TIMEFRAME_ORDER
    return [tf for tf in candidates if _MINUTES[tf] > _MINUTES[timeframe]]


def lower_timeframes(
    timeframe: str, values: Optional[Sequence[str]] = None
) -> List[str]:
    """
    Return timeframes with *shorter* duration than the given one.

    e.g. ``lower_timeframes("H1") == ["M1", "M5", "M15", "M30"]``.
    """
    if not is_valid_timeframe(timeframe):
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    candidates = list(values) if values else TIMEFRAME_ORDER
    return [tf for tf in candidates if _MINUTES[tf] < _MINUTES[timeframe]]
