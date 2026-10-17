"""Swing (fractal) point detection with configurable strength.

Extends the Phase 1 ``detect_swing_points`` helper with a ``min_strength``
filter: a pivot is only considered a swing when it dominates its surrounding
window by a configurable margin. Deterministic and vectorisable.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .models import SwingPoint


def _window_strength(window: np.ndarray, center_idx: int) -> float:
    """0..1 dominance of the centre value over the rest of the window."""
    center = window[center_idx]
    others = np.delete(window, center_idx)
    if others.size == 0:
        return 1.0
    others_max = others.max()
    if center <= others_max:
        return 0.0  # not strictly dominant at all
    rng = others_max - others.min()
    if rng == 0:
        return 1.0
    return float((center - others_max) / rng)


def detect_swings(
    highs: Sequence[float],
    lows: Sequence[float],
    left: int = 5,
    right: int = 5,
    min_strength: float = 0.0,
    timestamps: Optional[Sequence[datetime]] = None,
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Detect swing highs and swing lows (fractal pivots).

    A bar at index ``i`` is a swing high when its high is the strict maximum
    of the window ``[i-left, i+right]`` and its dominance (see
    ``_window_strength``) is at least ``min_strength``.

    Returns ``(swing_highs, swing_lows)``.
    """
    highs_arr = np.asarray(highs, dtype=float)
    lows_arr = np.asarray(lows, dtype=float)
    n = len(highs_arr)

    swing_highs: List[SwingPoint] = []
    swing_lows: List[SwingPoint] = []

    for i in range(left, n - right):
        hi_window = highs_arr[i - left : i + right + 1]
        lo_window = lows_arr[i - left : i + right + 1]

        if _window_strength(hi_window, left) >= min_strength:
            ts = timestamps[i] if timestamps is not None else None
            swing_highs.append(
                SwingPoint(
                    index=i,
                    price=float(highs_arr[i]),
                    kind="high",
                    strength=_window_strength(hi_window, left),
                    timestamp=ts,
                )
            )
        # Mirror: a swing low is a strict minimum of the (negated) low window.
        if _window_strength(-lo_window, left) >= min_strength:
            ts = timestamps[i] if timestamps is not None else None
            swing_lows.append(
                SwingPoint(
                    index=i,
                    price=float(lows_arr[i]),
                    kind="low",
                    strength=_window_strength(-lo_window, left),
                    timestamp=ts,
                )
            )

    return swing_highs, swing_lows


__all__ = ["detect_swings", "_window_strength"]