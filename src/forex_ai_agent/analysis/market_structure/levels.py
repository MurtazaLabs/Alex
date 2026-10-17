"""Support / resistance level identification from swing points."""
from __future__ import annotations

from typing import List, Optional

from .models import Level, SwingPoint


def _cluster_levels(
    swings: List[SwingPoint], kind: str, tolerance: float
) -> List[Level]:
    """Greedily cluster swing values within ``tolerance`` absolute price."""
    if not swings:
        return []

    clusters: List[List[SwingPoint]] = [[swings[0]]]
    for point in swings[1:]:
        placed = False
        for cluster in clusters:
            centroid = sum(s.price for s in cluster) / len(cluster)
            if abs(point.price - centroid) <= tolerance:
                cluster.append(point)
                placed = True
                break
        if not placed:
            clusters.append([point])

    max_touches = max(len(c) for c in clusters)
    levels: List[Level] = []
    for cluster in clusters:
        prices = sorted(s.price for s in cluster)
        # Use the extreme touch (max for resistance, min for support)
        level_price = prices[-1] if kind == "resistance" else prices[0]
        touches = len(cluster)
        levels.append(
            Level(
                price=level_price,
                kind=kind,
                touches=touches,
                strength=round(touches / max_touches, 2),
            )
        )
    return levels


def identify_levels(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    tolerance: float = 0.0008,
) -> List[Level]:
    """
    Cluster swing points into support (from swing lows) and resistance (from
    swing highs) levels.

    ``tolerance`` is an absolute price distance (default ~8 pips on a 4-dp
    pair). Returns levels sorted by price ascending.
    """
    levels: List[Level] = []
    levels.extend(_cluster_levels(swing_highs, "resistance", tolerance))
    levels.extend(_cluster_levels(swing_lows, "support", tolerance))
    levels.sort(key=lambda lv: lv.price)

    # De-duplicate near-identical levels of opposite kind (keep stronger).
    deduped: List[Level] = []
    for level in levels:
        merged = False
        for existing in deduped:
            if abs(existing.price - level.price) <= tolerance:
                if level.strength > existing.strength or (
                    level.strength == existing.strength
                    and level.touches > existing.touches
                ):
                    existing.price = level.price
                    existing.touches = level.touches
                    existing.strength = level.strength
                merged = True
                break
        if not merged:
            deduped.append(level)
    deduped.sort(key=lambda lv: lv.price)
    return deduped


def key_levels(
    levels: List[Level], current_price: float
) -> tuple[Optional[float], Optional[float]]:
    """
    Return ``(key_support, key_resistance)`` around ``current_price``.

    Key support: strongest support below price. Key resistance: strongest
    resistance above price.
    """
    supports = [lv for lv in levels if lv.kind == "support" and lv.price < current_price]
    resistances = [lv for lv in levels if lv.kind == "resistance" and lv.price > current_price]

    def _strongest(candidates: List[Level]) -> Optional[Level]:
        if not candidates:
            return None
        return max(candidates, key=lambda lv: (lv.strength, lv.touches))

    s = _strongest(supports)
    r = _strongest(resistances)
    return (s.price if s else None, r.price if r else None)


__all__ = ["identify_levels", "key_levels"]