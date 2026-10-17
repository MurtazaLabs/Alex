"""Liquidity sweep detection (equal extremes taken out and reclaimed)."""
from __future__ import annotations

from typing import List

from .models import Direction, LiquiditySweep, SwingPoint


def detect_liquidity_sweeps(
    swing_lows: List[SwingPoint],
    swing_highs: List[SwingPoint],
    atr_pips: float,
    pip_size: float = 0.0001,
    min_depth_atr: float = 0.5,
) -> List[LiquiditySweep]:
    """
    Detect liquidity sweeps on the recent structure.

    A sweep occurs when price briefly takes out a run of equal extremes:
    - equal swing lows taken out (new lower low beyond a tight cluster) and then
      reclaimed = buy-side liquidity sweep (Direction.LONG)
    - equal swing highs taken out and reclaimed = sell-side sweep (SHORT)

    ``min_depth_atr`` controls how deep the sweep must extend below the prior
    equal lows (in ATR pips) to be considered a genuine sweep rather than noise.
    """
    sweeps: List[LiquiditySweep] = []
    depth = max(min_depth_atr, 0.25) * atr_pips * pip_size

    for lows in (swing_lows,):
        if len(lows) < 3:
            continue
        # Cluster recent lows that are near-equal (within one sweep depth).
        recent = lows[-3:]
        if _tight_cluster([s.price for s in recent], depth):
            cluster_low = max(s.price for s in recent)
            # A sweep LONG requires the final flare below the cluster...
            last = recent[-1]
            if last.price < cluster_low - depth:
                sweeps.append(
                    LiquiditySweep(
                        index=last.index,
                        direction=Direction.LONG,
                        swept_price=last.price,
                        depth_atr=round(
                            (cluster_low - last.price) / (atr_pips * pip_size), 2
                        ),
                        confidence=0.8,
                    )
                )

    for highs in (swing_highs,):
        if len(highs) < 3:
            continue
        recent = highs[-3:]
        if _tight_cluster([s.price for s in recent], depth):
            cluster_high = min(s.price for s in recent)
            last = recent[-1]
            if last.price > cluster_high + depth:
                sweeps.append(
                    LiquiditySweep(
                        index=last.index,
                        direction=Direction.SHORT,
                        swept_price=last.price,
                        depth_atr=round(
                            (last.price - cluster_high) / (atr_pips * pip_size), 2
                        ),
                        confidence=0.8,
                    )
                )

    return sweeps


def _tight_cluster(prices: List[float], tolerance: float) -> bool:
    """Whether a price run is clustered within ``tolerance``."""
    return (max(prices) - min(prices)) <= tolerance


__all__ = ["detect_liquidity_sweeps"]