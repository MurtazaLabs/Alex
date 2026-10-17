"""Market structure signal rules (setup timeframe)."""

from __future__ import annotations

from typing import Optional
from ..signal_rules.base_rule import SignalRule, MarketContext, Direction


def make_structure_rules() -> list[SignalRule]:
    """Create all market structure rules."""

    return [
        SignalRule(
            name="h1_bos_confirmed",
            condition=lambda ctx: _has_confirmed_bos(ctx.structure, ctx.direction),
            weight=1.0,
            required=True,
            timeframe="H1",
            description="H1 confirmed break of structure in trade direction",
            category="STRUCTURE",
        ),
        SignalRule(
            name="h1_pullback_to_key_level",
            condition=lambda ctx: _pullback_to_level(ctx),
            weight=0.9,
            required=True,
            timeframe="H1",
            description="Price pulling back to broken resistance (now support) or vice versa",
            category="STRUCTURE",
        ),
        SignalRule(
            name="liquidity_sweep_detected",
            condition=lambda ctx: _has_liquidity_sweep(ctx.structure, ctx.direction),
            weight=0.7,
            required=False,
            timeframe="H1",
            description="Liquidity sweep of prior extremes before reversal",
            category="STRUCTURE",
        ),
        SignalRule(
            name="consolidation_breakout",
            condition=lambda ctx: _has_breakout(ctx.structure, ctx.direction),
            weight=0.8,
            required=False,
            timeframe="H1",
            description="Breakout from consolidation in trade direction",
            category="STRUCTURE",
        ),
        SignalRule(
            name="retest_successful",
            condition=lambda ctx: _has_successful_retest(ctx.structure, ctx.direction),
            weight=0.7,
            required=False,
            timeframe="H1",
            description="Successful retest of broken level",
            category="STRUCTURE",
        ),
        SignalRule(
            name="swing_structure_intact",
            condition=lambda ctx: _swing_structure_intact(ctx.structure, ctx.direction),
            weight=0.6,
            required=True,
            timeframe="H1",
            description="Swing structure (HH/HL or LH/LL) remains intact",
            category="STRUCTURE",
        ),
    ]


def _has_confirmed_bos(structure: "StructureSnapshot", direction: Direction) -> bool:
    if not structure or not structure.last_bos:
        return False
    return structure.last_bos.get("direction") == direction.value


def _pullback_to_level(ctx: MarketContext) -> bool:
    """Check if price is at/near a key structural level for entry."""
    structure = ctx.structure
    if not structure:
        return False

    current = ctx.current_price

    if ctx.direction == Direction.LONG:
        level = structure.key_support
        if level is None:
            return False
        # Price within 0.1% of support (or at support)
        return abs(current - level) / level <= 0.001
    else:
        level = structure.key_resistance
        if level is None:
            return False
        return abs(current - level) / level <= 0.001


def _has_liquidity_sweep(structure: "StructureSnapshot", direction: Direction) -> bool:
    if not structure or not structure.liquidity_sweeps:
        return False
    for sweep in structure.liquidity_sweeps:
        if sweep.get("direction") == direction.value:
            return True
    return False


def _has_breakout(structure: "StructureSnapshot", direction: Direction) -> bool:
    if not structure or not structure.consolidation:
        return False
    if not structure.consolidation.get("broken", False):
        return False
    return structure.consolidation.get("breakout_direction") == direction.value


def _has_successful_retest(structure: "StructureSnapshot", direction: Direction) -> bool:
    if not structure or not structure.retests:
        return False
    for retest in structure.retests:
        if retest.get("direction") == direction.value and retest.get("success", False):
            return True
    return False


def _swing_structure_intact(structure: "StructureSnapshot", direction: Direction) -> bool:
    """Verify the swing structure hasn't been invalidated."""
    if not structure:
        return True

    if direction == Direction.LONG:
        # Need HH/HL pattern - last swing low should be higher than previous
        lows = structure.swing_lows
        if len(lows) >= 2:
            return lows[-1].get("price", 0) > lows[-2].get("price", 0)
        return True
    else:
        # Need LH/LL pattern
        highs = structure.swing_highs
        if len(highs) >= 2:
            return highs[-1].get("price", 0) < highs[-2].get("price", 0)
        return True