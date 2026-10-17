"""Trend-based signal rules (HTF alignment)."""

from __future__ import annotations

from ..signal_rules.base_rule import SignalRule, MarketContext, Direction


def make_trend_rules() -> list[SignalRule]:
    """Create all trend alignment rules."""

    return [
        SignalRule(
            name="h4_structure_matches_direction",
            condition=lambda ctx: _matches_direction(ctx.htf_context.get("H4"), ctx.direction),
            weight=1.0,
            required=True,
            timeframe="H4",
            description="H4 market structure (HH/HL or LH/LL) matches trade direction",
            category="TREND",
        ),
        SignalRule(
            name="h4_no_choch_against",
            condition=lambda ctx: not _has_choch_against(ctx.htf_context.get("H4"), ctx.direction),
            weight=0.8,
            required=True,
            timeframe="H4",
            description="No recent CHoCH opposing trade direction on H4",
            category="TREND",
        ),
        SignalRule(
            name="d1_trend_aligned",
            condition=lambda ctx: _is_aligned(ctx.htf_context.get("D1"), ctx.direction),
            weight=0.6,
            required=False,
            timeframe="D1",
            description="D1 macro trend aligned with trade direction",
            category="TREND",
        ),
        SignalRule(
            name="h1_trend_aligned",
            condition=lambda ctx: _is_aligned(ctx.htf_context.get("H1"), ctx.direction),
            weight=0.8,
            required=True,
            timeframe="H1",
            description="H1 trend aligned with trade direction",
            category="TREND",
        ),
        SignalRule(
            name="m15_trend_aligned",
            condition=lambda ctx: _is_aligned(ctx.htf_context.get("M15"), ctx.direction),
            weight=0.5,
            required=False,
            timeframe="M15",
            description="M15 trend aligned for entry precision",
            category="TREND",
        ),
    ]


def _matches_direction(ctx: "HTFContext", direction: Direction) -> bool:
    """H4 trend agrees with the trade direction (NEUTRAL is acceptable, UNKNOWN blocks)."""
    if not ctx:
        return False  # No H4 data at all → cannot confirm required trend
    if direction == Direction.LONG:
        return ctx.trend in ("BULLISH", "NEUTRAL")
    if direction == Direction.SHORT:
        return ctx.trend in ("BEARISH", "NEUTRAL")
    return False


def _is_aligned(ctx: "HTFContext", direction: Direction) -> bool:
    if not ctx:
        return True  # No data = don't block
    if direction == Direction.LONG:
        return ctx.trend in ("BULLISH", "NEUTRAL")
    if direction == Direction.SHORT:
        return ctx.trend in ("BEARISH", "NEUTRAL")
    return True


def _has_choch_against(ctx: "HTFContext", direction: Direction) -> bool:
    if not ctx or not ctx.structure:
        return False
    choch = ctx.structure.last_choch
    if not choch:
        return False
    if direction == Direction.LONG:
        return choch.get("direction") == "SHORT"
    if direction == Direction.SHORT:
        return choch.get("direction") == "LONG"
    return False