"""Default signal rule sets.

These implement the rules described in ``docs/Forex docmentation/SIGNAL_FLOW.md``:
HTF alignment, structure confirmation, momentum support, volatility filters and
news/session gates. Weights and required flags are configuration, not LLM output.
"""
from __future__ import annotations

from typing import List

from ..market_structure.models import Trend
from .base_rule import AND, NOT, OR, MarketContext, SignalRule


# ---------------------------------------------------------------------------
# Trend rules (HTF alignment)
# ---------------------------------------------------------------------------


def trend_alignment_rules() -> List[SignalRule]:
    """Required alignment between candidate direction and HTF structure."""

    def _trend_bullish(ctx: MarketContext) -> bool:
        return (
            (ctx.structure_trend or ctx.htf_trend or "").upper()
            == Trend.BULLISH.value
        )

    def _trend_bearish(ctx: MarketContext) -> bool:
        return (
            (ctx.structure_trend or ctx.htf_trend or "").upper()
            == Trend.BEARISH.value
        )

    return [
        SignalRule(
            name="long_htf_bullish_structure",
            condition=AND(
                SignalRule(
                    "dir_long", lambda c: c.direction.value == "LONG", required=True
                ),
                SignalRule("htf_bullish", _trend_bullish),
            ),
            weight=1.0,
            required=True,
            category="trend",
            timeframe="H4",
            description="HTF market structure is bullish (HH/HL)",
        ),
        SignalRule(
            name="short_htf_bearish_structure",
            condition=AND(
                SignalRule(
                    "dir_short", lambda c: c.direction.value == "SHORT", required=True
                ),
                SignalRule("htf_bearish", _trend_bearish),
            ),
            weight=1.0,
            required=True,
            category="trend",
            timeframe="H4",
            description="HTF market structure is bearish (LH/LL)",
        ),
        SignalRule(
            name="no_choch_against_trade",
            condition=NOT(
                SignalRule(
                    "choch_against",
                    lambda c: (
                        c.structure_choch_direction or c.htf_last_choch or ""
                    )
                    == {"LONG": "SHORT", "SHORT": "LONG"}.get(
                        c.direction.value, ""
                    ),
                )
            ),
            weight=0.8,
            required=True,
            category="trend",
            description="No CHoCH against the trade direction",
        ),
    ]