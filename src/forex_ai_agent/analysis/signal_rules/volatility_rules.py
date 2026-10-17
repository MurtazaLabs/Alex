"""Volatility and session signal rules."""

from __future__ import annotations

from typing import Optional
from ..signal_rules.base_rule import SignalRule, MarketContext, Direction


def make_volatility_rules() -> list[SignalRule]:
    """Create all volatility rules."""

    return [
        SignalRule(
            name="atr_sufficient_for_sl",
            condition=lambda ctx: _atr_ok(ctx.atr_pips, ctx.indicators.atr),
            weight=0.8,
            required=True,
            timeframe="H1",
            description="ATR provides enough room for stop loss placement",
            category="VOLATILITY",
        ),
        SignalRule(
            name="spread_acceptable",
            condition=lambda ctx: _spread_ok(ctx.spread_pips),
            weight=1.0,
            required=True,
            timeframe="ALL",
            description="Current spread within acceptable limits",
            category="VOLATILITY",
        ),
        SignalRule(
            name="bb_width_reasonable",
            condition=lambda ctx: _bb_width_ok(ctx.indicators.bb_upper, ctx.indicators.bb_lower),
            weight=0.4,
            required=False,
            timeframe="H1",
            description="Bollinger Band width within normal range",
            category="VOLATILITY",
        ),
    ]


def _atr_ok(current_atr: float, indicator_atr: Optional[float]) -> bool:
    if indicator_atr is None or indicator_atr <= 0:
        return True  # No data, don't block
    return current_atr >= indicator_atr * 0.5  # Current ATR >= 50% of recent ATR


def _spread_ok(spread_pips: float) -> bool:
    return spread_pips <= 3.0  # Default max spread threshold


def _bb_width_ok(upper: Optional[float], lower: Optional[float]) -> bool:
    if upper is None or lower is None:
        return True
    width = upper - lower
    # Normal Bollinger Band width for typical pairs
    return 1.0 <= width <= 5.0


def make_session_rules() -> list[SignalRule]:
    """Create all session rules."""

    return [
        SignalRule(
            name="in_trading_session",
            condition=lambda ctx: _session_ok(ctx.session),
            weight=0.7,
            required=False,
            timeframe="ALL",
            description="Trading session has sufficient liquidity",
            category="SESSION",
        ),
        SignalRule(
            name="london_session",
            condition=lambda ctx: _session_is(ctx.session, "London"),
            weight=0.3,
            required=False,
            timeframe="ALL",
            description="London session active (high vol, major pairs)",
            category="SESSION",
        ),
        SignalRule(
            name="new_york_session",
            condition=lambda ctx: _session_is(ctx.session, "NewYork"),
            weight=0.3,
            required=False,
            timeframe="ALL",
            description="NY session active (high volume, overlap with London)",
            category="SESSION",
        ),
    ]


def _session_ok(session: str) -> bool:
    active = {"London", "NewYork", "Overlap"}
    return session in active


def _session_is(session: str, session_name: str) -> bool:
    return session == session_name