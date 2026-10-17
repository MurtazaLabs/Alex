"""Momentum signal rules."""

from __future__ import annotations

from typing import Optional
from ..signal_rules.base_rule import SignalRule, MarketContext, Direction


def make_momentum_rules() -> list[SignalRule]:
    """Create all momentum rules."""

    return [
        SignalRule(
            name="rsi_not_overbought",
            condition=lambda ctx: _rsi_ok(ctx.indicators.rsi, "LONG" if ctx.direction == Direction.LONG else "SHORT"),
            weight=0.6,
            required=True,
            timeframe="H1",
            description="RSI in healthy range (40-70), not overbought (>70) or oversold (<40)",
            category="MOMENTUM",
        ),
        SignalRule(
            name="rsi_not_oversold",
            condition=lambda ctx: _rsi_ok(ctx.indicators.rsi, "SHORT" if ctx.direction == Direction.LONG else "LONG"),
            weight=0.5,
            required=False,
            timeframe="H1",
            description="RSI not below 30 (oversold) for short entries",
            category="MOMENTUM",
        ),
        SignalRule(
            name="macd_histogram_positive",
            condition=lambda ctx: _macd_ok(ctx.indicators.macd_histogram),
            weight=0.5,
            required=False,
            timeframe="H1",
            description="MACD histogram positive = bullish momentum",
            category="MOMENTUM",
        ),
        SignalRule(
            name="macd_histogram_negative",
            condition=lambda ctx: _macd_ok(ctx.indicators.macd_histogram, negative=True),
            weight=0.4,
            required=False,
            timeframe="H1",
            description="MACD histogram negative = bearish momentum",
            category="MOMENTUM",
        ),
        SignalRule(
            name="price_momentum_direction",
            condition=lambda ctx: _price_momentum(ctx),
            weight=0.5,
            required=False,
            timeframe="H1",
            description="Recent price momentum supports trade direction",
            category="MOMENTUM",
        ),
    ]


def _rsi_ok(rsi_val: Optional[float], side: str) -> bool:
    if rsi_val is None:
        return True  # No data, don't block
    if side == "LONG":
        return 40.0 <= rsi_val <= 70.0
    else:
        return 30.0 <= rsi_val <= 70.0


def _macd_ok(hist: Optional[float], negative: bool = False) -> bool:
    if hist is None:
        return True
    if negative:
        return hist < 0
    return hist > 0


def _price_momentum(ctx: MarketContext) -> bool:
    """Simple momentum check based on recent price action."""
    if not ctx.indicators:
        return True
    # Check M15 close direction relative to open (if available)
    # Simple: close > open suggests bullish momentum
    # This would need M15 candle data; placeholder
    return True