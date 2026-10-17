"""Tests for the SignalAgent fixes: candidate formation, sane SL/TP brackets,
failed-rules-as-missing (not contradicting), and the state machine."""

from __future__ import annotations

import pytest

from forex_ai_agent.agents.signal_agent import SignalAgent
from forex_ai_agent.analysis.signal_rules.base_rule import (
    Direction,
    HTFContext,
    IndicatorsSnapshot,
    MarketContext,
    StructureSnapshot,
)


def _ctx(direction: Direction, trend: str = "BULLISH", rsi: float = 55.0) -> MarketContext:
    """A synthetic market context where every required rule can pass."""
    structure = StructureSnapshot(
        trend=trend,
        swing_highs=[{"index": 10, "price": 1.0900}, {"index": 30, "price": 1.0950}],
        swing_lows=[{"index": 5, "price": 1.0800}, {"index": 25, "price": 1.0830}],
        last_bos={"kind": "BOS", "direction": direction.value, "level": 1.0900, "price": 1.0910, "confidence": 0.8},
        last_choch=None,
        key_support=1.0830,
        key_resistance=1.0950,
        liquidity_sweeps=[],
        consolidation=None,
        retests=[],
        events=[],
    )
    return MarketContext(
        pair="EUR/USD",
        timeframe="H1",
        current_price=1.0840,
        spread_pips=1.0,
        atr_pips=12.0,
        session="London",
        news_state="LOW",
        direction=direction,
        htf_context={
            "H4": HTFContext(timeframe="H4", trend=trend, structure=structure,
                             key_support=1.0830, key_resistance=1.0950),
            "D1": HTFContext(timeframe="D1", trend=trend),
            "H1": HTFContext(timeframe="H1", trend=trend),
        },
        indicators=IndicatorsSnapshot(rsi=rsi, macd_histogram=0.0002, atr=0.0012),
        structure=structure,
    )


@pytest.fixture
def agent() -> SignalAgent:
    return SignalAgent()


@pytest.mark.asyncio
async def test_candidate_forms_on_aligned_context(agent: SignalAgent) -> None:
    candidate = await agent.evaluate_signal("EUR/USD", Direction.LONG, "H1", _ctx(Direction.LONG))
    assert candidate.setup_type == "STRUCTURED_SETUP"
    assert candidate.entry_zone_low > 0
    assert candidate.stop_loss > 0
    assert candidate.take_profit_1 > 0
    assert candidate.risk_reward >= 1.0
    assert candidate.strategy_version == "v1.0"


@pytest.mark.asyncio
async def test_failed_rules_are_missing_not_contradicting(agent: SignalAgent) -> None:
    """Rule failures must not poison the evidence pool as contradictions."""
    ctx = _ctx(Direction.LONG, rsi=90.0)  # overbought → momentum rules fail
    candidate = await agent.evaluate_signal("EUR/USD", Direction.LONG, "H1", ctx)
    descriptions = [e.description for e in ctx.evidence.contradicting]
    assert not any("FAILED" in d for d in descriptions)
    assert ctx.evidence.missing  # failures recorded as missing info


@pytest.mark.asyncio
async def test_no_trade_when_trend_opposes(agent: SignalAgent) -> None:
    candidate = await agent.evaluate_signal("EUR/USD", Direction.LONG, "H1", _ctx(Direction.LONG, trend="BEARISH"))
    assert candidate.setup_type == "NO_TRADE"
    assert agent.get_state("EUR/USD_H1_LONG") is not None


def test_sane_bracket_long(agent: SignalAgent) -> None:
    from forex_ai_agent.agents.signal_agent import SignalCandidate

    c = SignalCandidate(
        pair="EUR/USD", direction=Direction.LONG, timeframe="H1", setup_type="X",
        entry_zone_low=1.0900, entry_zone_high=1.0910, stop_loss=1.0950,  # broken: SL above
        take_profit_1=1.0880,  # broken: TP below
    )
    SignalAgent._enforce_sane_bracket(c)
    assert c.stop_loss < c.entry_zone_low <= c.entry_zone_high < c.take_profit_1


def test_sane_bracket_short(agent: SignalAgent) -> None:
    from forex_ai_agent.agents.signal_agent import SignalCandidate

    c = SignalCandidate(
        pair="EUR/USD", direction=Direction.SHORT, timeframe="H1", setup_type="X",
        entry_zone_low=1.0900, entry_zone_high=1.0910, stop_loss=1.0800,  # broken: SL below
        take_profit_1=1.0960,  # broken: TP above
    )
    SignalAgent._enforce_sane_bracket(c)
    assert c.take_profit_1 < c.entry_zone_low <= c.entry_zone_high < c.stop_loss


@pytest.mark.asyncio
async def test_bracket_invariant_on_real_candidate(agent: SignalAgent) -> None:
    for direction in (Direction.LONG, Direction.SHORT):
        c = await agent.evaluate_signal("EUR/USD", direction, "H1", _ctx(direction))
        if c.setup_type == "STRUCTURED_SETUP":
            if direction == Direction.LONG:
                assert c.stop_loss < c.entry_zone_low < c.take_profit_1
            else:
                assert c.take_profit_1 < c.entry_zone_high < c.stop_loss


@pytest.mark.asyncio
async def test_watch_without_key_levels(agent: SignalAgent) -> None:
    ctx = _ctx(Direction.LONG)
    ctx.structure.key_support = None
    ctx.structure.key_resistance = None
    candidate = await agent.evaluate_signal("EUR/USD", Direction.LONG, "H1", ctx)
    assert candidate.setup_type == "NO_TRADE"


@pytest.mark.asyncio
async def test_event_emission_does_not_crash_without_loop(agent: SignalAgent) -> None:
    ctx = _ctx(Direction.LONG)
    candidate = await agent.evaluate_signal("EUR/USD", Direction.LONG, "H1", ctx)
    assert candidate is not None
