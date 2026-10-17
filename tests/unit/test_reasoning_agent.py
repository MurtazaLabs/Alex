"""Tests for the ReasoningAgent: template fallbacks and context building."""

from __future__ import annotations

import pytest

from forex_ai_agent.agents.reasoning_agent import ReasoningAgent
from forex_ai_agent.llm import LLMClient, LLMUnavailable
from forex_ai_agent.config.settings import Settings


@pytest.fixture
def agent() -> ReasoningAgent:
    settings = Settings(
        environment="test",
        llm_enabled=False,  # force the deterministic fallback path
        _env_file=None,  # type: ignore[call-arg]
    )
    return ReasoningAgent(llm=LLMClient(settings))


def _context() -> dict:
    return {
        "pair": "EUR/USD",
        "direction": "LONG",
        "timeframe": "H1",
        "session": "London",
        "data_source": "mt5",
        "structure": {
            "trend": "BULLISH",
            "last_bos": {"level": 1.0830, "direction": "LONG"},
        },
        "indicators": {"rsi": 58.2, "atr": 12.0},
        "risk_checks": {
            "passed": False,
            "checks": {"spread": {"passed": True, "message": "ok"},
                       "news_filter": {"passed": False, "message": "blocked"}},
        },
        "signal": {
            "setup_type": "STRUCTURED_SETUP",
            "entry_zone_low": 1.0825,
            "entry_zone_high": 1.0845,
            "stop_loss": 1.0800,
            "take_profit_1": 1.0900,
            "risk_reward": 2.9,
            "invalidation": "H1 candle closes below 1.0800",
        },
    }


@pytest.mark.asyncio
async def test_template_signal_fallback(agent: ReasoningAgent) -> None:
    out = await agent.narrate_signal(_context())
    assert "EUR/USD" in out
    assert "2.9" in out  # R:R from the deterministic signal
    assert "news_filter" in out  # failing check named
    # Fallback must be deterministic.
    assert out == await agent.narrate_signal(_context())


@pytest.mark.asyncio
async def test_template_explanation_fallback(agent: ReasoningAgent) -> None:
    out = await agent.narrate_explanation(_context(), "WHY THIS SETUP EXISTS:\n  1. trend")
    assert "EUR/USD" in out
    assert "WHY THIS SETUP EXISTS" in out


def test_confidence_word_calibration() -> None:
    assert ReasoningAgent._confidence_word(0.9) == "high"
    assert ReasoningAgent._confidence_word(0.75) == "moderate"
    assert ReasoningAgent._confidence_word(0.55) == "low"
    assert ReasoningAgent._confidence_word(0.2) == "very low"
    assert ReasoningAgent._confidence_word(None) == "low"


def test_build_context_shape(agent: ReasoningAgent) -> None:
    ctx = agent.build_context(
        pair="GBP/JPY",
        direction="SHORT",
        timeframe="H4",
        session="Overlap",
        structure_dict={"trend": "BEARISH"},
        indicators_dict={"rsi": 35.0},
        risk_checks={"passed": True, "checks": {}},
        signal_candidate={"risk_reward": 3.1},
        data_source="yahoo",
    )
    assert ctx["pair"] == "GBP/JPY"
    assert ctx["data_source"] == "yahoo"
    assert "signal" in ctx and "structure" in ctx
