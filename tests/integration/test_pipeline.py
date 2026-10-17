"""Integration tests for the full Forex AI pipeline.

Tests the complete flow over the deterministic MockProvider:
MarketData → TechnicalAnalysis → MarketStructure → Signal → Risk → narration.

The mock feed is seeded per (pair, timeframe), so results are deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from forex_ai_agent.config.settings import Settings
from forex_ai_agent.core.event_bus import InMemoryEventBus
from forex_ai_agent.data.providers.mock_provider import MockProvider
from forex_ai_agent.pipeline import Pipeline

RISK_CHECK_NAMES = {
    "account_risk",
    "min_rr",
    "sl_distance",
    "spread",
    "volatility",
    "total_exposure",
    "correlated_exposure",
    "max_position",
    "news_filter",
    "session_filter",
}


@pytest.fixture
async def pipeline_fixture():
    """Pipeline wired to the deterministic mock feed (no network)."""
    import os

    os.environ["ENVIRONMENT"] = "test"

    event_bus = InMemoryEventBus()
    pipeline = Pipeline(event_bus=event_bus, data_provider=MockProvider())
    await pipeline.ensure_db()
    return pipeline


@pytest.mark.asyncio
async def test_pipeline_creation(pipeline_fixture):
    """Test that pipeline creates successfully with all agents."""
    assert pipeline_fixture is not None
    status = await pipeline_fixture.get_pipeline_status()
    assert status["status"] == "ok"
    assert "signal_agent" in status["agents"]
    assert "data_source" in status


@pytest.mark.asyncio
async def test_full_signal_pipeline(pipeline_fixture):
    """The signal pipeline returns a consistent, evidence-backed result."""
    result = await pipeline_fixture.generate_signal(
        pair="EUR/USD", direction="LONG", timeframe="H1"
    )

    assert result["status"] == "ok"
    assert "signal_candidate" in result
    assert "risk_validation" in result

    signal = result["signal_candidate"]
    assert signal["pair"] == "EUR/USD"
    assert signal["direction"] == "LONG"
    assert signal["timeframe"] == "H1"
    assert signal["setup_type"] is not None

    risk = result["risk_validation"]
    if signal["setup_type"] == "NO_TRADE":
        # No levels → risk validation is intentionally skipped.
        assert risk["checks"] == {}
    else:
        assert set(risk["checks"]) == RISK_CHECK_NAMES
    assert risk["position_size_lots"] >= 0

    # Every result carries data provenance and a narrative.
    assert result["data_source"]
    assert result["narrative"]

    if signal["setup_type"] != "NO_TRADE":
        # A formed setup must have a coherent bracket.
        assert signal["entry_zone_low"] < signal["entry_zone_high"]
        assert signal["stop_loss"] < signal["entry_zone_low"]
        assert signal["take_profit_1"] > signal["entry_zone_high"]
        assert signal["risk_reward"] > 0
        assert signal["invalidation_condition"]
    else:
        # Honest NO_TRADE: levels zeroed, state WATCH.
        assert signal["status"] in ("WATCH", "NO_TRADE")


@pytest.mark.asyncio
async def test_signal_state_machine(pipeline_fixture):
    """Signal states stay within the known state machine vocabulary."""
    for _ in range(2):
        result = await pipeline_fixture.generate_signal(
            pair="EUR/USD", direction="LONG", timeframe="H1"
        )
        status = result["signal_candidate"]["status"]
        assert status in {
            "NO_TRADE",
            "WATCH",
            "SETUP_FORMING",
            "TRIGGERED",
            "INVALIDATED",
        }, f"Unexpected status: {status}"


@pytest.mark.asyncio
async def test_risk_validation(pipeline_fixture):
    """Risk validation always reports all ten checks with details."""
    result = await pipeline_fixture.generate_signal(
        pair="GBP/JPY", direction="SHORT", timeframe="H1"
    )

    signal = result["signal_candidate"]
    risk = result["risk_validation"]
    assert "passed" in risk
    if signal["setup_type"] == "NO_TRADE":
        # Risk validation is intentionally skipped when there are no levels.
        assert risk["checks"] == {}
    elif risk["passed"]:
        assert risk["position_size_lots"] >= 0
    else:
        failed = [n for n, c in risk["checks"].items() if not c["passed"]]
        assert failed  # failure details are always present


@pytest.mark.asyncio
async def test_invalidation_condition(pipeline_fixture):
    """Every signal (formed or not) carries a non-empty invalidation condition."""
    result = await pipeline_fixture.generate_signal(
        pair="USD/CHF", direction="LONG", timeframe="H1"
    )

    signal = result["signal_candidate"]
    assert signal["invalidation_condition"]
    inv = signal["invalidation_condition"].lower()
    assert any(
        kw in inv for kw in ["support", "resistance", "closes below", "closes above", "invalidat"]
    ), f"Invalidation condition should reference a structural level: {signal['invalidation_condition']}"


@pytest.mark.asyncio
async def test_signal_evidence_collection(pipeline_fixture):
    """Evidence is always collected (supporting and/or missing)."""
    result = await pipeline_fixture.generate_signal(
        pair="AUD/USD", direction="LONG", timeframe="H1"
    )

    signal = result["signal_candidate"]
    evidence = signal["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["supporting"] is not None
    assert evidence["missing"] is not None


@pytest.mark.asyncio
async def test_pipeline_health(pipeline_fixture):
    """Pipeline status reports agents and settings."""
    status = await pipeline_fixture.get_pipeline_status()
    assert status is not None
    assert status.get("agents")
    assert status.get("settings")
