"""Unit tests for the orchestrator pipeline."""

from __future__ import annotations

import asyncio

import pytest

from forex_ai_agent.agents.market_agent import MarketAgent
from forex_ai_agent.core.event_bus import InMemoryEventBus
from forex_ai_agent.core.orchestrator import Orchestrator
from forex_ai_agent.core.permission_manager import PermissionLevel, PermissionManager


@pytest.mark.asyncio
async def test_identifies_analysis_intent():
    orch = Orchestrator(PermissionManager())
    assert orch._identify_intent("Analyze EUR/USD trend") == "ANALYSIS"
    assert orch._identify_intent("Research liquidity sweeps") == "RESEARCH"
    assert orch._identify_intent("hello there") == "GENERAL"


@pytest.mark.asyncio
async def test_analysis_request_routes_to_market_specialist():
    bus = InMemoryEventBus()
    orch = Orchestrator(
        PermissionManager(current_level=PermissionLevel.MARKET_MONITORING),
        event_bus=bus,
        specialists=[MarketAgent(event_bus=bus)],
    )
    outcome = await orch.process_request("Analyze EUR/USD trend")
    assert outcome["status"] == "completed"
    assert outcome["intent"] == "ANALYSIS"
    assert outcome["classification"]["domains"] == ["market"]
    assert "market_agent" in outcome["result"]["findings"][0]["specialist"]


@pytest.mark.asyncio
async def test_execution_denied_at_level_zero():
    orch = Orchestrator(
        PermissionManager(current_level=PermissionLevel.RESEARCH_ONLY),
        specialists=[MarketAgent()],
    )
    outcome = await orch.process_request("open trade EUR/USD now")
    assert outcome["status"] == "permission_denied"


@pytest.mark.asyncio
async def test_research_intent_uses_research_domain():
    orch = Orchestrator(
        PermissionManager(),
    )
    classification = orch._classify_task("RESEARCH", "Research order blocks")
    assert classification["category"] == "RESEARCH"
    assert classification["required_level"] == 0


@pytest.mark.asyncio
async def test_conflict_detection():
    class Bull:
        name = "bull_agent"

        async def run(self, task, context=None):
            return {"status": "SUCCESS", "direction": "LONG"}

    class Bear:
        name = "bear_agent"

        async def run(self, task, context=None):
            return {"status": "SUCCESS", "direction": "SHORT"}

    orch = Orchestrator(
        PermissionManager(current_level=PermissionLevel.SIGNAL_GENERATION),
        specialists=[Bull(), Bear()],
    )
    # Force both specialists onto the request
    context = {"force": ["bull_agent", "bear_agent"]}
    outcome = await orch.process_request("Analyze EUR/USD", context)
    # conflict makes the safety gate fail -> state BLOCKED or PARTIAL
    assert outcome["final_state"] in ("BLOCKED", "PARTIAL")
    assert outcome["status"] == "completed"


@pytest.mark.asyncio
async def test_parallel_and_sequential_runs():
    class Slow:
        name = "slow_agent"

        async def run(self, task, context=None):
            await asyncio.sleep(0.01)
            return {"status": "SUCCESS", "summary": "done"}

    orch = Orchestrator(
        PermissionManager(current_level=PermissionLevel.MARKET_MONITORING),
        specialists=[Slow()],
    )
    outcome = await orch.process_request("Analyze EUR/USD")
    assert outcome["result"]["findings"][0]["summary"] == "done"
