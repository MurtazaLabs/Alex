"""Tests for the agent tool registry: permissions, schemas, arg validation."""

from __future__ import annotations

import pytest

from forex_ai_agent.agent.tools import get_tool_registry
from forex_ai_agent.agent.confirmations import RiskClass, get_tool_risk_class


EXPECTED_CORE_TOOLS = {
    # read-only market/analysis
    "get_market_data", "get_current_price", "analyze_structure", "get_indicators",
    "get_active_signals", "get_risk_state", "get_news",
    # strategy/research reads
    "search_strategies", "get_strategy", "list_probation", "list_research_history",
    # explanations (Phase 6)
    "explain_probation", "explain_research", "explain_gate", "answer_why",
    "search_knowledge", "get_strategy_history", "get_rejection_reasons",
    # health (Stage B1)
    "get_strategy_health",
    # robustness (Stage B3)
    "run_robustness_sweep",
    # research director (Stage B4)
    "director_enqueue", "director_run_cycle", "director_status",
    # research graph (Stage B2)
    "hypothesis_test_count", "aggregate_by_parameter", "research_graph_summary",
    # research plans (Stage B5)
    "research_plan",
    # LLM hypothesis generation (Stage C1)
    "generate_hypotheses",
    # LLM experiment selection (Stage C2)
    "select_experiments",
    # LLM research reports (Stage C3)
    "generate_research_report",
    # probation promotion review (Stage C4)
    "review_probation_promotions",
    # probation monitoring (Stage D3)
    "monitor_probation_health",
    # experiments (Phase 3)
    "backtest_strategy", "walk_forward_test", "compare_strategies",
    "create_experiment", "list_experiments", "get_experiment", "conclude_experiment",
    # actions (Phase 2)
    "start_research", "remove_strategy", "archive_strategy", "reject_strategy",
    "create_strategy_draft",
    # autonomy (Phase 5)
    "check_viability",
}


def test_all_expected_tools_registered():
    reg = get_tool_registry(0)
    assert EXPECTED_CORE_TOOLS <= set(reg._tools.keys()), (
        f"missing: {EXPECTED_CORE_TOOLS - set(reg._tools.keys())}"
    )
    assert len(reg._tools) == 49  # 40 + 3 B4 + B5 plan + C1 + C2 + C3 report + C4 review + D3 monitor


def test_registry_respects_permission_levels():
    # Every tool here should be usable at OBSERVE (0); if an action tool declares
    # a higher minimum, it must be absent at level 0.
    reg0 = get_tool_registry(0)
    reg1 = get_tool_registry(1)
    assert set(reg1._tools.keys()) >= set(reg0._tools.keys())


def test_schemas_for_prompt_lists_names_and_args():
    reg = get_tool_registry(0)
    docs = reg.get_schemas_for_prompt(min_level=0)
    assert "get_current_price" in docs
    assert "list_probation" in docs
    # docs must be plain text describing tools — used in the system prompt
    assert isinstance(docs, str) and len(docs) > 500


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_raises_valueerror():
    reg = get_tool_registry(0)
    with pytest.raises(ValueError):
        await reg.dispatch("not_a_tool", {})


@pytest.mark.asyncio
async def test_dispatch_validates_args_via_pydantic():
    reg = get_tool_registry(0)
    with pytest.raises(Exception):  # pydantic ValidationError on bad args
        await reg.dispatch("get_current_price", {"pair": 12345, "bogus": "field"})


def test_risk_class_registry_defaults_to_low():
    assert get_tool_risk_class("totally_unknown_tool") == RiskClass.LOW
