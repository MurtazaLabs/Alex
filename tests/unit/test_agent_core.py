"""Unit tests for AgentCore's action loop.

Uses a stub LLM with canned JSON replies so the loop mechanics — tool dispatch,
observation accumulation, invalid-reply bailouts, fallbacks, confirmation
surfacing — are tested deterministically without any network or DB.
"""

from __future__ import annotations

import pytest

from forex_ai_agent.agent.core import AgentCore, AgentTurnResult, PendingConfirmation
from forex_ai_agent.agent.memory import ConversationMemory


@pytest.fixture(autouse=True)
def _fresh_registry():
    """get_tool_registry returns a shared singleton; tests that stub its dispatch
    must not leak into other tests. Rebuild it fresh after every test."""
    from forex_ai_agent.agent import tools as tools_mod
    yield
    if tools_mod._registry is not None:
        lvl = tools_mod._registry.user_permission_level
        tools_mod._registry = tools_mod.ToolRegistry(user_permission_level=lvl)


class StubLLM:
    """Returns canned replies in order; records prompts for assertions."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def reason_json(self, system_prompt, context):
        self.prompts.append(system_prompt)
        if not self.replies:
            return {"final": "done (stub exhausted)"}
        return self.replies.pop(0)


def make_core(replies, max_steps=6) -> AgentCore:
    return AgentCore(
        llm=StubLLM(replies),
        memory=ConversationMemory(),
        max_steps=max_steps,
        user_permission_level=0,
    )


class FakeResult:
    """Mimics a pydantic tool result just enough for the loop."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def stub_dispatch(core, payload=None, fail=False):
    """Replace registry.dispatch with a deterministic fake."""
    calls = []

    async def fake(name, args, confirm_token=None):
        calls.append((name, args))
        if fail:
            raise RuntimeError(f"{name} exploded")
        return FakeResult(payload or {"ok": True, "tool": name})

    core.tools.dispatch = fake
    return calls


@pytest.mark.asyncio
async def test_tool_loop_collects_observations_then_final():
    core = make_core([
        {"action": "get_current_price", "args": {"pair": "EUR/USD"}},
        {"final": "EUR/USD is trading."},
    ])
    calls = stub_dispatch(core, {"pair": "EUR/USD", "bid": 1.08})
    result = await core.run_turn("What is EUR/USD doing?")

    assert isinstance(result, AgentTurnResult)
    assert result.success is True
    assert result.tools_used == ["get_current_price"]
    assert calls == [("get_current_price", {"pair": "EUR/USD"})]
    assert len(result.evidence) == 1
    assert result.evidence[0]["status"] == "success"
    assert result.evidence[0]["result"]["bid"] == 1.08
    assert result.answer  # composed answer non-empty
    # The tool's observation must appear in the final-compose prompt (stub logged it)
    final_prompt = core.llm.prompts[-1]
    assert "get_current_price" in final_prompt


@pytest.mark.asyncio
async def test_unknown_tool_becomes_error_observation_not_crash():
    core = make_core([
        {"action": "nonexistent_tool", "args": {}},
        {"final": "handled"},
    ])
    result = await core.run_turn("do something weird")
    assert result.success is True
    assert any(o["status"] == "error" for o in result.evidence)
    # audit records the failed attempt
    assert any(e["tool"] == "nonexistent_tool" and e["status"] == "error" for e in core.get_audit_log())


@pytest.mark.asyncio
async def test_invalid_replies_terminate_within_budget():
    # Endless malformed replies must not loop forever.
    core = make_core([{"garbage": True}] * 10, max_steps=6)
    result = await core.run_turn("nonsense question")
    assert result.steps <= 6
    assert result.answer  # still composed something
    # Invalid replies do not consume steps but the consecutive-bailout triggers.
    assert result.evidence == []  # no tool ran


@pytest.mark.asyncio
async def test_llm_unavailable_falls_back_deterministically():
    core = AgentCore(
        llm=StubLLM([]),  # reason_json returns {"final": "done (stub exhausted)"} — not None
        memory=ConversationMemory(),
        max_steps=3,
        user_permission_level=0,
    )
    # Simulate a dead LLM: reason_json returns falsy.
    async def dead_llm(system, ctx):
        return None
    core.llm.reason_json = dead_llm

    result = await core.run_turn("anything")
    assert result.success is False
    assert "can't reach the language model" in result.answer or "LLM unavailable" in (result.error or "")


@pytest.mark.asyncio
async def test_permission_denied_is_recorded_and_counts_as_step():
    # get_registry at level 0: all listed tools are allowed, so force denial via a
    # tool requiring a higher level (registry filters, so emulate by monkeypatching).
    core = make_core([
        {"action": "get_current_price", "args": {"pair": "EUR/USD"}},
        {"final": "ok"},
    ], max_steps=2)

    async def deny(name, args, confirm_token=None):
        raise PermissionError("Tool requires permission level 3")

    core.tools.dispatch = deny
    result = await core.run_turn("try something restricted")
    assert any(o["status"] == "permission_denied" for o in result.evidence)
    assert result.steps == 1  # denied attempt consumed a step — cannot loop forever


@pytest.mark.asyncio
async def test_memory_records_both_sides():
    core = make_core([
        {"final": "short answer"},
    ])
    await core.run_turn("remember this question")
    turns = core.memory.turns
    assert turns[-1]["role"] == "assistant"
    assert any(t["role"] == "user" for t in turns)


@pytest.mark.asyncio
async def test_audit_log_entries_have_latency_and_status():
    core = make_core([
        {"action": "get_current_price", "args": {"pair": "EUR/USD"}},
        {"final": "ok"},
    ])
    stub_dispatch(core)
    await core.run_turn("price check")
    entries = core.get_audit_log()
    assert entries and all({"tool", "status", "latency_ms", "timestamp"} <= set(e) for e in entries)
    assert entries[0]["status"] == "success"
