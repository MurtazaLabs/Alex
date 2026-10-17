"""AgentCore — the brain behind both chat surfaces (dashboard + CLI).

Runs the JSON action loop: intent → tool call → observation → repeat (max 6 steps)
→ final answer composed from observations only.

Constitution: the LLM decides *what to ask*, deterministic tools decide *what is
true*, and every number in the final answer traces to a tool observation.

This file was rewritten after review found the original loop referenced an
undefined ``messages`` variable (every tool-using turn raised NameError) and
could loop forever on malformed LLM replies.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from forex_ai_agent.agent.prompts import AGENT_LOOP_SYSTEM, FINAL_SYSTEM
from forex_ai_agent.agent.memory import ConversationMemory
from forex_ai_agent.agent.tools import ToolRegistry, get_tool_registry
from forex_ai_agent.llm.client import LLMClient

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_INVALID = 3  # bail out of the loop if the LLM keeps replying garbage
_MAX_STEPS_DEFAULT = 8  # enough for probe → backtest → explain chains

# Question shapes that are about system/market state. A final answer to any of
# these with ZERO tool observations is a constitution violation — the model is
# inventing data — so run_turn rejects it and forces a tool call instead.
_DATA_SEEKING_RE = re.compile(
    r"\b(price|pair|pairs|eur|gbp|usd|jpy|chf|aud|nzd|cad|strategy|strategies|"
    r"backtest|probation|experiment|research|signal|signals|news|risk|feed|"
    r"watchlist|account|equity|balance|chart|candle|indicator|structure|"
    r"profit|drawdown|sharpe|trade|trades)\b",
    re.IGNORECASE,
)


@dataclass
class PendingConfirmation:
    """A medium-risk action awaiting user confirmation."""

    tool: str
    args: Dict[str, Any]
    token: str
    impact: str
    expires_in_seconds: int


@dataclass
class AgentTurnResult:
    """Result of one conversational turn."""

    answer: str
    tools_used: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    success: bool = True
    error: Optional[str] = None
    pending_confirmation: Optional[PendingConfirmation] = None


class AgentCore:
    """Conversational agent core that routes to deterministic tools.

    Surface-agnostic: the same core drives the dashboard Chat page and the
    CLI ``chat`` command.
    """

    def __init__(
        self,
        pipeline=None,
        llm: Optional[LLMClient] = None,
        memory: Optional[ConversationMemory] = None,
        max_steps: int = _MAX_STEPS_DEFAULT,
        user_permission_level: int = 0,
    ):
        self.pipeline = pipeline
        self.llm = llm or LLMClient()
        self.memory = memory or ConversationMemory()
        self.max_steps = max_steps
        self.tools = get_tool_registry(user_permission_level)
        self._audit_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Main turn loop
    # ------------------------------------------------------------------

    async def run_turn(self, user_message: str) -> AgentTurnResult:
        """Run one full conversational turn.

        1. LLM proposes an action (tool call) or a final answer, as strict JSON.
        2. Tools run deterministically; observations accumulate as evidence.
        3. Final answers are composed from observations only.
        4. Medium-risk tools surface a PendingConfirmation instead of executing;
           the caller confirms via /api/agent/confirm (token, single-use, 10-min TTL).
        """
        start_time = time.time()
        self.memory.add_user(user_message)

        tool_docs = self.tools.get_schemas_for_prompt(min_level=self.tools.user_permission_level)
        system_prompt = AGENT_LOOP_SYSTEM.format(
            tool_docs=tool_docs,
            max_steps=self.max_steps,
            question=user_message,
        )

        observations: List[Dict[str, Any]] = []
        tools_used: List[str] = []
        invalid_replies = 0
        steps = 0

        try:
            while steps < self.max_steps:
                context = {
                    "conversation_history": self.memory.turns,
                    "observations": observations,
                }
                response = await self.llm.reason_json(system_prompt, context)
                logger.debug("Agent step %s response: %s", steps + 1, response)

                if not response:
                    # LLM unavailable or returned nothing parseable.
                    fallback = self._fallback_response(user_message, observations)
                    return AgentTurnResult(
                        answer=fallback,
                        tools_used=tools_used,
                        evidence=observations,
                        steps=steps,
                        success=bool(observations),  # degraded but still useful
                        error="LLM unavailable",
                    )

                # --- Final answer requested -------------------------------
                if "final" in response:
                    # Constitution guard: with zero observations, the model
                    # cannot know anything about live market/system state — a
                    # data-seeking question answered anyway means fabrication.
                    if not observations and _DATA_SEEKING_RE.search(user_message):
                        logger.warning(
                            "Rejected zero-observation final answer for a "
                            "data-seeking question; forcing a tool call."
                        )
                        invalid_replies += 1
                        if invalid_replies >= _MAX_CONSECUTIVE_INVALID:
                            final_answer = await self._compose_final_answer(
                                user_message, observations,
                                "No tool evidence could be collected. Explain that "
                                "no data is available and do not invent any.",
                            )
                            self.memory.add_assistant(final_answer, tools_used)
                            return AgentTurnResult(
                                answer=final_answer,
                                tools_used=tools_used,
                                evidence=observations,
                                steps=steps,
                                success=False,
                                error="agent refused to use tools for a data question",
                            )
                        observations.append({
                            "system": "Your previous reply gave a final answer with "
                            "no tool evidence, which is not allowed for data "
                            "questions. You MUST call a tool first. Choose one "
                            "tool whose data can answer the question and return "
                            "its JSON action now.",
                            "status": "guard",
                        })
                        continue

                    final_answer = await self._compose_final_answer(
                        user_message, observations,
                        response["final"] if isinstance(response["final"], str) else "",
                    )
                    self.memory.add_assistant(final_answer, tools_used)
                    return AgentTurnResult(
                        answer=final_answer,
                        tools_used=tools_used,
                        evidence=observations,
                        steps=steps,
                        success=True,
                    )

                # --- Tool action ------------------------------------------
                if "action" not in response:
                    invalid_replies += 1
                    logger.warning("Invalid agent reply (no action/final): %s", response)
                    if invalid_replies >= _MAX_CONSECUTIVE_INVALID:
                        final_answer = await self._compose_final_answer(
                            user_message, observations,
                            "The assistant could not form a valid next action. "
                            "Answer with the evidence collected so far.",
                        )
                        self.memory.add_assistant(final_answer, tools_used)
                        return AgentTurnResult(
                            answer=final_answer,
                            tools_used=tools_used,
                            evidence=observations,
                            steps=steps,
                            success=bool(observations),
                            error="agent replied with invalid format repeatedly",
                        )
                    continue

                invalid_replies = 0
                action = response["action"]
                args = response.get("args", {}) or {}

                tool_start = time.time()
                try:
                    result = await self.tools.dispatch(action, args)
                    latency_ms = int((time.time() - tool_start) * 1000)

                    obs = {
                        "tool": action,
                        "args": args,
                        "result": self._serialize_result(result),
                        "latency_ms": latency_ms,
                        "status": "success",
                    }

                    # Medium-risk tools return a proposal instead of executing.
                    pending = self._extract_pending(action, args, obs["result"])
                    if pending:
                        obs["status"] = "awaiting_confirmation"
                        observations.append(obs)
                        return AgentTurnResult(
                            answer=(
                                f"⚠️ This action needs your confirmation:\n\n{pending.impact}\n\n"
                                f"Expires in {pending.expires_in_seconds // 60} minutes. "
                                "Confirm or cancel in the dialog above."
                            ),
                            tools_used=tools_used + [action],
                            evidence=observations,
                            steps=steps,
                            success=True,
                            pending_confirmation=pending,
                        )

                    observations.append(obs)
                    tools_used.append(action)

                    self._audit_log.append({
                        "tool": action,
                        "args_summary": str(args)[:200],
                        "status": "success",
                        "latency_ms": latency_ms,
                        "timestamp": time.time(),
                    })
                    steps += 1
                    logger.info("Tool '%s' completed in %sms", action, latency_ms)

                except PermissionError as e:
                    observations.append({
                        "tool": action,
                        "args": args,
                        "error": str(e),
                        "status": "permission_denied",
                    })
                    self._audit_log.append({
                        "tool": action,
                        "args_summary": str(args)[:200],
                        "status": "permission_denied",
                        "error": str(e),
                        "timestamp": time.time(),
                    })
                    steps += 1  # count it — otherwise a tool above the user's level loops forever

                except Exception as e:  # noqa: BLE001 — tool errors are observations, not crashes
                    observations.append({
                        "tool": action,
                        "args": args,
                        "error": str(e),
                        "status": "error",
                    })
                    self._audit_log.append({
                        "tool": action,
                        "args_summary": str(args)[:200],
                        "status": "error",
                        "error": str(e),
                        "timestamp": time.time(),
                    })
                    steps += 1
                    logger.warning("Tool '%s' failed: %s", action, e)

            # Max steps reached without a final answer.
            final_answer = await self._compose_final_answer(
                user_message, observations,
                "Maximum tool calls reached. Answer with available evidence.",
            )
            self.memory.add_assistant(final_answer, tools_used)
            return AgentTurnResult(
                answer=final_answer,
                tools_used=tools_used,
                evidence=observations,
                steps=steps,
                success=True,
            )

        except Exception as e:  # noqa: BLE001 — the turn itself must never raise
            logger.exception("Agent turn failed")
            return AgentTurnResult(
                answer=f"Agent error: {e}",
                tools_used=tools_used,
                evidence=observations,
                steps=steps,
                success=False,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_result(result: Any) -> Any:
        """Tool results → JSON-safe structures for prompts, evidence, and UI."""
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "__dict__"):
            return vars(result)
        return str(result)

    @staticmethod
    def _extract_pending(tool: str, args: Dict[str, Any], serialized: Any) -> Optional[PendingConfirmation]:
        """If a medium-risk tool returned a confirmation proposal, surface it."""
        if not isinstance(serialized, dict):
            return None
        proposal = serialized.get("proposal")
        if not isinstance(proposal, dict) or not proposal.get("token"):
            return None
        return PendingConfirmation(
            tool=tool,
            args=args,
            token=proposal["token"],
            impact=proposal.get("impact", "Unknown impact"),
            expires_in_seconds=int(proposal.get("expires_in_seconds", 600)),
        )

    async def _compose_final_answer(
        self,
        question: str,
        observations: List[Dict[str, Any]],
        llm_draft: str,
    ) -> str:
        """Compose the final answer using the FINAL_SYSTEM prompt."""
        if not observations:
            return (
                "I don't have enough information to answer that — no tool observations "
                "were collected. Try rephrasing, or ask about a pair, your strategies, "
                "the probation queue, or the feed."
            )

        obs_lines = []
        for i, obs in enumerate(observations):
            if obs.get("status") == "success":
                obs_lines.append(f"Tool {i + 1} ({obs['tool']}): {obs['result']}")
            elif obs.get("status") == "awaiting_confirmation":
                obs_lines.append(f"Tool {i + 1} ({obs['tool']}): awaiting user confirmation")
            else:
                obs_lines.append(
                    f"Tool {i + 1} ({obs['tool']}): {obs.get('status', 'error')} — {obs.get('error', 'unknown')}"
                )
        obs_text = "\n\n".join(obs_lines)

        final_prompt = FINAL_SYSTEM.format(question=question, observations=obs_text)
        response = await self.llm.reason_json(final_prompt, {"observations": observations})
        if response and isinstance(response.get("final"), str):
            return response["final"]
        # Some free models nest the answer (e.g. {"final": {"answer": ...}}) —
        # never let a non-string reach the API layer; degrade to the draft.
        if response and isinstance(response.get("final"), dict):
            nested = response["final"].get("answer") or response["final"].get("text")
            if isinstance(nested, str):
                return nested
        return llm_draft

    def _fallback_response(self, question: str, observations: List[Dict[str, Any]]) -> str:
        """Deterministic fallback when the LLM is unavailable."""
        if not observations:
            return (
                "I can't reach the language model right now, so I can't phrase an answer — "
                "but the deterministic core is fine. Try again shortly, or use the "
                "dashboard pages (Dashboard/Strategies/Research) which don't need the LLM."
            )
        if observations and all(
            obs.get("status") == "error" and "start_research" == obs.get("tool")
            for obs in observations
        ):
            return (
                "Research was launched in the background, but it does not return "
                "results immediately — it logs to the Research page when done. "
                "If you want a SPECIFIC strategy backtested right now, name its "
                "rules (e.g. an Asian-range London sweep) and I will run it "
                "through backtest_strategy_spec, which returns full results "
                "and saves the strategy if it passes the gate."
            )
        lines = ["Based on the data I collected (LLM unavailable for narration):"]
        for obs in observations:
            if obs.get("status") == "success":
                lines.append(f"- {obs['tool']}: {obs['result']}")
            else:
                lines.append(f"- {obs['tool']}: {obs.get('status', 'error')}")
        return "\n".join(lines)

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return the audit log for this session."""
        return list(self._audit_log)

    def clear_memory(self) -> None:
        """Clear conversation memory and audit log."""
        self.memory.clear()
        self._audit_log.clear()


# ---------- Convenience function for surfaces ----------

async def run_chat_turn(
    message: str,
    pipeline=None,
    llm: Optional[LLMClient] = None,
    memory: Optional[ConversationMemory] = None,
    max_steps: int = 6,
    user_permission_level: int = 0,
) -> AgentTurnResult:
    """Run a single chat turn — convenience for dashboard bridge / CLI."""
    core = AgentCore(
        pipeline=pipeline,
        llm=llm,
        memory=memory,
        max_steps=max_steps,
        user_permission_level=user_permission_level,
    )
    return await core.run_turn(message)
