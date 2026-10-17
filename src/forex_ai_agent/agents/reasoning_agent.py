"""Reasoning Agent — LLM narration over deterministic evidence.

Alex's constitution, enforced mechanically:
- The LLM narrates WHY a setup exists; it never outputs entries, SL/TP,
  sizes or decisions. All numbers come from SignalAgent/RiskAgent.
- No LLM available (no key, budget exhausted, offline) → deterministic
  template narration from the same context. The signal never blocks on the LLM.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, Optional

from ..config import get_settings
from ..llm import LLMClient, LLMUnavailable

logger = logging.getLogger(__name__)

_SIGNAL_SYSTEM_PROMPT = """You are Alex, a forex research agent. You are narrating an \
already-generated trade setup. HARD RULES:
- You NEVER decide trades and NEVER invent numbers. Only use the numbers in the CONTEXT.
- Narrate why this setup exists using ONLY the provided evidence.
- Use calibrated language: "very low", "low", "moderate", "high", "very high" — never \
exact probabilities.
- If evidence contradicts the setup, say so explicitly.
- If information is missing, name it.
- Maximum 180 words. Plain prose, no bullet lists, no disclaimers.
- Your reply must be ONLY the final narration paragraph — no thinking process, no \
analysis, no preamble, no "Here's", just the narration itself."""

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LEADING_PREAMBLE_RE = re.compile(r"^(here'?s?|here is)\b.*:\s*$", re.IGNORECASE)
_LEAK_RE = re.compile(r"thinking process|chain[- ]of[- ]thought|step \d", re.IGNORECASE)


def _strip_reasoning_leakage(text: str) -> Optional[str]:
    """Return the clean narration, or None if the model dumped its reasoning.

    Some open models (notably reasoning-tuned ones) prepend their chain of
    thought despite instructions. If what remains after stripping ``<think>``
    blocks and a leading "Here's ...:" preamble still looks like a reasoning
    dump (markdown/numbered list start, leak phrases, or >220 words), we
    reject it and the caller falls back to the deterministic template.
    """
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    first_line, _, rest = cleaned.partition("\n")
    if _LEADING_PREAMBLE_RE.match(first_line.strip()):
        cleaned = rest.strip()
    if not cleaned:
        return None
    looks_like_dump = (
        _LEAK_RE.search(cleaned[:400]) is not None
        or cleaned.lstrip().startswith(("**", "1.", "#", "- ", "* "))
        or len(cleaned.split()) > 220
    )
    if looks_like_dump:
        return None
    return cleaned

_EXPLANATION_SYSTEM_PROMPT = """You are Alex, a forex research agent. Below is a \
deterministic seven-section explanation of a trade setup, followed by market context. \
Write a short natural summary (max 150 words) that a human trader can skim. HARD RULES: \
never invent numbers, never add advice beyond the provided evidence, name contradicting \
evidence and missing data if present."""


class ReasoningAgent:
    """Narrates signals and explanations using the LLM, with template fallback."""

    def __init__(self, llm: Optional[LLMClient] = None, settings: Any = None) -> None:
        self.llm = llm or LLMClient()
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def build_context(
        self,
        *,
        pair: str,
        direction: str,
        timeframe: str,
        session: str,
        structure_dict: Dict[str, Any],
        indicators_dict: Dict[str, Any],
        risk_checks: Dict[str, Any],
        signal_candidate: Dict[str, Any],
        data_source: str = "unknown",
    ) -> Dict[str, Any]:
        """Assemble the sanitized, LLM-safe narration context."""
        return {
            "pair": pair,
            "direction": direction,
            "timeframe": timeframe,
            "session": session,
            "data_source": data_source,
            "structure": structure_dict,
            "indicators": indicators_dict,
            "risk_checks": risk_checks,
            "signal": signal_candidate,
        }

    # ------------------------------------------------------------------
    # Narration
    # ------------------------------------------------------------------

    async def narrate_signal(self, context: Dict[str, Any], prefer_llm: bool = True) -> str:
        """One-paragraph narration of a signal; template fallback on failure.

        ``prefer_llm=False`` skips the LLM entirely (used for empty NO_TRADE
        states, where the deterministic template is complete and instant).
        """
        if not prefer_llm:
            return self._template_signal(context)
        try:
            raw = await asyncio.wait_for(
                self.llm.narrate(_SIGNAL_SYSTEM_PROMPT, context),
                timeout=self._settings.llm_narrate_timeout_s + 5.0,
            )
        except (LLMUnavailable, asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
            logger.info("LLM narration unavailable (%s); using deterministic template", exc)
            return self._template_signal(context)
        cleaned = _strip_reasoning_leakage(raw)
        if cleaned is None:
            logger.info("LLM narration was unparseable reasoning output; using template")
            return self._template_signal(context)
        return cleaned

    async def narrate_explanation(
        self, context: Dict[str, Any], seven_section: str
    ) -> str:
        """Short summary over the 7-section explanation; template fallback."""
        enriched = {**context, "deterministic_explanation": seven_section}
        try:
            return await asyncio.wait_for(
                self.llm.narrate(_EXPLANATION_SYSTEM_PROMPT, enriched), timeout=4.0
            )
        except (LLMUnavailable, asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
            logger.info("LLM explanation unavailable (%s); using deterministic template", exc)
            return self._template_explanation(context, seven_section)

    # ------------------------------------------------------------------
    # Deterministic fallbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _confidence_word(score_pct: Any) -> str:
        try:
            pct = float(score_pct)
        except (TypeError, ValueError):
            return "low"
        if pct >= 0.85:
            return "high"
        if pct >= 0.70:
            return "moderate"
        if pct >= 0.5:
            return "low"
        return "very low"

    @classmethod
    def _template_signal(cls, context: Dict[str, Any]) -> str:
        signal = context.get("signal", {}) or {}
        structure = context.get("structure", {}) or {}
        indicators = context.get("indicators", {}) or {}
        risk = context.get("risk_checks", {}) or {}

        pair = context.get("pair", "the pair")
        direction = str(context.get("direction", "")).lower()
        timeframe = context.get("timeframe", "")
        trend = structure.get("trend", "neutral")
        bos = structure.get("last_bos") or {}
        rsi = indicators.get("rsi")
        atr = indicators.get("atr")
        rr = signal.get("risk_reward")
        checks = risk.get("checks") or {}
        failed = [k for k, v in checks.items() if isinstance(v, dict) and not v.get("passed")]

        parts: list[str] = []
        parts.append(
            f"{pair} {direction} on {timeframe}: {trend.lower()} higher-timeframe structure"
            + (
                f" with a confirmed break of structure at {bos.get('level'):.5f}"
                if bos and isinstance(bos.get("level"), (int, float))
                else (f" with a confirmed break of structure at {bos.get('level')}" if bos else "")
            )
            + "."
        )
        if rsi is not None:
            parts.append(f"RSI at {rsi:.0f} {'supports' if 40 <= rsi <= 70 else 'is stretched for'} the setup.")
        if atr is not None:
            parts.append(f"ATR {atr:.1f} pips sets the volatility envelope.")
        if rr:
            parts.append(f"Reward-to-risk is {rr:.1f} to 1.")
        if failed:
            parts.append(f"Risk checks currently failing: {', '.join(failed)} — do not act until resolved.")
        elif checks:
            parts.append("All risk checks pass.")
        else:
            parts.append("No setup formed, so risk validation was not run.")
        if context.get("data_source"):
            parts.append(f"(Data: {context['data_source']} feed.)")
        return " ".join(parts)

    @staticmethod
    def _template_explanation(context: Dict[str, Any], seven_section: str) -> str:
        signal = context.get("signal", {}) or {}
        pair = context.get("pair", "")
        direction = str(context.get("direction", "")).lower()
        rr = signal.get("risk_reward")
        summary = (
            f"Deterministic summary for {pair} {direction}: setup built from the evidence "
            "below; entry, stop and targets are structural, not model outputs."
        )
        if rr:
            summary += f" Reward-to-risk {rr:.1f}:1."
        return f"{summary}\n\n{seven_section}"
