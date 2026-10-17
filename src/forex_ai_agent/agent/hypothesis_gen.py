"""LLM hypothesis generation (Stage C1) — the LLM chooses WHAT to research.

Roadmap Stage C: "Director gains LLM-driven hypothesis generation and
experiment selection over the B4 deterministic skeleton. Alex chooses **what
to research** … All budget guardrails and hard blocks carry over verbatim."

Constitution applied here:
- The LLM only ever sees deterministic market-structure briefs and only ever
  produces CANDIDATE hypotheses (text + confidence). It computes no number
  that anyone acts on — every candidate is validated by deterministic code
  and only then enqueued into the B4 Director queue, where the existing
  propose → backtest → gate pipeline (deterministic) owns every outcome.
- Free-model reality: the reasoning chain can rate-limit or return garbage.
  ``llm.reason_json`` parses defensively and this module treats any LLM
  failure as a signal to fall back to template hypotheses derived from the
  same briefs — the loop never stalls on credits (B4's promise, kept).
- Hard blocks by construction: this module can only reach the Director
  queue — no orders, no risk mutation, no permission changes (source-scanned
  by a test, like B4).
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from forex_ai_agent.data.database import get_session_factory

logger = logging.getLogger(__name__)

HYPO_SCHEMA_VERSION = 1

DEFAULT_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY"]
_MAX_PAIRS = 4
_MAX_HYPOTHESES = 5
_MIN_CONFIDENCE = 0.6

_LLM_SYSTEM_PROMPT = (
    "You are Alex's research director. You receive deterministic market-structure "
    "briefs for forex pairs (trend, swings, BOS/CHoCH, support/resistance, "
    "consolidation, liquidity sweeps) computed by deterministic code. Propose "
    "testable, specific trading hypotheses worth researching NEXT on those pairs. "
    "A hypothesis must state a concrete, falsifiable idea (entry logic family, "
    "regime, expected behaviour) — not trading advice. Confidence is your 0..1 "
    "estimate of how promising the idea is to TEST. Never invent numbers about "
    "profitability — you have no backtest data. If past-rejection lessons are "
    "provided, treat them as stored facts: avoid re-proposing the exact failing "
    "spec shapes they describe. If past-success lessons are provided, treat "
    "those as stored facts too: lean toward the spec shapes that PASSED the "
    "gate where the current briefs make them plausible — without copying any "
    "numbers from them."
)

_LLM_SCHEMA_HINT = (
    '{"hypotheses": [{"pair": "EUR/USD", "text": "specific falsifiable hypothesis '
    '(<=500 chars)", "rationale": "why this is worth testing now (<=300 chars)", '
    '"confidence": 0.75, "expected_regime": "trending|ranging|volatile"}]}'
)


@dataclass
class HypothesisGenConfig:
    """Budget + scope caps for one generation run. All deterministic."""

    pairs: Optional[List[str]] = None     # default universe, capped at 4
    timeframe: str = "H1"
    max_hypotheses: int = 3               # enqueued candidates per run
    min_confidence: float = _MIN_CONFIDENCE
    use_llm: bool = True                  # False → deterministic fallback only
    enqueue: bool = True                  # False → propose only, touch nothing
    priority: int = 2                     # NORMAL


@dataclass
class HypothesisGenReport:
    """What one run did — every number traceable to a brief or a queue row."""

    generated_at: str
    timeframe: str
    pairs_scanned: List[str]
    llm_used: bool = False
    llm_error: Optional[str] = None
    fallback_used: bool = False
    lessons_used: int = 0
    successes_used: int = 0
    raw_candidates: int = 0
    accepted: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    enqueued_count: int = 0
    schema_version: int = HYPO_SCHEMA_VERSION

    @property
    def disclaimer(self) -> str:
        return (
            "The LLM chose what to research; deterministic code validated every "
            "candidate and the backtest gate owns every outcome. Research only — "
            "no live execution, no risk changes."
        )

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["disclaimer"] = self.disclaimer
        return data


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _normalize_pair(raw: Any) -> Optional[str]:
    """``eurusd``/``EUR-USD``/``EUR/USD`` → ``EUR/USD``; None if unusable."""
    import re

    compact = re.sub(r"[^A-Za-z]", "", str(raw or "")).upper()
    if len(compact) != 6 or not compact.isalpha():
        return None
    return f"{compact[:3]}/{compact[3:]}"


async def _default_structure_fetcher(pair: str, timeframe: str) -> Dict[str, Any]:
    """The real brief source — the deterministic structure engine."""
    from forex_ai_agent.agent.tools import _build_structure_snapshot

    return await _build_structure_snapshot(pair, timeframe)


async def _default_lessons_loader(session_factory: Optional[Any] = None):
    """The real lesson source — both polarities mined from stored Director
    runs as ``(rejections, successes)``; plain lists stay back-compat."""
    from forex_ai_agent.agent.research_lessons import load_run_lessons

    return await load_run_lessons(session_factory=session_factory)


def build_brief(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Trim a structure snapshot to the fields the LLM may see (no opinions)."""
    sweeps = snapshot.get("liquidity_sweeps") or []
    return {
        "symbol": snapshot.get("symbol"),
        "timeframe": snapshot.get("timeframe"),
        "current_price": snapshot.get("current_price"),
        "trend": snapshot.get("trend"),
        "key_support": snapshot.get("key_support"),
        "key_resistance": snapshot.get("key_resistance"),
        "last_bos": snapshot.get("last_bos"),
        "last_choch": snapshot.get("last_choch"),
        "consolidation": snapshot.get("consolidation"),
        "recent_sweeps": [
            {"direction": s.get("direction"), "depth_atr": s.get("depth_atr")}
            for s in sweeps[-3:]
        ],
    }


def _fallback_hypotheses(briefs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Template hypotheses from the briefs when the LLM is unavailable.

    Clearly marked ``source="fallback"``; confidence is None (the confidence
    gate exists to police LLM self-assessment, not deterministic templates).
    """
    out: List[Dict[str, Any]] = []
    for brief in briefs:
        pair = brief.get("symbol") or "?"
        trend = str(brief.get("trend") or "").lower()
        support = brief.get("key_support")
        resistance = brief.get("key_resistance")
        consolidation = brief.get("consolidation") or {}
        if trend in ("up", "bullish", "uptrend"):
            text = (
                f"Buy pullbacks toward key support on {pair} {brief.get('timeframe')}: "
                f"the structure engine labels the trend {trend}; test whether "
                f"displacement-continuation entries at support hold out-of-sample."
            )
        elif trend in ("down", "bearish", "downtrend"):
            text = (
                f"Sell rallies toward key resistance on {pair} {brief.get('timeframe')}: "
                f"the structure engine labels the trend {trend}; test whether "
                f"displacement-continuation entries at resistance hold out-of-sample."
            )
        elif consolidation and consolidation.get("top") and consolidation.get("bottom"):
            text = (
                f"Fade range edges on {pair} {brief.get('timeframe')} between "
                f"{consolidation.get('bottom')} and {consolidation.get('top')}: "
                f"test whether mean-reversion at range extremes survives costs."
            )
        else:
            text = (
                f"Baseline trend-following on {pair} {brief.get('timeframe')}: "
                f"establish the pair's baseline edge (if any) before refining entry logic."
            )
        out.append(
            {
                "pair": pair,
                "text": text[:500],
                "rationale": "deterministic template from the structure brief (LLM unavailable)",
                "confidence": None,
                "expected_regime": trend or "unspecified",
                "source": "fallback",
            }
        )
    return out


def _validate_candidate(
    cand: Any, allowed_pairs: Dict[str, str], config: HypothesisGenConfig
) -> tuple:
    """Deterministic validation of one LLM candidate → (normalized, reject_reason)."""
    if not isinstance(cand, dict):
        return None, "not a JSON object"
    pair = _normalize_pair(cand.get("pair"))
    if pair is None:
        return None, "missing/unusable pair"
    if pair not in allowed_pairs:
        return None, f"pair {pair} not in this run's universe"
    text = str(cand.get("text") or "").strip()
    if len(text) < 10:
        return None, "hypothesis text too short (<10 chars)"
    if len(text) > 500:
        text = text[:500]
    confidence = cand.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return None, "confidence not a number"
        if not 0.0 <= confidence <= 1.0:
            return None, f"confidence {confidence} outside 0..1"
        if confidence < config.min_confidence:
            return None, f"confidence {confidence:.2f} below threshold {config.min_confidence:.2f}"
    regime = str(cand.get("expected_regime") or "").strip().lower()
    if regime not in ("trending", "ranging", "volatile", "unspecified", ""):
        regime = "unspecified"
    # Provenance is honest: LLM candidates carry no source (→ "llm"), while
    # deterministic fallback templates keep their "fallback" tag.
    source = str(cand.get("source") or "llm")
    if source not in ("llm", "fallback"):
        source = "llm"
    return {
        "pair": pair,
        "text": text,
        "rationale": str(cand.get("rationale") or "")[:300],
        "confidence": confidence,
        "expected_regime": regime or "unspecified",
        "source": source,
    }, None


async def generate_hypotheses(
    config: Optional[HypothesisGenConfig] = None,
    session_factory: Optional[Any] = None,
    *,
    structure_fetcher: Optional[Callable[..., Any]] = None,
    llm: Optional[Any] = None,
    lessons_loader: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """One generation run: briefs → LLM candidates → validation → Director queue.

    ``structure_fetcher(pair, timeframe)`` and ``llm`` are injection seams
    (tests substitute fakes; production lazy-imports the real ones). Returns
    a JSON-safe report dict; never raises for LLM failures — only a truly
    unusable universe (no valid pairs) raises ``ValueError``.
    """
    config = config or HypothesisGenConfig()
    factory = session_factory or get_session_factory()
    structure_fetcher = structure_fetcher or _default_structure_fetcher

    clean_pairs: List[str] = []
    for p in config.pairs or DEFAULT_PAIRS:
        pair = _normalize_pair(p)
        if pair and pair not in clean_pairs:
            clean_pairs.append(pair)
    clean_pairs = clean_pairs[:_MAX_PAIRS]
    if not clean_pairs:
        raise ValueError("no valid pairs in the generation universe")
    allowed = {p: p for p in clean_pairs}

    report = HypothesisGenReport(
        generated_at=_now_iso(),
        timeframe=config.timeframe,
        pairs_scanned=clean_pairs,
    )

    # 1. Deterministic briefs (feed, not opinion).
    briefs: List[Dict[str, Any]] = []
    for pair in clean_pairs:
        try:
            snapshot = await structure_fetcher(pair, config.timeframe)
            briefs.append(build_brief(snapshot))
        except Exception as exc:  # noqa: BLE001 — one dead pair must not stop the run
            report.rejected.append(
                {"pair": pair, "reason": f"brief unavailable: {type(exc).__name__}: {exc}"}
            )
    if not briefs:
        report.llm_error = "no structure briefs could be built for any pair"
        return report.to_dict()

    # 2. Candidates: LLM first (budget to the config), deterministic fallback
    #    on any LLM failure.
    candidates: List[Dict[str, Any]] = []
    if config.use_llm:
        if llm is None:
            from forex_ai_agent.llm.client import LLMClient

            llm = LLMClient()
        # Learning loop (deterministic): mined lessons from past gate
        # rejections AND gate-passed runs ride into the context as FACTS.
        # They must never break the run — any loader failure simply means no
        # lessons this run.
        lessons_loader = lessons_loader or _default_lessons_loader
        try:
            loaded = lessons_loader(session_factory=factory)
            if inspect.isawaitable(loaded):
                loaded = await loaded
            if isinstance(loaded, tuple) and len(loaded) == 2:
                rejections, successes = loaded
            else:
                rejections, successes = list(loaded or []), []
            report.lessons_used = len(rejections or [])
            report.successes_used = len(successes or [])
        except Exception as exc:  # noqa: BLE001 — lessons are optional fuel
            logger.warning("lesson loading failed (continuing without): %s", exc)
            rejections, successes = [], []
        context: Dict[str, Any] = {"briefs": briefs, "requested_pairs": clean_pairs}
        if rejections:
            from forex_ai_agent.agent.research_lessons import render_lessons_for_context

            context["past_rejections"] = render_lessons_for_context(rejections)
        if successes:
            from forex_ai_agent.agent.research_lessons import render_successes_for_context

            context["past_successes"] = render_successes_for_context(successes)
        try:
            raw = await llm.reason_json(
                _LLM_SYSTEM_PROMPT,
                context,
                schema_hint=_LLM_SCHEMA_HINT,
            )
            raw_list = raw.get("hypotheses") if isinstance(raw, dict) else None
            if not isinstance(raw_list, list):
                raise ValueError("LLM response missing 'hypotheses' list")
            report.llm_used = True
            candidates = list(raw_list)
        except Exception as exc:  # noqa: BLE001 — credits/rate-limit/garbage JSON
            report.llm_error = f"{type(exc).__name__}: {exc}"
            logger.warning("hypothesis generation LLM failed: %s: %s", type(exc).__name__, exc)
    if not candidates:
        candidates = _fallback_hypotheses(briefs)
        report.fallback_used = not report.llm_used

    report.raw_candidates = len(candidates)

    # 3. Deterministic validation, best-first, capped.
    validated: List[Dict[str, Any]] = []
    for cand in candidates:
        normalized, reason = _validate_candidate(cand, allowed, config)
        if reason is not None:
            report.rejected.append(
                {"pair": str((cand or {}).get("pair", "?")) if isinstance(cand, dict) else "?",
                 "reason": reason}
            )
            continue
        validated.append(normalized)
    validated.sort(key=lambda h: (-(h["confidence"] or 0.0), h["pair"]))
    if len(validated) > config.max_hypotheses:
        for extra in validated[config.max_hypotheses:]:
            report.rejected.append(
                {"pair": extra["pair"], "reason": "beyond max_hypotheses for this run"}
            )
        validated = validated[: config.max_hypotheses]

    # 4. Enqueue into the B4 Director (the only side effect this module has).
    for hyp in validated:
        entry = dict(hyp)
        if config.enqueue:
            from forex_ai_agent.agent.director import enqueue_hypothesis

            try:
                res = await enqueue_hypothesis(
                    pair=hyp["pair"],
                    hypothesis_text=hyp["text"],
                    timeframe=config.timeframe,
                    priority=config.priority,
                    session_factory=factory,
                )
            except ValueError as exc:
                res = {"enqueued": False, "reason": str(exc)}
            entry["enqueued"] = bool(res.get("enqueued"))
            entry["item_id"] = res.get("item_id")
            if not res.get("enqueued"):
                entry["enqueue_reason"] = res.get("reason")
                report.rejected.append(
                    {"pair": hyp["pair"], "reason": f"queue refused: {res.get('reason')}"}
                )
            else:
                report.enqueued_count += 1
        else:
            entry["enqueued"] = False
            entry["item_id"] = None
        report.accepted.append(entry)

    return report.to_dict()
