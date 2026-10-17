"""Probation promotion (Stage C4) — deterministic ranking, LLM narration only.

Roadmap Stage C4: "weak-hypothesis rejection, probation promotion". The
probation store already owns the objective gates (win rate / profit factor /
daily loss / quota / window — ``autonomy/probation.py``); this module closes
the last gap: a probation candidate whose gates are MET can sit unnoticed in
the queue. ``review_promotions`` deterministically ranks every probation row
that is ready for the human decision and produces the promotion package —
numbers from stored records, narration by the LLM at most as PROSE.

Constitution mapping:
- Every number here is read from ``strategy_probations`` (and gate metrics
  stored by the probation runner); the LLM computes nothing, decides nothing
  and cannot promote anything. It only renders a short narration per
  candidate; any narration failure degrades to a deterministic one-liner.
- Promotion itself stays HUMAN-ONLY: this module never calls
  ``ProbationStore.approve``/``reject``. It marks nothing, writes nothing —
  the existing human buttons keep the exclusive transition to ``approved``.
- Hard blocks by construction: read-only over stored records (no orders, no
  risk changes, no permission changes — source-scanned by a test).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROMOTION_SCHEMA_VERSION = 1
_MAX_CANDIDATES = 10


@dataclass
class PromotionConfig:
    """Caps for one review run. All deterministic."""

    limit: int = 5                        # ranked candidates returned
    include_narration: bool = True        # LLM prose per candidate (fallback-safe)
    include_expired: bool = False         # expired rows are informational, not ready


@dataclass
class PromotionReview:
    """What one review produced — every number traceable to a stored row."""

    generated_at: str
    reviewed: int = 0
    ready_count: int = 0
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    narration_used: bool = False
    narration_error: Optional[str] = None
    schema_version: int = PROMOTION_SCHEMA_VERSION

    @property
    def disclaimer(self) -> str:
        return (
            "Deterministic review of stored probation records. No candidate was "
            "promoted: approval and rejection remain human-only buttons. "
            "Narration (if any) is LLM prose over stored numbers — never a decision."
        )

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["disclaimer"] = self.disclaimer
        return data


def _now_iso() -> str:
    from datetime import datetime

    return datetime.utcnow().isoformat()


def _metric(status: Any, key: str) -> Optional[float]:
    value = getattr(status, key, None)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def rank_ready(rows: List[Any]) -> List[Any]:
    """Ready-first ordering (deterministic): ``passed`` before ``active``
    closest-to-quota; ties broken by version_id. ``expired``/terminal rows
    sort last without being called ready. Storage order never leaks in."""
    def key(r: Any) -> tuple:
        status = str(getattr(r, "status", "") or "")
        if status == "passed":
            band, progress = 0, 1.0
        elif status == "active":
            req = int(getattr(r, "required_trades", 0) or 0)
            obs = int(getattr(r, "observed_trades", 0) or 0)
            band, progress = 1, (obs / req) if req > 0 else 0.0
        else:
            band, progress = 2, 0.0
        return (band, -progress, str(getattr(r, "version_id", "") or ""))

    return sorted(rows, key=key)


def _candidate_dict(status: Any) -> Dict[str, Any]:
    """One candidate's stored facts — no derived opinions."""
    ready = str(getattr(status, "status", "") or "") == "passed"
    return {
        "version_id": str(getattr(status, "version_id", "") or ""),
        "strategy_name": str(getattr(status, "strategy_name", "") or ""),
        "pair": str(getattr(status, "pair", "") or ""),
        "status": str(getattr(status, "status", "") or ""),
        "ready": ready,
        "observed_trades": int(getattr(status, "observed_trades", 0) or 0),
        "required_trades": int(getattr(status, "required_trades", 0) or 0),
        "observed_days": int(getattr(status, "observed_days", 0) or 0),
        "required_days": int(getattr(status, "required_days", 0) or 0),
        "days_elapsed": int(getattr(status, "days_elapsed", 0) or 0),
        "win_rate": _metric(status, "win_rate"),
        "profit_factor": _metric(status, "profit_factor"),
        "worst_daily_loss_pct": _metric(status, "worst_daily_loss_pct"),
        "notes": str(getattr(status, "notes", "") or "")[:200],
        "started_at": str(getattr(status, "started_at", "") or ""),
        "narration": None,
    }


_PROMOTION_SYSTEM_PROMPT = (
    "You are Alex's research assistant. You receive deterministic promotion "
    "packages for strategy versions whose paper-probation gates are MET "
    "(stored numbers only). Write ONE short factual narration per candidate "
    "(<=2 sentences): what was tested, how it performed against its stored "
    "probation thresholds, and that the human decides next. Use ONLY the "
    "stored numbers given. Never add advice, promises, or numbers."
)

_PROMOTION_SCHEMA_HINT = '{"narrations": [{"version_id": "...", "narration": "..."}]}'


async def _narrate(candidates: List[Dict[str, Any]], llm: Optional[Any] = None) -> tuple[bool, Optional[str]]:
    """LLM narration seam: fills ``narration`` per candidate. Any failure
    degrades to the deterministic one-liner — never raises."""
    try:
        if llm is None:
            from forex_ai_agent.llm.client import LLMClient

            llm = LLMClient()
        context = {
            "candidates": [
                {k: c[k] for k in (
                    "version_id", "strategy_name", "pair", "status",
                    "observed_trades", "required_trades", "win_rate",
                    "profit_factor", "worst_daily_loss_pct", "notes",
                )}
                for c in candidates
            ]
        }
        raw = await llm.reason_json(
            _PROMOTION_SYSTEM_PROMPT, context, schema_hint=_PROMOTION_SCHEMA_HINT
        )
        items = raw.get("narrations") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            raise ValueError("LLM response missing 'narrations' list")
        by_id: Dict[str, str] = {}
        for item in items:
            if isinstance(item, dict) and item.get("version_id") and item.get("narration"):
                by_id[str(item["version_id"])] = str(item["narration"])[:300]
        for c in candidates:
            c["narration"] = by_id.get(c["version_id"]) or _fallback_narration(c)
        return True, None
    except Exception as exc:  # noqa: BLE001 — narration is optional prose
        logger.warning("promotion narration failed: %s: %s", type(exc).__name__, exc)
        for c in candidates:
            c["narration"] = _fallback_narration(c)
        return False, f"{type(exc).__name__}: {exc}"


def _fallback_narration(c: Dict[str, Any]) -> str:
    """Deterministic one-liner from the stored numbers (narration fallback)."""
    if c.get("ready"):
        return (
            f"Probation gates MET: {c['observed_trades']}/{c['required_trades']} "
            f"trades on {c['pair']}; awaiting the human approval decision."
        )
    return (
        f"Probation {c['status']}: {c['observed_trades']}/{c['required_trades']} "
        f"trades on {c['pair']}; not ready for approval."
    )


async def review_promotions(
    config: Optional[PromotionConfig] = None,
    *,
    store: Optional[Any] = None,
    session_factory: Optional[Any] = None,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """One deterministic review: rank stored probation rows, package the ones
    whose gates are MET, narrate (LLM prose at most). Never promotes.

    ``store`` and ``llm`` are injection seams (tests pass fakes); production
    lazy-imports the real ``ProbationStore`` / ``LLMClient``. The review never
    raises for data problems — a broken store yields an empty, honest report.
    """
    config = config or PromotionConfig()
    if store is None:
        from forex_ai_agent.autonomy.probation import ProbationStore

        store = ProbationStore(session_factory=session_factory)

    review = PromotionReview(generated_at=_now_iso())
    try:
        rows = await store.list()
    except Exception as exc:  # noqa: BLE001 — an unreadable queue is not fatal
        review.narration_error = f"probation store unavailable: {type(exc).__name__}: {exc}"
        return review.to_dict()

    ready_rows = [
        r for r in rows
        if str(getattr(r, "status", "") or "") == "passed"
        or (config.include_expired and str(getattr(r, "status", "") or "") == "expired")
    ]
    ranked = rank_ready(ready_rows)[: max(1, min(int(config.limit), _MAX_CANDIDATES))]
    review.reviewed = len(rows)
    review.candidates = [_candidate_dict(r) for r in ranked]
    review.ready_count = sum(1 for c in review.candidates if c["ready"])

    if review.candidates and config.include_narration:
        review.narration_used, review.narration_error = await _narrate(review.candidates, llm=llm)

    return review.to_dict()
