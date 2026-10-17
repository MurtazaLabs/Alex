"""Research log: every autonomy-loop run becomes a ``ResearchTask`` row.

Gives the Research page (and the API) a durable record of what Alex
researched, what was proposed, and what the deterministic gate decided.
Best-effort by design: a logging failure must never break a research run.

Mapping to the existing model:
- ``concept_name``    → "``<PAIR>: <spec name>``"
- ``status``          → COMPLETED (research finished; verdict lives in the
  conclusion/recommendation) or FAILED (the run itself errored)
- ``recommendation``  → PAPER_TEST (gate passed) / DO_NOT_USE (gate rejected)
- ``conclusion_json`` → full evidence: spec, gate report, messages, version
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..data.database import get_session_factory
from ..data.models.research import ResearchTask
from ..data.models.user import User

logger = logging.getLogger(__name__)

_ALEX_USERNAME = "alex"
_ALEX_EMAIL = "alex@localhost"


def _safe(obj: Any) -> Any:
    """JSON-round-trip so numpy/decimal/uuid artifacts can't break the column."""
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        return {"error": "unserializable research payload"}


async def _get_or_create_alex_user(session: Any) -> Any:
    """The loop runs headless — it needs a system user row for the FK."""
    result = await session.execute(select(User).where(User.username == _ALEX_USERNAME))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(
            username=_ALEX_USERNAME,
            email=_ALEX_EMAIL,
            password_hash="!",  # not a login account; column is NOT NULL
            permission_level=0,
        )
        session.add(user)
        await session.flush()
    return user


def _summary_for(result: Any) -> str:
    """One-line human summary of the run's outcome."""
    status = getattr(result, "status", "error")
    if status == "validated":
        return "Gate PASSED — versioned and entered paper probation awaiting human approval."
    if status == "rejected":
        reasons = getattr(result.validation, "reasons", None) or []
        # Stage C4 weak-hypothesis veto: never call a veto a "gate" rejection —
        # no backtest ran; the loop simply refused to repeat a known-weak shape.
        if any(str(r).startswith("weak-hypothesis veto:") for r in reasons):
            return "Weak-hypothesis veto: " + "; ".join(reasons[:2])
        head = "; ".join(reasons[:3]) if reasons else "gate criteria not met"
        return f"Gate REJECTED: {head}"
    messages = getattr(result, "messages", None) or []
    return "; ".join(messages[:3]) or "research run failed"


async def log_research_run(
    result: Any, session_factory: Optional[Any] = None
) -> Optional[Any]:
    """Persist one :class:`~autonomy.proposal.ProposalResult`.

    Returns the created ``ResearchTask`` or ``None`` on failure (never raises).
    """
    factory = session_factory or get_session_factory()
    started_at: Optional[datetime] = None
    proposed_at = getattr(result, "proposed_at", None)
    if isinstance(proposed_at, str):
        try:
            started_at = datetime.fromisoformat(proposed_at)
        except ValueError:
            started_at = None

    status = getattr(result, "status", "error")
    spec_dict = getattr(result, "spec_dict", None)
    spec_name = (spec_dict or {}).get("name", "unproposed")
    pair = getattr(result, "pair", "?")

    if status == "error":
        task_status, step, recommendation = "FAILED", "FAILED", None
    else:
        task_status = "COMPLETED"
        step = "COMPLETED"
        recommendation = "PAPER_TEST" if status == "validated" else "DO_NOT_USE"

    conclusion: Dict[str, Any] = {
        "summary": _summary_for(result),
        "verdict": status,
        "used_llm": bool(getattr(result, "used_llm", False)),
        "pair": pair,
    }
    if getattr(result, "llm_error", None):
        conclusion["llm_error"] = result.llm_error
    if spec_dict:
        conclusion["spec"] = _safe(spec_dict)
    validation = getattr(result, "validation", None)
    if validation is not None:
        conclusion["validation"] = _safe(validation.to_dict())
    version = getattr(result, "version", None)
    if version is not None:
        conclusion["version_id"] = str(version.id)
    messages = getattr(result, "messages", None) or []
    if messages:
        conclusion["messages"] = [str(m) for m in messages]
    brief = getattr(result, "brief", None)
    if brief:
        conclusion["brief"] = _safe(brief)

    try:
        async with factory() as session:
            user = await _get_or_create_alex_user(session)
            task = ResearchTask(
                user_id=user.id,
                concept_name=f"{pair}: {spec_name}"[:200],
                questions_json=[
                    f"What entry logic suits {pair} in the observed regime?",
                    "Does the proposal survive walk-forward out-of-sample validation?",
                ],
                status=task_status,
                current_step=step,
                progress_percent=100,
                conclusion_json=_safe(conclusion),
                recommendation=recommendation,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )
            session.add(task)
            await session.commit()
            return task
    except Exception as exc:  # noqa: BLE001 — logging must never break research
        logger.warning("research log write failed: %s: %s", type(exc).__name__, exc)
        return None


async def recent_research_tasks(
    limit: int = 25, session_factory: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Most recent research runs (all statuses) for dashboards/API."""
    factory = session_factory or get_session_factory()
    out: List[Dict[str, Any]] = []
    async with factory() as session:
        stmt = (
            select(ResearchTask)
            .order_by(ResearchTask.created_at.desc())
            .limit(limit)
        )
        rows = list((await session.execute(stmt)).scalars().all())

    for task in rows:
        conclusion = task.conclusion_json or {}
        summary = conclusion.get("summary", "In progress")
        verdict = conclusion.get("verdict", "")
        created = task.created_at.strftime("%Y-%m-%d %H:%M") if task.created_at else "Unknown"
        out.append(
            {
                "id": str(task.id),
                "topic": task.concept_name,
                "status": task.status,
                "progress": task.progress_percent or 0,
                "conclusion": summary,
                "created": created,
                "recommendation": task.recommendation or "",
                "verdict": verdict,
                "version_id": conclusion.get("version_id", ""),
            }
        )
    return out
