"""Structured research plans (Stage B5) — the RESEARCH #id card.

Roadmap §3 B5: ``start_research`` upgrades to a structured plan persisted as
its own research-task row: hypothesis, pairs, timeframes, period, an
experiment checklist (baseline → displacement → entry → exit → session → cost
sensitivity), a validation checklist and a status. Progress updates as the
Director (B4) works the plan; ``research_plan`` returns the conclusion card.

Design rules inherited from B1–B4:
- Deterministic code owns every number. The LLM may narrate the card later;
  this module only reads and writes stored records.
- Best-effort writes (a logging failure must never break a research run) for
  the *Director-side* refresh; explicit failures on the *create* path.
- SQLite is create_all-managed: ``research_plans`` is a new table on the
  shared Base (B2's edge-table pattern), created idempotently by
  :func:`ensure_plan_schema` at boot — no ALTER, no migration.
- Per-pair Director items carry ``plan_id`` in ``result_json``; refresh joins
  on that, so a plan survives process restarts (all state is in the DB).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import String, select
from sqlalchemy.orm import Mapped, mapped_column

from forex_ai_agent.data.database import get_session_factory
from forex_ai_agent.data.models.base import (
    Base,
    JSONType,
    TimestampMixin,
    UUIDPkMixin,
)

logger = logging.getLogger(__name__)

PLAN_SCHEMA_VERSION = 1

# The roadmap's fixed experiment checklist (order matters — it is the
# methodology, not a menu).
EXPERIMENT_STEPS = (
    "baseline",
    "displacement",
    "entry",
    "exit",
    "session",
    "cost_sensitivity",
)

VALIDATION_STEPS = (
    "walk_forward",
    "robustness_sweep",
    "parameter_sensitivity",
    "probation",
)

# Human-readable labels for the card.
_EXPERIMENT_LABELS = {
    "baseline": "Baseline backtest",
    "displacement": "Displacement parameter",
    "entry": "Entry logic",
    "exit": "Exit logic",
    "session": "Session filter",
    "cost_sensitivity": "Cost sensitivity",
}
_VALIDATION_LABELS = {
    "walk_forward": "Walk-forward OOS",
    "robustness_sweep": "Cross-pair robustness",
    "parameter_sensitivity": "Parameter sensitivity",
    "probation": "Paper probation",
}

STATUS_PENDING = "PENDING"
STATUS_IN_PROGRESS = "IN_PROGRESS"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_CANCELLED = "CANCELLED"

# Pair is exhausted only in a terminal state; the loop never rewinds one.
_PAIR_TERMINAL = {"validated", "rejected", "error", "cancelled"}

# Director row status → plan pair status (result_json.status is absent for
# error rows, so the row's own terminal status decides honestly).
_ITEM_STATUS_TO_PAIR = {
    "QUEUED": "queued",
    "RUNNING": "running",
    "DONE_VALIDATED": "validated",
    "DONE_REJECTED": "rejected",
    "DONE_ERROR": "error",
    "CANCELLED": "cancelled",
}

_MESSAGES_KEPT = 5


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _safe(obj: Any) -> Any:
    """JSON-round-trip so numpy/decimal/uuid artifacts can't break the column."""
    import json

    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        return {"error": "unserializable plan payload"}


# ---------------------------------------------------------------------------
# Model — research_plans (create_all-managed, like B2's research_graph_edges)
# ---------------------------------------------------------------------------


class ResearchPlan(UUIDPkMixin, TimestampMixin, Base):
    """One structured research task (the RESEARCH #id card row).

    ``plan_json`` carries the full plan document (checklists, per-pair
    progress, conclusion); the scalar columns exist so plain SQL and other
    pages can list plans without parsing JSON.
    """

    __tablename__ = "research_plans"

    topic: Mapped[str] = mapped_column(String(200), nullable=False)
    hypothesis: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    pairs: Mapped[list] = mapped_column(JSONType, nullable=False, default=list)
    timeframes: Mapped[list] = mapped_column(JSONType, nullable=False, default=list)
    period_days: Mapped[int] = mapped_column(nullable=False, default=365)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=STATUS_PENDING, index=True)
    progress_percent: Mapped[int] = mapped_column(nullable=False, default=0)
    current_step: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    plan_json: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    schema_version: Mapped[int] = mapped_column(nullable=False, default=PLAN_SCHEMA_VERSION)


async def ensure_plan_schema(session_factory: Optional[Any] = None) -> Dict[str, Any]:
    """Create ``research_plans`` if the DB predates B5. Idempotent.

    Safe to call on every boot and from tests: existing data is untouched,
    repeated calls are no-ops. Returns what was done (for logging/tests).
    """
    factory = session_factory or get_session_factory()
    created = False
    async with factory() as session:
        conn = await session.connection()

        def _ensure(sync_conn: Any) -> None:
            nonlocal created
            before = {
                row[0]
                for row in sync_conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            Base.metadata.create_all(sync_conn, tables=[ResearchPlan.__table__])
            after = {
                row[0]
                for row in sync_conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            created = "research_plans" in after and "research_plans" not in before

        await conn.run_sync(_ensure)
        await session.commit()
    if created:
        logger.info("research_plan_schema_ensured %s", {"created": "research_plans"})
    return {"created_table": created}


# ---------------------------------------------------------------------------
# Plan build / progress derivation (pure functions over the plan document)
# ---------------------------------------------------------------------------


def build_plan_payload(
    *,
    topic: str,
    pairs: List[str],
    timeframes: List[str],
    hypothesis_text: str = "",
    period_days: int = 365,
) -> Dict[str, Any]:
    """The stored plan document — a plain JSON-able dict, versioned."""
    now = _now_iso()
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "topic": topic,
        "hypothesis": (hypothesis_text or topic)[:500],
        "pairs": list(pairs),
        "timeframes": list(timeframes),
        "period_days": int(period_days),
        "status": STATUS_PENDING,
        "created_at": now,
        "updated_at": now,
        "experiment_checklist": [
            {"step": step, "label": _EXPERIMENT_LABELS.get(step, step), "state": "pending"}
            for step in EXPERIMENT_STEPS
        ],
        "validation_checklist": [
            {"step": step, "label": _VALIDATION_LABELS.get(step, step), "state": "pending"}
            for step in VALIDATION_STEPS
        ],
        "pairs_progress": {},
        "conclusion": None,
    }


def _pair_status(result: Dict[str, Any]) -> str:
    return str(result.get("status") or "unknown").lower()


def _derive_plan_status(queued: bool, n_pairs: int, done_pairs: int) -> str:
    if n_pairs == 0:
        return STATUS_PENDING
    if done_pairs == 0:
        return STATUS_IN_PROGRESS if queued else STATUS_PENDING
    if done_pairs < n_pairs:
        return STATUS_IN_PROGRESS
    return STATUS_COMPLETED


def _apply_plan_progress(
    payload: Dict[str, Any], rows: List[Dict[str, Any]], *, queued: bool = False
) -> Tuple[Dict[str, Any], int]:
    """Fold Director item rows into the plan (pure; returns (new payload, touched)).

    A Director row belongs to this plan when its ``result_json.plan_id`` equals
    the plan id — the join Director writes at enqueue time. Terminal pair
    states never rewind; checklist steps only ever advance.
    """
    payload = dict(payload)
    payload.setdefault("pairs_progress", {})
    # Heal plans written by older schema versions so refresh never dies.
    if not isinstance(payload.get("experiment_checklist"), list) or not payload["experiment_checklist"]:
        payload["experiment_checklist"] = [
            {"step": s, "label": _EXPERIMENT_LABELS.get(s, s), "state": "pending"}
            for s in EXPERIMENT_STEPS
        ]
    if not isinstance(payload.get("validation_checklist"), list) or not payload["validation_checklist"]:
        payload["validation_checklist"] = [
            {"step": s, "label": _VALIDATION_LABELS.get(s, s), "state": "pending"}
            for s in VALIDATION_STEPS
        ]

    touched = 0
    for row in rows:
        result = dict(row.get("result") or {})
        if not result or str(result.get("plan_id") or "") != str(payload.get("task_id", "")):
            continue
        touched += 1
        pair = row.get("pair") or "?"
        before = dict(payload["pairs_progress"].get(pair) or {})
        # A freshly-enqueued item carries plan_id but no run outcome yet —
        # show the Director row's own status instead of a bare "unknown".
        status = _pair_status(result)
        if status == "unknown":
            status = _ITEM_STATUS_TO_PAIR.get(
                str(row.get("item_status") or "").upper(), "unknown"
            )
        if before.get("status") in _PAIR_TERMINAL:
            status = before["status"]  # terminal wins; the loop never rewinds
        if (
            before.get("status") == status
            and before.get("version_id") == result.get("version_id")
            and before.get("run_finished_at") == row.get("run_finished_at")
        ):
            continue  # observation already folded — refresh stays idempotent
        entry = {
            "status": status,
            "used_llm": bool(result.get("used_llm")),
            "version_id": result.get("version_id"),
            "messages": [str(m) for m in (result.get("messages") or [])][:_MESSAGES_KEPT],
            "run_finished_at": row.get("run_finished_at"),
        }
        payload["pairs_progress"][pair] = entry

        # Experiment checklist (shared across pairs — one methodology):
        # the first run opens step 1; each finished pair completes the open
        # step; the next step opens only while pairs remain to be worked.
        checklist = payload["experiment_checklist"]
        states = [s["state"] for s in checklist]
        if not any(s == "in_progress" for s in states):
            for s in checklist:
                if s["state"] == "pending":
                    s["state"] = "in_progress"
                    break
        if status in _PAIR_TERMINAL:
            for s in checklist:
                if s["state"] == "in_progress":
                    s["state"] = "done"
                    break
            remaining = sum(
                1
                for p in payload["pairs_progress"].values()
                if p.get("status") not in _PAIR_TERMINAL
            )
            if remaining:
                for s in checklist:
                    if s["state"] == "pending":
                        s["state"] = "in_progress"
                        break

        # Validation checklist: deterministic, outcome-driven states.
        val = payload["validation_checklist"]
        ok = status == "validated"

        def _set(step: str, state: str) -> None:
            for s in val:
                if s["step"] == step and s["state"] != "done":
                    s["state"] = state

        if ok:
            _set("walk_forward", "done")
            _set("probation", "in_progress")
        elif status in _PAIR_TERMINAL:
            _set("walk_forward", "skipped")

        payload["conclusion"] = {
            "verdict": status,
            "summary": entry["messages"][0] if entry["messages"] else "",
            "version_id": entry["version_id"],
        }

    n_pairs = len(payload["pairs_progress"])
    done_pairs = sum(
        1 for p in payload["pairs_progress"].values() if p.get("status") in _PAIR_TERMINAL
    )
    payload["status"] = _derive_plan_status(queued, n_pairs, done_pairs)
    payload["progress_percent"] = int(round(100.0 * done_pairs / n_pairs)) if n_pairs else 0
    payload["current_step"] = next(
        (s["label"] for s in payload["experiment_checklist"] if s["state"] == "in_progress"),
        payload["experiment_checklist"][-1]["label"] if payload["experiment_checklist"] else "COMPLETED",
    )
    payload["updated_at"] = _now_iso()
    return payload, touched


def _plan_to_card(payload: Dict[str, Any], plan_id: str, row_status: Optional[str] = None) -> Dict[str, Any]:
    """The RESEARCH #id card — every field from a persisted record."""
    checklist = payload.get("experiment_checklist") or []
    val = payload.get("validation_checklist") or []
    progress = payload.get("pairs_progress") or {}
    exp_done = sum(1 for s in checklist if s["state"] == "done")

    def _items(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {"step": s["step"], "label": s.get("label", s["step"]), "state": s["state"]}
            for s in steps
        ]

    return {
        "task_id": plan_id,
        "topic": payload.get("topic", ""),
        "hypothesis": payload.get("hypothesis", ""),
        "pairs": list(payload.get("pairs") or []),
        "timeframes": list(payload.get("timeframes") or []),
        "period_days": payload.get("period_days"),
        "status": payload.get("status", STATUS_PENDING),
        "row_status": row_status,
        "progress_percent": payload.get("progress_percent", 0),
        "current_step": payload.get("current_step"),
        "experiment_checklist": _items(checklist),
        "experiment_done": exp_done,
        "experiment_total": len(checklist),
        "validation_checklist": _items(val),
        "pairs_progress": {
            pair: {
                "status": p.get("status"),
                "used_llm": bool(p.get("used_llm")),
                "version_id": p.get("version_id"),
                "run_finished_at": p.get("run_finished_at"),
            }
            for pair, p in progress.items()
        },
        "conclusion": payload.get("conclusion"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "schema_version": payload.get("schema_version"),
    }


def _item_to_row(item: Any) -> Dict[str, Any]:
    finished = getattr(item, "run_finished_at", None)
    return {
        "pair": getattr(item, "pair", "?"),
        "item_status": getattr(item, "status", None),
        "result": dict(getattr(item, "result_json", None) or {}),
        "run_finished_at": finished.isoformat() if finished else None,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


async def create_research_plan(
    *,
    topic: str,
    pairs: List[str],
    timeframes: Optional[List[str]] = None,
    hypothesis_text: str = "",
    period_days: int = 365,
    enqueued: bool = True,
    priority: int = 2,
    config: Optional[Any] = None,
    session_factory: Optional[Any] = None,
) -> Dict[str, Any]:
    """Persist a plan row and (optionally) queue its pairs with the Director.

    Returns ``{"task_id", "plan", "enqueue", "card", "rendered"}``. The card
    is built from the persisted row only — no in-memory shortcuts. Raises
    ``ValueError`` on a bad topic/pair (caller surfaces it honestly).
    """
    if not topic or not str(topic).strip():
        raise ValueError("topic is required")
    clean_pairs: List[str] = []
    for p in pairs or []:
        p = str(p).strip().upper()
        if not p:
            continue
        if "/" not in p and len(p) == 6:
            p = f"{p[:3]}/{p[3:]}"
        if "/" not in p or len(p) != 7:
            raise ValueError(f"pair must look like EUR/USD, got {p!r}")
        if p not in clean_pairs:
            clean_pairs.append(p)
    if not clean_pairs:
        raise ValueError("at least one valid pair is required")
    tfs = [str(t).strip().upper() for t in (timeframes or ["H1"]) if str(t).strip()] or ["H1"]
    period_days = max(60, min(730, int(period_days or 365)))
    clean_topic = str(topic).strip()

    factory = session_factory or get_session_factory()
    payload = build_plan_payload(
        topic=clean_topic,
        pairs=clean_pairs,
        timeframes=tfs,
        hypothesis_text=hypothesis_text,
        period_days=period_days,
    )
    async with factory() as session:
        plan = ResearchPlan(
            topic=clean_topic[:200],
            hypothesis=(hypothesis_text or clean_topic)[:500],
            pairs=clean_pairs,
            timeframes=tfs,
            period_days=period_days,
            status=STATUS_PENDING,
            progress_percent=0,
            current_step=_EXPERIMENT_LABELS[EXPERIMENT_STEPS[0]],
            plan_json=_safe(payload),
        )
        session.add(plan)
        await session.flush()
        task_id = str(plan.id)
        payload["task_id"] = task_id
        plan.plan_json = _safe(payload)
        await session.commit()

    enqueue_result: Dict[str, Any] = {"requested": bool(enqueued), "items": []}
    if enqueued:
        from forex_ai_agent.agent.director import enqueue_hypothesis

        for pair in clean_pairs:
            # ValueError (bad priority) propagates: a contract error on the
            # create path must surface honestly. Queue-full / duplicates come
            # back as {"enqueued": False, "reason": ...}, not exceptions.
            res = await enqueue_hypothesis(
                pair=pair,
                hypothesis_text=(hypothesis_text or clean_topic)[:500],
                timeframe=tfs[0],
                priority=priority,
                config=config,
                session_factory=factory,  # resolved ONCE above — one store
                plan_id=task_id,
            )
            res["pair"] = pair
            enqueue_result["items"].append(res)
        # Director rows joined back on plan_id (written by the cycle hook) —
        # refresh once so the card already reflects queued/started work.
        await refresh_plan_from_director(task_id, session_factory=factory)

    card = await get_research_plan(task_id, session_factory=factory)
    if card is None:  # pragma: no cover — the row was just committed
        card = _plan_to_card(payload, task_id)
    return {
        "task_id": task_id,
        "plan": payload,
        "enqueue": enqueue_result,
        "card": card,
        "rendered": render_plan_card(card),
    }


async def refresh_plan_from_director(
    task_id: str, session_factory: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """Fold Director rows (joined on ``result_json.plan_id``) into the plan.

    Progress comes ONLY from stored records. Returns the refreshed card, or
    ``None`` if the plan id is unknown. Best-effort: never raises — the
    Director's cycle must not die because a plan row is missing.
    """
    try:
        key = uuid.UUID(str(task_id))
    except (TypeError, ValueError):
        return None
    factory = session_factory or get_session_factory()
    try:
        async with factory() as session:
            plan = await session.get(ResearchPlan, key)
            if plan is None:
                return None
            from forex_ai_agent.agent.director import DirectorQueueItem

            rows = (
                (
                    await session.execute(
                        select(DirectorQueueItem).order_by(DirectorQueueItem.created_at)
                    )
                )
                .scalars()
                .all()
            )
            open_rows = [
                i for i in rows if getattr(i, "status", None) not in ("CANCELLED",)
            ]
            queued = any(getattr(i, "status", None) in ("QUEUED", "RUNNING") for i in open_rows)
            item_rows = [_item_to_row(i) for i in rows]
            payload, _touched = _apply_plan_progress(
                dict(plan.plan_json or {}), item_rows, queued=queued
            )
            plan.plan_json = _safe(payload)
            plan.status = payload["status"]
            plan.progress_percent = int(payload.get("progress_percent") or 0)
            plan.current_step = (payload.get("current_step") or "")[:50] or None
            await session.commit()
        return await get_research_plan(str(task_id), session_factory=factory)
    except Exception as exc:  # noqa: BLE001 — refresh must never break the cycle
        logger.warning("research plan refresh failed: %s: %s", type(exc).__name__, exc)
        return None


async def get_research_plan(
    task_id: str, session_factory: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """The RESEARCH #id card, built from the persisted row (no re-runs)."""
    try:
        key = uuid.UUID(str(task_id))
    except (TypeError, ValueError):
        return None
    factory = session_factory or get_session_factory()
    async with factory() as session:
        plan = await session.get(ResearchPlan, key)
        if plan is None:
            return None
        return _plan_to_card(dict(plan.plan_json or {}), str(plan.id), plan.status)


async def list_research_plans(
    limit: int = 20, session_factory: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Card summaries (most recent first) for the plans index."""
    factory = session_factory or get_session_factory()
    async with factory() as session:
        plans = (
            (
                await session.execute(
                    select(ResearchPlan)
                    .order_by(ResearchPlan.created_at.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        snapshots = [
            _plan_to_card(dict(p.plan_json or {}), str(p.id), p.status) for p in plans
        ]
    return snapshots


def render_plan_card(card: Dict[str, Any]) -> str:
    """Deterministic text rendering of the RESEARCH #id card."""
    icon = {
        STATUS_COMPLETED: "✅",
        STATUS_IN_PROGRESS: "🔄",
        STATUS_PENDING: "⏳",
        STATUS_FAILED: "⚠️",
        STATUS_CANCELLED: "🚫",
    }.get(card.get("status"), "⏳")
    _MARKS = {"done": "[x]", "in_progress": "[~]", "skipped": "[-]", "pending": "[ ]"}

    lines = [
        f"RESEARCH #{str(card.get('task_id', ''))[:8]} — {card.get('topic', '')} {icon}",
        f"Hypothesis: {card.get('hypothesis', '')}",
        f"Pairs: {', '.join(card.get('pairs') or [])} · Timeframes: "
        f"{', '.join(card.get('timeframes') or [])} · Period: {card.get('period_days')} days",
        f"Status: {card.get('status')} · progress {card.get('progress_percent', 0)}%",
        "Experiments:",
    ]
    for s in card.get("experiment_checklist") or []:
        lines.append(f"  {_MARKS.get(s['state'], '[ ]')} {s['label']}")
    lines.append("Validation:")
    for s in card.get("validation_checklist") or []:
        lines.append(f"  {_MARKS.get(s['state'], '[ ]')} {s['label']}")
    pairs_progress = card.get("pairs_progress") or {}
    if pairs_progress:
        lines.append("Pairs:")
        for pair, p in pairs_progress.items():
            lines.append(f"  {pair}: {p.get('status', '?')}")
    concl = card.get("conclusion")
    if concl:
        lines.append(f"Conclusion: {concl.get('verdict', '?')} — {concl.get('summary', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Director-side hook (director.py imports this; failures never break a cycle)
# ---------------------------------------------------------------------------


def conclusion_card_from_plan(result_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A compact conclusion card for one finished Director item.

    Attached to the item's ``result_json`` so every number on the plan's
    conclusion card traces back to a stored record. ``None`` for non-plan rows.
    """
    plan_id = (result_json or {}).get("plan_id")
    if not plan_id:
        return None
    messages = [str(m) for m in (result_json.get("messages") or [])]
    return {
        "plan_id": str(plan_id),
        "pair": result_json.get("pair_name") or result_json.get("pair"),
        "verdict": result_json.get("status"),
        "summary": messages[0] if messages else "",
        "version_id": result_json.get("version_id"),
    }


async def attach_conclusion_card(
    item_result_json: Dict[str, Any], pair: str, session_factory: Optional[Any] = None
) -> None:
    """Best-effort: append the conclusion card and refresh the owning plan.

    ``session_factory`` is the caller's (the Director cycle's) factory so the
    refresh reads the same store the cycle writes — never the ambient global.
    """
    try:
        card = conclusion_card_from_plan({**item_result_json, "pair_name": pair})
        if card is not None:
            item_result_json["conclusion_card"] = card
            refreshed = await refresh_plan_from_director(
                str(card["plan_id"]), session_factory=session_factory
            )
            if refreshed is None:
                logger.warning("research plan %s not found for refresh", card["plan_id"])
    except Exception as exc:  # noqa: BLE001 — refresh must never break the cycle
        logger.warning("conclusion card attach failed: %s: %s", type(exc).__name__, exc)
