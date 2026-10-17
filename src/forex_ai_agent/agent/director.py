"""Research Director (Stage B4) — orchestrator over the existing research loop.

Roadmap §3 B4: hypothesis queue → prioritization → run (``propose_for_pair``)
→ record → next. Deterministic core; the LLM only proposes/summarizes inside
each run. Budget-capped (items per cycle, LLM calls per cycle, queue size);
the permission surface is research-tools-only; the hard blocks are enforced
in code and repeated in the audit trail: no self-approval, no risk changes,
no audit deletion, no permission self-promotion, no live trading.

Storage (SQLite, shared Base — no graph DB, consistent with B2):
- ``director_queue``    persisted hypothesis queue (deduplicated per pair)
- ``director_events``   append-only audit trail (one row per action)

ACCEPT: a full cycle runs unattended within budget; every action is audited;
nothing outside the research path is reachable from here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import String, select
from sqlalchemy.orm import Mapped, mapped_column

from forex_ai_agent.data.database import get_session_factory
from forex_ai_agent.data.models.base import (
    Base,
    JSONType,
    TimestampMixin,
    UUIDPkMixin,
)

DIRECTOR_SCHEMA_VERSION = 1

QUEUED = "QUEUED"
RUNNING = "RUNNING"
DONE_REJECTED = "DONE_REJECTED"      # gate rejected — recorded, not retried
DONE_VALIDATED = "DONE_VALIDATED"    # gate passed — versioned + probation
DONE_ERROR = "DONE_ERROR"
CANCELLED = "CANCELLED"

TERMINAL = {DONE_REJECTED, DONE_VALIDATED, DONE_ERROR, CANCELLED}

# Priorities: lower number = run first.
PRIORITY_HIGH = 1
PRIORITY_NORMAL = 2
PRIORITY_LOW = 3
_PRIORITY_NAMES = {PRIORITY_HIGH: "HIGH", PRIORITY_NORMAL: "NORMAL", PRIORITY_LOW: "LOW"}


class DirectorQueueItem(UUIDPkMixin, TimestampMixin, Base):
    """One hypothesis-to-test in the Director's persisted queue."""

    __tablename__ = "director_queue"

    pair: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False, default="H1")
    hypothesis_text: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    priority: Mapped[int] = mapped_column(nullable=False, default=PRIORITY_NORMAL)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=QUEUED, index=True)
    # Result of the run (verdict / version id / notes), JSON for flexibility.
    result_json: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    run_started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    run_finished_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    schema_version: Mapped[int] = mapped_column(nullable=False, default=DIRECTOR_SCHEMA_VERSION)


class DirectorEvent(UUIDPkMixin, TimestampMixin, Base):
    """Append-only audit row. Never updated, never deleted by the agent."""

    __tablename__ = "director_events"

    cycle_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(40), nullable=False)
    pair: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    item_id: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    detail: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)


@dataclass
class DirectorConfig:
    """Budget caps for one unattended cycle. All deterministic."""

    max_items_per_cycle: int = 3        # pairs researched per cycle
    max_llm_calls_per_cycle: int = 3    # LLM proposals per cycle (fallback after)
    max_queue_size: int = 25            # enqueue refused beyond this
    history_days: int = 180             # bars fetched per pair run
    use_llm: bool = True                # Stage B keeps the deterministic fallback


@dataclass
class CycleReport:
    """What one cycle did — every number traceable to a queue row or event."""

    cycle_id: str
    started_at: str
    finished_at: str
    items_run: int = 0
    validated: int = 0
    rejected: int = 0
    errors: int = 0
    llm_calls_used: int = 0
    budget_exhausted: bool = False
    skipped_stale_running: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


async def _alex_user_id(session: Any) -> Optional[str]:
    from forex_ai_agent.data.models.user import User

    uid = (
        await session.execute(select(User.id).where(User.username == "alex"))
    ).scalar_one_or_none()
    if uid is None:
        uid = (await session.execute(select(User.id).limit(1))).scalar_one_or_none()
    return uid


# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------


async def enqueue_hypothesis(
    pair: str,
    hypothesis_text: str = "",
    timeframe: str = "H1",
    priority: int = PRIORITY_NORMAL,
    config: Optional[DirectorConfig] = None,
    session_factory: Optional[Any] = None,
    plan_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a hypothesis to the queue (deduplicated on open rows per pair).

    ``plan_id`` (B5) links the item to a structured research plan; it lands in
    ``result_json`` immediately so the plan's refresh can find the row from
    the moment it is enqueued.
    """
    config = config or DirectorConfig()
    if priority not in _PRIORITY_NAMES:
        raise ValueError(f"priority must be one of {sorted(_PRIORITY_NAMES)}, got {priority!r}")
    factory = session_factory or get_session_factory()
    async with factory() as session:
        open_count = (
            await session.execute(
                select(DirectorQueueItem).where(
                    DirectorQueueItem.status.notin_(TERMINAL)
                )
            )
        ).scalars().all()
        if len(open_count) >= config.max_queue_size:
            return {
                "enqueued": False,
                "reason": f"queue full ({config.max_queue_size} open items)",
            }
        dup = next(
            (i for i in open_count if i.pair == pair and i.timeframe == timeframe),
            None,
        )
        if dup is not None:
            # B5: link the existing open row to the requesting plan (the plan
            # must still see this pair's progress even though no new row is
            # created — one open row per pair stays the invariant).
            if plan_id and not (dup.result_json or {}).get("plan_id"):
                dup.result_json = {**(dup.result_json or {}), "plan_id": str(plan_id)}
                await session.commit()
            return {
                "enqueued": False,
                "reason": f"pair {pair} {timeframe} already {dup.status} in queue",
                "item_id": str(dup.id),
            }
        item = DirectorQueueItem(
            pair=pair,
            timeframe=timeframe,
            hypothesis_text=hypothesis_text[:500],
            priority=priority,
            result_json={"plan_id": str(plan_id)} if plan_id else {},
        )
        session.add(item)
        await session.commit()
        return {"enqueued": True, "item_id": str(item.id), "position": len(open_count) + 1}


async def cancel_item(item_id: str, session_factory: Optional[Any] = None) -> bool:
    """Cancel a queued item (human action). Terminal rows are untouched."""
    try:  # callers pass string ids; the Uuid column needs uuid objects
        key = uuid.UUID(str(item_id))
    except (TypeError, ValueError):
        return False
    factory = session_factory or get_session_factory()
    async with factory() as session:
        item = await session.get(DirectorQueueItem, key)
        if item is None or item.status in TERMINAL:
            return False
        item.status = CANCELLED
        item.run_finished_at = datetime.utcnow()
        await session.commit()
        return True


async def queue_snapshot(
    session_factory: Optional[Any] = None,
) -> Dict[str, Any]:
    """Full queue state for the status tool / UI (stored state only)."""
    factory = session_factory or get_session_factory()
    async with factory() as session:
        rows = (
            (
                await session.execute(
                    select(DirectorQueueItem).order_by(
                        DirectorQueueItem.priority,
                        DirectorQueueItem.created_at,
                    )
                )
            )
            .scalars()
            .all()
        )
        events = (
            (
                await session.execute(
                    select(DirectorEvent)
                    .order_by(DirectorEvent.created_at.desc())
                    .limit(20)
                )
            )
            .scalars()
            .all()
        )

    def _row(i: DirectorQueueItem) -> Dict[str, Any]:
        return {
            "id": str(i.id),
            "pair": i.pair,
            "timeframe": i.timeframe,
            "hypothesis": i.hypothesis_text,
            "priority": _PRIORITY_NAMES.get(i.priority, str(i.priority)),
            "status": i.status,
            "result": dict(i.result_json or {}),
            "created_at": i.created_at.isoformat() if i.created_at else None,
            "run_started_at": i.run_started_at.isoformat() if i.run_started_at else None,
            "run_finished_at": (
                i.run_finished_at.isoformat() if i.run_finished_at else None
            ),
        }

    open_items = [i for i in rows if i.status not in TERMINAL]
    return {
        "queue": [_row(i) for i in rows],
        "open_items": len(open_items),
        "by_status": {
            s: sum(1 for i in rows if i.status == s)
            for s in (QUEUED, RUNNING, DONE_VALIDATED, DONE_REJECTED, DONE_ERROR, CANCELLED)
        },
        "recent_events": [
            {
                "cycle_id": e.cycle_id,
                "action": e.action,
                "pair": e.pair,
                "detail": dict(e.detail or {}),
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ],
    }


# ---------------------------------------------------------------------------
# The cycle — the deterministic core
# ---------------------------------------------------------------------------


def _next_cycle_id(session_factory: Any) -> str:
    """B4-YYYYMMDD-HHMMSS — unique enough for the audit trail."""
    stamp = datetime.utcnow().strftime("B4-%Y%m%d-%H%M%S")
    return stamp


async def _record_event(
    session_factory: Any,
    cycle_id: str,
    action: str,
    pair: Optional[str] = None,
    item_id: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> None:
    async with session_factory() as session:
        session.add(
            DirectorEvent(
                cycle_id=cycle_id,
                action=action,
                pair=pair,
                item_id=item_id,
                detail=dict(detail or {}),
            )
        )
        await session.commit()


async def run_cycle(
    config: Optional[DirectorConfig] = None,
    session_factory: Optional[Any] = None,
    *,
    history_fetcher: Optional[Any] = None,
    proposal_runner: Optional[Any] = None,
    pipeline: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one supervised research cycle over the queue.

    Per item (in priority, then FIFO order):
      1. mark RUNNING, audit ``item_started``
      2. fetch history, run ``propose_for_pair`` (LLM while budget lasts,
         then the deterministic fallback — the loop NEVER stalls on credits)
      3. record the outcome on the row, audit ``item_finished``
    Every action lands in ``director_events``; nothing outside the research
    path is invoked. Budget exhaustion ends the cycle honestly.

    ``history_fetcher`` / ``proposal_runner`` / ``pipeline`` default to the
    real production implementations (lazy-imported); tests inject fakes.
    """
    config = config or DirectorConfig()
    factory = session_factory or get_session_factory()
    cycle_id = _next_cycle_id(factory)
    report = CycleReport(cycle_id=cycle_id, started_at=_now_iso(), finished_at="")

    await _record_event(factory, cycle_id, "cycle_started", detail=config.__dict__)

    if history_fetcher is None or proposal_runner is None:
        from forex_ai_agent.autonomy.proposal import (
            fetch_history_for as _real_fetch,
            propose_for_pair as _real_propose,
        )

        history_fetcher = history_fetcher or _real_fetch
        proposal_runner = proposal_runner or _real_propose
    if pipeline is None:
        from forex_ai_agent.core.event_bus import InMemoryEventBus
        from forex_ai_agent.pipeline import Pipeline

        pipeline = Pipeline(event_bus=InMemoryEventBus())
        await pipeline.initialize()

    async with factory() as session:
        # Recovery FIRST: a previous cycle may have died mid-run (process
        # restart). Requeue those rows so this cycle can pick them up —
        # crashed work must not wait for yet another cycle.
        stale_rows = (
            await session.execute(
                select(DirectorQueueItem).where(DirectorQueueItem.status == RUNNING)
            )
        ).scalars().all()
        for stale in stale_rows:
            stale.status = QUEUED
            stale.result_json = {
                **(stale.result_json or {}),
                "note": "requeued after interrupted cycle",
            }
            report.skipped_stale_running += 1
        await session.commit()

        items = (
            (
                await session.execute(
                    select(DirectorQueueItem)
                    .where(DirectorQueueItem.status == QUEUED)
                    .order_by(DirectorQueueItem.priority, DirectorQueueItem.created_at)
                )
            )
            .scalars()
            .all()
        )
        batch = list(items[: config.max_items_per_cycle])

    if report.skipped_stale_running:
        await _record_event(
            factory, cycle_id, "stale_running_requeued",
            detail={"count": report.skipped_stale_running},
        )

    for item in batch:
        use_llm = config.use_llm and report.llm_calls_used < config.max_llm_calls_per_cycle
        item_id = str(item.id)
        async with factory() as session:
            live = await session.get(DirectorQueueItem, item.id)
            live.status = RUNNING
            live.run_started_at = datetime.utcnow()
            await session.commit()
        await _record_event(
            factory, cycle_id, "item_started", pair=item.pair, item_id=item_id,
            detail={"use_llm": use_llm, "timeframe": item.timeframe},
        )

        # plan_id (B5) rides in result_json from enqueue; it must survive the
        # result_json replacement below, so it is carried into detail.
        _plan_id = (getattr(item, "result_json", None) or {}).get("plan_id")
        detail: Dict[str, Any] = {"use_llm": use_llm}
        if _plan_id:
            detail["plan_id"] = str(_plan_id)
        try:
            history = await history_fetcher(
                pipeline, item.pair, days=config.history_days, timeframe=item.timeframe
            )
            if use_llm:
                report.llm_calls_used += 1
            result = await proposal_runner(
                pipeline,
                item.pair,
                history,
                use_llm=use_llm,
                timeframe=item.timeframe,
            )
            detail.update(
                {
                    "status": result.status,
                    "used_llm": bool(result.used_llm),
                    "version_id": str(result.version.id) if result.version else None,
                    "messages": list(result.messages)[:5],
                }
            )
            if result.status == "validated":
                report.validated += 1
                new_status = DONE_VALIDATED
            elif result.status == "rejected":
                report.rejected += 1
                new_status = DONE_REJECTED
            else:
                report.errors += 1
                new_status = DONE_ERROR
        except Exception as exc:  # noqa: BLE001 — one bad pair must not kill the cycle
            report.errors += 1
            new_status = DONE_ERROR
            detail["error"] = f"{type(exc).__name__}: {exc}"

        async with factory() as session:
            live = await session.get(DirectorQueueItem, item.id)
            live.status = new_status
            live.run_finished_at = datetime.utcnow()
            live.result_json = detail
            await session.commit()
        # B5: structured-plan bookkeeping — AFTER the stored row is final, so
        # the plan's refresh reads committed state. plan_id rides in result_json
        # from enqueue; card attach + plan refresh are best-effort and can
        # never break the cycle. The card is persisted back onto the row.
        if detail.get("plan_id"):
            try:
                from forex_ai_agent.agent.research_plan import attach_conclusion_card

                await attach_conclusion_card(detail, item.pair, session_factory=factory)
                async with factory() as session:
                    live = await session.get(DirectorQueueItem, item.id)
                    live.result_json = detail
                    await session.commit()
            except Exception as exc:  # noqa: BLE001 — plan bookkeeping is optional
                logger.warning(
                    "director plan bookkeeping failed for %s: %s: %s",
                    item_id, type(exc).__name__, exc,
                )
        await _record_event(
            factory, cycle_id, "item_finished", pair=item.pair, item_id=item_id,
            detail=detail,
        )
        report.items_run += 1
        report.results.append({"item_id": item_id, "pair": item.pair, **detail})

    # Honest final open-queue count (single query, outside the item loop):
    # if anything is still queued, the cycle stopped on budget, not on work.
    async with factory() as session:
        still_queued = (
            await session.execute(
                select(DirectorQueueItem.id).where(DirectorQueueItem.status == QUEUED)
            )
        ).all()
    report.budget_exhausted = bool(still_queued) and report.items_run > 0

    report.finished_at = _now_iso()
    await _record_event(
        factory, cycle_id, "cycle_finished",
        detail={
            "items_run": report.items_run,
            "validated": report.validated,
            "rejected": report.rejected,
            "errors": report.errors,
            "llm_calls_used": report.llm_calls_used,
            "still_queued": len(still_queued),
        },
    )
    return report.to_dict()
