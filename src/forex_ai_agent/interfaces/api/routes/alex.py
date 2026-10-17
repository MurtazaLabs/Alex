"""Alex routes: the endpoints the alex-ui SPA consumes.

Groups:
- ``/alex/overview``        dashboard summary (feed, account, probation, signals)
- ``/alex/signals``         stored signals + on-demand generation (narration off)
- ``/alex/strategies/...``  probation queue + human-only approve/reject
- ``/alex/research/...``    autonomy-loop runs + history
- ``/alex/account``         MT5 broker account snapshot (read-only)
- ``/alex/status``          pipeline status (feeds, LLM, settings)
- ``/alex/chat``            Phase-1 chat endpoint (deterministic overview reply;
                            AgentCore lands here in Phase 1b of the agent plan)

Agent endpoints (Phase 1b+):
- ``/agent/chat``           AgentCore JSON tool loop (POST)
- ``/agent/confirm``        Confirmation token consumption (POST)
- ``/agent/experiments``    Experiment CRUD (GET, POST)
- ``/agent/experiments/{id}`` Get single experiment (GET)
- ``/agent/experiments/{id}/conclude`` Conclude experiment (POST)

Read-only surface: nothing here places orders or changes permission levels.
Approvals stay human-only and are additionally confirmed in the UI.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from forex_ai_agent.config import get_settings
from forex_ai_agent.data.database import get_session
from forex_ai_agent.data.models import User
from forex_ai_agent.data.repositories import WatchlistRepository
from forex_ai_agent.auth import get_current_user

router = APIRouter(prefix="/alex", tags=["alex"])

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared pipeline accessor (one Pipeline per process, on the app's loop)
# ---------------------------------------------------------------------------

_pipeline: Any = None


def get_pipeline() -> Any:
    """Build the shared Pipeline lazily (FastAPI runs one event loop)."""
    global _pipeline
    if _pipeline is None:
        from forex_ai_agent.core.event_bus import InMemoryEventBus
        from forex_ai_agent.pipeline import Pipeline

        _pipeline = Pipeline(event_bus=InMemoryEventBus())
    return _pipeline


async def _ensure_pipeline_init() -> Any:
    pipeline = get_pipeline()
    if not getattr(pipeline, "_agent_initialized", False):
        try:
            await pipeline.initialize()
        except Exception:  # noqa: BLE001 — degrade to no-persistence mode
            try:
                await pipeline.ensure_db()
            except Exception:  # noqa: BLE001
                pass
        pipeline._agent_initialized = True
    return pipeline


async def _watched_pairs() -> List[str]:
    """DB watchlist (alex-local user), falling back to a sane default."""
    try:
        from forex_ai_agent.data.database import get_session_factory
        from forex_ai_agent.data.repositories import UserRepository

        async with get_session_factory()() as session:
            users = UserRepository(session)
            user = await users.get_by_username("alex-local")
            if user is None:
                return ["EUR/USD"]
            lists = WatchlistRepository(session)
            wl = await lists.get_by_name(user.id, "default")
            return list(wl.pairs or []) if wl else ["EUR/USD"]
    except Exception:  # noqa: BLE001
        return ["EUR/USD"]


# ---------------------------------------------------------------------------
# Watchlist management (shared with the CLI's `watch` store)
# ---------------------------------------------------------------------------

# Mirrors the Markets page defaults — used when the watchlist is empty.
DEFAULT_WATCHLIST_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "GBP/JPY", "EUR/JPY", "AUD/USD",
]


class WatchlistPairsPayload(BaseModel):
    """Body for watchlist mutations — a single pair, e.g. ``EUR/USD``."""

    pair: str = Field(..., min_length=6, max_length=7)


class WatchlistMutateResponse(BaseModel):
    """Resulting pair list plus the source of truth for the UI."""

    pairs: List[str]
    source: str  # "watchlist" | "defaults"


def _normalize_symbol(raw: str) -> str:
    """Accept ``eurusd``/``eur-usd``/``EUR/USD`` and canonicalize to ``EUR/USD``."""
    compact = re.sub(r"[^A-Za-z]", "", raw).upper()
    if len(compact) != 6 or not compact.isalpha():
        raise HTTPException(status_code=422, detail="Pair must look like EUR/USD")
    return f"{compact[:3]}/{compact[3:]}"


async def _mutate_default_watchlist(symbol: str, add: bool) -> None:
    """Add/remove ``symbol`` on the alex-local/default watchlist in one session.

    The same store the CLI's ``watch``/``unwatch`` commands use. Creates the
    alex-local user and the default watchlist if missing (mirrors cli.py).
    """
    from forex_ai_agent.data.database import get_session_factory
    from forex_ai_agent.data.models import User
    from forex_ai_agent.data.repositories import UserRepository

    factory = get_session_factory()
    async with factory() as session:
        users = UserRepository(session)
        user = await users.get_by_username("alex-local")
        if user is None:
            user = await users.add(
                User(
                    username="alex-local",
                    email="alex-local@localhost",
                    password_hash="!",
                    permission_level=0,
                )
            )
        lists = WatchlistRepository(session)
        wl = await lists.get_by_name(user.id, "default")
        if wl is None:
            wl = await lists.create(user.id, "default", [symbol] if add else [])
        elif add:
            await lists.add_pairs(wl, [symbol])
        else:
            await lists.remove_pairs(wl, [symbol])


@router.post("/watchlist/pairs", response_model=WatchlistMutateResponse)
async def add_watched_pair(
    payload: WatchlistPairsPayload,
    current_user: User = Depends(get_current_user),
) -> WatchlistMutateResponse:
    """Add a pair to the shared default watchlist (idempotent)."""
    symbol = _normalize_symbol(payload.pair)
    try:
        await _mutate_default_watchlist(symbol, add=True)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to add pair: {exc}")
    pairs = await _watched_pairs()
    return WatchlistMutateResponse(
        pairs=pairs, source="watchlist" if pairs else "defaults"
    )


@router.delete("/watchlist/pairs", response_model=WatchlistMutateResponse)
async def remove_watched_pair(
    symbol: str = Query(..., min_length=6, max_length=7),
    current_user: User = Depends(get_current_user),
) -> WatchlistMutateResponse:
    """Remove a pair from the shared default watchlist (ignored if absent).

    The symbol is a query param — ``EUR/USD`` contains a slash and must not
    travel in a path segment.
    """
    canonical = _normalize_symbol(symbol)
    try:
        await _mutate_default_watchlist(canonical, add=False)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to remove pair: {exc}")
    pairs = await _watched_pairs()
    return WatchlistMutateResponse(
        pairs=pairs, source="watchlist" if pairs else "defaults"
    )


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


@router.get("/overview")
async def overview(current_user: User = Depends(get_current_user)) -> dict:
    """Dashboard summary: feed, account, probation queue, recent signals."""
    pipeline = await _ensure_pipeline_init()
    status = await pipeline.get_pipeline_status()
    pairs = await _watched_pairs()

    account: Dict[str, Any] = await _account_snapshot_safe()
    probation = await _probation_rows_safe()

    from forex_ai_agent.autonomy.research_log import recent_research_tasks

    try:
        research = await recent_research_tasks(limit=10)
    except Exception:  # noqa: BLE001
        research = []

    return {
        "data_source": status.get("data_source", "unknown"),
        "feed_health": status.get("feed_health", []),
        "account": account,
        "open_positions": account.get("positions") if account.get("available") else 0,
        "probation_active": sum(
            1 for r in probation if r.get("status") in {"proposing", "active"}
        ),
        "probation_passed": sum(1 for r in probation if r.get("eligible_for_approval")),
        "research_runs": len(research),
        "recent_signals": _recent_signals_from_tasks(research[:5]),
        "watchlist": pairs,
        "llm": status.get("llm", {}),
    }


def _recent_signals_from_tasks(research: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Derive dashboard cards from the latest research runs (honest signal proxy)."""
    out: List[Dict[str, Any]] = []
    for t in research:
        out.append(
            {
                "pair": t.get("topic", "?"),
                "direction": "LONG",
                "setup": t.get("conclusion", "")[:120],
                "timeframe": "",
                "created": t.get("created", ""),
                "score_pct": None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


@router.get("/signals")
async def list_signals(
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Signals recorded in the research log (the durable signal trail)."""
    from forex_ai_agent.autonomy.research_log import recent_research_tasks

    try:
        rows = await recent_research_tasks(limit=limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"signal history unavailable: {exc}")
    signals = [
        {
            "pair": r.get("topic", "?"),
            "direction": "LONG",
            "setup": r.get("conclusion", ""),
            "timeframe": "",
            "created": r.get("created", ""),
            "score_pct": None,
            "verdict": r.get("verdict", ""),
        }
        for r in rows
    ]
    return {"signals": signals}


class SignalGenerateRequest(BaseModel):
    pair: str = Field(..., examples=["EUR/USD"])
    direction: str = Field("LONG", examples=["LONG", "SHORT"])
    timeframe: str = Field("H1", examples=["M15", "H1", "H4", "D1"])
    narrate: bool = Field(False, description="Slow LLM narration off by default for chat-speed responses")


@router.post("/signals/generate")
async def generate_signal(
    payload: SignalGenerateRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Run the full deterministic pipeline for one pair/direction/timeframe."""
    pipeline = await _ensure_pipeline_init()
    direction = payload.direction.upper()
    if direction not in {"LONG", "SHORT"}:
        raise HTTPException(status_code=422, detail="direction must be LONG or SHORT")
    try:
        result = await asyncio.wait_for(
            pipeline.generate_signal(
                payload.pair.upper(), direction, payload.timeframe.upper(), narrate=payload.narrate
            ),
            timeout=120.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="signal generation timed out")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"pipeline error: {exc}")
    if result.get("status") != "ok":
        raise HTTPException(status_code=502, detail=result.get("message", "generation failed"))
    return result


# ---------------------------------------------------------------------------
# Strategies: probation queue + human-only approval
# ---------------------------------------------------------------------------


async def _probation_rows_safe() -> List[Dict[str, Any]]:
    try:
        from forex_ai_agent.autonomy.probation import ProbationStore

        rows = await ProbationStore().list()
        return [
            {
                "version_id": r.version_id,
                "strategy_name": r.strategy_name,
                "pair": getattr(r, "pair", ""),
                "status": r.status,
                "observed_trades": r.observed_trades,
                "required_trades": r.required_trades,
                "eligible_for_approval": r.eligible_for_approval,
                "notes": r.notes or "",
            }
            for r in rows
        ]
    except Exception:  # noqa: BLE001
        return []


@router.get("/strategies/probation")
async def probation_queue(current_user: User = Depends(get_current_user)) -> list:
    """Probation queue for the Strategies approval panel."""
    return await _probation_rows_safe()


@router.post("/strategies/probation/{version_id}/approve")
async def approve_strategy(
    version_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Human-only approval of a probation candidate (UI double-confirms)."""
    from forex_ai_agent.autonomy.probation import ProbationStore

    store = ProbationStore()
    rows = await store.list()
    matches = [r for r in rows if r.version_id == version_id or r.version_id.startswith(version_id)]
    if not matches:
        raise HTTPException(status_code=404, detail=f"no probation matching {version_id!r}")
    if len(matches) > 1:
        raise HTTPException(status_code=409, detail="ambiguous version prefix")
    if not matches[0].eligible_for_approval:
        raise HTTPException(
            status_code=409,
            detail=f"not eligible: {matches[0].observed_trades}/{matches[0].required_trades} trades, status={matches[0].status}",
        )
    try:
        status = await store.approve(matches[0].version_id, approver=current_user.username)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "approved": True,
        "version_id": status.version_id,
        "strategy_name": status.strategy_name,
        "status": status.status,
    }


@router.post("/strategies/probation/{version_id}/reject")
async def reject_strategy(
    version_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    from forex_ai_agent.autonomy.probation import ProbationStore

    store = ProbationStore()
    rows = await store.list()
    matches = [r for r in rows if r.version_id == version_id or r.version_id.startswith(version_id)]
    if not matches:
        raise HTTPException(status_code=404, detail=f"no probation matching {version_id!r}")
    if len(matches) > 1:
        raise HTTPException(status_code=409, detail="ambiguous version prefix")
    try:
        status = await store.reject(matches[0].version_id, approver=current_user.username)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"rejected": True, "version_id": status.version_id, "status": status.status}


# ---------------------------------------------------------------------------
# Strategy Health (Stage B1): deterministic, stored-state only
# ---------------------------------------------------------------------------


@router.get("/strategies/health")
async def strategies_health_all(
    current_user: User = Depends(get_current_user),
) -> list:
    """Health summary for every stored strategy version (one card per row)."""
    from forex_ai_agent.agent.health import get_strategy_health

    out: List[dict] = []
    try:
        from forex_ai_agent.strategy import create_version_manager

        vm = await create_version_manager()
        ids = list(vm.versions.keys())
    except Exception:  # noqa: BLE001 — no version store → no cards
        return []
    for sid in ids:
        try:
            out.append(await get_strategy_health(sid))
        except Exception:  # noqa: BLE001 — skip unreadable versions, keep serving
            continue

    # Actionable badges first (RETIRE? demands attention, then candidates,
    # then probation, then research-only noise). Stable sort keeps the store's
    # creation order within each rank.
    badge_rank = {"RETIRE?": 0, "CANDIDATE": 1, "PROBATION": 2, "RESEARCH ONLY": 3}
    out.sort(key=lambda r: badge_rank.get((r.get("health") or {}).get("badge"), 4))
    return out


@router.get("/strategies/health/{strategy_id}")
async def strategies_health_one(
    strategy_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Health card for one strategy version, from stored state only."""
    from forex_ai_agent.agent.health import get_strategy_health

    try:
        return await get_strategy_health(strategy_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/strategies/robustness/{strategy_id}")
async def strategies_robustness_one(
    strategy_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Stored Stage-B3 robustness sweep for one strategy (no re-runs)."""
    from forex_ai_agent.strategy import create_version_manager

    vm = await create_version_manager()
    version = vm.get_version(strategy_id) or (
        vm.get_version(strategy_id[:8]) if len(strategy_id) >= 8 else None
    )
    if version is None:
        raise HTTPException(status_code=404, detail=f"unknown strategy version {strategy_id!r}")
    sweep = (version.backtest_results or {}).get("robustness")
    if not sweep:
        raise HTTPException(
            status_code=404,
            detail="no robustness sweep stored — POST /strategies/robustness/{id} to run one",
        )
    return sweep


@router.post("/strategies/robustness/{strategy_id}")
async def strategies_robustness_run(
    strategy_id: str,
    current_user: User = Depends(get_current_user),
    body: Optional[dict] = None,
) -> dict:
    """Run the cross-pair / regime robustness sweep and store it on the version.

    Heavy (up to 4 pairs x 8 windows of backtests) — requires ASSIST-level
    access, matching the tool's permission floor.
    """
    if current_user.permission_level < 1:
        raise HTTPException(
            status_code=403,
            detail="running a robustness sweep requires ASSIST permission (level 1+)",
        )
    from forex_ai_agent.agent.robustness import run_robustness_sweep

    body = body or {}
    try:
        return await run_robustness_sweep(
            strategy_id=strategy_id,
            pairs=body.get("pairs"),
            start_date=body.get("start_date"),
            end_date=body.get("end_date"),
            timeframe=body.get("timeframe"),
            window_months=int(body.get("window_months", 6)),
            step_months=int(body.get("step_months", 2)),
            save=bool(body.get("save", True)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/director/queue")
async def director_queue_status(
    current_user: User = Depends(get_current_user),
) -> dict:
    """Research Director queue snapshot + recent audit events (B4)."""
    from forex_ai_agent.agent.director import queue_snapshot

    return await queue_snapshot()


@router.post("/director/queue")
async def director_queue_enqueue(
    body: dict,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Enqueue a pair/hypothesis for the Director (ASSIST+)."""
    if current_user.permission_level < 1:
        raise HTTPException(
            status_code=403,
            detail="enqueueing research requires ASSIST permission (level 1+)",
        )
    from forex_ai_agent.agent.director import enqueue_hypothesis

    pair = str(body.get("pair") or "").strip()
    if not pair:
        raise HTTPException(status_code=422, detail="pair is required")
    try:
        return await enqueue_hypothesis(
            pair=pair,
            hypothesis_text=str(body.get("hypothesis") or ""),
            timeframe=str(body.get("timeframe") or "H1"),
            priority=int(body.get("priority", 2)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/director/cycle")
async def director_run_cycle(
    current_user: User = Depends(get_current_user),
    body: Optional[dict] = None,
) -> dict:
    """Run one budget-capped unattended research cycle (ASSIST+).

    Deterministic core: propose_for_pair per queued item; LLM budget-capped;
    research tools only; every action audited; no live execution.
    """
    if current_user.permission_level < 1:
        raise HTTPException(
            status_code=403,
            detail="running research cycles requires ASSIST permission (level 1+)",
        )
    from forex_ai_agent.agent.director import DirectorConfig, run_cycle

    body = body or {}
    config = DirectorConfig(
        max_items_per_cycle=max(1, int(body.get("max_items", 3))),
        max_llm_calls_per_cycle=max(0, int(body.get("max_llm_calls", 3))),
        use_llm=bool(body.get("use_llm", True)),
    )
    report = await run_cycle(config)

    # B5: any plan-linked item that finished this cycle refreshes its plan
    # (idempotent, deterministic; per-plan failures never fail the request).
    try:
        from forex_ai_agent.agent.research_plan import refresh_plan_from_director

        for r in report.get("results", []):
            plan_id = (r or {}).get("plan_id")
            if plan_id:
                await refresh_plan_from_director(str(plan_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning("director_cycle_plan_refresh_failed: %s", exc)
    return report


# ---------------------------------------------------------------------------
# Research Plans (Stage B5): structured plan cards, persisted
# ---------------------------------------------------------------------------


@router.post("/director/generate-hypotheses")
async def director_generate_hypotheses(
    current_user: User = Depends(get_current_user),
    body: Optional[dict] = None,
) -> dict:
    """LLM proposes research directions; deterministic code validates + enqueues (ASSIST+).

    Stage C1: the LLM only chooses WHAT to research from deterministic
    structure briefs; every candidate is validated in code and the backtest
    gate still owns every outcome. No orders, no risk changes.
    """
    if current_user.permission_level < 1:
        raise HTTPException(
            status_code=403,
            detail="hypothesis generation requires ASSIST permission (level 1+)",
        )
    from forex_ai_agent.agent.hypothesis_gen import HypothesisGenConfig, generate_hypotheses

    body = body or {}
    pairs = body.get("pairs")
    if isinstance(pairs, str):
        pairs = [pairs]
    try:
        config = HypothesisGenConfig(
            pairs=pairs,
            timeframe=str(body.get("timeframe") or "H1"),
            max_hypotheses=max(1, min(int(body.get("max_hypotheses", 3)), 5)),
            min_confidence=float(body.get("min_confidence", 0.6)),
            use_llm=bool(body.get("use_llm", True)),
            enqueue=bool(body.get("enqueue", True)),
        )
        return await generate_hypotheses(config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/agent/experiments/select")
async def experiments_select(
    current_user: User = Depends(get_current_user),
    body: Optional[dict] = None,
) -> dict:
    """LLM chooses which experiment to run next; deterministic code validates
    and executes it (ASSIST+).

    Stage C2: the LLM only picks candidates from a deterministic menu
    (strategies, declared parameter ranges, prior tests); parameter-range
    clamping, budget caps and the backtest/walk-forward engines are all owned
    by code. No orders, no risk changes, no version creation.
    """
    if current_user.permission_level < 1:
        raise HTTPException(
            status_code=403,
            detail="experiment selection requires ASSIST permission (level 1+)",
        )
    from forex_ai_agent.agent.experiment_selector import (
        ExperimentSelConfig,
        select_experiments,
    )

    body = body or {}
    strategies = body.get("strategies")
    if isinstance(strategies, str):
        strategies = [strategies]
    try:
        config = ExperimentSelConfig(
            strategies=strategies,
            symbol=str(body.get("symbol") or "EUR/USD"),
            timeframe=str(body.get("timeframe") or "H1"),
            start_date=str(body.get("start_date") or "2025-01-01"),
            end_date=str(body.get("end_date") or "2025-12-31"),
            max_experiments=max(1, min(int(body.get("max_experiments", 1)), 3)),
            max_variants=max(1, min(int(body.get("max_variants", 4)), 6)),
            window_months=max(1, min(int(body.get("window_months", 6)), 24)),
            step_months=max(1, min(int(body.get("step_months", 2)), 12)),
            max_oos_windows=max(1, min(int(body.get("max_oos_windows", 3)), 4)),
            use_llm=bool(body.get("use_llm", True)),
            dry_run=bool(body.get("dry_run", False)),
        )
        return await select_experiments(config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ---------------------------------------------------------------------------
# Research reports (Stage C3): LLM narration, code-owned numbers, persisted
# ---------------------------------------------------------------------------


@router.post("/research/report")
async def research_report_generate(
    current_user: User = Depends(get_current_user),
    body: Optional[dict] = None,
) -> dict:
    """Generate a research report over stored evidence (ASSIST+).

    Stage C3: the LLM narrates; every number is rendered from stored records
    by code, every LLM section is traceability-checked, and the report is
    persisted as a first-class record. No orders, no risk changes.
    """
    if current_user.permission_level < 1:
        raise HTTPException(
            status_code=403,
            detail="report generation requires ASSIST permission (level 1+)",
        )
    from forex_ai_agent.agent.report_gen import ReportConfig, generate_research_report

    body = body or {}
    experiment_ids = body.get("experiment_ids")
    if isinstance(experiment_ids, str):
        experiment_ids = [experiment_ids]
    strategy_ids = body.get("strategy_ids")
    if isinstance(strategy_ids, str):
        strategy_ids = [strategy_ids]
    try:
        config = ReportConfig(
            topic=str(body.get("topic") or "Research summary"),
            experiment_ids=experiment_ids,
            strategy_ids=strategy_ids,
            limit_experiments=max(1, min(int(body.get("limit_experiments", 10)), 20)),
            use_llm=bool(body.get("use_llm", True)),
            persist=bool(body.get("persist", True)),
        )
        return await generate_research_report(config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/research/report/{report_id}")
async def research_report_get(
    report_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """One persisted report by id/prefix — from the stored row, no re-runs."""
    from forex_ai_agent.agent.report_gen import get_research_report

    doc = await get_research_report(report_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"report {report_id} not found")
    return doc


@router.get("/research/reports")
async def research_reports_index(
    current_user: User = Depends(get_current_user),
    limit: int = 20,
) -> dict:
    """Recent report summaries (most recent first)."""
    from forex_ai_agent.agent.report_gen import list_research_reports

    reports = await list_research_reports(limit=limit)
    return {"reports": reports, "disclaimer": "LLM narration; numbers rendered from stored records by code."}


# ---------------------------------------------------------------------------
# Probation promotion review (Stage C4): deterministic package for the human
# approval decision. Promotion itself stays on the human-only buttons below.
# ---------------------------------------------------------------------------


@router.get("/research/promotions")
async def research_promotions_review(
    current_user: User = Depends(get_current_user),
    limit: int = Query(5, ge=1, le=10),
    include_narration: bool = True,
) -> dict:
    """Rank probation candidates whose objective gates are MET (Stage C4).

    Deterministic review over stored rows; narration (if any) is LLM prose.
    This endpoint NEVER promotes — approve/reject remain the human-only
    probation transitions above.
    """
    from forex_ai_agent.agent.probation_promotion import PromotionConfig, review_promotions

    review = await review_promotions(
        PromotionConfig(limit=limit, include_narration=include_narration)
    )
    return review


@router.get("/probation/monitor")
async def probation_monitor(
    current_user: User = Depends(get_current_user),
    version_id: Optional[str] = Query(None, description="Probation version id (prefix ok)"),
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    """Continuous probation health snapshot (Stage D3).

    Deterministic flags over accrual progress, gate-floor standing, accrual
    infrastructure freshness and feed integrity — arithmetic over stored
    records, read-only: this endpoint NEVER mutates probation state.
    Approve/reject remain the human-only transitions above.
    """
    from forex_ai_agent.autonomy.probation_monitor import ProbationMonitor

    monitor = ProbationMonitor()
    reports = await monitor.monitor_once(publish=False)
    if version_id:
        reports = [
            r for r in reports if r["version_id"] == version_id or r["version_id"].startswith(version_id)
        ]
    shown = reports[:limit]
    return {
        "summary": ProbationMonitor.summarize(reports),
        "probations": shown,
        "rendered": ProbationMonitor.render_reports(shown),
    }


@router.get("/research/plans")
async def research_plans_index(
    limit: int = Query(20, ge=1, le=50),
    current_user: User = Depends(get_current_user),
) -> list:
    """Recent structured research plans (B5) — newest first."""
    from forex_ai_agent.agent.research_plan import list_research_plans

    return await list_research_plans(limit=limit)


@router.get("/research/plan/{task_id}")
async def research_plan_card(
    task_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """One RESEARCH #id card from persisted records (prefix match ok)."""
    from forex_ai_agent.agent.research_plan import get_research_plan
    from forex_ai_agent.data.database import get_session_factory
    from sqlalchemy import select as _select

    from forex_ai_agent.agent.research_plan import ResearchPlan

    async def _resolve() -> Optional[str]:
        try:
            import uuid as _uuid

            return str(_uuid.UUID(task_id))
        except (TypeError, ValueError):
            pass
        if len(task_id) < 4:
            raise HTTPException(status_code=422, detail="plan id prefix too short")
        factory = get_session_factory()
        async with factory() as session:
            rows = (
                (
                    await session.execute(
                        _select(ResearchPlan.id).order_by(ResearchPlan.created_at.desc()).limit(200)
                    )
                )
                .scalars()
                .all()
            )
        matches = [r for r in rows if str(r).lower().startswith(task_id.strip().lower())]
        if not matches:
            raise HTTPException(status_code=404, detail=f"no research plan matching {task_id!r}")
        if len(matches) > 1:
            raise HTTPException(status_code=409, detail="ambiguous plan prefix")
        return str(matches[0])

    resolved = await _resolve()
    card = await get_research_plan(resolved)
    if card is None:
        raise HTTPException(status_code=404, detail=f"no research plan matching {task_id!r}")
    return card


@router.post("/research/plan")
async def research_plan_create(
    body: dict,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Create a structured research plan and queue its pairs (ASSIST+).

    Deterministic core: the plan row + Director queue entries are persisted;
    progress self-reports from stored records as cycles complete.
    """
    if current_user.permission_level < 1:
        raise HTTPException(
            status_code=403,
            detail="creating research plans requires ASSIST permission (level 1+)",
        )
    from forex_ai_agent.agent.research_plan import create_research_plan

    body = body or {}
    topic = str(body.get("topic") or "").strip()
    if not topic:
        raise HTTPException(status_code=422, detail="topic is required")
    pairs = body.get("pairs") or []
    if isinstance(pairs, str):
        pairs = [pairs]
    try:
        return await create_research_plan(
            topic=topic,
            pairs=[str(p) for p in pairs],
            timeframes=[str(t) for t in (body.get("timeframes") or ["H1"])],
            hypothesis_text=str(body.get("hypothesis") or ""),
            period_days=int(body.get("period_days", 365)),
            enqueued=bool(body.get("enqueued", True)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ---------------------------------------------------------------------------
# Research: autonomy loop + history
# ---------------------------------------------------------------------------


class ResearchRunRequest(BaseModel):
    pair: str = Field(..., examples=["EUR/USD"])
    days: int = Field(365, ge=60, le=730)


@router.post("/research/run")
async def run_research(
    payload: ResearchRunRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """One autonomy-loop iteration: brief → LLM proposal → backtest gate."""
    from forex_ai_agent.autonomy.proposal import fetch_history_for, propose_for_pair

    pipeline = await _ensure_pipeline_init()
    pair = payload.pair.upper()
    if "/" not in pair or len(pair) != 8:
        raise HTTPException(status_code=422, detail="pair must look like EUR/USD")

    try:
        history = await fetch_history_for(pipeline, pair, days=payload.days)
        result = await asyncio.wait_for(
            propose_for_pair(pipeline, pair, history, use_llm=True), timeout=300.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="research run timed out")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"research failed: {exc}")

    report = result.validation
    return {
        "pair": pair,
        "messages": list(result.messages),
        "validation": (
            {
                "passed": report.passed,
                "full_backtest": report.full_backtest or {},
                "walk_forward": report.walk_forward or {},
                "reasons": list(report.reasons),
                "data_range": report.data_range or {},
            }
            if report is not None
            else None
        ),
        "version": (
            {"id": result.version.id, "name": result.version.name} if result.version else None
        ),
    }


@router.get("/research/history")
async def research_history(
    limit: int = Query(25, ge=1, le=100),
    current_user: User = Depends(get_current_user),
) -> list:
    from forex_ai_agent.autonomy.research_log import recent_research_tasks

    try:
        return await recent_research_tasks(limit=limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"history unavailable: {exc}")


# ---------------------------------------------------------------------------
# Account (MT5 broker snapshot, read-only)
# ---------------------------------------------------------------------------


async def _account_snapshot_safe() -> Dict[str, Any]:
    """MT5 account info; never raises — the dashboard must always render."""
    try:
        from forex_ai_agent.data.providers import MT5BrokerProvider

        if MT5BrokerProvider is None:
            return {"available": False, "note": "MT5 broker module unavailable on this machine"}
        broker = MT5BrokerProvider()
        connected = await broker.connect()
        if not connected:
            return {"available": False, "note": "MT5 terminal not reachable (is it running?)"}
        info = await broker.get_account_info()
        positions = await broker.get_positions()
        g = lambda name: getattr(info, name, None)
        return {
            "available": True,
            "source": "mt5",
            "login": g("login"),
            "server": g("server"),
            "balance": g("balance"),
            "equity": g("equity"),
            "margin": g("margin"),
            "free_margin": g("free_margin"),
            "margin_level": g("margin_level"),
            "leverage": g("leverage"),
            "currency": g("currency"),
            "positions": len(positions),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "note": f"MT5 account unavailable: {exc}"}


@router.get("/account")
async def account(current_user: User = Depends(get_current_user)) -> dict:
    """Broker account snapshot (read-only; no order endpoints by design)."""
    snap = await _account_snapshot_safe()
    snap.setdefault("note", "")
    return snap


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@router.get("/status")
async def status(current_user: User = Depends(get_current_user)) -> dict:
    pipeline = await _ensure_pipeline_init()
    return await pipeline.get_pipeline_status()


# ---------------------------------------------------------------------------
# Markets page — deterministic core data for the UI (no LLM involved)
# ---------------------------------------------------------------------------


@router.get("/markets/candles")
async def markets_candles_endpoint(
    symbol: str = Query(..., min_length=3),
    timeframe: str = Query(default="H1"),
    limit: int = Query(default=180, ge=20, le=1000),
    current_user: User = Depends(get_current_user),
) -> dict:
    """OHLCV candles from the resilient feed chain for the Markets page."""
    from forex_ai_agent.agent.tools import _fetch_candles

    try:
        candles = await _fetch_candles(symbol.upper(), timeframe.upper(), limit)
    except Exception as exc:  # noqa: BLE001 — feed failures surface as 502
        raise HTTPException(status_code=502, detail=f"feed error: {exc}")
    if not candles:
        raise HTTPException(status_code=404, detail=f"no candles for {symbol} {timeframe}")
    source = None
    try:
        from forex_ai_agent.data.providers import get_feed_chain
        source = getattr(get_feed_chain(), "active_source", None)
    except Exception:  # noqa: BLE001
        pass
    return {"symbol": symbol.upper(), "timeframe": timeframe.upper(), "source": source, "candles": candles}


@router.get("/markets/structure")
async def markets_structure_endpoint(
    symbol: str = Query(..., min_length=3),
    timeframe: str = Query(default="H1"),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Market-structure snapshot (trend, swings, BOS/CHoCH, S/R, sweeps)."""
    from forex_ai_agent.agent.tools import _build_structure_snapshot

    try:
        return await _build_structure_snapshot(symbol.upper(), timeframe.upper())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"structure error: {exc}")


@router.get("/markets/indicators")
async def markets_indicators_endpoint(
    symbol: str = Query(..., min_length=3),
    timeframe: str = Query(default="H1"),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Latest indicator values (RSI, EMAs, MACD, ATR) for the Markets page."""
    from forex_ai_agent.agent.tools import _compute_indicators

    try:
        return await _compute_indicators(
            symbol.upper(), timeframe.upper(), ["rsi", "ema", "macd", "atr"]
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"indicators error: {exc}")


# ---------------------------------------------------------------------------
# Chat (Phase-1 deterministic reply; AgentCore replaces this in Phase 1b)
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Phase-1 chat: deterministic market/strategy overview reply.

    The full AgentCore (LLM tool loop) plugs in here next; this endpoint's
    contract ({"answer", "tools_used"}) is stable so the UI won't change.
    """
    pipeline = await _ensure_pipeline_init()
    status = await pipeline.get_pipeline_status()
    pairs = await _watched_pairs()
    probation = await _probation_rows_safe()

    lines = [
        f"Feed: {status.get('data_source', 'unknown')} · "
        f"watching {', '.join(pairs) if pairs else 'nothing'} · "
        f"risk {status.get('settings', {}).get('risk_percent', '?')}/trade.",
    ]
    if probation:
        active = [r for r in probation if r.get("status") in {"proposing", "active"}]
        passed = [r for r in probation if r.get("eligible_for_approval")]
        lines.append(
            f"Probation: {len(active)} active candidate(s)"
            + (f", {len(passed)} awaiting YOUR approval" if passed else "")
            + "."
        )
    else:
        lines.append("Probation queue is empty — run Research to propose a strategy.")
    lines.append(
        "Ask for a specific pair's signal via the Signals page; deeper conversational "
        "reasoning (tool-calling Alex) lands with the agent layer."
    )
    return {"answer": " ".join(lines), "tools_used": ["get_status", "list_probation"]}


# ---------------------------------------------------------------------------
# AgentCore Chat (Phase 1b+) — JSON tool loop
# ---------------------------------------------------------------------------


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    max_steps: int = Field(6, ge=1, le=10)


class AgentChatResponse(BaseModel):
    answer: str
    tools_used: List[str] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    steps: int = 0
    success: bool = True
    error: Optional[str] = None


# Per-user memory store (Phase 4: SQLite persistence)
_agent_memories: Dict[str, Any] = {}


def _get_user_memory(user_id: int):
    """Get or create conversation memory for user."""
    if user_id not in _agent_memories:
        from forex_ai_agent.agent.memory import ConversationMemory
        _agent_memories[user_id] = ConversationMemory()
    return _agent_memories[user_id]


@router.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(
    payload: AgentChatRequest,
    current_user: User = Depends(get_current_user),
) -> AgentChatResponse:
    """AgentCore conversational turn: intent → tool loop → final answer from observations."""
    from forex_ai_agent.agent import AgentCore, get_tool_registry
    from forex_ai_agent.llm.client import LLMClient

    # Get user's permission level (from settings or user record)
    user_permission_level = getattr(current_user, 'permission_level', 0)

    # Build AgentCore with user's memory
    memory = _get_user_memory(current_user.id)
    core = AgentCore(
        llm=LLMClient(),
        memory=memory,
        max_steps=payload.max_steps,
        user_permission_level=user_permission_level,
    )

    # Run the turn
    result = await core.run_turn(payload.message)

    return AgentChatResponse(
        answer=result.answer,
        tools_used=result.tools_used,
        evidence=result.evidence,
        steps=result.steps,
        success=result.success,
        error=result.error,
    )


# ---------------------------------------------------------------------------
# Confirmation Token Consumption (Phase 2+)
# ---------------------------------------------------------------------------


class ConfirmRequest(BaseModel):
    token: str = Field(..., min_length=1)
    tool_name: str = Field(..., min_length=1)
    args: Dict[str, Any] = Field(default_factory=dict)


class ConfirmResponse(BaseModel):
    success: bool
    tool_name: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/agent/confirm", response_model=ConfirmResponse)
async def agent_confirm(
    payload: ConfirmRequest,
    current_user: User = Depends(get_current_user),
) -> ConfirmResponse:
    """Consume a confirmation token for a medium-risk tool."""
    from forex_ai_agent.agent import get_tool_registry
    from forex_ai_agent.core.permission_manager import PermissionLevel
    from forex_ai_agent.agent.confirmations import get_confirmation_policy

    policy = get_confirmation_policy()
    token_obj = policy.consume_token(payload.token)

    if not token_obj:
        return ConfirmResponse(
            success=False,
            tool_name=payload.tool_name,
            error="Invalid or expired confirmation token",
        )

    if token_obj.tool_name != payload.tool_name:
        return ConfirmResponse(
            success=False,
            tool_name=payload.tool_name,
            error=f"Token is for tool '{token_obj.tool_name}', not '{payload.tool_name}'",
        )

    # Dispatch with token - this will execute the action
    registry = get_tool_registry(user_permission_level=PermissionLevel.ASSIST)
    try:
        result = await registry.dispatch(payload.tool_name, payload.args, confirm_token=payload.token)
        return ConfirmResponse(
            success=True,
            tool_name=payload.tool_name,
            result=result.model_dump() if hasattr(result, 'model_dump') else str(result),
        )
    except Exception as exc:
        return ConfirmResponse(
            success=False,
            tool_name=payload.tool_name,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Experiment Endpoints (Phase 3+)
# ---------------------------------------------------------------------------


class CreateExperimentRequest(BaseModel):
    strategy_id: str = Field(..., min_length=1)
    param_mutations: Dict[str, Any] = Field(..., description="param_path -> new_value")
    symbol: str = Field(..., min_length=1)
    timeframe: str = Field("H1")
    start_date: str = Field(..., description="ISO format")
    end_date: str = Field(..., description="ISO format")
    hypothesis_id: Optional[str] = None


class CreateExperimentResponse(BaseModel):
    experiment_id: str
    verdict: str
    recommendation: str
    baseline_metrics: Dict[str, Any]
    variant_count: int


@router.post("/agent/experiments", response_model=CreateExperimentResponse)
async def create_experiment_endpoint(
    payload: CreateExperimentRequest,
    current_user: User = Depends(get_current_user),
) -> CreateExperimentResponse:
    """Create and run a parameter mutation experiment."""
    from forex_ai_agent.agent.experiments import create_experiment

    result = await create_experiment(
        strategy_id=payload.strategy_id,
        param_mutations=payload.param_mutations,
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        start_date=payload.start_date,
        end_date=payload.end_date,
        hypothesis_id=payload.hypothesis_id,
    )
    return CreateExperimentResponse(**result)


class ListExperimentsResponse(BaseModel):
    experiments: List[Dict[str, Any]]


@router.get("/agent/experiments", response_model=ListExperimentsResponse)
async def list_experiments_endpoint(
    strategy_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
) -> ListExperimentsResponse:
    """List experiment records."""
    from forex_ai_agent.agent.experiments import list_experiments

    exps = await list_experiments(strategy_id=strategy_id, limit=limit)
    return ListExperimentsResponse(experiments=exps)


class GetExperimentResponse(BaseModel):
    experiment: Optional[Dict[str, Any]]


@router.get("/agent/experiments/{experiment_id}", response_model=GetExperimentResponse)
async def get_experiment_endpoint(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
) -> GetExperimentResponse:
    """Get experiment details with full metrics."""
    from forex_ai_agent.agent.experiments import get_experiment

    exp = await get_experiment(experiment_id)
    return GetExperimentResponse(experiment=exp)


class ConcludeExperimentRequest(BaseModel):
    decision: str = Field(..., pattern="^(adopt|reject|defer)$")
    notes: str = Field(default="", max_length=1000)


class WalkForwardRequest(BaseModel):
    strategy_id: str = Field(..., min_length=1)
    symbol: str = Field(..., min_length=1)
    start_date: str = Field(..., min_length=1)
    end_date: str = Field(..., min_length=1)
    timeframe: str = Field(default="H1")
    window_months: int = Field(default=3, ge=1, le=24)
    step_months: int = Field(default=3, ge=1, le=12)


@router.post("/agent/experiments/walk-forward")
async def walk_forward_endpoint(payload: WalkForwardRequest, current_user: User = Depends(get_current_user)) -> dict:
    """Run a real walk-forward analysis (rolling IS->OOS windows) for a strategy."""
    from forex_ai_agent.agent.experiments import walk_forward_test

    try:
        return await walk_forward_test(
            strategy_id=payload.strategy_id,
            symbol=payload.symbol,
            start_date=payload.start_date,
            end_date=payload.end_date,
            timeframe=payload.timeframe,
            window_months=payload.window_months,
            step_months=payload.step_months,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    limit: int = Field(default=10, ge=1, le=50)


@router.post("/agent/knowledge/search")
async def knowledge_search_endpoint(payload: KnowledgeSearchRequest, current_user: User = Depends(get_current_user)) -> dict:
    """Full-text search over research tasks, experiments, hypotheses and rejections."""
    from forex_ai_agent.agent.explain import KnowledgeRetriever

    retriever = KnowledgeRetriever()
    results = await retriever.search(payload.query, limit=payload.limit)
    return {"results": results}


class ConcludeExperimentResponse(BaseModel):
    experiment_id: str
    final_verdict: str
    concluded_at: Optional[str]


@router.post("/agent/experiments/{experiment_id}/conclude", response_model=ConcludeExperimentResponse)
async def conclude_experiment_endpoint(
    experiment_id: str,
    payload: ConcludeExperimentRequest,
    current_user: User = Depends(get_current_user),
) -> ConcludeExperimentResponse:
    """Record final decision on an experiment."""
    from forex_ai_agent.agent.experiments import conclude_experiment

    result = await conclude_experiment(
        experiment_id=experiment_id,
        decision=payload.decision,
        notes=payload.notes,
    )
    return ConcludeExperimentResponse(**result)
