"""FastAPI application factory.

Serves the alex-ui SPA from the same origin as the API (no CORS needed):
- ``/api/...``  JSON API (auth, pairs, users, watchlists, alex routes)
- ``/ws``       WebSocket (price/signal pushes; pings answered)
- ``/``         the alex-ui single-page app (static mount, added last)
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from forex_ai_agent import __version__ as _version
from ...config import configure_logging, get_settings
from ...config.logging import get_logger
from ...core.event_bus import EventBus, create_event_bus
from ...data.database import dispose_engine, init_db
from .routes import (
    health_router,
    pairs_router,
    users_router,
    watchlist_router,
    auth_router,
    execution_router,
)
from .routes.alex import router as alex_router

logger = get_logger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Root of the SPA (repo_root/alex-ui)
_UI_DIR = Path(__file__).resolve().parents[4] / "alex-ui"


def create_app(settings=None, event_bus: EventBus = None) -> FastAPI:
    """
    Build the FastAPI application.

    Args:
        settings: Optional settings override (tests use this).
        event_bus: Optional event bus instance (tests inject InMemoryEventBus).
    """
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        configure_logging(
            json_output=app_settings.is_production,
            level=logging.INFO if not app_settings.is_test else logging.WARNING,
        )
        bus = event_bus or create_event_bus(use_redis=not app_settings.is_test)
        _app.state.event_bus = bus
        await init_db(app_settings.database_url)

        # B2: research graph edges — idempotent (edge table + link columns).
        try:
            from ...agent.research_graph import ensure_graph_schema

            await ensure_graph_schema()
        except Exception as exc:  # noqa: BLE001 — must not block startup
            logger.warning("research_graph_schema_error", error=str(exc))

        # B5: structured research plans — plan_json column, idempotent.
        try:
            from ...agent.research_plan import ensure_plan_schema

            await ensure_plan_schema()
        except Exception as exc:  # noqa: BLE001 — must not block startup
            logger.warning("research_plan_schema_error", error=str(exc))

        # C3: LLM research reports — report_json column, idempotent.
        try:
            from ...agent.report_gen import ensure_report_schema

            await ensure_report_schema()
        except Exception as exc:  # noqa: BLE001 — must not block startup
            logger.warning("research_report_schema_error", error=str(exc))

        # Background probation accrual: periodically replay active probations
        # over freshly closed bars (idempotent via watermark) so candidates
        # work toward their trade quota without any human action.
        async def _probation_loop(interval_s: float = 900.0) -> None:
            while True:
                await asyncio.sleep(interval_s)
                try:
                    from ...autonomy.probation_runner import ProbationRunner

                    outcomes = await ProbationRunner().run_once()
                    for o in outcomes:
                        if o.get("accrued"):
                            logger.info(
                                "probation_accrued",
                                version_id=o["version_id"],
                                trades=o["accrued"],
                                observed=o.get("observed_trades"),
                            )
                except Exception as exc:  # noqa: BLE001 — the loop must survive
                    logger.warning("probation_loop_error", error=str(exc))

        # Stage D2: event-driven probation accrual — a pump publishes each
        # newly CLOSED bar as a market event and a subscriber accrues
        # probation trades incrementally through the same engine the gate
        # uses. The 900s replay loop above remains the fallback/reconciler.
        async def _market_event_loop(pump, interval_s: float = 30.0) -> None:
            while True:
                await asyncio.sleep(interval_s)
                try:
                    published = await pump.publish_new_bars()
                    if published:
                        logger.info("market_bars_published", bars=published)
                except Exception as exc:  # noqa: BLE001 — the loop must survive
                    logger.warning("market_event_loop_error", error=str(exc))

        # Stage D3: continuous deterministic monitoring of every active
        # probation — logs, bus events on flag change, readable via
        # GET /alex/probation/monitor and the monitor_probation_health tool.
        async def _monitor_loop(monitor, interval_s: float = 300.0) -> None:
            while True:
                await asyncio.sleep(interval_s)
                try:
                    reports = await monitor.monitor_once()  # emits on flag change
                    summary = monitor.summarize(reports)
                    if summary.get("warn") or summary.get("critical"):
                        logger.warning("probation_monitor_summary", **summary)
                    else:
                        logger.info("probation_monitor_summary", **summary)
                except Exception as exc:  # noqa: BLE001 — the loop must survive
                    logger.warning("probation_monitor_loop_error", error=str(exc))

        if not app_settings.is_test:
            _app.state.probation_task = asyncio.create_task(_probation_loop())
            try:
                from ...autonomy.probation_events import (
                    EventDrivenProbationAccrual,
                    MarketBarEventPump,
                )
                from ...autonomy.probation_monitor import ProbationMonitor

                accrual = EventDrivenProbationAccrual(event_bus=bus)
                await accrual.start()
                _app.state.accrual = accrual
                pump = MarketBarEventPump(bus, subscriptions=accrual.subscriptions)
                _app.state.market_event_task = asyncio.create_task(
                    _market_event_loop(pump)
                )
                monitor = ProbationMonitor(event_bus=bus)
                _app.state.monitor = monitor
                _app.state.monitor_task = asyncio.create_task(
                    _monitor_loop(monitor)
                )
            except Exception as exc:  # noqa: BLE001 — must not block startup
                logger.warning("event_accrual_startup_error", error=str(exc))

        logger.info("application_started", env=app_settings.environment)
        yield
        task = getattr(_app.state, "probation_task", None)
        if task is not None:
            task.cancel()
        market_event_task = getattr(_app.state, "market_event_task", None)
        if market_event_task is not None:
            market_event_task.cancel()
        monitor_task = getattr(_app.state, "monitor_task", None)
        if monitor_task is not None:
            monitor_task.cancel()
        accrual = getattr(_app.state, "accrual", None)
        if accrual is not None:
            await accrual.stop()
        await bus.close()
        await dispose_engine()
        logger.info("application_stopped")

    app = FastAPI(
        title="Forex AI Agent",
        description="Forex Research & Intelligence Agent - Phase 1",
        version=_version,
        lifespan=lifespan,
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ---------------- JSON API (all under /api) ----------------
    api_prefix = "/api"

    app.include_router(health_router, prefix=api_prefix)
    app.include_router(pairs_router, prefix=api_prefix)
    app.include_router(users_router, prefix=api_prefix)
    app.include_router(watchlist_router, prefix=api_prefix)
    app.include_router(auth_router, prefix=api_prefix)
    app.include_router(alex_router, prefix=api_prefix)
    app.include_router(execution_router, prefix=api_prefix)

    # Service overview (the mount below owns "/" for the SPA).
    @app.get("/api")
    async def root() -> dict:
        """Service overview endpoint."""
        return {
            "name": "forex-ai-agent",
            "version": _version,
            "phase": 1,
            "docs": "/docs",
        }

    # ---------------- WebSocket: /ws ----------------

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """Live push channel: pings answered; client messages logged.

        Price/signal broadcasts hook into the event bus in a later phase; the
        SPA tolerates a quiet channel (its UI badges reflect connection state).
        """
        await websocket.accept()
        logger.info("ws_client_connected")
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                mtype = msg.get("type")
                if mtype == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "ts": msg.get("ts")}))
                elif mtype == "order":
                    # Live execution is permission-locked; acknowledge but do nothing.
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "response",
                                "requestId": msg.get("requestId"),
                                "error": "order execution is disabled (permission level 0)",
                            }
                        )
                    )
                else:
                    logger.info("ws_message", type=mtype)
        except WebSocketDisconnect:
            logger.info("ws_client_disconnected")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ws_error", error=str(exc))

    # ---------------- SPA: serve alex-ui (added LAST — catch-all) ----------------

    if _UI_DIR.is_dir():

        class _NoCacheStaticFiles(StaticFiles):
            """Serve SPA files with ``no-cache`` so code changes always reach the browser.

            Responses still carry an ETag, so unchanged files get a cheap
            304 Not Modified instead of a re-download — caching stays fast,
            but never stale.
            """

            def file_response(self, *args, **kwargs):
                resp = super().file_response(*args, **kwargs)
                resp.headers["Cache-Control"] = "no-cache"
                return resp

        app.mount("/", _NoCacheStaticFiles(directory=str(_UI_DIR), html=True), name="spa")
    else:  # pragma: no cover — repo layout change
        logger.warning("alex-ui directory not found; SPA not mounted", path=str(_UI_DIR))

    return app


app = create_app()
