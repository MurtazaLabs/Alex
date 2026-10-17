"""Unit tests for the research log (ResearchTask persistence)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from forex_ai_agent.data.models.base import Base as ModelsBase
from forex_ai_agent.data.models.research import ResearchTask


@pytest.fixture()
async def session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(ModelsBase.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


def _result(status: str, **kw):
    """A minimal stand-in for ProposalResult (avoids engine imports)."""

    base = {
        "pair": "EUR/USD",
        "status": status,
        "spec_dict": {"name": "TestSpec", "direction": "long"},
        "used_llm": False,
        "llm_error": None,
        "validation": None,
        "version": None,
        "brief": None,
        "messages": ["using deterministic fallback spec"],
        "proposed_at": "2026-09-19T10:00:00",
    }
    base.update(kw)
    return SimpleNamespace(**base)


class _Report:
    def __init__(self, reasons):
        self.reasons = reasons

    def to_dict(self):
        return {"reasons": self.reasons, "passed": not self.reasons}


@pytest.mark.asyncio
async def test_validated_run_logged_as_paper_test(session_factory):
    from forex_ai_agent.autonomy.research_log import log_research_run

    task = await log_research_run(
        _result(
            "validated",
            validation=_Report([]),
            version=SimpleNamespace(id="v-123"),
        ),
        session_factory=session_factory,
    )
    assert task is not None
    assert task.status == "COMPLETED"
    assert task.recommendation == "PAPER_TEST"
    assert "EUR/USD: TestSpec" in task.concept_name
    assert task.conclusion_json["verdict"] == "validated"


@pytest.mark.asyncio
async def test_rejected_run_logged_as_do_not_use(session_factory):
    from forex_ai_agent.autonomy.research_log import log_research_run

    task = await log_research_run(
        _result("rejected", validation=_Report(["profit factor 0.65 < 1.1"])),
        session_factory=session_factory,
    )
    assert task is not None
    assert task.recommendation == "DO_NOT_USE"
    assert "Gate REJECTED" in task.conclusion_json["summary"]
    assert task.conclusion_json["validation"]["reasons"]


@pytest.mark.asyncio
async def test_error_run_logged_as_failed(session_factory):
    from forex_ai_agent.autonomy.research_log import log_research_run

    task = await log_research_run(
        _result("error", spec_dict=None, messages=["brief failed: BoomError: x"]),
        session_factory=session_factory,
    )
    assert task is not None
    assert task.status == "FAILED"
    assert task.recommendation is None
    assert task.conclusion_json["summary"].startswith("brief failed")


@pytest.mark.asyncio
async def test_alex_user_created_once_and_reused(session_factory):
    from sqlalchemy import func, select

    from forex_ai_agent.autonomy.research_log import log_research_run
    from forex_ai_agent.data.models.user import User

    await log_research_run(_result("rejected"), session_factory=session_factory)
    await log_research_run(_result("validated"), session_factory=session_factory)

    async with session_factory() as session:
        count = (
            await session.execute(
                select(func.count()).select_from(User).where(User.username == "alex")
            )
        ).scalar()
        tasks = (
            await session.execute(select(func.count()).select_from(ResearchTask))
        ).scalar()
    assert count == 1
    assert tasks == 2


@pytest.mark.asyncio
async def test_recent_tasks_lists_all_statuses(session_factory):
    from forex_ai_agent.autonomy import research_log

    await research_log.log_research_run(
        _result("rejected"), session_factory=session_factory
    )
    await research_log.log_research_run(
        _result("validated"), session_factory=session_factory
    )
    await research_log.log_research_run(
        _result("error"), session_factory=session_factory
    )

    rows = await research_log.recent_research_tasks(
        limit=10, session_factory=session_factory
    )
    statuses = {r["status"] for r in rows}
    assert {"COMPLETED", "FAILED"} <= statuses
    rejected = next(r for r in rows if r["verdict"] == "rejected")
    assert rejected["recommendation"] == "DO_NOT_USE"
    assert rejected["progress"] == 100
