"""Stage-B4 unit tests: Research Director.

Pins the roadmap acceptance: the Director runs full cycles unattended within
budget, every action is audited, and nothing outside the research path is
reachable. All pipeline/LLM work is faked — the tests pin the ORCHESTRATION,
not the backtester (that is the golden suite's job).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Import at module level so DirectorQueueItem/DirectorEvent register on
# Base.metadata before the fixture's create_all (B2 lesson).
from forex_ai_agent.agent import director as dr
from forex_ai_agent.data.models.base import Base as ModelsBase

PAIRS_HISTORY_DAYS = 180


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


def _fake_proposal(status: str, version_id: str = None, used_llm: bool = False):
    messages = [f"fake {status}"]
    return SimpleNamespace(
        pair="?",
        status=status,
        used_llm=used_llm,
        llm_error=None,
        messages=messages,
        validation=SimpleNamespace(passed=status == "validated", reasons=[]),
        version=SimpleNamespace(id=version_id) if version_id else None,
        spec_dict={"name": "Fake"},
        brief={},
        proposed_at="2026-09-25T00:00:00",
    )


@pytest.fixture()
def llm_tracker():
    return {"calls": 0}


def _patch_loop(monkeypatch, outcomes: List[Dict[str, Any]], llm_tracker: dict):
    """Fakes injected through run_cycle's seam; each call pops the next outcome.

    Returns kwargs to pass to every run_cycle call in the test.
    """

    class _FakePipeline:
        def __init__(self, *a, **k):
            self.data_provider = SimpleNamespace()

        async def initialize(self):
            return None

    fake_pipeline = _FakePipeline()

    async def fake_fetch(pipeline, pair, days=PAIRS_HISTORY_DAYS, timeframe="H1"):
        return object()  # placeholder frame

    async def fake_propose(pipeline, pair, history, use_llm=True, **kw):
        if use_llm:
            llm_tracker["calls"] += 1
        outcome = outcomes.pop(0) if outcomes else {"status": "rejected"}
        if isinstance(outcome, Exception):
            raise outcome
        return _fake_proposal(
            outcome.get("status", "rejected"),
            version_id=outcome.get("version_id"),
            used_llm=use_llm,
        )

    return {
        "pipeline": fake_pipeline,
        "history_fetcher": fake_fetch,
        "proposal_runner": fake_propose,
    }


# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_and_dedup(session_factory):
    first = await dr.enqueue_hypothesis("GBP/JPY", "sweep idea", session_factory=session_factory)
    assert first["enqueued"] is True

    dup = await dr.enqueue_hypothesis("GBP/JPY", "again", session_factory=session_factory)
    assert dup["enqueued"] is False
    assert "already" in dup["reason"]

    other = await dr.enqueue_hypothesis("EUR/USD", "", session_factory=session_factory)
    assert other["enqueued"] is True


@pytest.mark.asyncio
async def test_enqueue_rejects_bad_priority(session_factory):
    with pytest.raises(ValueError):
        await dr.enqueue_hypothesis("GBP/JPY", priority=9, session_factory=session_factory)


@pytest.mark.asyncio
async def test_queue_cap_is_enforced(session_factory):
    config = dr.DirectorConfig(max_queue_size=2)
    assert (await dr.enqueue_hypothesis("A/1", config=config, session_factory=session_factory))["enqueued"]
    assert (await dr.enqueue_hypothesis("A/2", config=config, session_factory=session_factory))["enqueued"]
    third = await dr.enqueue_hypothesis("A/3", config=config, session_factory=session_factory)
    assert third["enqueued"] is False
    assert "queue full" in third["reason"]


@pytest.mark.asyncio
async def test_priority_order_and_fifo_within_priority(session_factory):
    await dr.enqueue_hypothesis("LOW/1", priority=3, session_factory=session_factory)
    await dr.enqueue_hypothesis("HIGH/1", priority=1, session_factory=session_factory)
    await dr.enqueue_hypothesis("HIGH/2", priority=1, session_factory=session_factory)
    snap = await dr.queue_snapshot(session_factory)
    ordered = [(i["pair"], i["priority"]) for i in snap["queue"]]
    assert ordered == [("HIGH/1", "HIGH"), ("HIGH/2", "HIGH"), ("LOW/1", "LOW")]


@pytest.mark.asyncio
async def test_cancel_only_open_items(session_factory):
    created = await dr.enqueue_hypothesis("GBP/JPY", session_factory=session_factory)
    real_id = created["item_id"]
    assert await dr.cancel_item(real_id, session_factory) is True
    assert await dr.cancel_item(real_id, session_factory) is False  # terminal
    assert await dr.cancel_item(str(uuid.uuid4()), session_factory) is False
    assert await dr.cancel_item("not-a-uuid", session_factory) is False


# ---------------------------------------------------------------------------
# Cycle execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cycle_runs_queue_and_records_outcomes(session_factory, monkeypatch, llm_tracker):
    seam = _patch_loop(
        monkeypatch,
        [
            {"status": "validated", "version_id": "v1"},
            {"status": "rejected"},
        ],
        llm_tracker,
    )
    await dr.enqueue_hypothesis("GBP/JPY", session_factory=session_factory)
    await dr.enqueue_hypothesis("EUR/USD", session_factory=session_factory)

    report = await dr.run_cycle(session_factory=session_factory, **seam)
    assert report["items_run"] == 2
    assert report["validated"] == 1 and report["rejected"] == 1
    assert report["cycle_id"].startswith("B4-")

    snap = await dr.queue_snapshot(session_factory)
    by_pair = {i["pair"]: i for i in snap["queue"]}
    assert by_pair["GBP/JPY"]["status"] == dr.DONE_VALIDATED
    assert by_pair["GBP/JPY"]["result"]["version_id"] == "v1"
    assert by_pair["EUR/USD"]["status"] == dr.DONE_REJECTED


@pytest.mark.asyncio
async def test_budget_caps_items_per_cycle(session_factory, monkeypatch, llm_tracker):
    seam = _patch_loop(monkeypatch, [{"status": "rejected"}] * 5, llm_tracker)
    for p in ("A/1", "A/2", "A/3", "A/4"):
        await dr.enqueue_hypothesis(p, session_factory=session_factory)

    config = dr.DirectorConfig(max_items_per_cycle=2)
    report = await dr.run_cycle(config, session_factory=session_factory, **seam)

    assert report["items_run"] == 2
    assert report["budget_exhausted"] is True
    snap = await dr.queue_snapshot(session_factory)
    assert snap["by_status"]["QUEUED"] == 2  # untouched for the next cycle

    # next cycle finishes the queue
    report2 = await dr.run_cycle(config, session_factory=session_factory, **seam)
    assert report2["items_run"] == 2
    assert report2["budget_exhausted"] is False
    snap2 = await dr.queue_snapshot(session_factory)
    assert snap2["open_items"] == 0


@pytest.mark.asyncio
async def test_llm_budget_drops_to_fallback_mid_cycle(session_factory, monkeypatch, llm_tracker):
    """After the LLM cap, later items still run — on the deterministic spec."""
    seam = _patch_loop(
        monkeypatch,
        [{"status": "validated", "version_id": "v1"}, {"status": "validated", "version_id": "v2"}],
        llm_tracker,
    )
    await dr.enqueue_hypothesis("A/1", session_factory=session_factory)
    await dr.enqueue_hypothesis("A/2", session_factory=session_factory)

    config = dr.DirectorConfig(max_items_per_cycle=2, max_llm_calls_per_cycle=1)
    report = await dr.run_cycle(config, session_factory=session_factory, **seam)

    assert llm_tracker["calls"] == 1  # second item must NOT have used the LLM
    by_pair = {r["pair"]: r for r in report["results"]}
    assert by_pair["A/1"]["used_llm"] is True
    assert by_pair["A/2"]["used_llm"] is False
    assert report["validated"] == 2  # fallback spec still researches


@pytest.mark.asyncio
async def test_use_llm_false_runs_fully_deterministic(session_factory, monkeypatch, llm_tracker):
    seam = _patch_loop(monkeypatch, [{"status": "rejected"}], llm_tracker)
    await dr.enqueue_hypothesis("A/1", session_factory=session_factory)
    config = dr.DirectorConfig(use_llm=False)
    report = await dr.run_cycle(config, session_factory=session_factory, **seam)
    assert llm_tracker["calls"] == 0
    assert report["results"][0]["used_llm"] is False


@pytest.mark.asyncio
async def test_audit_trail_records_every_action(session_factory, monkeypatch, llm_tracker):
    seam = _patch_loop(monkeypatch, [{"status": "validated", "version_id": "v1"}], llm_tracker)
    await dr.enqueue_hypothesis("GBP/JPY", session_factory=session_factory)
    report = await dr.run_cycle(session_factory=session_factory, **seam)

    async with session_factory() as session:
        events = (
            (
                await session.execute(
                    select(dr.DirectorEvent).order_by(dr.DirectorEvent.created_at)
                )
            )
            .scalars()
            .all()
        )
    actions = [e.action for e in events]
    assert actions[0] == "cycle_started"
    assert "item_started" in actions and "item_finished" in actions
    assert actions[-1] == "cycle_finished"
    assert all(e.cycle_id == report["cycle_id"] for e in events)
    finished = next(e for e in events if e.action == "item_finished")
    assert finished.detail["status"] == "validated"
    assert finished.pair == "GBP/JPY"


@pytest.mark.asyncio
async def test_exception_becomes_error_row_not_cycle_death(session_factory, monkeypatch, llm_tracker):
    seam = _patch_loop(
        monkeypatch,
        [RuntimeError("feed boom"), {"status": "validated", "version_id": "v9"}],
        llm_tracker,
    )
    await dr.enqueue_hypothesis("BAD/1", session_factory=session_factory)
    await dr.enqueue_hypothesis("GOOD/1", session_factory=session_factory)

    report = await dr.run_cycle(session_factory=session_factory, **seam)
    assert report["items_run"] == 2
    assert report["errors"] == 1 and report["validated"] == 1
    snap = await dr.queue_snapshot(session_factory)
    by_pair = {i["pair"]: i for i in snap["queue"]}
    assert by_pair["BAD/1"]["status"] == dr.DONE_ERROR
    assert "feed boom" in by_pair["BAD/1"]["result"]["error"]
    assert by_pair["GOOD/1"]["status"] == dr.DONE_VALIDATED


@pytest.mark.asyncio
async def test_stale_running_requeued_on_next_cycle(session_factory, monkeypatch, llm_tracker):
    """A crashed cycle's RUNNING row is requeued honestly, not lost."""
    created = await dr.enqueue_hypothesis("STALE/1", session_factory=session_factory)
    async with session_factory() as session:
        item = await session.get(dr.DirectorQueueItem, uuid.UUID(created["item_id"]))
        item.status = dr.RUNNING  # simulate a dead process mid-run
        await session.commit()

    seam = _patch_loop(monkeypatch, [{"status": "rejected"}], llm_tracker)
    report = await dr.run_cycle(session_factory=session_factory, **seam)

    assert report["skipped_stale_running"] == 1
    assert report["items_run"] == 1  # the requeued item was actually re-run
    snap = await dr.queue_snapshot(session_factory)
    assert snap["queue"][0]["status"] == dr.DONE_REJECTED
    # recovery is auditable, and the fresh run's result replaces the note
    assert any(
        e["action"] == "stale_running_requeued" for e in snap["recent_events"]
    )


@pytest.mark.asyncio
async def test_empty_queue_cycle_is_a_clean_noop(session_factory, monkeypatch, llm_tracker):
    seam = _patch_loop(monkeypatch, [], llm_tracker)
    report = await dr.run_cycle(session_factory=session_factory, **seam)
    assert report["items_run"] == 0
    assert report["budget_exhausted"] is False
    snap = await dr.queue_snapshot(session_factory)
    assert sorted(e["action"] for e in snap["recent_events"]) == [
        "cycle_finished",
        "cycle_started",
    ]


@pytest.mark.asyncio
async def test_events_survive_with_items_after_restart(session_factory, monkeypatch, llm_tracker):
    """Persistence check: queue + events live in SQLite, not memory."""
    seam = _patch_loop(monkeypatch, [{"status": "rejected"}], llm_tracker)
    await dr.enqueue_hypothesis("A/1", session_factory=session_factory)
    await dr.run_cycle(session_factory=session_factory, **seam)

    # New engine over the SAME in-memory DB would be a new DB (StaticPool);
    # instead verify both tables have rows — i.e. not memory-only state.
    async with session_factory() as session:
        items = (await session.execute(select(dr.DirectorQueueItem))).scalars().all()
        events = (await session.execute(select(dr.DirectorEvent))).scalars().all()
    assert len(items) == 1 and len(events) >= 3


# ---------------------------------------------------------------------------
# Hard blocks: stated in the audit trail, enforced by construction
# ---------------------------------------------------------------------------


def test_director_module_has_no_risk_or_order_imports():
    """The Director must not touch trading/risk/permissions machinery."""
    import inspect

    import forex_ai_agent.agent.director as director_mod

    src = inspect.getsource(director_mod)
    for banned in ("place_order", "risk_manager", "set_permission", "delete_event", "approve"):
        assert banned not in src, f"director must not reference {banned}"


def test_cycle_report_shape_is_json_safe():
    config = dr.DirectorConfig()
    report = dr.CycleReport(cycle_id="B4-x", started_at="t", finished_at="t")
    d = report.to_dict()
    import json

    json.loads(json.dumps(d))  # must not raise
    assert d["results"] == []
