"""Stage-B5 unit tests: structured research plans (the RESEARCH #id card).

Pins the roadmap acceptance: a plan created now reports its own checklist
progress and conclusion later — all from persisted records. The Director
pipeline is faked through run_cycle's injection seam, exactly like B4: the
tests pin the ORCHESTRATION and the persistence join, not the backtester.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Import at module level so ResearchPlan / DirectorQueueItem / DirectorEvent
# register on Base.metadata before the fixture's create_all (B2 lesson).
from forex_ai_agent.agent import director as dr
from forex_ai_agent.agent import research_plan as rp
from forex_ai_agent.data.models.base import Base as ModelsBase

HISTORY_DAYS = 180


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
    return SimpleNamespace(
        pair="?",
        status=status,
        used_llm=used_llm,
        llm_error=None,
        messages=[f"fake {status} on plan data"],
        validation=SimpleNamespace(passed=status == "validated", reasons=[]),
        version=SimpleNamespace(id=version_id) if version_id else None,
        spec_dict={"name": "Fake"},
        brief={},
        proposed_at="2026-09-25T00:00:00",
    )


def _cycle_seam(outcome: Dict[str, Any]):
    """Injection-seam fakes: one call runs one item, then records its plan_id."""

    class _FakePipeline:
        def __init__(self, *a, **k):
            self.data_provider = SimpleNamespace()

        async def initialize(self):
            return None

    async def fake_fetch(pipeline, pair, days=HISTORY_DAYS, timeframe="H1"):
        return object()

    seen: Dict[str, Any] = {}

    async def fake_propose(pipeline, pair, history, use_llm=True, timeframe="H1"):
        seen[pair] = use_llm
        return _fake_proposal(outcome["status"], version_id=outcome.get("version_id"))

    return fake_fetch, fake_propose, _FakePipeline, seen


async def _mk_plan(sf, pairs=("EUR/USD", "GBP/USD"), **kw) -> Dict[str, Any]:
    return await rp.create_research_plan(
        topic="Does displacement predict continuation?",
        pairs=list(pairs),
        hypothesis_text="Strong displacement bars precede continuation",
        session_factory=sf,
        **kw,
    )


# ---------------------------------------------------------------------------
# Creation + persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_persists_plan_row(session_factory):
    created = await _mk_plan(session_factory)
    card = await rp.get_research_plan(created["task_id"], session_factory=session_factory)
    assert card is not None
    assert card["topic"] == "Does displacement predict continuation?"
    assert card["pairs"] == ["EUR/USD", "GBP/USD"]
    # create() refreshes once after enqueueing, so the pairs already show as
    # queued and the plan is IN_PROGRESS from the moment it exists.
    assert card["status"] == rp.STATUS_IN_PROGRESS
    assert card["experiment_total"] == 6
    assert [s["step"] for s in card["experiment_checklist"]] == [
        "baseline", "displacement", "entry", "exit", "session", "cost_sensitivity",
    ]
    assert [s["step"] for s in card["validation_checklist"]] == [
        "walk_forward", "robustness_sweep", "parameter_sensitivity", "probation",
    ]
    assert all(s["state"] == "pending" for s in card["experiment_checklist"][1:])
    assert card["experiment_checklist"][0]["state"] == "in_progress"


@pytest.mark.asyncio
async def test_create_enqueues_pairs_with_plan_id(session_factory):
    created = await _mk_plan(session_factory)
    snap = await dr.queue_snapshot(session_factory=session_factory)
    assert snap["open_items"] == 2
    linked = [
        i for i in snap["queue"]
        if (i["result"] or {}).get("plan_id") == created["task_id"]
    ]
    assert len(linked) == 2
    # The plan row joined its own items back: pairs visible as queued.
    card = await rp.get_research_plan(created["task_id"], session_factory=session_factory)
    assert set(card["pairs_progress"]) == {"EUR/USD", "GBP/USD"}
    assert card["pairs_progress"]["EUR/USD"]["status"] == "queued"
    assert card["status"] == rp.STATUS_IN_PROGRESS


@pytest.mark.asyncio
async def test_create_validates_inputs(session_factory):
    with pytest.raises(ValueError):
        await rp.create_research_plan(topic="  ", pairs=["EUR/USD"], session_factory=session_factory)
    with pytest.raises(ValueError):
        await rp.create_research_plan(topic="x", pairs=[], session_factory=session_factory)
    with pytest.raises(ValueError):
        await rp.create_research_plan(topic="x", pairs=["NOPE"], session_factory=session_factory)
    # 6-letter pairs are normalized, not rejected.
    created = await rp.create_research_plan(
        topic="x", pairs=["eurusd"], enqueued=False, session_factory=session_factory
    )
    assert created["card"]["pairs"] == ["EUR/USD"]


@pytest.mark.asyncio
async def test_create_rejects_bad_priority(session_factory):
    with pytest.raises(ValueError):
        await _mk_plan(session_factory, priority=7)


@pytest.mark.asyncio
async def test_prefix_lookup_and_list(session_factory):
    created = await _mk_plan(session_factory, enqueued=False)
    plans = await rp.list_research_plans(session_factory=session_factory)
    assert [p["task_id"] for p in plans] == [created["task_id"]]
    card = await rp.get_research_plan(created["task_id"][:8], session_factory=session_factory)
    assert card is None  # get_ is strict; the tool layer does prefix resolution


# ---------------------------------------------------------------------------
# The Director works the plan — progress from persisted records only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cycle_rejected_pair_updates_checklist_and_conclusion(session_factory):
    created = await _mk_plan(session_factory)
    fetch, propose, FakePipeline, _seen = _cycle_seam({"status": "rejected"})
    report = await dr.run_cycle(
        session_factory=session_factory,
        history_fetcher=fetch,
        proposal_runner=propose,
        pipeline=FakePipeline(),
    )
    assert report["items_run"] == 2 and report["rejected"] == 2

    card = await rp.get_research_plan(created["task_id"], session_factory=session_factory)
    assert card["status"] == rp.STATUS_COMPLETED  # all pairs terminal → done
    assert card["progress_percent"] == 100
    assert card["pairs_progress"]["EUR/USD"]["status"] == "rejected"
    # One methodology step completed per finished pair, in order.
    done = [s["step"] for s in card["experiment_checklist"] if s["state"] == "done"]
    assert done == ["baseline", "displacement"]
    # Honest validation: walk-forward skipped when the gate rejected.
    wf = next(s for s in card["validation_checklist"] if s["step"] == "walk_forward")
    assert wf["state"] == "skipped"
    assert card["conclusion"]["verdict"] == "rejected"
    assert "fake rejected" in card["conclusion"]["summary"]
    # The conclusion card rode on the stored Director row.
    snap = await dr.queue_snapshot(session_factory=session_factory)
    row = next(i for i in snap["queue"] if i["pair"] == "EUR/USD")
    cc = row["result"].get("conclusion_card")
    assert cc and cc["plan_id"] == created["task_id"] and cc["verdict"] == "rejected"


@pytest.mark.asyncio
async def test_cycle_validated_pair_marks_walk_forward_done(session_factory):
    created = await _mk_plan(session_factory, pairs=("USD/JPY",))
    fetch, propose, FakePipeline, _seen = _cycle_seam(
        {"status": "validated", "version_id": "v-plan-1"}
    )
    await dr.run_cycle(
        session_factory=session_factory,
        history_fetcher=fetch,
        proposal_runner=propose,
        pipeline=FakePipeline(),
    )
    card = await rp.get_research_plan(created["task_id"], session_factory=session_factory)
    assert card["status"] == rp.STATUS_COMPLETED
    assert card["pairs_progress"]["USD/JPY"]["version_id"] == "v-plan-1"
    # Single pair, single cycle: step 1 completes; the next methodology step
    # only opens while pairs remain to be worked — nothing remains, so the
    # rest stay pending (the plan is exhausted, honestly).
    assert [s["state"] for s in card["experiment_checklist"]] == [
        "done", "pending", "pending", "pending", "pending", "pending",
    ]
    wf = next(s for s in card["validation_checklist"] if s["step"] == "walk_forward")
    assert wf["state"] == "done"
    prob = next(s for s in card["validation_checklist"] if s["step"] == "probation")
    assert prob["state"] == "in_progress"  # version entered probation


@pytest.mark.asyncio
async def test_progress_partial_then_complete(session_factory):
    created = await _mk_plan(session_factory)
    fetch, propose, FakePipeline, _seen = _cycle_seam({"status": "validated"})

    # Budget-capped first cycle: only the first (FIFO) pair is worked.
    report = await dr.run_cycle(
        session_factory=session_factory,
        history_fetcher=fetch,
        proposal_runner=propose,
        pipeline=FakePipeline(),
        config=dr.DirectorConfig(max_items_per_cycle=1),
    )
    assert report["items_run"] == 1
    card = await rp.get_research_plan(created["task_id"], session_factory=session_factory)
    assert card["status"] == rp.STATUS_IN_PROGRESS
    assert card["progress_percent"] == 50
    assert card["pairs_progress"]["GBP/USD"]["status"] == "queued"

    # Morning→evening acceptance: the second cycle finishes the plan.
    await dr.run_cycle(
        session_factory=session_factory,
        history_fetcher=fetch,
        proposal_runner=propose,
        pipeline=FakePipeline(),
    )
    card = await rp.get_research_plan(created["task_id"], session_factory=session_factory)
    assert card["status"] == rp.STATUS_COMPLETED
    assert card["progress_percent"] == 100
    assert card["conclusion"]["verdict"] == "validated"


@pytest.mark.asyncio
async def test_cycle_error_is_terminal_and_audited(session_factory):
    created = await _mk_plan(session_factory, pairs=("AUD/USD",))

    async def boom(pipeline, pair, days=HISTORY_DAYS, timeframe="H1"):
        raise RuntimeError("feed down")

    await dr.run_cycle(
        session_factory=session_factory,
        history_fetcher=boom,
        proposal_runner=_fake_proposal,
        pipeline=SimpleNamespace(data_provider=SimpleNamespace()),
    )
    card = await rp.get_research_plan(created["task_id"], session_factory=session_factory)
    assert card["status"] == rp.STATUS_COMPLETED  # exhausted, honestly
    assert card["pairs_progress"]["AUD/USD"]["status"] == "error"


@pytest.mark.asyncio
async def test_unrelated_queue_rows_never_leak_into_plan(session_factory):
    # A pre-existing Director row without plan_id must not join this plan.
    await dr.enqueue_hypothesis(pair="NZD/USD", session_factory=session_factory)
    created = await _mk_plan(session_factory)
    card = await rp.get_research_plan(created["task_id"], session_factory=session_factory)
    assert set(card["pairs_progress"]) == {"EUR/USD", "GBP/USD"}


@pytest.mark.asyncio
async def test_refresh_never_raises_on_garbage_id(session_factory):
    assert await rp.refresh_plan_from_director("not-a-uuid") is None
    assert await rp.refresh_plan_from_director(
        "00000000-0000-0000-0000-000000000000", session_factory=session_factory
    ) is None


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------


def test_render_plan_card_lists_checklists():
    payload = rp.build_plan_payload(
        topic="t", pairs=["EUR/USD"], timeframes=["H1"], hypothesis_text="h"
    )
    payload["task_id"] = "abcdef12-34"
    payload, _touched = rp._apply_plan_progress(
        payload,
        [
            {
                "pair": "EUR/USD",
                "item_status": "DONE_REJECTED",
                "result": {"plan_id": "abcdef12-34", "status": "rejected", "messages": ["no edge"]},
            }
        ],
    )
    text = rp.render_plan_card(rp._plan_to_card(payload, "abcdef12-34"))
    assert text.startswith("RESEARCH #abcdef12 — t")
    assert "[x] Baseline backtest" in text
    assert "[ ] Cost sensitivity" in text
    assert "Walk-forward OOS" in text
    assert "EUR/USD: rejected" in text
    assert "Conclusion: rejected" in text


def test_conclusion_card_requires_plan_id():
    assert rp.conclusion_card_from_plan({"status": "validated"}) is None
    card = rp.conclusion_card_from_plan(
        {"plan_id": "p1", "status": "validated", "messages": ["m"], "version_id": "v1"}
    )
    assert card == {
        "plan_id": "p1", "pair": None, "verdict": "validated",
        "summary": "m", "version_id": "v1",
    }


@pytest.mark.asyncio
async def test_ensure_plan_schema_idempotent(session_factory):
    first = await rp.ensure_plan_schema(session_factory)
    second = await rp.ensure_plan_schema(session_factory)
    assert first == {"created_table": False}  # create_all made it in the fixture
    assert second == {"created_table": False}


@pytest.mark.asyncio
async def test_ensure_plan_schema_creates_missing_table(session_factory):
    # Simulate a pre-B5 DB: drop the table, then ensure must recreate it.
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(ModelsBase.metadata.create_all)
        from sqlalchemy import text

        await conn.execute(text("DROP TABLE research_plans"))
    sf = async_sessionmaker(engine, expire_on_commit=False)
    result = await rp.ensure_plan_schema(sf)
    assert result == {"created_table": True}
    await engine.dispose()


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_research_tool_persists_plan(monkeypatch, session_factory):
    from forex_ai_agent.agent import tools as t

    # research_plan.py resolved get_session_factory at import time, so patch
    # the name the module actually uses (B2/B4 lesson).
    monkeypatch.setattr(rp, "get_session_factory", lambda: session_factory)
    res = await t.start_research(
        t.StartResearchArgs(topic="Session filters on GBP/JPY", pair="GBP/JPY", timeframe="H1")
    )
    assert res.status == "started"
    card = res.proposal
    assert card["task_id"] == res.task_id
    assert card["pairs"] == ["GBP/JPY"]
    assert card["experiment_total"] == 6
    # Persisted, not just returned:
    again = await rp.get_research_plan(res.task_id, session_factory=session_factory)
    assert again is not None and again["topic"] == "Session filters on GBP/JPY"


@pytest.mark.asyncio
async def test_start_research_tool_reports_bad_pair(session_factory, monkeypatch):
    from forex_ai_agent.agent import tools as t

    monkeypatch.setattr(rp, "get_session_factory", lambda: session_factory)
    res = await t.start_research(t.StartResearchArgs(topic="x", pair="NOTAPAIR"))
    assert res.status == "error"
    assert "pair must look like" in res.proposal["error"]


@pytest.mark.asyncio
async def test_research_plan_tool_by_id_and_prefix(monkeypatch, session_factory):
    from forex_ai_agent.agent import tools as t

    # research_plan_tool imports get_session_factory from the database module
    # at call time, while get_research_plan uses rp's import-time global —
    # patch both so every store access hits the fixture DB.
    monkeypatch.setattr(
        "forex_ai_agent.data.database.get_session_factory", lambda: session_factory
    )
    monkeypatch.setattr(rp, "get_session_factory", lambda: session_factory)
    created = await _mk_plan(session_factory)
    # Full id
    res = await t.research_plan_tool(t.ResearchPlanArgs(task_id=created["task_id"]))
    assert res.found and res.plans[0]["task_id"] == created["task_id"]
    assert "RESEARCH #" in res.rendered
    # Prefix
    res = await t.research_plan_tool(t.ResearchPlanArgs(task_id=created["task_id"][:8]))
    assert res.found and res.task_id == created["task_id"]
    # Unknown
    res = await t.research_plan_tool(t.ResearchPlanArgs(task_id="ffffffff"))
    assert not res.found
    # Ambiguous: two plans sharing an 8-char prefix is (astronomically)
    # unlikely, so pin the deterministic list path instead.
    res = await t.research_plan_tool(t.ResearchPlanArgs())
    assert res.found and len(res.plans) == 1


def test_registry_has_research_plan_at_observe():
    from forex_ai_agent.agent.tools import get_tool_registry

    reg = get_tool_registry(0)
    tool = reg._tools["research_plan"]
    assert tool.min_permission_level == 0  # OBSERVE
    from forex_ai_agent.agent.confirmations import get_tool_risk_class

    assert get_tool_risk_class("research_plan").value == "low"
