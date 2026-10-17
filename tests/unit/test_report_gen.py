"""Stage-C3 unit tests: LLM research reports with deterministic traceability.

Pins the Stage C constitution on the WRITTEN-FINDINGS path: the LLM only sees
a deterministic evidence catalog (ids + facts + numbers from stored records)
and only narrates; code renders the numbers ledger, validates that every
prose number traces to the cited evidence, replaces untraceable sections with
deterministic summaries, and persists the report as a first-class record.
LLM and evidence loader are faked via injection seams — the tests pin the
ORCHESTRATION, not the network.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Dict, List, Optional

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from forex_ai_agent.agent import report_gen as rg
from forex_ai_agent.data.models.base import Base as ModelsBase


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


class _FakeLLM:
    """Records the context it is shown; returns a canned narration."""

    def __init__(self, response: Any = None, fail: bool = False):
        self.response = response
        self.fail = fail
        self.calls: List[Dict[str, Any]] = []

    async def reason_json(self, system, context, schema_hint=None):
        self.calls.append({"system": system, "context": context, "schema_hint": schema_hint})
        if self.fail:
            raise RuntimeError("LLM unavailable (fake)")
        return self.response


def _experiment(eid="exp1", sid="s1", verdict="FAIL", pf=0.7613, sharpe=-0.8133):
    """Shape mirrors ``experiments.get_experiment`` (flat metric dicts)."""
    return {
        "id": eid,
        "strategy_id": sid,
        "verdict": verdict,
        "recommendation": "No variant improved over baseline. Do not adopt changes.",
        "parameter_changes": {"atr_sl_mult": 1.5},
        "baseline_metrics": {"profit_factor": pf, "sharpe_ratio": sharpe},
        "variant_metrics": {"profit_factor": pf},
    }


def _pack() -> Dict[str, rg.EvidenceItem]:
    return rg.build_evidence_pack([_experiment()], [], [], {})


def _narrative() -> Dict[str, Any]:
    return {
        "title": "September research review",
        "summary": "One experiment was run; its verdict was FAIL.",
        "sections": [
            {
                "heading": "Experiments",
                "narrative": (
                    "Experiment exp1 on strategy s1 ended with verdict FAIL: the "
                    "variant's profit_factor 0.76 and baseline sharpe_ratio -0.81 "
                    "show no adoptable edge."
                ),
                "evidence_ids": ["exp:exp1"],
            }
        ],
    }


def _run(llm=None, config=None, pack=None, session_factory=None):
    return rg.generate_research_report(
        config or rg.ReportConfig(topic="t"),
        evidence_loader=lambda cfg: pack if pack is not None else _pack(),
        llm=llm,
        session_factory=session_factory,
    )


# ---------------------------------------------------------------------------
# Evidence pack (the citable universe)
# ---------------------------------------------------------------------------


def test_evidence_pack_trims_to_citable_facts():
    pack = _pack()
    assert "exp:exp1" in pack
    item = pack["exp:exp1"]
    assert item.kind == "experiment"
    assert item.metrics["baseline_profit_factor"] == pytest.approx(0.7613)
    assert item.metrics["variant_profit_factor"] == pytest.approx(0.7613)
    assert item.facts["verdict"] == "FAIL"
    assert item.facts["parameter_changes"] == {"atr_sl_mult": 1.5}


def test_evidence_catalog_is_json_safe():
    catalog = rg.render_evidence_catalog(_pack())
    json.loads(json.dumps(catalog))
    assert catalog[0]["id"] == "exp:exp1"


def test_empty_records_are_dropped():
    pack = rg.build_evidence_pack([{"id": "", "verdict": "FAIL"}], [], [], {"s9": {}})
    assert pack == {}


# ---------------------------------------------------------------------------
# Traceability — the heart of C3
# ---------------------------------------------------------------------------


def test_trace_accepts_numbers_from_cited_evidence():
    item = _pack()["exp:exp1"]
    ok, issues = rg.trace_narrative(
        "profit_factor 0.76 and sharpe_ratio -0.81 both failed", [item]
    )
    assert ok, issues


def test_trace_rejects_invented_numbers():
    item = _pack()["exp:exp1"]
    ok, issues = rg.trace_narrative(
        "the win_rate was 0.63 across 1410 trades", [item]
    )
    assert not ok
    assert "0.63" in issues and "1410" in issues


def test_trace_ignores_small_ints_and_years():
    item = _pack()["exp:exp1"]
    ok, issues = rg.trace_narrative(
        "3 conditions were tested in the 2025 window on 4 pairs", [item]
    )
    assert ok, issues


def test_trace_prefix_matching():
    item = _pack()["exp:exp1"]
    ok, _ = rg.trace_narrative("profit_factor 0.7613 collapsed", [item])
    assert ok


# ---------------------------------------------------------------------------
# Document assembly: untraceable sections replaced
# ---------------------------------------------------------------------------


def test_untraceable_llm_section_replaced_by_fallback():
    narrative = {
        "title": "t",
        "summary": "s",
        "sections": [
            {
                "heading": "Experiments",
                "narrative": "The strategy shows win_rate 0.63 over 1410 trades — clearly promising.",
                "evidence_ids": ["exp:exp1"],
            }
        ],
    }
    doc = rg.build_report_document(_pack(), narrative, True, None, rg.ReportConfig(topic="t"))
    sec = doc["sections"][0]
    assert sec["source"] == "fallback"
    assert sec["trace_issues"], "untraced numbers must be recorded"
    assert doc["traceability"]["sections_replaced"] == 1


def test_traceable_llm_section_kept_and_marked_llm():
    doc = rg.build_report_document(_pack(), _narrative(), True, None, rg.ReportConfig(topic="t"))
    sec = doc["sections"][0]
    assert sec["source"] == "llm"
    assert sec["trace_issues"] == []
    assert doc["traceability"]["sections_replaced"] == 0


def test_unknown_evidence_ids_fall_to_fallback():
    narrative = _narrative()
    narrative["sections"][0]["evidence_ids"] = ["exp:does-not-exist"]
    doc = rg.build_report_document(_pack(), narrative, True, None, rg.ReportConfig(topic="t"))
    assert doc["sections"][0]["source"] == "fallback"
    assert doc["traceability"]["discarded_sections"], "unresolvable citations must be recorded"


def test_prefix_citations_resolve_to_pack_ids():
    # Live ledger ids can be long UUIDs; the LLM cites the short form.
    long_id = "task:20fcd565-7400-4251-948b-fe3948526f22"
    pack = rg.build_evidence_pack([], [{"id": long_id[5:], "topic": "T", "status": "COMPLETED"}], [], {})
    narrative = {
        "title": "t",
        "summary": "s",
        "sections": [
            {
                "heading": "Tasks",
                "narrative": "Research task 20fcd565 completed without any invented numbers here.",
                "evidence_ids": ["task:20fcd565"],
            }
        ],
    }
    doc = rg.build_report_document(pack, narrative, True, None, rg.ReportConfig(topic="t"))
    sec = next(s for s in doc["sections"] if s["heading"] == "Tasks")
    assert sec["source"] == "llm"
    assert sec["evidence_ids"] == [long_id]


def test_mixed_kind_sections_assigned_to_first_matching_kind_once():
    pack = rg.build_evidence_pack([_experiment()], [], [], {})
    narrative = {
        "title": "t",
        "summary": "s",
        "sections": [
            {
                "heading": "Experiments and tasks",
                "narrative": "Everything at once, traced against cited evidence values 0.76.",
                "evidence_ids": ["exp:exp1"],
            }
        ],
    }
    doc = rg.build_report_document(pack, narrative, True, None, rg.ReportConfig(topic="t"))
    # The section lands on the experiment kind (its only resolved kind) once;
    # no double-counting, remaining kinds fall back deterministically.
    exp_secs = [s for s in doc["sections"] if s["heading"] == "Experiments and tasks"]
    assert len(exp_secs) == 1 and exp_secs[0]["source"] == "llm"


def test_ledger_is_rendered_from_records_not_llm():
    doc = rg.build_report_document(_pack(), _narrative(), True, None, rg.ReportConfig(topic="t"))
    assert doc["evidence"][0]["metrics"]["baseline_profit_factor"] == pytest.approx(0.7613)
    assert doc["evidence"][0]["id"] == "exp:exp1"


def test_fallback_document_covers_all_kinds():
    pack = rg.build_evidence_pack([_experiment()], [], [], {})
    doc = rg.build_report_document(pack, {}, False, "boom", rg.ReportConfig(topic="t"))
    assert doc["llm_used"] is False
    assert doc["llm_error"] == "boom"
    assert all(s["source"] == "fallback" for s in doc["sections"])


def test_render_report_text_is_deterministic():
    doc = rg.build_report_document(_pack(), _narrative(), True, None, rg.ReportConfig(topic="t"))
    assert rg.render_report_text(doc) == rg.render_report_text(doc)


# ---------------------------------------------------------------------------
# One run: orchestration + persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_report_persists_real_row(session_factory):
    doc = await _run(llm=_FakeLLM(_narrative()), session_factory=session_factory)
    assert doc["llm_used"] is True
    assert doc["report_id"]
    saved = await rg.get_research_report(doc["report_id"], session_factory=session_factory)
    assert saved["title"] == "September research review"
    assert saved["sections"][0]["source"] == "llm"


@pytest.mark.asyncio
async def test_llm_failure_falls_back_and_persists(session_factory):
    doc = await _run(llm=_FakeLLM(fail=True), session_factory=session_factory)
    assert doc["llm_used"] is False
    assert "RuntimeError" in doc["llm_error"]
    assert all(s["source"] == "fallback" for s in doc["sections"])
    saved = await rg.get_research_report(doc["report_id"], session_factory=session_factory)
    assert saved["report_id"] == doc["report_id"]


@pytest.mark.asyncio
async def test_use_llm_false_skips_llm(session_factory):
    llm = _FakeLLM()

    async def boom(*a, **k):
        raise AssertionError("LLM must not be called")

    llm.reason_json = boom
    doc = await _run(
        llm=llm,
        config=rg.ReportConfig(topic="t", use_llm=False),
        session_factory=session_factory,
    )
    assert doc["llm_used"] is False


@pytest.mark.asyncio
async def test_garbage_llm_shape_falls_back(session_factory):
    doc = await _run(llm=_FakeLLM({"nope": 1}), session_factory=session_factory)
    assert doc["llm_used"] is False
    assert "sections" in doc["llm_error"]


@pytest.mark.asyncio
async def test_persist_false_writes_nothing(session_factory):
    doc = await _run(
        llm=_FakeLLM(_narrative()),
        config=rg.ReportConfig(topic="t", persist=False),
        session_factory=session_factory,
    )
    assert "report_id" not in doc
    assert await rg.list_research_reports(session_factory=session_factory) == []


@pytest.mark.asyncio
async def test_no_evidence_raises_valueerror():
    with pytest.raises(ValueError):
        await _run(pack={})


@pytest.mark.asyncio
async def test_report_index_lists_summaries(session_factory):
    await _run(llm=_FakeLLM(_narrative()), session_factory=session_factory)
    reports = await rg.list_research_reports(session_factory=session_factory)
    assert len(reports) == 1
    assert reports[0]["evidence_count"] == 1
    assert reports[0]["llm_used"] is True


# ---------------------------------------------------------------------------
# Constitution + JSON safety
# ---------------------------------------------------------------------------


def test_hard_blocks_by_construction_source_scan():
    src = inspect.getsource(rg)
    for banned in ("place_order", "execute_trade", "risk_manager", "MAX_PERMISSION_LEVEL = 4"):
        assert banned not in src, banned


def test_document_is_json_safe():
    doc = rg.build_report_document(_pack(), _narrative(), True, None, rg.ReportConfig(topic="t"))
    json.loads(json.dumps(doc, default=str))
