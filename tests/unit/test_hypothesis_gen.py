"""Stage-C1 unit tests: LLM hypothesis generation with deterministic validation.

Pins the Stage C constitution on the FIRST LLM-chooses-what-to-research path:
the LLM only sees deterministic briefs and only proposes candidates; code
validates every one; only validated candidates reach the B4 Director queue;
the backtest gate owns every outcome. LLM and structure fetcher are faked via
injection seams — the tests pin the ORCHESTRATION, not the network.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from forex_ai_agent.agent import director as dr
from forex_ai_agent.agent import hypothesis_gen as hg
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


def _brief_for(pair: str, trend: str = "up") -> Dict[str, Any]:
    return hg.build_brief(
        {
            "symbol": pair,
            "timeframe": "H1",
            "current_price": 1.1,
            "trend": trend,
            "key_support": 1.09,
            "key_resistance": 1.12,
            "last_bos": None,
            "last_choch": None,
            "consolidation": {"top": 1.11, "bottom": 1.09},
            "liquidity_sweeps": [],
        }
    )


def _make_llm(candidates: List[Dict[str, Any]], fail: bool = False):
    class _FakeLLM:
        def __init__(self):
            self.calls = 0

        async def reason_json(self, system, context, schema_hint=""):
            self.calls += 1
            if fail:
                raise RuntimeError("simulated LLM outage")
            return {"hypotheses": candidates}

    return _FakeLLM()


async def _run(sf, llm=None, pairs=None, **kw):
    cfg = hg.HypothesisGenConfig(
        pairs=pairs or ["EUR/USD"],
        max_hypotheses=kw.pop("max_hypotheses", 3),
        **{k: v for k, v in kw.items() if k in ("use_llm", "enqueue", "min_confidence", "timeframe", "priority")},
    )

    async def fetcher(pair: str, timeframe: str) -> Dict[str, Any]:
        return _brief_for(pair)

    return await hg.generate_hypotheses(
        cfg,
        session_factory=sf,
        structure_fetcher=fetcher,
        llm=llm,
    )


# ---------------------------------------------------------------------------
# Briefs
# ---------------------------------------------------------------------------


def test_build_brief_is_deterministic_and_trimmed():
    b1 = _brief_for("EUR/USD")
    b2 = _brief_for("EUR/USD")
    assert b1 == b2
    assert set(b1) <= {
        "symbol", "timeframe", "current_price", "trend", "key_support",
        "key_resistance", "last_bos", "last_choch", "consolidation", "recent_sweeps",
    }


def test_fallback_hypotheses_cover_all_trend_shapes():
    out = hg._fallback_hypotheses(
        [
            _brief_for("EUR/USD", "up"),
            _brief_for("GBP/USD", "down"),
            _brief_for("USD/JPY", "sideways"),
        ]
    )
    assert len(out) == 3
    assert all(h["source"] == "fallback" and h["confidence"] is None for h in out)
    assert "support" in out[0]["text"] and "resistance" in out[1]["text"]
    assert "range" in out[2]["text"].lower()


def test_normalize_pair():
    assert hg._normalize_pair("eurusd") == "EUR/USD"
    assert hg._normalize_pair("gbp-usd") == "GBP/USD"
    assert hg._normalize_pair("nonsense") is None
    assert hg._normalize_pair(None) is None


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_candidates_validated_and_enqueued(session_factory):
    llm = _make_llm(
        [
            {"pair": "EUR/USD", "text": "Displacement continuation after sweeps of session highs.", "confidence": 0.9},
            {"pair": "gbpjpy", "text": "Range fade at consolidation edges.", "confidence": 0.7},
        ]
    )
    report = await _run(
        sf=session_factory, llm=llm, pairs=["EUR/USD", "GBP/JPY"], max_hypotheses=2
    )
    assert report["llm_used"] is True and report["fallback_used"] is False
    assert len(report["accepted"]) == 2 and report["enqueued_count"] == 2
    assert report["accepted"][0]["pair"] == "EUR/USD"  # confidence 0.9 sorts first
    assert report["accepted"][0]["source"] == "llm"
    snap = await dr.queue_snapshot(session_factory=session_factory)
    assert snap["open_items"] == 2


@pytest.mark.asyncio
async def test_validation_rejects_out_of_universe_and_bad_confidence(session_factory):
    llm = _make_llm(
        [
            {"pair": "AUD/USD", "text": "Not in this run's universe.", "confidence": 0.9},
            {"pair": "EUR/USD", "text": "Confidence above 1 is nonsense.", "confidence": 1.5},
            {"pair": "EUR/USD", "text": "Confidence below threshold.", "confidence": 0.2},
            {"pair": "EUR/USD", "text": "too short"},
            "not even a dict",
            {"pair": "EUR/USD", "text": "Valid hypothesis worth testing now.", "confidence": 0.8},
        ]
    )
    report = await _run(sf=session_factory, llm=llm)
    assert len(report["accepted"]) == 1
    reasons = {r["reason"] for r in report["rejected"]}
    assert any("not in this run's universe" in r for r in reasons)
    assert any("outside 0..1" in r for r in reasons)
    assert any("below threshold" in r for r in reasons)
    assert any("too short" in r for r in reasons)
    assert any("not a JSON object" in r for r in reasons)


@pytest.mark.asyncio
async def test_llm_failure_falls_back_deterministically(session_factory):
    llm = _make_llm([], fail=True)
    report = await _run(sf=session_factory, llm=llm)
    assert report["llm_used"] is False
    assert report["fallback_used"] is True
    assert report["llm_error"] and "simulated LLM outage" in report["llm_error"]
    assert len(report["accepted"]) == 1
    assert report["accepted"][0]["source"] == "fallback"
    assert report["accepted"][0]["enqueued"] is True  # fallback still useful


@pytest.mark.asyncio
async def test_use_llm_false_skips_llm_entirely(session_factory):
    llm = _make_llm([])
    report = await _run(sf=session_factory, llm=llm, use_llm=False)
    assert llm.calls == 0
    assert report["fallback_used"] is True and report["llm_used"] is False


@pytest.mark.asyncio
async def test_propose_only_mode_touches_nothing(session_factory):
    llm = _make_llm(
        [{"pair": "EUR/USD", "text": "Propose but do not enqueue this one.", "confidence": 0.9}]
    )
    report = await _run(sf=session_factory, llm=llm, enqueue=False)
    assert len(report["accepted"]) == 1
    assert report["accepted"][0]["enqueued"] is False
    assert report["enqueued_count"] == 0
    snap = await dr.queue_snapshot(session_factory=session_factory)
    assert snap["open_items"] == 0


@pytest.mark.asyncio
async def test_dead_pair_brief_rejection_details(session_factory):
    async def broken_fetcher(pair, timeframe):
        if pair == "EUR/USD":
            raise RuntimeError("feed down")
        return _brief_for(pair)

    cfg = hg.HypothesisGenConfig(pairs=["EUR/USD", "GBP/USD"])
    report = await hg.generate_hypotheses(
        cfg,
        session_factory=session_factory,
        structure_fetcher=broken_fetcher,
        llm=_make_llm([{"pair": "GBP/USD", "text": "Valid hypothesis for the live pair.", "confidence": 0.8}]),
    )
    assert {"pair": "EUR/USD", "reason": "brief unavailable: RuntimeError: feed down"} in report["rejected"]
    assert [h["pair"] for h in report["accepted"]] == ["GBP/USD"]


@pytest.mark.asyncio
async def test_no_valid_pairs_raises_valueerror(session_factory):
    with pytest.raises(ValueError):
        await hg.generate_hypotheses(
            hg.HypothesisGenConfig(pairs=["garbage"]),
            session_factory=session_factory,
            structure_fetcher=_brief_for,  # never called — pairs rejected first
            llm=_make_llm([]),
        )


@pytest.mark.asyncio
async def test_llm_garbage_shape_falls_back(session_factory):
    class _GarbageLLM:
        async def reason_json(self, *a, **k):
            return {"oops": "wrong shape"}

    report = await _run(sf=session_factory, llm=_GarbageLLM())
    assert report["llm_used"] is False
    assert report["fallback_used"] is True
    assert "hypotheses" in (report["llm_error"] or "")


# ---------------------------------------------------------------------------
# Director integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueued_hypothesis_is_real_queue_row(session_factory):
    llm = _make_llm(
        [{"pair": "EUR/USD", "text": "This becomes a real Director row.", "confidence": 0.85}]
    )
    report = await _run(sf=session_factory, llm=llm)
    item_id = report["accepted"][0]["item_id"]
    snap = await dr.queue_snapshot(session_factory=session_factory)
    row = next(i for i in snap["queue"] if i["id"] == item_id)
    assert row["pair"] == "EUR/USD"
    assert "This becomes a real Director row." in row["hypothesis"]


# ---------------------------------------------------------------------------
# Constitution: hard blocks by construction (B4-style source scan)
# ---------------------------------------------------------------------------


def test_hard_blocks_by_construction_source_scan():
    src = inspect.getsource(hg)
    for banned in ("place_order", "risk_manager", "execute_trade", "MAX_PERMISSION_LEVEL = 4"):
        assert banned not in src, banned


def test_report_is_json_safe():
    report = hg.HypothesisGenReport(
        generated_at="2026-09-25T00:00:00",
        timeframe="H1",
        pairs_scanned=["EUR/USD"],
        llm_used=True,
        accepted=[{"pair": "EUR/USD", "text": "x", "confidence": 0.9}],
    )
    import json

    json.loads(json.dumps(report.to_dict()))
