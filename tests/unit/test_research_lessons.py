"""Learning-loop unit tests: deterministic gate-rejection lessons feeding C1.

Pins: the miner aggregates stored rejected runs into stable lessons (counts,
verbatim reasons, spec shapes) with junk tolerance; C1 carries lessons into
its context as FACTS only, with lessons_used reported, and any loader failure
degrades to zero lessons without breaking the run. The gate itself is
untouched — lessons bias WHAT is proposed, never WHETHER it is accepted.

Positive lessons extend the loop to gate-PASSED (validated) runs: mined into
times_passed + spec shape + produced version ids, rendered as past_successes
facts, and reported via successes_used — leaning C1 toward shapes that
worked, still without touching the gate.
"""

from __future__ import annotations

import inspect
import json
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from forex_ai_agent.agent import hypothesis_gen as hg
from forex_ai_agent.agent import research_lessons as rl
from forex_ai_agent.data.models.research import ResearchTask


# ---------------------------------------------------------------------------
# Miner
# ---------------------------------------------------------------------------


def _row(verdict="rejected", reasons=None, spec=None, conclusion=None):
    if conclusion is None:
        conclusion = {
            "verdict": verdict,
            "validation": {"reasons": reasons or []},
            "spec": spec or {},
        }
    return SimpleNamespace(conclusion_json=conclusion)


def test_mine_groups_by_reason_class_and_spec_shape():
    rows = [
        _row(reasons=["spec invalid: name: Field required"],
             spec={"direction": "long", "timeframe": "H1",
                   "entry_conditions": [{"verb": "above"}]}),
        _row(reasons=["spec invalid: entry_conditions: Field required"],
             spec={"direction": "long", "timeframe": "H1",
                   "entry_conditions": [{"verb": "above"}]}),
        _row(reasons=["spec invalid: name: Field required"],
             spec={"direction": "short", "timeframe": "H4", "entry_conditions": []}),
    ]
    lessons = rl.mine_gate_lessons(rows)
    assert len(lessons) == 2
    assert lessons[0].count == 2  # grouped: same shape, bucket, verbs
    assert lessons[0].reason_bucket == "spec_invalid"
    assert lessons[0].direction == "long"
    assert lessons[0].entry_verbs == ["above"]
    # verbatim reason samples are kept
    assert "name: Field required" in lessons[0].sample_reason or \
           "entry_conditions: Field required" in lessons[0].sample_reason


def test_mine_ignores_non_rejected_and_reasonless_rows():
    rows = [
        _row(verdict="validated", reasons=["too good"], spec={"direction": "long"}),
        _row(verdict="rejected", reasons=[], spec={}),
        _row(conclusion={"verdict": "rejected"}),  # no validation block at all
        SimpleNamespace(conclusion_json=None),
        SimpleNamespace(conclusion_json="junk"),
    ]
    assert rl.mine_gate_lessons(rows) == []


# ---------------------------------------------------------------------------
# Miner — positive (gate-passed) lessons
# ---------------------------------------------------------------------------


def _pass_row(spec=None, version_id=None):
    conclusion = {
        "verdict": "validated",
        "validation": {"passed": True, "reasons": []},
        "spec": spec or {},
    }
    if version_id is not None:
        conclusion["version_id"] = version_id
    return SimpleNamespace(conclusion_json=conclusion)


def test_mine_success_groups_by_spec_shape_and_counts_runs():
    shape = {"direction": "long", "timeframe": "H1",
             "entry_conditions": [{"verb": "crosses_above"}]}
    rows = [
        _pass_row(spec=shape, version_id="aaaaaaaa"),
        _pass_row(spec=shape, version_id="bbbbbbbb"),
        _pass_row(spec={"direction": "short", "timeframe": "H4", "entry_conditions": []}),
    ]
    lessons = rl.mine_success_lessons(rows)
    assert len(lessons) == 2
    assert lessons[0].times_passed == 2  # grouped by shape, one count per run
    assert lessons[0].direction == "long"
    assert lessons[0].entry_verbs == ["crosses_above"]
    assert lessons[0].version_ids == ["aaaaaaaa", "bbbbbbbb"]


def test_mine_success_ignores_non_validated_and_junk_rows():
    rows = [
        _row(verdict="rejected", reasons=["spec invalid: x"], spec={}),
        _row(verdict="pending", spec={}),
        SimpleNamespace(conclusion_json=None),
        SimpleNamespace(conclusion_json="junk"),
    ]
    assert rl.mine_success_lessons(rows) == []


def test_mine_success_is_deterministic_and_capped():
    rows = [
        _pass_row(spec={"direction": "long", "timeframe": "H1", "entry_conditions": []})
        for _ in range(12)
    ]
    a = rl.mine_success_lessons(rows)
    b = rl.mine_success_lessons(list(reversed(rows)))
    assert a == b
    assert a[0].times_passed == 12
    assert len(a) <= rl._MAX_LESSONS


def test_mine_success_dedupes_and_caps_version_ids():
    shape = {"direction": "long", "timeframe": "H1", "entry_conditions": []}
    rows = [
        _pass_row(spec=shape, version_id="aaaaaaaa"),
        _pass_row(spec=shape, version_id="aaaaaaaa"),
        _pass_row(spec=shape, version_id="bbbbbbbb"),
        _pass_row(spec=shape, version_id="cccccccc"),
    ]
    lesson = rl.mine_success_lessons(rows)[0]
    assert lesson.version_ids == ["aaaaaaaa", "bbbbbbbb", "cccccccc"]  # dedup + cap at 3


def test_render_successes_contains_only_stored_values():
    rows = [_pass_row(
        spec={"direction": "long", "timeframe": "H1", "entry_conditions": []},
        version_id="aaaaaaaa",
    )]
    rendered = rl.render_successes_for_context(rl.mine_success_lessons(rows))
    json.loads(json.dumps(rendered))  # JSON-safe
    lesson = rendered["lessons"][0]
    assert lesson["times_passed"] == 1
    assert lesson["version_ids"] == ["aaaaaaaa"]
    assert lesson["spec_shape"]["direction"] == "long"
    assert "VALIDATED" in rendered["note"]


async def test_combined_loader_returns_both_polarities_from_one_scan(session_factory):
    factory = session_factory
    from forex_ai_agent.autonomy.research_log import _get_or_create_alex_user

    async with factory() as session:
        user = await _get_or_create_alex_user(session)
        session.add(ResearchTask(
            id=uuid.uuid4(), user_id=user.id, concept_name="seed rejected",
            questions_json=["seed"],
            status="COMPLETED", current_step="COMPLETED", progress_percent=100,
            conclusion_json={"verdict": "rejected", "validation":
                             {"reasons": ["spec invalid: name: Field required"]},
                             "spec": {"direction": "long", "timeframe": "H1",
                                      "entry_conditions": []}},
        ))
        session.add(ResearchTask(
            id=uuid.uuid4(), user_id=user.id, concept_name="seed validated",
            questions_json=["seed"],
            status="COMPLETED", current_step="COMPLETED", progress_percent=100,
            conclusion_json={"verdict": "validated", "validation": {"passed": True},
                             "spec": {"direction": "short", "timeframe": "H4",
                                      "entry_conditions": []},
                             "version_id": "deadbeef"},
        ))
        await session.commit()

    rejections, successes = await rl.load_run_lessons(session_factory=factory)
    assert len(rejections) == 1 and rejections[0].reason_bucket == "spec_invalid"
    assert len(successes) == 1 and successes[0].times_passed == 1
    assert successes[0].version_ids == ["deadbeef"]
    # the back-compat wrapper still yields rejections only
    assert len(await rl.load_rejected_run_lessons(session_factory=factory)) == 1


def test_mine_is_deterministic_and_capped():
    rows = [
        _row(reasons=["gate threshold not met"],
             spec={"direction": "long", "timeframe": "H1", "entry_conditions": []})
        for i in range(12)
    ]
    a = rl.mine_gate_lessons(rows)
    b = rl.mine_gate_lessons(list(reversed(rows)))
    assert a == b  # order-independent: aggregation, not iteration
    assert a[0].count == 12
    assert len(a) <= rl._MAX_LESSONS


def test_render_contains_only_stored_values():
    rows = [_row(reasons=["spec invalid: name: Field required"],
                 spec={"direction": "long", "timeframe": "H1", "entry_conditions": []})]
    rendered = rl.render_lessons_for_context(rl.mine_gate_lessons(rows))
    json.loads(json.dumps(rendered))
    lesson = rendered["lessons"][0]
    assert lesson["times_rejected"] == 1
    assert lesson["reason_sample"] == "spec invalid: name: Field required"
    assert lesson["spec_shape"]["direction"] == "long"


async def test_loader_never_raises_on_broken_factory():
    async def broken_factory():
        raise RuntimeError("db down")

    lessons = await rl.load_rejected_run_lessons(session_factory=broken_factory)
    assert lessons == []


# ---------------------------------------------------------------------------
# C1 wiring
# ---------------------------------------------------------------------------


def _brief_for(pair: str, trend: str = "up") -> Dict[str, Any]:
    return hg.build_brief(
        {
            "symbol": pair,
            "timeframe": "H1",
            "current_price": 1.1,
            "trend": trend,
            "key_support": 1.09,
            "key_resistance": 1.11,
            "last_bos": None,
            "last_choch": None,
            "consolidation": None,
            "liquidity_sweeps": [],
        }
    )


async def _brief_fetch(pair: str, tf: str) -> Dict[str, Any]:
    """C1's structure_fetcher seam is awaited directly — provide it async."""
    return _brief_for(pair)


def _success_lesson():
    return rl.SuccessLesson(times_passed=2, direction="long", timeframe="H1",
                            entry_verbs=["crosses_above"], version_ids=["aaaaaaaa"])


class _RecordingLLM:
    def __init__(self, response):
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    async def reason_json(self, system, context, schema_hint=None):
        self.calls.append({"system": system, "context": context})
        return self.response


@pytest.fixture()
async def session_factory():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool

    from forex_ai_agent.data.models.base import Base as ModelsBase

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(ModelsBase.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.mark.asyncio
async def test_c1_carries_lessons_into_context(session_factory):
    llm = _RecordingLLM({"hypotheses": []})
    report = await hg.generate_hypotheses(
        hg.HypothesisGenConfig(pairs=["EUR/USD"], use_llm=True, enqueue=False),
        session_factory=session_factory,
        structure_fetcher=_brief_fetch,
        llm=llm,
        lessons_loader=lambda session_factory=None: [
            rl.GateLesson(count=3, reason_bucket="spec_invalid",
                          sample_reason="spec invalid: name: Field required",
                          direction="long", timeframe="H1", entry_verbs=["above"])
        ],
    )
    assert report["lessons_used"] == 1
    ctx = llm.calls[0]["context"]
    assert "past_rejections" in ctx
    lesson = ctx["past_rejections"]["lessons"][0]
    assert lesson["times_rejected"] == 3
    assert "unlikely to reproduce" in ctx["past_rejections"]["note"]
    # the system prompt tells the LLM lessons are stored facts
    assert "past-rejection lessons" in llm.calls[0]["system"]


@pytest.mark.asyncio
async def test_c1_carries_successes_into_context(session_factory):
    llm = _RecordingLLM({"hypotheses": []})
    report = await hg.generate_hypotheses(
        hg.HypothesisGenConfig(pairs=["EUR/USD"], use_llm=True, enqueue=False),
        session_factory=session_factory,
        structure_fetcher=_brief_fetch,
        llm=llm,
        lessons_loader=lambda session_factory=None: ([], [_success_lesson()]),
    )
    assert report["lessons_used"] == 0
    assert report["successes_used"] == 1  # one success lesson (times_passed=2)
    ctx = llm.calls[0]["context"]
    assert "past_rejections" not in ctx
    assert "past_successes" in ctx
    lesson = ctx["past_successes"]["lessons"][0]
    assert lesson["times_passed"] == 2
    assert lesson["version_ids"] == ["aaaaaaaa"]
    assert "Lean toward" in ctx["past_successes"]["note"]
    assert "past-success lessons" in llm.calls[0]["system"]


@pytest.mark.asyncio
async def test_c1_without_lessons_has_no_context_key(session_factory):
    llm = _RecordingLLM({"hypotheses": []})
    report = await hg.generate_hypotheses(
        hg.HypothesisGenConfig(pairs=["EUR/USD"], use_llm=True, enqueue=False),
        session_factory=session_factory,
        structure_fetcher=_brief_fetch,
        llm=llm,
        lessons_loader=lambda session_factory=None: ([], []),
    )
    assert report["lessons_used"] == 0
    assert report["successes_used"] == 0
    assert "past_rejections" not in llm.calls[0]["context"]
    assert "past_successes" not in llm.calls[0]["context"]


@pytest.mark.asyncio
async def test_c1_survives_exploding_lessons_loader(session_factory):
    llm = _RecordingLLM({"hypotheses": []})

    async def boom(session_factory=None):
        raise RuntimeError("miner down")

    report = await hg.generate_hypotheses(
        hg.HypothesisGenConfig(pairs=["EUR/USD"], use_llm=True, enqueue=False),
        session_factory=session_factory,
        structure_fetcher=_brief_fetch,
        llm=llm,
        lessons_loader=boom,
    )
    assert report["lessons_used"] == 0
    assert report["llm_used"] is True  # the run itself was unaffected


@pytest.mark.asyncio
async def test_lessons_do_not_bypass_validation(session_factory):
    llm = _RecordingLLM(
        {
            "hypotheses": [
                {"pair": "USD/CHF", "text": "Out-of-universe idea with lessons present.",
                 "confidence": 0.9}
            ]
        }
    )
    report = await hg.generate_hypotheses(
        hg.HypothesisGenConfig(pairs=["EUR/USD"], use_llm=True, enqueue=False),
        session_factory=session_factory,
        structure_fetcher=_brief_fetch,
        llm=llm,
        lessons_loader=lambda session_factory=None: [
            rl.GateLesson(count=1, reason_bucket="gate_threshold",
                          sample_reason="oos sharpe below threshold",
                          direction="long", timeframe="H1", entry_verbs=[])
        ],
    )
    # the LLM saw lessons, but validation still rejects the out-of-universe pick
    assert report["lessons_used"] == 1
    assert report["accepted"] == []
    assert report["rejected"] and "universe" in report["rejected"][0]["reason"]


@pytest.mark.asyncio
async def test_successes_do_not_bypass_validation(session_factory):
    llm = _RecordingLLM(
        {"hypotheses": [
            {"pair": "USD/JPY", "text": "Out-of-universe idea with successes present.",
             "confidence": 0.9}
        ]}
    )
    report = await hg.generate_hypotheses(
        hg.HypothesisGenConfig(pairs=["EUR/USD"], use_llm=True, enqueue=False),
        session_factory=session_factory,
        structure_fetcher=_brief_fetch,
        llm=llm,
        lessons_loader=lambda session_factory=None: ([], [_success_lesson()]),
    )
    # the LLM saw past successes, but validation still polices every candidate
    assert report["successes_used"] == 1
    assert report["accepted"] == []
    assert report["rejected"] and "universe" in report["rejected"][0]["reason"]


# ---------------------------------------------------------------------------
# Stage C4 — weak-shape policy (deterministic decisions over stored verdicts)
# ---------------------------------------------------------------------------


def _reject_conclusion(direction="both", timeframe="H1", verbs=("above",),
                       reason="profit factor 0.71 below floor 1.00", verdict="rejected"):
    return {
        "verdict": verdict,
        "validation": {"passed": False, "reasons": [reason]},
        "spec": {
            "direction": direction,
            "timeframe": timeframe,
            "entry_conditions": [{"left": "price", "verb": v, "right": "sma"} for v in verbs],
        },
    }


def test_spec_shape_is_the_shared_key():
    spec = {"direction": "Long", "timeframe": "h1",
            "entry_conditions": [{"verb": "above"}, {"verb": "crosses_above"}, {"verb": "above"}]}
    assert rl.spec_shape(spec) == ("long", "H1", ["above", "crosses_above"])


def test_weak_shapes_threshold_and_determinism():
    concl = [_reject_conclusion() for _ in range(3)] + [
        _reject_conclusion(direction="short", verbs=("below",)) for _ in range(2)
    ]
    shapes = rl.weak_shapes(concl, min_attempts=3)
    assert len(shapes) == 1
    assert shapes[0].count == 3
    assert shapes[0].direction == "both" and shapes[0].timeframe == "H1"
    assert "profit factor 0.71" in shapes[0].sample_reason
    # order-independent + below-threshold shapes excluded
    assert rl.weak_shapes(list(reversed(concl)), min_attempts=3) == shapes
    assert rl.weak_shapes(concl, min_attempts=4) == []


def test_veto_reason_is_honest_and_verbatim_styled():
    lesson = rl.GateLesson(count=3, reason_bucket="gate_threshold",
                           sample_reason="profit factor 0.71 below floor 1.00",
                           direction="both", timeframe="H1", entry_verbs=["above"])
    reason = rl.veto_reason(lesson)
    assert reason.startswith("weak-hypothesis veto:")
    assert "rejected 3x" in reason and "profit factor 0.71 below floor 1.00" in reason


def test_shape_nudges_rank_and_bounded():
    lesson = rl.GateLesson(count=5, reason_bucket="gate_threshold",
                           sample_reason="too few trades: 0 < 20",
                           direction="both", timeframe="H1", entry_verbs=["above"])
    spec = {"direction": "both", "timeframe": "H1", "entry_conditions": [{"verb": "above"}]}
    nudges = rl.shape_nudges(spec, [lesson])
    assert len(nudges) == 1 and "resembles weak shape" in nudges[0]
    unrelated = {"direction": "short", "timeframe": "H4", "entry_conditions": [{"verb": "falling"}]}
    assert rl.shape_nudges(unrelated, [lesson]) == []


@pytest.mark.asyncio
async def test_load_rejected_conclusions_filters_verdicts(session_factory):
    from forex_ai_agent.autonomy.research_log import _get_or_create_alex_user

    async with session_factory() as session:
        user = await _get_or_create_alex_user(session)
        session.add(ResearchTask(
            id=uuid.uuid4(), user_id=user.id, concept_name="rej",
            questions_json=["seed"], status="COMPLETED", current_step="COMPLETED",
            progress_percent=100,
            conclusion_json=_reject_conclusion(),
        ))
        session.add(ResearchTask(
            id=uuid.uuid4(), user_id=user.id, concept_name="val",
            questions_json=["seed"], status="COMPLETED", current_step="COMPLETED",
            progress_percent=100,
            conclusion_json=_reject_conclusion(direction="long", verdict="validated"),
        ))
        await session.commit()

    concl = await rl.load_rejected_conclusions(session_factory=session_factory)
    assert len(concl) == 1 and concl[0]["verdict"] == "rejected"


@pytest.mark.asyncio
async def test_load_rejected_conclusions_never_raises(session_factory):
    async def broken_factory():
        raise RuntimeError("db down")

    assert await rl.load_rejected_conclusions(session_factory=broken_factory) == []


# ---------------------------------------------------------------------------
# Stage C4 — the weak-hypothesis veto in the proposal loop
# ---------------------------------------------------------------------------


async def _async_return(*_a, **_k):
    return None


async def _fake_brief(*_a, **_k):
    return SimpleNamespace(to_prompt_dict=lambda: {})


class _FakeValidation:
    def __init__(self, passed=True, reasons=None):
        self.passed = passed
        self.reasons = reasons or []
        self.spec_name = "S"
        self.full_backtest = {}
        self.walk_forward = {}
        self.data_range = {}
        self.repro = {}

    def to_dict(self):
        return {"passed": self.passed, "reasons": self.reasons}


@pytest.mark.asyncio
async def test_proposal_veto_skips_gate_on_weak_shape(monkeypatch):
    from forex_ai_agent.autonomy import proposal as prop

    async def _loader(**_kw):
        return [
            {"verdict": "rejected", "validation": {"passed": False,
             "reasons": ["profit factor 0.71 below floor 1.00"]},
             "spec": {"direction": "both", "timeframe": "H1",
                      "entry_conditions": [{"verb": "above"}]}},
        ] * 3

    monkeypatch.setattr(prop, "_load_veto_conclusions", _loader)

    called = {"gate": 0}

    async def _must_not_run(*_a, **_k):
        called["gate"] += 1
        raise AssertionError("gate must not run for a vetoed shape")

    monkeypatch.setattr(prop.BacktestValidator, "validate_spec", _must_not_run)

    class _Spec:
        name = "FallbackTrend EUR/USD"
        description = "d"
        direction = "both"
        timeframe = "H1"
        atr_sl_mult = 1.5
        atr_tp_mult = 3.0

        def model_dump(self):
            return {"direction": "both", "timeframe": "H1",
                    "entry_conditions": [{"verb": "above"}]}

    class _Res:
        pass

    monkeypatch.setattr(prop, "fallback_spec", lambda pair: _Spec())
    monkeypatch.setattr(prop, "build_pair_brief", _fake_brief)
    monkeypatch.setattr(prop, "_log_result", _async_return)

    result = await prop.propose_for_pair(
        pipeline=None, pair="EUR/USD", candles_df=None, use_llm=False,
        log_to_db=True, enter_probation=False,
    )
    assert result.status == "rejected"
    assert result.validation is not None and result.validation.passed is False
    assert result.validation.reasons and result.validation.reasons[0].startswith("weak-hypothesis veto:")
    assert any(m.startswith("weak-hypothesis veto:") for m in result.messages)
    assert called["gate"] == 0  # no backtest was burned


@pytest.mark.asyncio
async def test_proposal_veto_failure_falls_through_to_gate(monkeypatch):
    from forex_ai_agent.autonomy import proposal as prop

    async def _boom(**_kw):
        raise RuntimeError("lessons db down")

    monkeypatch.setattr(prop, "_load_veto_conclusions", _boom)

    gate_calls = {"n": 0}

    async def _gate(self, spec, df, symbol=None):
        gate_calls["n"] += 1
        return _FakeValidation(passed=True, reasons=[])

    monkeypatch.setattr(prop.BacktestValidator, "validate_spec", _gate)

    class _Spec:
        name = "S"
        description = "d"
        direction = "short"
        timeframe = "H4"
        atr_sl_mult = 1.5
        atr_tp_mult = 3.0

        def model_dump(self):
            return {"direction": "short", "timeframe": "H4", "entry_conditions": []}

    class _VM:
        def create_version(self, **kw):
            return SimpleNamespace(id="deadbeef")

    monkeypatch.setattr(prop, "fallback_spec", lambda pair: _Spec())
    monkeypatch.setattr(prop, "build_pair_brief", _fake_brief)
    monkeypatch.setattr(prop, "StrategyVersionManager", lambda _dir: _VM())
    monkeypatch.setattr(prop, "_log_result", _async_return)

    # probation store failure must not break the validated path either
    class _BoomStore:
        def __init__(self, *a, **k):
            pass

        async def start(self, *a, **k):
            raise RuntimeError("no db")

    import forex_ai_agent.autonomy.probation as _prob_mod
    monkeypatch.setattr(_prob_mod, "ProbationStore", _BoomStore, raising=False)

    result = await prop.propose_for_pair(
        pipeline=None, pair="EUR/USD", candles_df=None, use_llm=False,
        log_to_db=True, enter_probation=True,
    )
    assert gate_calls["n"] == 1  # the gate DID run
    assert result.status == "validated"


def test_hard_blocks_by_construction_source_scan():
    for mod in (rl, hg):
        src = inspect.getsource(mod)
        for banned in ("place_order", "execute_trade", "risk_manager",
                       "MAX_PERMISSION_LEVEL = 4"):
            assert banned not in src, (mod.__name__, banned)
