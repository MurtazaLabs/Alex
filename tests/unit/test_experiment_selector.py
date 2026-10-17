"""Stage-C2 unit tests: LLM experiment selection with deterministic validation.

Pins the Stage C constitution on the SECOND LLM-chooses-what-to-test path:
the LLM only sees a deterministic menu (strategies, declared parameter
ranges, prior tests) and only proposes candidates; code validates every one
(range clamping, budget caps, dedup) and only then executes through the
existing Phase-3 engines. LLM, menu and runner are faked via injection
seams — the tests pin the ORCHESTRATION, not the network or the engines.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

from forex_ai_agent.agent import experiment_selector as es


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Records the context it is shown; returns canned JSON payloads."""

    def __init__(self, responses: List[Any], fail: bool = False):
        self.responses = list(responses)
        self.fail = fail
        self.calls: List[Dict[str, Any]] = []

    async def reason_json(self, system: str, context: Any, schema_hint: Optional[str] = None):
        self.calls.append({"system": system, "context": context, "schema_hint": schema_hint})
        if self.fail:
            raise RuntimeError("LLM unavailable (fake)")
        return self.responses.pop(0)


def _strategy(sid: str, name: str, params: Optional[Dict[str, Any]] = None):
    return SimpleNamespace(id=sid, name=name, parameters=params or {"period": 20})


@pytest.fixture(autouse=True)
def _skip_real_spec_parse(monkeypatch):
    """Tests pin the ORCHESTRATION, not the spec parser — stub the check out;
    dedicated tests below re-enable it with controlled stubs."""
    monkeypatch.setattr(es, "_spec_error", lambda s: None)


def _menu_loader(strategies, experiments_by_strategy):
    def loader(_config):
        return strategies, experiments_by_strategy

    return loader


def _runner(calls: List[Tuple[str, Dict[str, float]]]):
    async def run(entry, proposal, config):
        calls.append((entry["strategy_id"], dict(proposal["parameter_mutations"])))
        return {"status": "FAKE_RUN", "strategy_id": entry["strategy_id"]}

    return run


def _cfg(**overrides) -> es.ExperimentSelConfig:
    base = dict(use_llm=True, max_experiments=2, symbol="EUR/USD")
    base.update(overrides)
    return es.ExperimentSelConfig(**base)


async def _run(
    strategies=None,
    experiments_by_strategy=None,
    llm=None,
    runner=None,
    config=None,
) -> Dict[str, Any]:
    strategies = strategies if strategies is not None else [_strategy("s1", "Strat One")]
    experiments_by_strategy = experiments_by_strategy or {}
    return await es.select_experiments(
        config or _cfg(),
        menu_loader=_menu_loader(strategies, experiments_by_strategy),
        llm=llm,
        runner=runner,
    )


# ---------------------------------------------------------------------------
# Menu (the only thing the LLM sees)
# ---------------------------------------------------------------------------


def test_build_menu_entry_is_deterministic_and_trimmed():
    strategy = _strategy("s1", "Strat One", {"period": 20, "active": True, "label": "x"})
    experiments = [
        {"parameter_changes": {"period": i}, "verdict": "FAIL", "created_at": f"2026-01-0{i}"}
        for i in range(1, 9)
    ]
    entry = es.build_menu_entry(strategy, experiments)
    assert entry["strategy_id"] == "s1"
    assert entry["name"] == "Strat One"
    # Only numeric params get derived ranges (±50%); non-numeric are facts only
    assert entry["param_ranges"] == {"period": [10.0, 30.0]}
    # tested_before is trimmed to the last 5
    assert len(entry["tested_before"]) == 5
    assert entry["tested_before"][-1]["verdict"] == "FAIL"
    # deterministic
    assert es.build_menu_entry(strategy, experiments) == entry


def test_build_menu_entry_explicit_ranges_override_derived():
    entry = es.build_menu_entry(
        _strategy("s1", "S"), [], param_ranges={"period": (5.0, 50.0)}
    )
    assert entry["param_ranges"] == {"period": [5.0, 50.0]}


def test_build_menu_drops_unusable_entries():
    menu = es.build_menu(
        [_strategy("", "no id"), _strategy("s2", ""), SimpleNamespace(id="s3", name="ok")],
        {},
    )
    assert [e["strategy_id"] for e in menu] == ["s3"]


async def test_llm_sees_only_the_menu():
    llm = _FakeLLM([{"experiments": []}])
    await _run(
        llm=llm,
        experiments_by_strategy={"s1": [{"parameter_changes": {"period": 30}, "verdict": "PASS"}]},
    )
    assert llm.calls, "LLM was not consulted"
    menu = llm.calls[0]["context"]["menu"]
    assert set(menu[0].keys()) == {
        "strategy_id", "name", "params", "param_ranges", "tested_before",
    }


# ---------------------------------------------------------------------------
# Fallback (deterministic, honest provenance)
# ---------------------------------------------------------------------------


def test_fallback_proposals_are_baseline_tagged():
    menu = es.build_menu([_strategy("s1", "A"), _strategy("s2", "B")], {})
    props = es._fallback_proposals(menu)
    assert len(props) == 2
    for p in props:
        assert p["source"] == "baseline_fallback"
        assert p["confidence"] is None
        assert p["parameter_mutations"] == {}


async def test_llm_failure_falls_back_and_runs_baseline():
    calls: List[Tuple[str, Dict[str, float]]] = []
    report = await _run(llm=_FakeLLM([], fail=True), runner=_runner(calls))
    assert report["llm_used"] is False
    assert report["fallback_used"] is True
    assert report["llm_error"] and "RuntimeError" in report["llm_error"]
    assert len(report["accepted"]) == 1
    assert report["accepted"][0]["source"] == "baseline_fallback"
    assert calls == [("s1", {})]


async def test_use_llm_false_skips_llm_entirely():
    llm = _FakeLLM([])
    llm.reason_json = lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM must not be called"))
    report = await _run(llm=llm, runner=_runner([]), config=_cfg(use_llm=False))
    assert report["llm_used"] is False
    assert report["fallback_used"] is True


# ---------------------------------------------------------------------------
# Deterministic validation
# ---------------------------------------------------------------------------


def test_validate_clamps_to_declared_range():
    entry = es.build_menu_entry(_strategy("s1", "S", {"period": 20}), [])
    normalized, reason = es._validate_proposal(
        {"strategy_id": "s1", "rationale": "test the upper band", "parameter_mutations": {"period": 99}},
        {"s1": entry},
        _cfg(),
    )
    assert reason is None
    assert normalized["parameter_mutations"] == {"period": 30.0}  # clamped to hi
    assert normalized["source"] == "llm"


def test_validate_rejects_unknown_strategy_and_param():
    entry = es.build_menu_entry(_strategy("s1", "S", {"period": 20}), [])
    _, reason = es._validate_proposal(
        {"strategy_id": "nope", "rationale": "not in menu", "parameter_mutations": {}},
        {"s1": entry},
        _cfg(),
    )
    assert "not in this run's menu" in reason
    _, reason = es._validate_proposal(
        {"strategy_id": "s1", "rationale": "unknown parameter", "parameter_mutations": {"magic": 1.0}},
        {"s1": entry},
        _cfg(),
    )
    assert "not in the strategy's declared ranges" in reason


def test_validate_rejects_bad_confidence_and_short_rationale():
    entry = es.build_menu_entry(_strategy("s1", "S"), [])
    for cand in (
        {"strategy_id": "s1", "rationale": "r", "parameter_mutations": {}, "confidence": 1.5},
        {"strategy_id": "s1", "rationale": "r", "parameter_mutations": {}, "confidence": "high"},
        {"strategy_id": "s1", "rationale": "too short", "parameter_mutations": {}},
    ):
        _, reason = es._validate_proposal(cand, {"s1": entry}, _cfg())
        assert reason is not None, cand


# ---------------------------------------------------------------------------
# Orchestration: validate → dedup → cap → execute
# ---------------------------------------------------------------------------


async def test_llm_proposals_validated_and_run():
    calls: List[Tuple[str, Dict[str, float]]] = []
    llm = _FakeLLM(
        [
            {
                "experiments": [
                    {"strategy_id": "s1", "rationale": "push period to the band top",
                     "parameter_mutations": {"period": 25}, "confidence": 0.7},
                    {"strategy_id": "s1", "rationale": "push period to the band bottom",
                     "parameter_mutations": {"period": 15}, "confidence": 0.6},
                ]
            }
        ]
    )
    report = await _run(llm=llm, runner=_runner(calls), config=_cfg(max_experiments=2))
    assert report["llm_used"] is True
    assert report["fallback_used"] is False
    assert report["raw_proposals"] == 2
    assert len(report["accepted"]) == 2
    assert [c for c, _ in calls] == ["s1", "s1"]
    assert calls[0][1] == {"period": 25.0}
    assert report["accepted"][0]["confidence"] == 0.7


async def test_run_caps_max_experiments():
    llm = _FakeLLM(
        [
            {
                "experiments": [
                    {"strategy_id": "s1", "rationale": f"experiment number {i}",
                     "parameter_mutations": {"period": 10 + i}, "confidence": 0.8}
                    for i in range(3)
                ]
            }
        ]
    )
    calls: List[Tuple[str, Dict[str, float]]] = []
    report = await _run(llm=llm, runner=_runner(calls), config=_cfg(max_experiments=1))
    assert len(report["accepted"]) == 1
    assert len(calls) == 1
    assert any("beyond max_experiments" in r["reason"] for r in report["rejected"])


async def test_duplicate_proposals_deduped():
    llm = _FakeLLM(
        [
            {
                "experiments": [
                    {"strategy_id": "s1", "rationale": "first proposal rationale",
                     "parameter_mutations": {"period": 25}, "confidence": 0.8},
                    {"strategy_id": "s1", "rationale": "identical duplicate here",
                     "parameter_mutations": {"period": 25}, "confidence": 0.9},
                ]
            }
        ]
    )
    report = await _run(llm=llm, runner=_runner([]), config=_cfg(max_experiments=2))
    assert any("duplicate" in r["reason"] for r in report["rejected"])


async def test_garbage_llm_shape_falls_back():
    llm = _FakeLLM([{"something_else": []}])
    report = await _run(llm=llm, runner=_runner([]))
    assert report["llm_used"] is False
    assert report["fallback_used"] is True
    assert "experiments" in report["llm_error"]


async def test_empty_menu_raises_valueerror():
    with pytest.raises(ValueError):
        await _run(strategies=[], experiments_by_strategy={})


async def test_unrunnable_strategies_rejected_before_llm(monkeypatch):
    monkeypatch.setattr(
        es,
        "_spec_error",
        lambda s: "name: Field required" if s.id == "broken" else None,
    )
    llm = _FakeLLM([{"experiments": []}])
    report = await _run(
        strategies=[_strategy("broken", "Broken"), _strategy("ok", "Ok")],
        llm=llm,
        runner=_runner([]),
    )
    assert any(
        r["strategy_id"] == "broken" and "runnable spec" in r["reason"]
        for r in report["rejected"]
    )
    # The LLM must never even SEE the unrunnable strategy.
    assert [e["strategy_id"] for e in llm.calls[0]["context"]["menu"]] == ["ok"]
    assert report["strategies_seen"] == 1


async def test_all_unrunnable_raises_valueerror(monkeypatch):
    monkeypatch.setattr(es, "_spec_error", lambda s: "not runnable")
    with pytest.raises(ValueError):
        await _run(strategies=[_strategy("bad", "Bad")])


async def test_one_runner_failure_does_not_stop_the_rest():
    llm = _FakeLLM(
        [
            {
                "experiments": [
                    {"strategy_id": "s1", "rationale": "will explode in runner",
                     "parameter_mutations": {}, "confidence": 0.8},
                ]
            }
        ]
    )

    async def exploding_runner(entry, proposal, config):
        raise RuntimeError("engine exploded")

    report = await _run(
        strategies=[_strategy("s1", "A"), _strategy("s2", "B")],
        llm=llm,
        runner=exploding_runner,
        config=_cfg(max_experiments=2),
    )
    # The runner's exception is contained: an honest ERROR row lands in the report.
    assert report["llm_used"] is True
    assert len(report["accepted"]) == 1
    assert report["accepted"][0]["status"] == "ERROR"
    assert "engine exploded" in report["accepted"][0]["error"]


# ---------------------------------------------------------------------------
# Constitution + JSON safety
# ---------------------------------------------------------------------------


def test_hard_blocks_by_construction_source_scan():
    src = inspect.getsource(es)
    for banned in ("place_order", "execute_trade", "risk_manager", "MAX_PERMISSION_LEVEL = 4"):
        assert banned not in src, banned


async def test_dry_run_executes_nothing():
    llm = _FakeLLM(
        [
            {
                "experiments": [
                    {"strategy_id": "s1", "rationale": "would be executed otherwise",
                     "parameter_mutations": {"period": 25}, "confidence": 0.8},
                ]
            }
        ]
    )

    async def boom_runner(entry, proposal, config):
        raise AssertionError("runner must not be called in dry_run")

    report = await _run(llm=llm, runner=boom_runner, config=_cfg(dry_run=True))
    assert report["accepted"][0]["status"] == "DRY_RUN_NOT_EXECUTED"
    assert report["accepted"][0]["parameter_mutations"] == {"period": 25.0}


def test_engine_output_is_json_safe():
    import numpy as np

    raw = {
        "f": np.float64(1.5),
        "i": np.int64(2),
        "b": np.bool_(True),
        "nested": [np.float64(0.1), {"x": np.int32(7)}],
        "plain": {"s": "ok", "n": None, "f2": 1.25},
    }
    cleaned = es._to_json_safe(raw)
    json.loads(json.dumps(cleaned))  # must not raise
    assert isinstance(cleaned["f"], float) and cleaned["f"] == 1.5
    assert isinstance(cleaned["i"], int)
    assert isinstance(cleaned["b"], bool)
    assert isinstance(cleaned["nested"][1]["x"], int)
    assert cleaned["plain"] == {"s": "ok", "n": None, "f2": 1.25}


def test_report_is_json_safe():
    report = es.ExperimentSelReport(
        generated_at="2026-09-26T00:00:00",
        timeframe="H1",
        period="2025-01-01..2025-12-31",
        llm_used=True,
        accepted=[{"strategy_id": "s1", "metrics": {"sharpe": 1.2}}],
    )
    json.loads(json.dumps(report.to_dict()))
