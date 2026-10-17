"""Stage-B1 unit tests: Strategy Health — five lights, badge, staleness.

Pins the roadmap acceptance criteria: health is computed deterministically
from STORED state only (no re-runs), every light carries an as-of date,
stale evidence auto-downgrades, and probation facts move the badge inside
the deterministic ladder.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from forex_ai_agent.agent.health import (
    DATA_HORIZON_DAYS,
    compute_strategy_health,
    get_strategy_health,
    health_summary,
    render_health_text,
)

_NOW = datetime(2026, 9, 24, 12, 0, 0)
_FRESH = _NOW - timedelta(days=3)
_STALE = _NOW - timedelta(days=DATA_HORIZON_DAYS + 10)


def _sweep(
    profitable: int = 3,
    failing: int = 0,
    regimes: dict | None = None,
    evaluated_at: str | None = None,
) -> dict:
    """A stored robustness-sweep result (Stage B3 shape, health-relevant keys)."""
    regimes = regimes if regimes is not None else {
        "trending": {"windows": 8, "trades": 40, "distinct_pairs": 3,
                     "mean_oos_sharpe": 0.9, "positive_windows": 7, "worst_oos_sharpe": 0.2},
        "ranging": {"windows": 4, "trades": 12, "distinct_pairs": 2,
                    "mean_oos_sharpe": 0.3, "positive_windows": 3, "worst_oos_sharpe": -0.1},
    }
    return {
        "schema_version": 1,
        "evaluated_at": evaluated_at or "2026-09-21T10:00:00",
        "base_pair": "GBP/JPY",
        "pairs": [
            {"symbol": f"P{i}", "base": i == 0, "status": "OK", "windows_tested": 4}
            for i in range(profitable)
        ]
        + [
            {"symbol": f"F{i}", "base": False, "status": "OK", "windows_tested": 4}
            for i in range(failing)
        ],
        "pairs_summary": {
            "tested": profitable + failing,
            "usable": profitable + failing,
            "profitable": profitable,
            "failing": failing,
            "profitable_pairs": [f"P{i}" for i in range(profitable)],
            "failing_pairs": [f"F{i}" for i in range(failing)],
        },
        "per_regime": regimes,
        "verdict": "ROBUST" if failing == 0 else "MIXED",
    }


def _windows(sharpes: List[float], trades: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    trades = trades if trades is not None else [5] * len(sharpes)
    return [
        {
            "window": i + 1,
            "is_sharpe": 1.2,
            "is_pf": 1.5,
            "is_trades": 8,
            "oos_sharpe": s,
            "oos_pf": 1.4 if s > 0 else 0.6,
            "oos_trades": t,
            "oos_max_dd": 0.05,
            "oos_return_pct": 1.0,
        }
        for i, (s, t) in enumerate(zip(sharpes, trades))
    ]


def _wf(
    sharpes: List[float] = (0.9, 0.8, 0.7, 0.6),
    trades: Optional[List[int]] = None,
    verdict: str = "STABLE",
    stability: float = 0.2,
) -> Dict[str, Any]:
    rows = _windows(list(sharpes), trades)
    total = sum(r["oos_trades"] for r in rows)
    dds = [0.06, 0.08, 0.07, 0.05]
    return {
        "windows": rows,
        "windows_tested": len(rows),
        "stability_score": stability,
        "verdict": verdict,
        "aggregate_metrics": {
            "is_sharpe_mean": 1.2,
            "oos_sharpe_mean": sum(sharpes) / len(sharpes),
            "oos_sharpe_std": 0.1,
            "oos_trades_total": total,
            "oos_max_drawdown": max(dds),
            "degradation_overfitting": 0.3,
            "oos_negative_window_frac": sum(1 for s in sharpes if s <= 0) / len(sharpes),
        },
        "overfit_analysis": {
            "components": {},
            "score": 0.35,
        },
    }


def _gate(
    wf: Optional[Dict[str, Any]] = None,
    pf: float = 1.45,
    win_rate: float = 0.52,
    total_trades: int = 60,
    max_dd: float = 0.08,
    evaluated_at: str = "2026-09-21T10:00:00",
    robustness: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    gate: Dict[str, Any] = {
        "profit_factor": pf,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": 1.1,
        "walk_forward": wf,
        "evaluated_at": evaluated_at,
    }
    if robustness is not None:
        gate["robustness"] = robustness
    return gate


# ---------------------------------------------------------------------------
# Lights and badge
# ---------------------------------------------------------------------------


def test_healthy_strategy_is_all_green_candidate():
    health = compute_strategy_health(_gate(_wf(), robustness=_sweep()), now=_NOW)
    assert health["badge"] == "CANDIDATE"
    assert all(l["status"] == "GREEN" for l in health["lights"])
    assert [l["name"] for l in health["lights"]] == [
        "PROFITABILITY", "ROBUSTNESS", "SAMPLE SIZE", "OOS CONSISTENCY", "DRAWDOWN",
        "PAIR GENERALITY", "REGIME ROBUSTNESS",
    ]
    assert not health["auto_downgraded"]
    assert health["refresh_command"] == "walk_forward_test"


def test_without_sweep_perfect_picture_still_has_grey_b3_lights():
    """B1-era fixtures carry no sweep — the two B3 lights stay GREY."""
    health = compute_strategy_health(_gate(_wf()), now=_NOW)
    by_name = {l["name"]: l for l in health["lights"]}
    assert by_name["PAIR GENERALITY"]["status"] == "GREY"
    assert by_name["REGIME ROBUSTNESS"]["status"] == "GREY"
    assert health["badge"] == "RESEARCH ONLY"
    assert "run_robustness_sweep" in health["recommendation"]


def test_unstable_walk_forward_is_red_retire():
    # Edge death: late OOS windows collapse (the f39b8d40 pattern).
    wf = _wf(sharpes=(1.4, 0.6, -0.5, -1.6), verdict="UNSTABLE", stability=0.8)
    wf["overfit_analysis"]["score"] = 0.64
    health = compute_strategy_health(_gate(wf), now=_NOW)
    by_name = {l["name"]: l for l in health["lights"]}
    assert by_name["ROBUSTNESS"]["status"] == "RED"
    assert by_name["OOS CONSISTENCY"]["status"] == "RED"
    assert health["badge"] == "RETIRE?"
    assert "OOS CONSISTENCY" in health["concerns"]


def test_missing_walk_forward_is_grey_research_only():
    health = compute_strategy_health(
        _gate(wf=None, robustness=_sweep()), now=_NOW
    )
    greys = [l["name"] for l in health["lights"] if l["status"] == "GREY"]
    # ROBUSTNESS / SAMPLE SIZE / OOS CONSISTENCY need WF windows; DRAWDOWN
    # still has the full-backtest DD, so it stays evidence-backed. The B3
    # sweep is present, so its lights stay evidence-backed too.
    assert greys == ["ROBUSTNESS", "SAMPLE SIZE", "OOS CONSISTENCY"]
    assert health["badge"] == "RESEARCH ONLY"


def test_no_gate_report_at_all_is_research_only():
    health = compute_strategy_health(None, now=_NOW)
    assert all(l["status"] == "GREY" for l in health["lights"])
    assert health["badge"] == "RESEARCH ONLY"


def test_sample_size_bandings():
    thin = _wf(trades=[1, 1, 1, 1])  # 4 OOS trades total → RED
    health = compute_strategy_health(_gate(thin), now=_NOW)
    by_name = {l["name"]: l for l in health["lights"]}
    assert by_name["SAMPLE SIZE"]["status"] == "RED"

    mid = _wf(trades=[3, 3, 3, 3])  # 12 total → AMBER
    health = compute_strategy_health(_gate(mid), now=_NOW)
    by_name = {l["name"]: l for l in health["lights"]}
    assert by_name["SAMPLE SIZE"]["status"] == "AMBER"

    good = _wf()  # 20 total → GREEN
    health = compute_strategy_health(_gate(good), now=_NOW)
    by_name = {l["name"]: l for l in health["lights"]}
    assert by_name["SAMPLE SIZE"]["status"] == "GREEN"


def test_drawdown_uses_worst_source():
    wf = _wf()
    wf["aggregate_metrics"]["oos_max_drawdown"] = 0.28  # worse than the gate DD
    health = compute_strategy_health(_gate(wf, max_dd=0.05), now=_NOW)
    by_name = {l["name"]: l for l in health["lights"]}
    assert by_name["DRAWDOWN"]["status"] == "RED"
    assert "OOS max DD" in by_name["DRAWDOWN"]["detail"]


def test_amber_only_picture_is_probation_badge():
    # PF 1.10 → amber profitability; everything else green.
    health = compute_strategy_health(
        _gate(_wf(), pf=1.10, robustness=_sweep()), now=_NOW
    )
    statuses = {l["name"]: l["status"] for l in health["lights"]}
    assert statuses["PROFITABILITY"] == "AMBER"
    assert health["badge"] == "PROBATION"


# ---------------------------------------------------------------------------
# Staleness (required by the roadmap: auto-downgrade + refresh offer)
# ---------------------------------------------------------------------------


def test_fresh_evidence_not_stale():
    health = compute_strategy_health(_gate(evaluated_at=_FRESH.isoformat()), now=_NOW)
    st = health["staleness"]
    assert st["age_days"] == 3
    assert st["stale"] is False and st["refresh_needed"] is False
    assert not health["auto_downgraded"]


def test_stale_candidate_auto_downgrades():
    gate = _gate(
        _wf(),
        robustness=_sweep(evaluated_at=_STALE.isoformat()),
        evaluated_at=_STALE.isoformat(),
    )
    health = compute_strategy_health(gate, now=_NOW)
    assert health["staleness"]["stale"] is True
    assert health["staleness"]["age_days"] == DATA_HORIZON_DAYS + 10
    # All lights would be green, but the record is beyond its data horizon.
    assert health["auto_downgraded"] is True
    assert health["badge"] == "RESEARCH ONLY"
    assert "walk_forward_test" in health["recommendation"]
    assert all(l["stale"] for l in health["lights"])


def test_sweep_evaluated_at_participates_in_staleness():
    # Fresh gate, but the sweep itself is beyond the data horizon.
    gate = _gate(
        _wf(),
        robustness=_sweep(evaluated_at=_STALE.isoformat()),
        evaluated_at=_FRESH.isoformat(),
    )
    health = compute_strategy_health(gate, now=_NOW)
    assert health["staleness"]["stale"] is True
    assert health["auto_downgraded"] is True
    assert health["badge"] == "RESEARCH ONLY"


def test_staleness_unknown_when_no_evaluated_at():
    gate = _gate(_wf())
    gate.pop("evaluated_at")
    health = compute_strategy_health(gate, now=_NOW)
    assert health["staleness"]["as_of"] is None
    assert health["staleness"]["stale"] is True
    assert "provenance" in health["staleness"]["note"]


# ---------------------------------------------------------------------------
# Probation interacts with the badge — never with the light math
# ---------------------------------------------------------------------------


def test_probation_failed_forces_retire_even_if_lights_green():
    prob = {"status": "failed", "observed_trades": 25, "required_trades": 20}
    health = compute_strategy_health(_gate(_wf()), probation=prob, now=_NOW)
    assert health["badge"] == "RETIRE?"


def test_probation_passed_promotes_to_candidate():
    prob = {"status": "passed", "observed_trades": 22, "required_trades": 20}
    health = compute_strategy_health(_gate(_wf(), pf=1.10), probation=prob, now=_NOW)
    assert health["badge"] == "CANDIDATE"


def test_probation_expired_keeps_research_only():
    prob = {"status": "expired", "observed_trades": 3, "required_trades": 20}
    health = compute_strategy_health(_gate(_wf(), pf=1.10), probation=prob, now=_NOW)
    assert health["badge"] == "RESEARCH ONLY"


def test_probation_active_counts_into_sample_size_detail():
    prob = {"status": "active", "observed_trades": 4, "required_trades": 20}
    health = compute_strategy_health(_gate(_wf()), probation=prob, now=_NOW)
    by_name = {l["name"]: l for l in health["lights"]}
    assert "probation 4/20" in by_name["SAMPLE SIZE"]["detail"]
    # Lights stay green — probation progress is narrative here, not a verdict.


def test_early_probation_without_oos_is_amber_not_red():
    # A strategy fresh into probation (0 trades accrued, no WF yet) is thin,
    # not dead — it must not carry a RED sample-size light.
    prob = {"status": "active", "observed_trades": 0, "required_trades": 5}
    health = compute_strategy_health(_gate(wf=None), probation=prob, now=_NOW)
    by_name = {l["name"]: l for l in health["lights"]}
    assert by_name["SAMPLE SIZE"]["status"] == "AMBER"
    assert "accruing" in by_name["SAMPLE SIZE"]["detail"]
    assert health["badge"] == "RESEARCH ONLY"


# ---------------------------------------------------------------------------
# Deterministic rendering + end-to-end stored-state load
# ---------------------------------------------------------------------------


def test_health_summary_and_text_card_deterministic():
    payload = {
        "strategy_name": "London Sweep",
        "strategy_id": "abc12345",
        "health": compute_strategy_health(
            _gate(_wf(), robustness=_sweep()), now=_NOW
        ),
    }
    text = render_health_text(payload)
    assert text == render_health_text(payload)
    assert "CANDIDATE" in text and "🟢" in text
    summary = health_summary(payload["health"])
    assert summary.startswith("CANDIDATE [🟢 🟢 🟢 🟢 🟢 🟢 🟢]")
    # Payload must be JSON-safe for the API layer.
    json.loads(json.dumps(payload["health"]))


def test_every_light_carries_as_of_date():
    gate_date = _FRESH.isoformat()
    health = compute_strategy_health(
        _gate(_wf(), robustness=_sweep(), evaluated_at=gate_date), now=_NOW
    )
    # Each light carries ITS OWN source's as-of date: gate-fed lights use the
    # gate report date, sweep-fed lights use the sweep date.
    gate_lights = ["PROFITABILITY", "ROBUSTNESS", "SAMPLE SIZE", "OOS CONSISTENCY", "DRAWDOWN"]
    for light in health["lights"]:
        expected = gate_date if light["name"] in gate_lights else "2026-09-21T10:00:00"
        assert light["as_of"] == expected, light["name"]
        assert light["stale"] is False


class _FakeVM:
    def __init__(self, version):
        self.versions = {version.id: version}

    def get_version(self, sid):
        return self.versions.get(sid) or self.versions.get(sid[:8])


class _FakeProbStore:
    def __init__(self, status_dict=None):
        self._status = status_dict

    async def get(self, version_id):
        if self._status is None:
            return None
        return SimpleNamespace(to_dict=lambda: dict(self._status))


def _stored_version(gate: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        id="abc12345",
        name="H4 Bullish Pullback",
        backtest_results=gate,
        parameters={"symbol": "GBP/JPY", "timeframe": "H1"},
        rules={},
    )


@pytest.mark.asyncio
async def test_get_strategy_health_reads_stored_state_only():
    gate = _gate(_wf(), robustness=_sweep(), evaluated_at=_FRESH.isoformat())
    version = _stored_version(gate)

    async def fake_vm(*args, **kwargs):
        return _FakeVM(version)

    # Patch at the source modules (health imports them lazily).
    import forex_ai_agent.strategy as strategy_pkg
    import forex_ai_agent.autonomy.probation as probation_mod

    with patch.object(strategy_pkg, "create_version_manager", fake_vm), \
            patch.object(probation_mod, "ProbationStore", _FakeProbStore):
        payload = await get_strategy_health("abc12345")

    assert payload["strategy_id"] == "abc12345"
    assert payload["pair"] == "GBP/JPY" and payload["timeframe"] == "H1"
    assert payload["health"]["badge"] == "CANDIDATE"
    assert payload["summary"].startswith("CANDIDATE [🟢")


@pytest.mark.asyncio
async def test_get_strategy_health_unknown_id_raises():
    async def fake_vm(*args, **kwargs):
        return _FakeVM(_stored_version(_gate(_wf())))

    import forex_ai_agent.strategy as strategy_pkg
    import forex_ai_agent.autonomy.probation as probation_mod

    with patch.object(strategy_pkg, "create_version_manager", fake_vm), \
            patch.object(probation_mod, "ProbationStore", _FakeProbStore):
        with pytest.raises(ValueError):
            await get_strategy_health("ffffffff")
