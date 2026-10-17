"""Strategy Health (Stage B1) — deterministic lights over STORED data.

Answers "is this strategy still healthy?" without re-running anything: every
number is read from persisted state (the latest gate report on the
``StrategyVersion`` — full backtest + walk-forward windows + overfit analysis —
plus the probation record when one exists). The LLM never computes these; it
may only narrate the card.

Lights (the roadmap's five) + overall badge + staleness:

- PROFITABILITY   profit factor / win rate from the stored full backtest
- ROBUSTNESS      walk-forward stability score + verdict (+ collapse flag)
- SAMPLE SIZE     OOS trades across windows (and probation quota progress)
- OOS CONSISTENCY negative-window fraction + worst OOS window
- DRAWDOWN        worst OOS drawdown / full-backtest drawdown
- PAIR GENERALITY      (B3) share of sweep pairs OOS-profitable
- REGIME ROBUSTNESS    (B3) share of measured regimes net-positive OOS

Overall badge ladder:
    RETIRE?        any RED light (robustness/probation failure dominate)
    RESEARCH ONLY  no full gate report, or gate rejected, or no WF evidence
    PROBATION      gate passed and probation still accruing
    CANDIDATE      all lights GREEN (incl. B3 sweep evidence) or probation PASSED

Staleness (required by the roadmap): each light carries an ``as_of`` date; if
the health record is older than its data horizon (default 60 days) the overall
badge auto-downgrades to RESEARCH ONLY with a ``refresh_needed`` flag — the
re-run (``walk_forward_test``) is the one-command refresh.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# How long a stored health picture stays credible before it must be refreshed.
# Backed by 6-month WF windows on ~2-year history: 60 days keeps the newest OOS
# window mostly inside its measured period.
DATA_HORIZON_DAYS = 60

# --- Light thresholds (deliberate, pinned by unit tests) -------------------
_PROFIT_FACTOR_GREEN = 1.20
_PROFIT_FACTOR_AMBER = 1.05
_WIN_RATE_GREEN = 0.45
_WIN_RATE_AMBER = 0.35

_STABILITY_GREEN = 0.45  # matches the WF STABLE band
_STABILITY_AMBER = 0.65  # matches the WF MIXED band

_MIN_OOS_TRADES_GREEN = 20
_MIN_OOS_TRADES_AMBER = 8

_NEG_FRAC_GREEN = 0.0
_NEG_FRAC_AMBER = 0.25
_WORST_OOS_SHARPE_GREEN = 0.0
_WORST_OOS_SHARPE_AMBER = -1.0

_DRAWDOWN_GREEN = 0.15  # ≤15%
_DRAWDOWN_AMBER = 0.25  # ≤25%

# --- B3 light thresholds (robustness sweep) -------------------------------
_PAIR_GENERALITY_GREEN = 0.75  # ≥75% of run pairs profitable
_PAIR_GENERALITY_AMBER = 0.5
_REGIME_WINDOWS_MIN = 3  # a regime bucket needs ≥3 windows to be evidence
_REGIME_POSITIVE_GREEN = 0.75  # ≥75% of measured regimes net-positive OOS
_REGIME_POSITIVE_AMBER = 0.5
_REGIME_COLLAPSE = -1.0  # any regime bucket with a window below this disqualifies


def _iso(value: Any) -> Optional[str]:
    """Best-effort ISO string from datetime or str; None if absent."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    try:
        datetime.fromisoformat(str(value))
        return str(value)
    except (TypeError, ValueError):
        return None


def _as_of_from_gates(gate: Optional[Dict[str, Any]]) -> Optional[str]:
    """The date the stored gate report (metrics + WF) was produced."""
    if not gate:
        return None
    evaluated = _iso(gate.get("evaluated_at"))
    if evaluated:
        return evaluated
    wf = gate.get("walk_forward") or {}
    return _iso(wf.get("evaluated_at"))


def _light(
    name: str,
    status: str,
    detail: str,
    as_of: Optional[str],
    stale: bool,
) -> Dict[str, Any]:
    return {
        "name": name,
        "status": status,  # GREEN / AMBER / RED / GREY (no evidence)
        "detail": detail,
        "as_of": as_of,
        "stale": bool(stale),
    }


def _profitability_light(
    gate: Optional[Dict[str, Any]],
    stale: bool,
    as_of: Optional[str],
) -> Dict[str, Any]:
    if not gate or "profit_factor" not in gate:
        return _light(
            "PROFITABILITY", "GREY",
            "no stored full-backtest metrics — run backtest_strategy_spec",
            as_of, stale,
        )
    pf = float(gate.get("profit_factor", 0) or 0)
    wr = gate.get("win_rate")
    detail = f"PF {pf:.2f}"
    if wr is not None:
        detail += f", win rate {float(wr) * 100:.1f}%"
    if pf >= _PROFIT_FACTOR_GREEN:
        status = "GREEN"
    elif pf >= _PROFIT_FACTOR_AMBER:
        status = "AMBER"
    else:
        status = "RED"
    return _light("PROFITABILITY", status, detail, as_of, stale)


def _robustness_light(
    wf: Optional[Dict[str, Any]],
    overfit: Optional[Dict[str, Any]],
    stale: bool,
    as_of: Optional[str],
) -> Dict[str, Any]:
    if not wf or not wf.get("windows"):
        return _light(
            "ROBUSTNESS", "GREY",
            "no stored walk-forward — run walk_forward_test for OOS evidence",
            as_of, stale,
        )
    stability = float(wf.get("stability_score", 1.0) or 0.0)
    verdict = str(wf.get("verdict", "UNSTABLE")).upper()
    if verdict == "STABLE" and stability <= _STABILITY_GREEN:
        status = "GREEN"
    elif verdict == "MIXED" and stability <= _STABILITY_AMBER:
        status = "AMBER"
    else:
        status = "RED"
    detail = f"WF verdict {verdict}, stability {stability:.2f}"
    score = (overfit or {}).get("score")
    if score is not None:
        detail += f", overfit score {float(score):.2f} (≤0.50 gate)"
    return _light("ROBUSTNESS", status, detail, as_of, stale)


def _sample_size_light(
    wf: Optional[Dict[str, Any]],
    probation: Optional[Dict[str, Any]],
    stale: bool,
    as_of: Optional[str],
) -> Dict[str, Any]:
    oos_trades = 0
    if wf and wf.get("windows"):
        agg = wf.get("aggregate_metrics") or {}
        oos_trades = int(agg.get("oos_trades_total", 0) or 0)
    bits = []
    has_wf = bool(wf and wf.get("windows"))
    if has_wf:
        bits.append(f"{oos_trades} OOS trades across {wf.get('windows_tested', len(wf['windows']))} windows")
    if probation:
        bits.append(
            f"probation {probation.get('observed_trades', 0)}/{probation.get('required_trades', '?')} trades"
        )
    detail = "; ".join(bits) or "no stored trade counts"
    if not bits:
        return _light("SAMPLE SIZE", "GREY", detail, as_of, stale)

    if has_wf:
        # OOS evidence is primary.
        status = "GREEN" if oos_trades >= _MIN_OOS_TRADES_GREEN else (
            "AMBER" if oos_trades >= _MIN_OOS_TRADES_AMBER else "RED"
        )
        return _light("SAMPLE SIZE", status, detail, as_of, stale)

    # No WF windows: the live probation accrual is the only sample evidence.
    # Accruing is NOT failing — a candidate at 0/N trades early in probation
    # is thin, not dead; failure shows up in the probation status itself.
    p_status = str((probation or {}).get("status", ""))
    p_obs = int((probation or {}).get("observed_trades", 0) or 0)
    p_req = int((probation or {}).get("required_trades", 0) or 0)
    if p_status == "passed" or (p_req and p_obs >= p_req):
        status = "GREEN"
    elif p_status in ("failed", "expired", "rejected"):
        status = "RED"
    else:
        status = "AMBER"
        detail = f"accruing — {detail} (no OOS evidence yet)"
    return _light("SAMPLE SIZE", status, detail, as_of, stale)


def _oos_consistency_light(
    wf: Optional[Dict[str, Any]],
    stale: bool,
    as_of: Optional[str],
) -> Dict[str, Any]:
    if not wf or not wf.get("windows"):
        return _light(
            "OOS CONSISTENCY", "GREY",
            "no stored walk-forward windows",
            as_of, stale,
        )
    windows: List[Dict[str, Any]] = wf["windows"]
    oos_sharpes = [float(w.get("oos_sharpe", 0.0) or 0.0) for w in windows]
    neg = sum(1 for s in oos_sharpes if s <= 0)
    neg_frac = neg / len(oos_sharpes) if oos_sharpes else 1.0
    worst = min(oos_sharpes) if oos_sharpes else 0.0
    detail = (
        f"{neg}/{len(oos_sharpes)} OOS windows ≤ 0 Sharpe "
        f"(worst {worst:.2f})"
    )
    if neg_frac <= _NEG_FRAC_GREEN and worst >= _WORST_OOS_SHARPE_GREEN:
        status = "GREEN"
    elif neg_frac <= _NEG_FRAC_AMBER and worst >= _WORST_OOS_SHARPE_AMBER:
        status = "AMBER"
    else:
        status = "RED"
    return _light("OOS CONSISTENCY", status, detail, as_of, stale)


def _drawdown_light(
    wf: Optional[Dict[str, Any]],
    gate: Optional[Dict[str, Any]],
    probation: Optional[Dict[str, Any]],
    stale: bool,
    as_of: Optional[str],
) -> Dict[str, Any]:
    candidates: List[tuple[float, str]] = []
    if wf and wf.get("windows"):
        agg = wf.get("aggregate_metrics") or {}
        candidates.append((float(agg.get("oos_max_drawdown", 0) or 0), "OOS max DD"))
    if gate and gate.get("max_drawdown_pct") is not None:
        candidates.append((float(gate.get("max_drawdown_pct") or 0), "backtest max DD"))
    if probation and probation.get("worst_daily_loss_pct") is not None:
        candidates.append((float(probation["worst_daily_loss_pct"]), "probation worst daily loss"))
    if not candidates:
        return _light("DRAWDOWN", "GREY", "no stored drawdown data", as_of, stale)
    worst_val, worst_src = max(candidates, key=lambda c: c[0])
    detail = f"{worst_src} {worst_val * 100:.1f}% (≤{_DRAWDOWN_GREEN * 100:.0f}% green)"
    if worst_val <= _DRAWDOWN_GREEN:
        status = "GREEN"
    elif worst_val <= _DRAWDOWN_AMBER:
        status = "AMBER"
    else:
        status = "RED"
    return _light("DRAWDOWN", status, detail, as_of, stale)


def _pair_generality_light(
    robustness: Optional[Dict[str, Any]],
    stale: bool,
    as_of: Optional[str],
) -> Dict[str, Any]:
    if not robustness or not robustness.get("pairs"):
        return _light(
            "PAIR GENERALITY", "GREY",
            "no stored robustness sweep — run run_robustness_sweep for cross-pair evidence",
            as_of, stale,
        )
    summary = robustness.get("pairs_summary") or {}
    usable = int(summary.get("usable", 0) or 0)
    profitable = int(summary.get("profitable", 0) or 0)
    if usable == 0:
        return _light(
            "PAIR GENERALITY", "GREY",
            "sweep stored but no pair produced walk-forward windows",
            as_of, stale,
        )
    frac = profitable / usable
    detail = (
        f"{profitable}/{usable} pairs OOS-profitable "
        f"({', '.join(summary.get('profitable_pairs') or []) or 'none'})"
    )
    if frac >= _PAIR_GENERALITY_GREEN:
        status = "GREEN"
    elif frac >= _PAIR_GENERALITY_AMBER:
        status = "AMBER"
    else:
        status = "RED"
    return _light("PAIR GENERALITY", status, detail, as_of, stale)


def _regime_robustness_light(
    robustness: Optional[Dict[str, Any]],
    stale: bool,
    as_of: Optional[str],
) -> Dict[str, Any]:
    if not robustness or not robustness.get("per_regime"):
        return _light(
            "REGIME ROBUSTNESS", "GREY",
            "no stored regime-tagged windows — run run_robustness_sweep",
            as_of, stale,
        )
    per_regime: Dict[str, Dict[str, Any]] = robustness.get("per_regime") or {}
    measured = {
        name: r for name, r in per_regime.items()
        if int(r.get("windows", 0) or 0) >= _REGIME_WINDOWS_MIN
    }
    if not measured:
        return _light(
            "REGIME ROBUSTNESS", "GREY",
            f"{len(per_regime)} regime(s) seen but none with ≥{_REGIME_WINDOWS_MIN} windows",
            as_of, stale,
        )
    positive = [n for n, r in measured.items() if float(r["mean_oos_sharpe"]) > 0]
    worst = min(float(r["worst_oos_sharpe"]) for r in measured.values())
    frac = len(positive) / len(measured)
    detail = (
        f"{len(positive)}/{len(measured)} regimes net-positive OOS "
        f"({', '.join(sorted(positive)) or 'none'}), worst window Sharpe {worst:.2f}"
    )
    if worst < _REGIME_COLLAPSE:
        status = "RED"  # a collapsed regime bucket disqualifies regardless of averages
    elif frac >= _REGIME_POSITIVE_GREEN:
        status = "GREEN"
    elif frac >= _REGIME_POSITIVE_AMBER:
        status = "AMBER"
    else:
        status = "RED"
    return _light("REGIME ROBUSTNESS", status, detail, as_of, stale)


_LIGHT_ORDER = (
    "PROFITABILITY",
    "ROBUSTNESS",
    "SAMPLE SIZE",
    "OOS CONSISTENCY",
    "DRAWDOWN",
    "PAIR GENERALITY",
    "REGIME ROBUSTNESS",
)


def _overall_badge(lights: List[Dict[str, Any]]) -> str:
    by_name = {l["name"]: l["status"] for l in lights}
    statuses = list(by_name.values())
    if any(s == "RED" for s in statuses):
        return "RETIRE?"
    if all(s == "GREEN" for s in statuses):
        return "CANDIDATE"
    if all(s in ("GREEN", "AMBER") for s in statuses):
        return "PROBATION"
    # Greys mean missing evidence — not promotable, not yet condemnable.
    return "RESEARCH ONLY"


def _staleness(as_of: Optional[str], now: Optional[datetime] = None) -> Dict[str, Any]:
    now = now or datetime.utcnow()
    if not as_of:
        return {
            "as_of": None,
            "age_days": None,
            "stale": True,
            "refresh_needed": True,
            "note": "no stored evaluation date — evidence provenance unknown",
        }
    try:
        evaluated = datetime.fromisoformat(str(as_of))
    except (TypeError, ValueError):
        return {
            "as_of": str(as_of),
            "age_days": None,
            "stale": True,
            "refresh_needed": True,
            "note": "unreadable evaluation date — treat as stale",
        }
    if evaluated.tzinfo is not None:
        evaluated = evaluated.replace(tzinfo=None)
    age_days = int((now - evaluated).total_seconds() // 86400)
    stale = age_days > DATA_HORIZON_DAYS
    return {
        "as_of": str(as_of),
        "age_days": age_days,
        "stale": stale,
        "refresh_needed": stale,
        "note": (
            f"evidence is {age_days} days old — beyond the {DATA_HORIZON_DAYS}-day "
            "data horizon; re-run walk_forward_test to refresh"
            if stale
            else f"evidence is {age_days} days old"
        ),
    }


def _probation_bump(
    badge: str,
    lights: List[Dict[str, Any]],
    probation: Optional[Dict[str, Any]],
) -> str:
    """Probation facts adjust the badge inside the deterministic ladder.

    Failed/expired probation is a RED-class event (RETIRE?); a PASSED
    probation awaiting the human is the strongest statement short of approval.
    """
    if not probation:
        return badge
    p_status = str(probation.get("status", ""))
    if p_status in ("failed", "rejected"):
        return "RETIRE?"
    if p_status == "expired":
        # Weak signal frequency — keep it out of the candidate set.
        return "RESEARCH ONLY"
    if p_status == "passed":
        if badge == "RETIRE?":
            return badge  # a red light still dominates
        return "CANDIDATE"
    return badge


def compute_strategy_health(
    gate: Optional[Dict[str, Any]],
    probation: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Compute the health card from STORED state only.

    ``gate`` is the strategy's stored ``backtest_results`` dict (full-backtest
    metrics + ``walk_forward`` incl. windows/overfit_analysis + ``repro`` +
    ``evaluated_at``). ``probation`` is a ``ProbationStatus.to_dict()``-shaped
    dict or None.
    """
    wf = (gate or {}).get("walk_forward") or {}
    overfit = wf.get("overfit_analysis") or {}
    robustness = (gate or {}).get("robustness") or None

    # Each evidence source carries its own date; a light is stale when ITS
    # source is stale, and the overall picture is as old as its oldest
    # evidence (stale sweep + fresh gate must still auto-downgrade).
    gate_as_of = _as_of_from_gates(gate)
    sweep_as_of = _iso(robustness.get("evaluated_at")) if robustness else None

    def _age_key(s: str) -> datetime:
        return datetime.fromisoformat(str(s)).replace(tzinfo=None)

    dated = [d for d in (gate_as_of, sweep_as_of) if d]
    overall_as_of = min(dated, key=_age_key) if dated else None
    staleness = _staleness(overall_as_of, now)
    stale = staleness["stale"]
    gate_stale = _staleness(gate_as_of, now)["stale"] if gate_as_of else True
    sweep_stale = _staleness(sweep_as_of, now)["stale"] if sweep_as_of else True

    lights = [
        _profitability_light(gate, gate_stale, gate_as_of),
        _robustness_light(wf, overfit, gate_stale, gate_as_of),
        _sample_size_light(wf, probation, gate_stale, gate_as_of),
        _oos_consistency_light(wf, gate_stale, gate_as_of),
        _drawdown_light(wf, gate, probation, gate_stale, gate_as_of),
        _pair_generality_light(robustness, sweep_stale, sweep_as_of),
        _regime_robustness_light(robustness, sweep_stale, sweep_as_of),
    ]
    # Keep the documented light order regardless of construction order.
    lights.sort(key=lambda l: _LIGHT_ORDER.index(l["name"]))

    badge = _overall_badge(lights)
    badge = _probation_bump(badge, lights, probation)

    # Required staleness rule: evidence older than its data horizon can no
    # longer certify a candidate — auto-downgrade and offer the refresh.
    auto_downgraded = False
    if stale and badge == "CANDIDATE":
        badge = "RESEARCH ONLY"
        auto_downgraded = True

    red = [l["name"] for l in lights if l["status"] == "RED"]
    grey = [l["name"] for l in lights if l["status"] == "GREY"]
    refresh_hint = (
        "run_robustness_sweep"
        if any(name in ("PAIR GENERALITY", "REGIME ROBUSTNESS") for name in grey)
        else "walk_forward_test"
    )
    if red:
        recommendation = (
            f"Red lights: {', '.join(red)}. Re-run walk_forward_test on fresh data "
            "before trusting any stored number."
        )
    elif grey:
        recommendation = (
            f"Missing evidence: {', '.join(grey)}. Run {refresh_hint} to "
            "complete the picture."
        )
    elif staleness["refresh_needed"]:
        recommendation = "Evidence is stale — refresh with walk_forward_test."
    else:
        recommendation = "Stored evidence is current; next checkpoint at the data horizon."

    return {
        "badge": badge,
        "lights": lights,
        "staleness": staleness,
        "auto_downgraded": auto_downgraded,
        "concerns": red + [f"{n} has no evidence" for n in grey],
        "recommendation": recommendation,
        "refresh_command": "walk_forward_test",
    }


def health_summary(health: Dict[str, Any]) -> str:
    """One-line deterministic summary for narration (no invented numbers)."""
    lights = " ".join(
        {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴", "GREY": "⚪"}.get(l["status"], "⚪")
        for l in health.get("lights", [])
    )
    age = (health.get("staleness") or {}).get("age_days")
    age_txt = f", {age}d old" if age is not None else ""
    return f"{health.get('badge', '?')} [{lights}]{age_txt}"


async def get_strategy_health(strategy_id: str) -> Dict[str, Any]:
    """Load stored state for a strategy and compute its health card.

    Reads the ``StrategyVersion.backtest_results`` (latest gate report) and the
    probation record when one exists. Never re-runs a backtest.
    """
    from forex_ai_agent.strategy import create_version_manager
    from forex_ai_agent.autonomy.probation import ProbationStore

    vm = await create_version_manager()
    version = vm.get_version(strategy_id) or (
        vm.get_version(strategy_id[:8]) if len(strategy_id) >= 8 else None
    )
    if version is None:
        raise ValueError(f"unknown strategy version {strategy_id!r}")

    gate = dict(version.backtest_results or {}) or None
    probation_dict: Optional[Dict[str, Any]] = None
    try:
        prob = await ProbationStore().get(version.id)
        probation_dict = prob.to_dict() if prob else None
    except Exception:  # noqa: BLE001 — probation store unavailable is not fatal
        probation_dict = None

    health = compute_strategy_health(gate, probation_dict)

    return {
        "strategy_id": version.id,
        "strategy_name": version.name,
        "pair": (gate or {}).get("symbol")
        or ((version.parameters or {}).get("symbol"))
        or ((version.rules or {}).get("symbol"))
        or ((probation_dict or {}).get("pair")),
        "timeframe": (gate or {}).get("timeframe")
        or ((version.parameters or {}).get("timeframe")),
        "health": health,
        "summary": health_summary(health),
    }


def render_health_text(payload: Dict[str, Any]) -> str:
    """Deterministic text card for the CLI/tool result narration."""
    h = payload["health"]
    lines = [
        f"STRATEGY HEALTH — {payload.get('strategy_name', payload.get('strategy_id', '?'))}",
        f"Badge: {h['badge']}"
        + (" (auto-downgraded: stale evidence)" if h.get("auto_downgraded") else ""),
    ]
    for l in h["lights"]:
        marker = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴", "GREY": "⚪"}.get(l["status"], "⚪")
        as_of = str(l.get("as_of") or "no date")[:10]
        stale_tag = " [STALE]" if l.get("stale") else ""
        lines.append(f"  {marker} {l['name']:<15} {l['detail']}  (as of {as_of}){stale_tag}")
    st = h["staleness"]
    lines.append(f"  Staleness: {st['note']}")
    lines.append(f"  Recommendation: {h['recommendation']}")
    return "\n".join(lines)
