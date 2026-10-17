"""Explainable overfitting score (Stage A2).

Replaces the single opaque ``degradation`` number with named components
computed from walk-forward windows — every value in the card traces to a
stored window row, so ``answer_why`` can explain a score without re-running
anything (roadmap A2, "why is the overfit score X?").
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Component weights for the composite score. Weights sum to 1.0.
_WEIGHTS = {
    "degradation": 0.4,
    "stability": 0.25,
    "negative_window_fraction": 0.2,
    "trade_concentration": 0.15,
}

# Thresholds for labeling components LOW / MEDIUM / HIGH.
_DEGRADATION_LABELS = ((0.6, "HIGH"), (0.3, "MEDIUM"), (-1.0, "LOW"))
_STABILITY_LABELS = ((0.5, "HIGH"), (0.25, "MEDIUM"), (-1.0, "LOW"))
_NEGFRAC_LABELS = ((0.5, "HIGH"), (0.25, "MEDIUM"), (-1.0, "LOW"))
_CONCENTRATION_LABELS = ((0.7, "HIGH"), (0.4, "MEDIUM"), (-1.0, "LOW"))


def _label(value: float, bands: Any) -> str:
    for cutoff, name in bands:
        if value >= cutoff:
            return name
    return "LOW"


def _param_sensitivity_placeholder(grid: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Shape of the A2b sensitivity component (measured when a grid was run)."""
    if not grid or not grid.get("variants"):
        return {
            "status": "NOT_MEASURED",
            "note": "run walk-forward with run_parameter_grid=True to measure",
            "worst_param_change": None,
            "oos_sharpe_range": None,
        }
    oos = [v.get("oos_sharpe") for v in grid["variants"] if v.get("oos_sharpe") is not None]
    return {
        "status": grid.get("status", "MEASURED"),
        "note": grid.get("note", ""),
        "variants_tested": len(grid.get("variants", [])),
        "oos_sharpe_min": min(oos) if oos else None,
        "oos_sharpe_max": max(oos) if oos else None,
        "worst_variant": grid.get("worst_variant"),
        "worst_param_change": grid.get("worst_param_change"),
        "oos_sharpe_range": (max(oos) - min(oos)) if oos else None,
    }


def compute_overfit_components(
    windows: List[Dict[str, Any]],
    oos_trades_total: Optional[int] = None,
    parameter_grid: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the component breakdown from stored WF window rows.

    Windows need ``is_sharpe`` and ``oos_sharpe`` keys (the same rows persisted
    by ``run_walk_forward_on_df`` and the gate's ``_walk_forward*`` helpers).
    """
    if not windows:
        return {
            "components": {},
            "score": None,
            "note": "no walk-forward windows available — score not computable",
        }

    is_sharpes = [float(w.get("is_sharpe", 0.0)) for w in windows]
    oos_sharpes = [float(w.get("oos_sharpe", 0.0)) for w in windows]

    mean_is = sum(is_sharpes) / len(is_sharpes)
    mean_oos = sum(oos_sharpes) / len(oos_sharpes)

    # 1. IS → OOS degradation. Replaces the old cliff (IS ≤ 0 forced 1.0) with
    # honest per-case reasoning: mean IS ≤ 0 means there was no edge in-sample
    # to degrade — flagged separately instead of saturating the score.
    if mean_is > 0:
        degradation = max(0.0, min(1.0, 1.0 - (mean_oos / mean_is)))
    else:
        degradation = 0.0 if mean_oos > 0 else 0.5

    # 2. Cross-window OOS stability (coefficient of variation, clipped 0–1).
    # A single catastrophic OOS window (collapse < −1.0 Sharpe) is instability
    # the CV formula can under-weight once it saturates, so a collapse forces
    # the component to HIGH — the exact "W3 collapse" pattern that exposed
    # real strategy f39b8d40.
    if len(oos_sharpes) > 1:
        variance = sum((s - mean_oos) ** 2 for s in oos_sharpes) / len(oos_sharpes)
        std_oos = variance ** 0.5
        stability = min(1.0, std_oos / abs(mean_oos)) if mean_oos != 0 else 1.0
    else:
        std_oos = 0.0
        stability = 0.0
    worst_oos = min(oos_sharpes)
    collapse = worst_oos < -1.0
    if collapse:
        stability = max(stability, 0.75)

    # 3. Fraction of OOS windows with no edge (Sharpe ≤ 0).
    neg_frac = sum(1 for s in oos_sharpes if s <= 0) / len(oos_sharpes)

    # 4. Trade concentration: share of all OOS trades captured by the best
    # window (A2c). Falls back to return concentration when trade counts are
    # missing (gate windows don't always carry oos_trades).
    trades = [w.get("oos_trades") for w in windows]
    if trades and all(isinstance(t, (int, float)) for t in trades):
        total = float(sum(trades))
        concentration = (max(trades) / total) if total > 0 else 0.0
        concentration_basis = "oos_trades"
    else:
        returns = [abs(float(w.get("oos_return_pct", w.get("oos_pf", 0.0)) or 0.0)) for w in windows]
        total_ret = sum(returns)
        concentration = (max(returns) / total_ret) if total_ret > 0 else 0.0
        concentration_basis = "abs(oos_return_pct)"

    components: Dict[str, Any] = {
        "degradation": {
            "value": round(degradation, 4),
            "label": _label(degradation, _DEGRADATION_LABELS),
            "detail": f"mean IS Sharpe {mean_is:.2f} → OOS {mean_oos:.2f}",
            "is_sharpe_mean": round(mean_is, 4),
            "oos_sharpe_mean": round(mean_oos, 4),
        },
        "stability": {
            "value": round(stability, 4),
            "label": _label(stability, _STABILITY_LABELS),
            "detail": (
                f"OOS Sharpe std {std_oos:.2f} across {len(windows)} windows"
                + (f"; collapse window {worst_oos:.1f}" if collapse else "")
            ),
            "oos_sharpe_std": round(std_oos, 4),
            "windows": len(windows),
            "worst_oos_sharpe": round(worst_oos, 4),
            "collapse_detected": collapse,
        },
        "negative_window_fraction": {
            "value": round(neg_frac, 4),
            "label": _label(neg_frac, _NEGFRAC_LABELS),
            "detail": (
                f"{sum(1 for s in oos_sharpes if s <= 0)} of {len(windows)} OOS "
                "windows had Sharpe ≤ 0"
            ),
            "negative_windows": sum(1 for s in oos_sharpes if s <= 0),
        },
        "trade_concentration": {
            "value": round(concentration, 4),
            "label": _label(concentration, _CONCENTRATION_LABELS),
            "detail": f"best window holds {concentration:.0%} of OOS activity ({concentration_basis})",
            "basis": concentration_basis,
        },
        "parameter_sensitivity": _param_sensitivity_placeholder(parameter_grid),
    }

    # Composite score: weighted sum of the four measured components. The
    # unmeasured sensitivity component contributes 0 but is reported — the
    # honest position is "sensitivity unknown", not "sensitivity great".
    score = sum(_WEIGHTS[name] * float(comp["value"]) for name, comp in components.items()
                if name in _WEIGHTS)
    score = max(0.0, min(1.0, score))

    note = None
    if mean_is <= 0:
        note = (
            "mean IS Sharpe ≤ 0 — there was no in-sample edge to degrade; "
            "degradation component set to "
            + ("0 (OOS still positive)" if mean_oos > 0 else "0.5 (no edge anywhere)")
        )

    return {
        "components": components,
        "weights": dict(_WEIGHTS),
        "score": round(score, 4),
        "window_count": len(windows),
        "note": note,
    }
