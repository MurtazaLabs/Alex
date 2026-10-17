"""Stage E4: 30-day evaluation — automated scoring vs declared criteria.

At the close of the declared evaluation window the human decides: extend
(new sign-off), widen (new sign-off), or retire. This module makes that
decision AUDITABLE by scoring exactly the criteria the Stage E plan
declared, from supplied numbers:

* (a) zero safety violations unresolved
* (b) live P&L within the shadow-predicted tolerance band
* (c) reconciliation exact throughout (no fill drift beyond tolerance,
       no volume mismatches)
* (d) the full window elapsed and the audit trail is complete
       (every live order accounted for)

Verdict is a RECOMMENDATION for the human, never an action: the module
recommends ELIGIBLE_FOR_DECISION when all criteria pass, else EXTEND_REVIEW
(or RETIRE_RECOMMENDED when safety criteria failed hard — the default is
always DOWN). Nothing here changes permission levels.

Pure arithmetic over supplied aggregates: no network, no broker, no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Optional

__all__ = [
    "CRITERION_SAFETY",
    "CRITERION_PNL_BAND",
    "CRITERION_RECONCILIATION",
    "CRITERION_AUDIT",
    "EVAL_ELIGIBLE",
    "EVAL_EXTEND_REVIEW",
    "EVAL_RETIRE_RECOMMENDED",
    "EvaluationCriterion",
    "EvaluationReport",
    "evaluate_window",
]

CRITERION_SAFETY = "safety_violations"
CRITERION_PNL_BAND = "pnl_within_tolerance"
CRITERION_RECONCILIATION = "reconciliation_exact"
CRITERION_AUDIT = "audit_complete"

EVAL_ELIGIBLE = "ELIGIBLE_FOR_DECISION"
EVAL_EXTEND_REVIEW = "EXTEND_REVIEW"
EVAL_RETIRE_RECOMMENDED = "RETIRE_RECOMMENDED"


@dataclass(frozen=True)
class EvaluationCriterion:
    """One declared criterion and its measured outcome."""

    name: str
    passed: bool
    detail: str
    measured: Optional[float] = None
    declared_limit: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "measured": self.measured,
            "declared_limit": self.declared_limit,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """The window's scoreboard — a recommendation for the human."""

    window_start: date
    window_end: date
    elapsed_days: int
    declared_days: int
    criteria: tuple
    verdict: str
    live_trades: int = 0
    live_pnl: float = 0.0
    shadow_predicted_pnl: float = 0.0
    tolerance_pct: float = 0.0

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.criteria)

    def failed_criteria(self):
        return [c for c in self.criteria if not c.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "elapsed_days": self.elapsed_days,
            "declared_days": self.declared_days,
            "criteria": [c.to_dict() for c in self.criteria],
            "verdict": self.verdict,
            "all_passed": self.all_passed,
            "live_trades": self.live_trades,
            "live_pnl": round(self.live_pnl, 2),
            "shadow_predicted_pnl": round(self.shadow_predicted_pnl, 2),
            "tolerance_pct": self.tolerance_pct,
        }


def evaluate_window(
    *,
    window_start: date,
    window_end: date,
    declared_days: int = 30,
    live_trades: int = 0,
    live_pnl: float = 0.0,
    shadow_predicted_pnl: float = 0.0,
    tolerance_pct: float = 20.0,
    unresolved_violations: int = 0,
    fill_drift_count: int = 0,
    volume_mismatch_count: int = 0,
    orders_sent: int = 0,
    orders_accounted: int = 0,
    safety_hard_fail: bool = False,
) -> EvaluationReport:
    """Score the window against the four declared acceptance criteria.

    ``tolerance_pct`` is the P&L band width as % of the shadow-predicted
    magnitude: |live - predicted| <= |predicted| * tolerance_pct/100.
    A zero predicted P&L requires live P&L 0.0 to pass (no band to hide in).
    """
    elapsed = (window_end - window_start).days

    # (a) safety: no unresolved violations
    safety_pass = unresolved_violations == 0 and not safety_hard_fail
    safety = EvaluationCriterion(
        name=CRITERION_SAFETY,
        passed=safety_pass,
        detail=(
            "all violations resolved" if safety_pass
            else f"{unresolved_violations} unresolved violation(s)"
            + (" + hard fail" if safety_hard_fail else "")
        ),
        measured=float(unresolved_violations),
        declared_limit=0.0,
    )

    # (b) P&L within the shadow-predicted tolerance band
    if shadow_predicted_pnl == 0.0:
        deviation = abs(live_pnl)
        pnl_pass = deviation == 0.0
        band_text = "no predicted band (0.0) => exact zero required"
    else:
        deviation_pct = (
            abs(live_pnl - shadow_predicted_pnl)
            / abs(shadow_predicted_pnl) * 100.0
        )
        deviation = deviation_pct
        pnl_pass = deviation_pct <= float(tolerance_pct)
        band_text = f"deviation {deviation_pct:.1f}% <= {tolerance_pct:.1f}%"
    pnl = EvaluationCriterion(
        name=CRITERION_PNL_BAND,
        passed=pnl_pass,
        detail=(
            f"live {live_pnl:+.2f} vs predicted {shadow_predicted_pnl:+.2f} "
            f"({band_text} => {'inside' if pnl_pass else 'outside'} band)"
        ),
        measured=round(deviation, 4),
        declared_limit=float(tolerance_pct),
    )

    # (c) reconciliation exact throughout
    recon_pass = fill_drift_count == 0 and volume_mismatch_count == 0
    recon = EvaluationCriterion(
        name=CRITERION_RECONCILIATION,
        passed=recon_pass,
        detail=(
            "no drift, no volume mismatch" if recon_pass
            else f"drift {fill_drift_count}, volume mismatches {volume_mismatch_count}"
        ),
        measured=float(fill_drift_count + volume_mismatch_count),
        declared_limit=0.0,
    )

    # (d) window elapsed + audit complete
    audit_pass = elapsed >= int(declared_days) and orders_sent == orders_accounted
    audit = EvaluationCriterion(
        name=CRITERION_AUDIT,
        passed=audit_pass,
        detail=(
            f"{elapsed}/{declared_days} days; orders {orders_accounted}/{orders_sent} accounted"
        ),
        measured=float(elapsed),
        declared_limit=float(declared_days),
    )

    criteria = (safety, pnl, recon, audit)

    if safety_hard_fail or unresolved_violations > 0:
        verdict = EVAL_RETIRE_RECOMMENDED
    elif all(c.passed for c in criteria):
        verdict = EVAL_ELIGIBLE
    else:
        verdict = EVAL_EXTEND_REVIEW

    return EvaluationReport(
        window_start=window_start,
        window_end=window_end,
        elapsed_days=elapsed,
        declared_days=int(declared_days),
        criteria=criteria,
        verdict=verdict,
        live_trades=int(live_trades),
        live_pnl=float(live_pnl),
        shadow_predicted_pnl=float(shadow_predicted_pnl),
        tolerance_pct=float(tolerance_pct),
    )
