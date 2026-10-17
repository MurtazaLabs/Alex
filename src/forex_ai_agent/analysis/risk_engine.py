"""
risk_engine.py — Trade validation for the Forex AI trading system.

All check_*() functions return (is_valid: bool, errors: List[ValidationError]).
Call validate_trade() for a single-pass full validation sweep.
"""

# ---------------------------------------------------------------------------
# Imports  (must come before constants and class definitions)
# ---------------------------------------------------------------------------

from __future__ import annotations  # enables PEP 604 "X | Y" unions on Python 3.9

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FIX: was MIN_R:R_RATIO — colons are illegal in Python identifiers.
MIN_RR_RATIO: float = 1.0

MAX_SL_DISTANCE_PIPS: float = 200.0
MIN_SL_DISTANCE_PIPS: float = 3.0
MAX_TOTAL_EXPOSURE_PCT: float = 20.0
MAX_LEVERAGE: float = 30.0

# ---------------------------------------------------------------------------
# Data classes  (must be defined before the check functions that reference them)
# ---------------------------------------------------------------------------


@dataclass
class ValidationError:
    """A single validation failure or advisory warning."""

    code: str
    message: str
    severity: str = "error"  # "error" | "warning"


@dataclass
class ValidationResult:
    """Aggregated result of a full validation sweep."""

    valid: bool
    errors: List[ValidationError] = field(default_factory=list)

    # Convenience accessors so callers can distinguish hard blocks from advisories.
    @property
    def hard_errors(self) -> List[ValidationError]:
        return [e for e in self.errors if e.severity == "error"]

    @property
    def warnings(self) -> List[ValidationError]:
        return [e for e in self.errors if e.severity == "warning"]


# ---------------------------------------------------------------------------
# Core validation checks
# ---------------------------------------------------------------------------


def check_max_risk_per_trade(
    account_equity: float,
    risk_percent: float,
    proposed_risk_usd: float,
) -> Tuple[bool, List[ValidationError]]:
    """Enforce max risk-per-trade as a percentage of account equity."""
    errors: List[ValidationError] = []
    max_risk = account_equity * (risk_percent / 100.0)
    if proposed_risk_usd > max_risk:
        errors.append(
            ValidationError(
                code="MAX_RISK_PER_TRADE",
                message=(
                    f"Proposed risk ${proposed_risk_usd:.2f} exceeds "
                    f"{risk_percent}% of account equity (${max_risk:.2f})"
                ),
            )
        )
    return len(errors) == 0, errors


def check_min_risk_reward(
    # FIX: was "r:r_ratio" — colons are illegal in Python parameter names.
    rr_ratio: float,
) -> Tuple[bool, List[ValidationError]]:
    """Enforce minimum risk : reward ratio."""
    errors: List[ValidationError] = []
    # FIX: was "if r:r_ratio < MIN_R:R_RATIO" — both tokens had illegal colons.
    if rr_ratio < MIN_RR_RATIO:
        errors.append(
            ValidationError(
                code="INSUFFICIENT_R:R",  # string literal — colon is valid here
                # FIX: was f"...{r:r_ratio:.2f}...{MIN_R:R_RATIO}" — Python parsed
                # "r" as the variable and ":r_ratio:.2f" as a malformed format spec.
                message=f"Risk:Reward = {rr_ratio:.2f} < minimum {MIN_RR_RATIO}",
            )
        )
    return len(errors) == 0, errors


def check_sl_distance(
    sl_pips: float,
) -> Tuple[bool, List[ValidationError]]:
    """Enforce stop-loss distance within sane pip bounds."""
    errors: List[ValidationError] = []
    if sl_pips > MAX_SL_DISTANCE_PIPS:
        errors.append(
            ValidationError(
                code="SL_TOO_FAR",
                message=(
                    f"Stop loss {sl_pips:.1f} pips exceeds "
                    f"maximum {MAX_SL_DISTANCE_PIPS} pips"
                ),
            )
        )
    if sl_pips < MIN_SL_DISTANCE_PIPS:
        errors.append(
            ValidationError(
                code="SL_TOO_TIGHT",
                message=(
                    f"Stop loss {sl_pips:.3f} pips is below "
                    f"minimum {MIN_SL_DISTANCE_PIPS} pips"
                ),
            )
        )
    return len(errors) == 0, errors


def check_max_exposure(
    current_exposure_pct: float,
    proposed_exposure_pct: float,
) -> Tuple[bool, List[ValidationError]]:
    """Enforce max total portfolio exposure."""
    errors: List[ValidationError] = []
    total = current_exposure_pct + proposed_exposure_pct
    if total > MAX_TOTAL_EXPOSURE_PCT:
        errors.append(
            ValidationError(
                code="EXPOSURE_EXCEEDED",
                message=(
                    f"Total exposure {total:.1f}% exceeds "
                    f"{MAX_TOTAL_EXPOSURE_PCT}% limit"
                ),
            )
        )
    return len(errors) == 0, errors


def check_leverage(
    leverage: float,
) -> Tuple[bool, List[ValidationError]]:
    """Enforce maximum leverage."""
    errors: List[ValidationError] = []
    if leverage > MAX_LEVERAGE:
        errors.append(
            ValidationError(
                code="LEVERAGE_EXCEEDED",
                message=f"Leverage {leverage:.1f}x exceeds maximum {MAX_LEVERAGE:.0f}x",
            )
        )
    return len(errors) == 0, errors


def check_news_filter(
    news_state: str,
) -> Tuple[bool, List[ValidationError]]:
    """Block trading when high-impact news is active."""
    errors: List[ValidationError] = []
    if news_state == "HIGH":
        errors.append(
            ValidationError(
                code="HIGH_IMPACT_NEWS",
                message="High-impact news detected — trading blocked",
                severity="warning",
            )
        )
    return len(errors) == 0, errors


def check_session_filter(
    session: str | None,
    allowed_sessions: List[str] | None,
) -> Tuple[bool, List[ValidationError]]:
    """Restrict trading to allowed market sessions."""
    errors: List[ValidationError] = []
    if session and allowed_sessions and session not in allowed_sessions:
        errors.append(
            ValidationError(
                code="SESSION_NOT_ALLOWED",
                message=(
                    f"Session '{session}' not in "
                    f"allowed sessions {allowed_sessions}"
                ),
            )
        )
    return len(errors) == 0, errors


def check_daily_loss_limit(
    daily_pnl: float,
    daily_loss_limit_pct: float = 5.0,
    account_equity: float = 10_000.0,
) -> Tuple[bool, List[ValidationError]]:
    """Hard stop when the daily loss limit has been breached."""
    errors: List[ValidationError] = []
    loss_pct = abs(daily_pnl) / account_equity * 100.0
    if loss_pct > daily_loss_limit_pct:
        errors.append(
            ValidationError(
                code="DAILY_LOSS_LIMIT",
                message=(
                    f"Daily loss {loss_pct:.1f}% exceeds "
                    f"{daily_loss_limit_pct}% limit"
                ),
            )
        )
    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Public API — single entry-point that runs all checks
# ---------------------------------------------------------------------------


def validate_trade(
    account_equity: float,
    risk_percent: float,
    proposed_risk_usd: float,
    rr_ratio: float,
    sl_pips: float,
    current_exposure_pct: float,
    proposed_exposure_pct: float,
    leverage: float,
    news_state: str,
    daily_pnl: float = 0.0,
    daily_loss_limit_pct: float = 5.0,
    session: Optional[str] = None,
    allowed_sessions: Optional[List[str]] = None,
) -> ValidationResult:
    """
    Run every validation check against a proposed trade in one call.

    Returns a ValidationResult whose .valid is True only when every check
    passes — including advisory warnings such as high-impact news, because
    in a risk engine any flagged condition should block execution.
    Callers can inspect .hard_errors and .warnings separately for UI display.
    """
    all_errors: List[ValidationError] = []

    # Order matters for UX: surfaced from most critical to most contextual.
    check_results = [
        check_daily_loss_limit(daily_pnl, daily_loss_limit_pct, account_equity),
        check_max_risk_per_trade(account_equity, risk_percent, proposed_risk_usd),
        check_min_risk_reward(rr_ratio),
        check_sl_distance(sl_pips),
        check_max_exposure(current_exposure_pct, proposed_exposure_pct),
        check_leverage(leverage),
        check_news_filter(news_state),
        check_session_filter(session, allowed_sessions),
    ]

    for _is_valid, errors in check_results:
        all_errors.extend(errors)

    # Any error or warning blocks the trade — this is a safety-first risk engine.
    return ValidationResult(valid=not bool(all_errors), errors=all_errors)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Data classes
    "ValidationError",
    "ValidationResult",
    # Public API
    "validate_trade",
    # Individual checks (exposed for unit testing and selective use)
    "check_max_risk_per_trade",
    "check_min_risk_reward",
    "check_sl_distance",
    "check_max_exposure",
    "check_leverage",
    "check_news_filter",
    "check_session_filter",
    "check_daily_loss_limit",
    # Constants
    "MIN_RR_RATIO",
    "MAX_SL_DISTANCE_PIPS",
    "MIN_SL_DISTANCE_PIPS",
    "MAX_TOTAL_EXPOSURE_PCT",
    "MAX_LEVERAGE",
]