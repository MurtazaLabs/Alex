"""Personal Trading Mandate v1.0 -- machine-readable risk constants.

The human's mandate text lives in ``ALEX_TRADING_MANDATE.md`` at the repo
root. This module is its single machine-readable form: every risk value the
execution stack enforces is defined here once and referenced from
``PolicyCaps`` (intent validation) and ``LiveCaps`` (live window).

Values (mandate v1.0, approved by the human 2026-10-18 for implementation
BEFORE the 30-day window opens):

- per-trade stop risk     <= 0.25% of equity
- aggregate open stop risk <= 0.75% of equity (all positions + pending)
- daily loss halt          >= 1.0% of the day's starting equity
- weekly loss halt         >= 2.0% of the week's starting equity
- drawdown halt            >= 5.0% from the equity high-water mark
- concurrent positions     <= 3

Deterministic constants in code: changing them requires a code edit
reviewed by the human. No tool, LLM call, or agent handler can change them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

__all__ = [
    "RiskMandate",
    "MANDATE_V1",
    "ACTIVE_MANDATE",
]


@dataclass(frozen=True)
class RiskMandate:
    """One frozen mandate revision -- every value a named constant."""

    #: Mandate document revision (ALEX_TRADING_MANDATE.md).
    version: str
    #: Per-trade stop risk ceiling, % of equity (0.25 -> 0.25%).
    max_risk_per_trade_pct: float
    #: Aggregate open stop risk ceiling, % of equity (positions + pending).
    max_aggregate_stop_risk_pct: float
    #: Daily loss halt threshold, % of the day's starting equity.
    max_daily_loss_pct: float
    #: Weekly loss halt threshold, % of the week's starting equity.
    max_weekly_loss_pct: float
    #: Drawdown halt threshold, % from the equity high-water mark.
    max_drawdown_pct: float
    #: Maximum concurrent open positions.
    max_concurrent_positions: int
    #: Date the human approved this revision.
    approved_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "max_risk_per_trade_pct": self.max_risk_per_trade_pct,
            "max_aggregate_stop_risk_pct": self.max_aggregate_stop_risk_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_weekly_loss_pct": self.max_weekly_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_concurrent_positions": self.max_concurrent_positions,
            "approved_at": self.approved_at,
        }


#: Mandate v1.0 -- the human's "Personal Trading Mandate" document.
MANDATE_V1 = RiskMandate(
    version="1.0",
    max_risk_per_trade_pct=0.25,
    max_aggregate_stop_risk_pct=0.75,
    max_daily_loss_pct=1.0,
    max_weekly_loss_pct=2.0,
    max_drawdown_pct=5.0,
    max_concurrent_positions=3,
    approved_at="2026-10-18",
)

#: The single canonical instance the execution stack defaults to.
ACTIVE_MANDATE = MANDATE_V1
