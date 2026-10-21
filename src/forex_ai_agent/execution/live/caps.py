"""Stage E4: live deployment hard caps — deterministic constants in code.

These are the NON-NEGOTIABLE limits for the extremely limited live
deployment window. They are reviewed by the human operator and, since
2026-10-18, mirror "Personal Trading Mandate v1.0"
(ALEX_TRADING_MANDATE.md): 0.25% per trade, 0.75% aggregate stop risk,
1% daily halt, 2% weekly halt, 5% drawdown halt, max 3 positions. The
machine-readable source of truth is execution/risk_mandate.py; changing
values requires a code edit reviewed by the human.

This module is pure data: no network, no broker, no LLM. The transmitter
and the daily report both read caps from here, so a breach can always be
traced to a single constant.

Note on the lot band: ``min_lot``/``max_lot`` (0.01) reflect the current
$100 STANDARD demo account, where the mandatory per-trade risk budget
($0.25) is smaller than one minimum lot ($1.00 at a 10-pip stop) -- the
policy therefore refuses every trade, correctly. Deploying on a cent
account (planned) widens this band to the sizes the mandate can express;
that is a reviewed code edit, not a config change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..risk_mandate import ACTIVE_MANDATE

__all__ = [
    "E4_PAIR",
    "ALEX_LIVE_MAGIC",
    "LiveCaps",
    "DEFAULT_LIVE_CAPS",
    "london_window",
    "new_york_window",
]

#: The one pair the E4 window is allowed to trade. Singular on purpose:
#: widening the pair list is a new human sign-off, not a config change.
E4_PAIR = "EUR/USD"

#: Order magic for E4 live orders. Distinct from the E2 shadow magic
#: (314159) on purpose: the daily E2 read-only check asserts the broker's
#: deal history carries NO shadow-magic deals, so live orders must be
#: identifiable by a different stamp. The daily report filters on this.
ALEX_LIVE_MAGIC = 314160


def london_window() -> tuple:
    """London session window in UTC hours (start inclusive, end exclusive)."""
    return (7, 16)


def new_york_window() -> tuple:
    """New York session window in UTC hours (start inclusive, end exclusive)."""
    return (12, 21)


@dataclass(frozen=True)
class LiveCaps:
    """E4 caps. Frozen dataclass — every value a named constant.

    Mandate v1.0 (approved 2026-10-18): EUR/USD only, <= 3 concurrent
    positions, 0.25% risk per trade, 0.75% aggregate open stop risk,
    1% daily loss halt, 2% weekly loss halt, 5% drawdown halt from the
    equity high-water mark, London/NY sessions only, +/-30 min news
    blackout.
    """

    #: The single tradable pair for the window.
    pair: str = E4_PAIR
    #: Min-lot volumes only — the E4 window never scales up.
    min_lot: float = 0.01
    max_lot: float = 0.01
    #: Concurrent open positions on the pair.
    max_concurrent_positions: int = ACTIVE_MANDATE.max_concurrent_positions
    #: Per-trade risk ceiling as % of equity (must match PolicyCaps).
    max_risk_per_trade_pct: float = ACTIVE_MANDATE.max_risk_per_trade_pct
    #: Aggregate open stop risk ceiling as % of equity (must match PolicyCaps).
    max_aggregate_stop_risk_pct: float = (
        ACTIVE_MANDATE.max_aggregate_stop_risk_pct
    )
    #: Daily loss ceiling as % of the day's starting equity (auto-halt).
    max_daily_loss_pct: float = ACTIVE_MANDATE.max_daily_loss_pct
    #: Weekly loss ceiling as % of the week's starting equity (auto-halt).
    max_weekly_loss_pct: float = ACTIVE_MANDATE.max_weekly_loss_pct
    #: Drawdown halt from the equity high-water mark (persists until clear).
    max_drawdown_pct: float = ACTIVE_MANDATE.max_drawdown_pct
    #: News blackout half-window in minutes (before AND after each event).
    news_blackout_minutes: int = 30
    #: Unfilled order expiry in seconds (cancel after this).
    order_expiry_seconds: int = 300
    #: Fill drift tolerance in pips (live vs expected).
    fill_drift_tolerance_pips: float = 2.0
    #: Intent rejection-rate tolerance as % of attempts (shadow band).
    rejection_rate_tolerance_pct: float = 2.0
    #: Evaluation window length in days.
    evaluation_window_days: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": self.pair,
            "min_lot": self.min_lot,
            "max_lot": self.max_lot,
            "max_concurrent_positions": self.max_concurrent_positions,
            "max_risk_per_trade_pct": self.max_risk_per_trade_pct,
            "max_aggregate_stop_risk_pct": self.max_aggregate_stop_risk_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_weekly_loss_pct": self.max_weekly_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "news_blackout_minutes": self.news_blackout_minutes,
            "order_expiry_seconds": self.order_expiry_seconds,
            "fill_drift_tolerance_pips": self.fill_drift_tolerance_pips,
            "rejection_rate_tolerance_pct": self.rejection_rate_tolerance_pct,
            "evaluation_window_days": self.evaluation_window_days,
            "london_utc": list(london_window()),
            "new_york_utc": list(new_york_window()),
        }


#: The single canonical instance every live component defaults to.
DEFAULT_LIVE_CAPS = LiveCaps()
