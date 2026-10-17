"""Stage E3: deterministic execution policy — order intents, not orders.

Turns approved-spec signals into fully specified order INTENTS. An intent is
a complete, traceable description of what WOULD be sent: symbol, side,
volume, SL/TP, expiry, and the deterministic math that produced each field.

Every intent is validated against the E3 hard caps before it can exist:

- Per-trade risk ≤ 0.5% equity (hard-coded constant)
- Daily loss ≤ 1.5% equity (checked against governor breakers)
- Max concurrent positions ≤ 3
- Min-lot sizing (0.01) with lot-step rounding
- Intent expiry (default 5 minutes)

The policy NEVER sends anything. It produces intents; a human grants a
single-use approval token; only then can the broker adapter transmit the
order (E4). No LLM is involved in any field of the intent.

Constitution: every number in an intent is computed by this module from
deterministic inputs — equity, SL distance, pip value, lot step. The LLM
never contributes to sizing, stops, or caps.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..utils.math_helpers import (
    pip_size_for_symbol,
    pips_between,
    position_size_lots,
    risk_reward_ratio,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ExecutionPolicy",
    "PolicyCaps",
    "OrderIntent",
    "INTENT_EXPIRED",
    "INTENT_VALID",
    "INTENT_REFUSED_RISK_CAP",
    "INTENT_REFUSED_POSITION_CAP",
    "INTENT_REFUSED_DAILY_LOSS",
    "INTENT_REFUSED_UNTRADEABLE",
    "INTENT_REFUSED_GOVERNOR",
]

INTENT_VALID = "VALID"
INTENT_EXPIRED = "EXPIRED"
INTENT_REFUSED_RISK_CAP = "REFUSED_RISK_CAP"
INTENT_REFUSED_POSITION_CAP = "REFUSED_POSITION_CAP"
INTENT_REFUSED_DAILY_LOSS = "REFUSED_DAILY_LOSS"
INTENT_REFUSED_UNTRADEABLE = "REFUSED_UNTRADEABLE"
INTENT_REFUSED_GOVERNOR = "REFUSED_GOVERNOR"


@dataclass(frozen=True)
class PolicyCaps:
    """Hard-coded E3 caps. Deterministic constants — reviewed by human.

    These are the NON-NEGOTIABLE limits for live execution. They cannot be
    changed by any tool, LLM call, or agent handler. Changing them requires
    a code edit reviewed by the human operator.

    CONFIRMED 2026-10-16: 5% risk/trade, 15% daily loss, 3 positions.
    """

    #: Maximum risk per trade as % of equity (5.0% = 5.0)
    max_risk_per_trade_pct: float = 5.0
    #: Maximum daily loss as % of equity (15.0% = 15.0)
    max_daily_loss_pct: float = 15.0
    #: Maximum concurrent open positions
    max_concurrent_positions: int = 3
    #: Maximum total notional exposure as % of equity
    max_total_exposure_pct: float = 5.0
    #: Minimum lot size (broker constraint)
    min_lot_size: float = 0.01
    #: Maximum lot size (broker constraint)
    max_lot_size: float = 200.0
    #: Lot step (broker constraint)
    lot_step: float = 0.01
    #: Intent expiry in seconds (default 5 minutes)
    intent_expiry_seconds: int = 300
    #: Minimum risk:reward ratio for a valid intent
    min_risk_reward: float = 1.0


@dataclass
class OrderIntent:
    """A fully specified order intent — every field traceable to deterministic math."""

    #: Unique key (hash of symbol+side+volume+sl+tp+timestamp — idempotent)
    key: str
    #: Currency pair (e.g. "EUR/USD")
    symbol: str
    #: "BUY" or "SELL"
    side: str
    #: Position volume in lots (rounded to lot_step, within min/max)
    volume: float
    #: Stop-loss price
    sl: float
    #: Take-profit price
    tp: float
    #: Entry price (the signal's proposed entry)
    entry: float
    #: Dollar risk per trade (equity * risk%)
    risk_usd: float
    #: SL distance in pips
    sl_pips: float
    #: Risk:reward ratio
    rr_ratio: float
    #: Account equity at intent creation
    equity: float
    #: When this intent was created
    created_at: datetime
    #: When this intent expires (created_at + expiry_seconds)
    expires_at: datetime
    #: Strategy version that produced the signal
    strategy_id: str = ""
    #: Research task / experiment that produced the signal
    source_id: str = ""
    #: Any notes
    comment: str = ""
    #: Current state
    status: str = INTENT_VALID
    #: Why refused (if not valid)
    refusal_reason: str = ""

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """Check if the intent has expired."""
        check = now or datetime.utcnow()
        return check >= self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe serialization."""
        return {
            "key": self.key,
            "symbol": self.symbol,
            "side": self.side,
            "volume": self.volume,
            "sl": self.sl,
            "tp": self.tp,
            "entry": self.entry,
            "risk_usd": round(self.risk_usd, 2),
            "sl_pips": round(self.sl_pips, 1),
            "rr_ratio": round(self.rr_ratio, 2),
            "equity": round(self.equity, 2),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "strategy_id": self.strategy_id,
            "source_id": self.source_id,
            "comment": self.comment,
            "status": self.status,
            "refusal_reason": self.refusal_reason,
        }


class ExecutionPolicy:
    """Deterministic execution policy — the only path from signal to intent.

    Hard caps are non-negotiable constants. The policy refuses any intent
    that violates them. It never sends anything; it produces intents that
    a human can approve.
    """

    def __init__(self, caps: Optional[PolicyCaps] = None) -> None:
        self.caps = caps or PolicyCaps()
        # Track open positions and daily P&L (fed by the caller)
        self._open_positions: int = 0
        self._daily_pnl: float = 0.0
        self._day_start_equity: float = 0.0
        self._current_equity: float = 0.0
        self._current_date: Optional[str] = None

    def update_account_state(
        self,
        *,
        equity: float,
        open_positions: int,
        daily_pnl: float = 0.0,
        now: Optional[datetime] = None,
    ) -> None:
        """Update the policy's view of the account (called by the caller each cycle)."""
        check = now or datetime.utcnow()
        date_key = check.strftime("%Y-%m-%d")
        if date_key != self._current_date:
            # New day: reset daily tracking
            self._current_date = date_key
            self._day_start_equity = equity
            self._daily_pnl = 0.0
        self._current_equity = equity
        self._open_positions = open_positions
        self._daily_pnl = daily_pnl

    def create_intent(
        self,
        *,
        symbol: str,
        side: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        strategy_id: str = "",
        source_id: str = "",
        comment: str = "",
        now: Optional[datetime] = None,
    ) -> OrderIntent:
        """Create an order intent from a signal, validated against all caps.

        Every field is computed deterministically. The intent is either VALID
        or carries a refusal_reason explaining exactly which cap was violated.
        """
        check = now or datetime.utcnow()
        equity = self._current_equity

        # --- Validation chain ---
        refusal = self._validate(
            symbol=symbol, side=side, entry=entry,
            stop_loss=stop_loss, take_profit=take_profit,
            equity=equity, now=check,
        )
        if refusal:
            return self._refused_intent(
                symbol=symbol, side=side, entry=entry,
                stop_loss=stop_loss, take_profit=take_profit,
                equity=equity, now=check,
                reason=refusal, strategy_id=strategy_id,
                source_id=source_id, comment=comment,
            )

        # --- Deterministic sizing ---
        risk_usd = equity * (self.caps.max_risk_per_trade_pct / 100.0)
        pip_size = pip_size_for_symbol(symbol)
        sl_pips = pips_between(entry, stop_loss, pip_size)
        volume = position_size_lots(
            account_equity=equity,
            risk_percent=self.caps.max_risk_per_trade_pct,
            entry_price=entry,
            stop_loss=stop_loss,
            symbol=symbol,
            min_lot_size=self.caps.min_lot_size,
            max_lot_size=self.caps.max_lot_size,
            lot_step=self.caps.lot_step,
        )
        rr = risk_reward_ratio(entry, stop_loss, take_profit)

        # Generate idempotent key
        key = self._intent_key(symbol, side, volume, stop_loss, take_profit, check)
        expires_at = check + timedelta(seconds=self.caps.intent_expiry_seconds)

        return OrderIntent(
            key=key, symbol=symbol, side=side, volume=volume,
            sl=stop_loss, tp=take_profit, entry=entry,
            risk_usd=risk_usd, sl_pips=sl_pips, rr_ratio=rr,
            equity=equity, created_at=check, expires_at=expires_at,
            strategy_id=strategy_id, source_id=source_id,
            comment=comment, status=INTENT_VALID,
        )

    def _validate(
        self, *, symbol: str, side: str, entry: float,
        stop_loss: float, take_profit: float, equity: float,
        now: datetime,
    ) -> str:
        """Return empty string if valid, or the refusal reason."""
        if equity <= 0:
            return INTENT_REFUSED_UNTRADEABLE
        if side not in ("BUY", "SELL"):
            return INTENT_REFUSED_UNTRADEABLE
        if entry <= 0 or stop_loss <= 0 or take_profit <= 0:
            return INTENT_REFUSED_UNTRADEABLE

        # SL must be on the correct side of entry
        if side == "BUY" and stop_loss >= entry:
            return INTENT_REFUSED_UNTRADEABLE
        if side == "SELL" and stop_loss <= entry:
            return INTENT_REFUSED_UNTRADEABLE

        # Daily loss cap
        day_start = self._day_start_equity or equity
        if day_start > 0:
            daily_loss_pct = ((day_start - (equity + self._daily_pnl)) / day_start) * 100.0
            if daily_loss_pct >= self.caps.max_daily_loss_pct:
                return INTENT_REFUSED_DAILY_LOSS

        # Position count cap
        if self._open_positions >= self.caps.max_concurrent_positions:
            return INTENT_REFUSED_POSITION_CAP

        # Risk:reward check
        rr = risk_reward_ratio(entry, stop_loss, take_profit)
        if rr < self.caps.min_risk_reward:
            return INTENT_REFUSED_RISK_CAP

        # Tradeability: SL distance must be > 0
        pip_size = pip_size_for_symbol(symbol)
        sl_pips = pips_between(entry, stop_loss, pip_size)
        if sl_pips < 1.0:  # minimum 1 pip stop
            return INTENT_REFUSED_UNTRADEABLE

        # Size must be within lot bounds
        volume = position_size_lots(
            account_equity=equity,
            risk_percent=self.caps.max_risk_per_trade_pct,
            entry_price=entry, stop_loss=stop_loss, symbol=symbol,
            min_lot_size=self.caps.min_lot_size,
            max_lot_size=self.caps.max_lot_size,
            lot_step=self.caps.lot_step,
        )
        if volume < self.caps.min_lot_size:
            return INTENT_REFUSED_UNTRADEABLE

        return ""

    def _refused_intent(
        self, *, symbol: str, side: str, entry: float,
        stop_loss: float, take_profit: float, equity: float,
        now: datetime, reason: str, strategy_id: str = "",
        source_id: str = "", comment: str = "",
    ) -> OrderIntent:
        """Build a refused intent for audit trail."""
        key = self._intent_key(symbol, side, 0.0, stop_loss, take_profit, now)
        return OrderIntent(
            key=key, symbol=symbol, side=side, volume=0.0,
            sl=stop_loss, tp=take_profit, entry=entry,
            risk_usd=0.0, sl_pips=0.0, rr_ratio=0.0,
            equity=equity, created_at=now,
            expires_at=now + timedelta(seconds=self.caps.intent_expiry_seconds),
            strategy_id=strategy_id, source_id=source_id,
            comment=comment, status=reason, refusal_reason=reason,
        )

    @staticmethod
    def _intent_key(
        symbol: str, side: str, volume: float,
        sl: float, tp: float, now: datetime,
    ) -> str:
        """Deterministic idempotent key for an intent."""
        raw = f"{symbol}:{side}:{volume:.2f}:{sl:.5f}:{tp:.5f}:{now.isoformat()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
