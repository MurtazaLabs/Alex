"""Stage E4: position tracker — live broker truth into one book.

The tracker is the single place where broker reads become the numbers the
policy and the governor act on:

* ``sync_positions(positions)`` — snapshot the broker's open positions
  (caller supplies the read; this module never calls the terminal)
* ``record_deal(profit, ts)`` — accumulate realized P&L per UTC day
* ``book`` — current count / volume / floating P&L per symbol

Two feeds the live loop needs:

1. ``open_count()`` -> the policy's ``update_account_state(open_positions=)``
2. ``realized_pnl_today()`` -> the governor's daily-loss breaker input

The tracker holds no clock of its own: every mutation takes an explicit
timestamp so tests are deterministic and an audit can replay the day.

Pure state over plain data: no network, no broker, no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from ...utils.math_helpers import pip_size_for_symbol, pip_value_per_lot

__all__ = [
    "PositionBook",
    "EquitySync",
    "PositionTracker",
    "sync_account_into",
    "stop_risk_usd",
]

PositionLike = Any  # PositionSnapshot or dict-shaped equivalent


def _attr(position: PositionLike, name: str, default: Any = None) -> Any:
    if isinstance(position, dict):
        return position.get(name, default)
    return getattr(position, name, default)


def _normalize_side(side: Any) -> str:
    text = str(side or "").strip().upper()
    if text in ("LONG", "BUY", "POSITION_TYPE_BUY"):
        return "BUY"
    if text in ("SHORT", "SELL", "POSITION_TYPE_SELL"):
        return "SELL"
    return text


@dataclass(frozen=True)
class PositionBook:
    """Point-in-time summary of the broker's open positions."""

    count: int
    buy_count: int
    sell_count: int
    net_volume: float
    floating_pnl: float
    tickets: Sequence[int]
    symbols: Sequence[str]
    synced_at: datetime

    def count_for(self, symbol: str) -> int:
        return sum(1 for s in self.symbols if s == symbol)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "net_volume": self.net_volume,
            "floating_pnl": round(self.floating_pnl, 2),
            "tickets": list(self.tickets),
            "symbols": list(self.symbols),
            "synced_at": self.synced_at.isoformat(),
        }


@dataclass(frozen=True)
class EquitySync:
    """One account-sync reading — what the policy and governor are fed."""

    equity: float
    balance: float
    daily_loss_limit_usd: float
    open_positions: int
    realized_pnl_today: float
    synced_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equity": round(self.equity, 2),
            "balance": round(self.balance, 2),
            "daily_loss_limit_usd": round(self.daily_loss_limit_usd, 2),
            "open_positions": self.open_positions,
            "realized_pnl_today": round(self.realized_pnl_today, 2),
            "synced_at": self.synced_at.isoformat(),
        }


class PositionTracker:
    """Broker-position sync + per-day realized P&L ledger."""

    def __init__(self, *, symbol: str = "") -> None:
        self.symbol = symbol
        self._book: Optional[PositionBook] = None
        self._realized: Dict[str, float] = {}
        self._deals: List[Dict[str, Any]] = []

    # -- positions -------------------------------------------------------

    def sync_positions(
        self,
        positions: Iterable[PositionLike],
        *,
        now: Optional[datetime] = None,
    ) -> PositionBook:
        """Rebuild the book from the broker's current positions (broker = truth)."""
        synced_at = now or datetime.utcnow()
        rows = list(positions)

        buy = sell = 0
        net_volume = 0.0
        floating = 0.0
        tickets: List[int] = []
        symbols: List[str] = []

        for row in rows:
            side = _normalize_side(_attr(row, "side"))
            volume = float(_attr(row, "volume", 0.0) or 0.0)
            profit = float(_attr(row, "profit", 0.0) or 0.0)
            if side == "BUY":
                buy += 1
                net_volume += volume
            elif side == "SELL":
                sell += 1
                net_volume -= volume
            floating += profit
            tickets.append(int(_attr(row, "ticket", 0) or 0))
            symbols.append(str(_attr(row, "symbol", "") or ""))

        self._book = PositionBook(
            count=len(rows),
            buy_count=buy,
            sell_count=sell,
            net_volume=round(net_volume, 8),
            floating_pnl=floating,
            tickets=tuple(tickets),
            symbols=tuple(symbols),
            synced_at=synced_at,
        )
        return self._book

    @property
    def book(self) -> Optional[PositionBook]:
        """Last synced book, or None if never synced."""
        return self._book

    def open_count(self) -> int:
        return self._book.count if self._book else 0

    def open_count_on_pair(self, symbol: Optional[str] = None) -> int:
        target = symbol or self.symbol
        if not self._book:
            return 0
        if not target:
            return self._book.count
        return self._book.count_for(target)

    # -- realized P&L ----------------------------------------------------

    def record_deal(
        self,
        *,
        profit: float,
        ts: datetime,
        ticket: Optional[int] = None,
        symbol: Optional[str] = None,
        volume: float = 0.0,
        comment: str = "",
    ) -> float:
        """Bank one closed deal's profit; returns the new day total."""
        day_key = ts.strftime("%Y-%m-%d")
        self._realized[day_key] = self._realized.get(day_key, 0.0) + float(profit)
        self._deals.append({
            "ticket": ticket,
            "symbol": symbol or self.symbol,
            "volume": float(volume),
            "profit": float(profit),
            "day": day_key,
            "ts": ts.isoformat(),
            "comment": comment,
        })
        return self._realized[day_key]

    def realized_pnl_today(self, now: Optional[datetime] = None) -> float:
        day_key = (now or datetime.utcnow()).strftime("%Y-%m-%d")
        return self._realized.get(day_key, 0.0)

    def realized_pnl(self, day: str) -> float:
        return self._realized.get(day, 0.0)

    def deals(self) -> List[Dict[str, Any]]:
        return list(self._deals)

    # -- combined feed ---------------------------------------------------

    def sync_account(
        self,
        account: Any,
        *,
        daily_loss_pct: float,
        now: Optional[datetime] = None,
    ) -> EquitySync:
        """Combine an account snapshot + book + realized ledger into one feed.

        ``daily_loss_limit_usd`` is recomputed from THIS equity each call
        (equity * pct), so the governor's dollar limit follows the account.
        """
        synced_at = now or datetime.utcnow()
        equity = float(_attr(account, "equity", 0.0) or 0.0)
        balance = float(_attr(account, "balance", 0.0) or 0.0)
        return EquitySync(
            equity=equity,
            balance=balance,
            daily_loss_limit_usd=round(equity * (float(daily_loss_pct) / 100.0), 2),
            open_positions=self.open_count(),
            realized_pnl_today=self.realized_pnl_today(synced_at),
            synced_at=synced_at,
        )


def stop_risk_usd(
    positions: Iterable[PositionLike],
    *,
    symbol: str = "",
    contract_size: float = 100_000.0,
) -> Optional[float]:
    """Total stop-loss risk across open positions, in account currency.

    risk per position = |price_open - sl| (pips) * pip_value_per_lot * volume

    Returns None (FAIL CLOSED) when any position has no stop loss or no
    usable entry price/symbol -- the mandate then refuses new trades
    because the aggregate open risk cannot be verified.
    """
    total = 0.0
    for row in positions:
        sl = _attr(row, "sl")
        entry_price = _attr(row, "price_open")
        volume = float(_attr(row, "volume", 0.0) or 0.0)
        row_symbol = str(_attr(row, "symbol", "") or "")
        pair = row_symbol if "/" in row_symbol else symbol
        if "/" not in pair:
            return None
        try:
            sl_value = float(sl)
            entry_value = float(entry_price)
        except (TypeError, ValueError):
            return None
        if sl_value <= 0 or entry_value <= 0:
            return None
        pip_size = pip_size_for_symbol(pair)
        pips = abs(entry_value - sl_value) / pip_size
        total += pips * pip_value_per_lot(pair, contract_size) * abs(volume)
    return round(total, 4)


def sync_account_into(
    tracker: PositionTracker,
    account: Any,
    *,
    policy: Any = None,
    governor: Any = None,
    daily_loss_pct: float = 15.0,
    daily_pnl: Optional[float] = None,
    weekly_pnl: float = 0.0,
    day_start_equity: Optional[float] = None,
    week_start_equity: Optional[float] = None,
    equity_hwm: Optional[float] = None,
    open_stop_risk_usd: Optional[float] = None,
    now: Optional[datetime] = None,
) -> EquitySync:
    """Push one account sync into the policy and the governor.

    The policy gets ``update_account_state(...)`` with the tracker's feed
    plus any mandate baselines the caller supplies (the live window runner
    passes persisted day/week start equity, the equity high-water mark,
    and the verified aggregate open stop risk in USD).

    Legacy callers pass nothing extra: ``daily_pnl=None`` keeps the old
    ``realized_pnl_today`` behavior; the mandate fields stay dormant.
    The governor's dollar breaker gets recomputed from the fresh equity.
    """
    feed = tracker.sync_account(account, daily_loss_pct=daily_loss_pct, now=now)

    if policy is not None and hasattr(policy, "update_account_state"):
        policy.update_account_state(
            equity=feed.equity,
            open_positions=feed.open_positions,
            daily_pnl=(
                feed.realized_pnl_today if daily_pnl is None else daily_pnl
            ),
            weekly_pnl=weekly_pnl,
            day_start_equity=day_start_equity,
            week_start_equity=week_start_equity,
            equity_hwm=equity_hwm,
            open_stop_risk_usd=open_stop_risk_usd,
            now=feed.synced_at,
        )

    if governor is not None:
        limits = getattr(governor, "limits", None)
        if limits is not None:
            limits.daily_loss_limit = feed.daily_loss_limit_usd

    return feed
