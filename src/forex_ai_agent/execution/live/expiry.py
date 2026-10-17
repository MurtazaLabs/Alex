"""Stage E4: order expiry — cancel unfilled orders after the declared TTL.

Market orders on an IOC filling mode fill or reject immediately, so this
tracker exists for the narrow case a broker accepts an order as PLACED
(pending) instead. Every placed order is tracked from its place time; when
it outlives ``expiry_seconds`` (default 300s = 5 minutes, from
``LiveCaps.order_expiry_seconds``) it is reported as DUE and the caller
must cancel it.

The tracker performs no broker call and holds no clock — the caller
supplies ``now`` and acts on ``due()`` results. Cancelling is a write
path and stays behind the same permission gate as everything else; this
module only answers "which orders have outlived their TTL".

Pure state over plain values: no network, no broker, no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

__all__ = [
    "TrackedOrder",
    "ExpiredOrder",
    "OrderExpiry",
]


@dataclass(frozen=True)
class TrackedOrder:
    """One placed-but-unresolved order."""

    ticket: int
    symbol: str
    placed_at: datetime
    side: str = ""
    volume: float = 0.0
    key: str = ""
    retcode: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "placed_at": self.placed_at.isoformat(),
            "side": self.side,
            "volume": self.volume,
            "key": self.key,
            "retcode": self.retcode,
        }


@dataclass(frozen=True)
class ExpiredOrder:
    """A tracked order past its TTL — cancel this."""

    order: TrackedOrder
    age_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.order.to_dict(),
            "age_seconds": round(self.age_seconds, 1),
        }


class OrderExpiry:
    """Deterministic TTL tracker for placed orders."""

    def __init__(self, expiry_seconds: int = 300) -> None:
        if int(expiry_seconds) <= 0:
            raise ValueError(
                f"expiry_seconds must be positive, got {expiry_seconds!r}"
            )
        self.expiry_seconds = int(expiry_seconds)
        self._orders: Dict[int, TrackedOrder] = {}

    # -- tracking --------------------------------------------------------

    def track(self, order: TrackedOrder) -> None:
        """Start (or refresh) TTL tracking for a placed order."""
        self._orders[int(order.ticket)] = order

    def resolve(self, ticket: int) -> Optional[TrackedOrder]:
        """Stop tracking (filled or cancelled); returns the dropped order."""
        return self._orders.pop(int(ticket), None)

    def resolve_all(self) -> List[TrackedOrder]:
        dropped = list(self._orders.values())
        self._orders.clear()
        return dropped

    # -- queries ---------------------------------------------------------

    def pending(self) -> List[TrackedOrder]:
        return list(self._orders.values())

    def age_seconds(self, ticket: int, now: Optional[datetime] = None) -> Optional[float]:
        order = self._orders.get(int(ticket))
        if order is None:
            return None
        check = now or datetime.utcnow()
        return (check - order.placed_at).total_seconds()

    def due(self, now: Optional[datetime] = None) -> List[ExpiredOrder]:
        """All tracked orders whose age >= expiry (oldest first)."""
        check = now or datetime.utcnow()
        ttl = timedelta(seconds=self.expiry_seconds)
        expired = [
            ExpiredOrder(order=o, age_seconds=(check - o.placed_at).total_seconds())
            for o in self._orders.values()
            if check - o.placed_at >= ttl
        ]
        expired.sort(key=lambda e: e.order.placed_at)
        return expired

    def __len__(self) -> int:
        return len(self._orders)
