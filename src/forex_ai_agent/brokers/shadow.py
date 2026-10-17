"""Stage E2: shadow ledger — the would-have-filled book.

Live signals become intents that run through the REAL validation chain
(governor → write-ahead journal → broker symbol spec → position caps) and
are then recorded here **instead of being sent**. Every monetary number the
ledger shows — fill price, commission, swap, SL/TP exit, P&L — is produced
by the same :class:`~forex_ai_agent.backtest.orders.OrderManager` and
:class:`~forex_ai_agent.backtest.orders.PositionManager` the backtest and
paper layers use, so the shadow book can never disagree with the engine
about what a fill would have cost. No second cost model exists anywhere in
this file.

Fill accounting (one bar, two spread sources):

* ``predicted_fill_price`` — the E0 cost model evaluated on the fill bar
  with the MODEL spread (``DEFAULT_SPREADS``, what the simulator assumed).
* ``actual_fill_price`` — the same cost model on that same real bar with
  the LIVE broker spread read from the terminal (``SymbolSpec``), rounded
  to the broker's own ``digits``.
* ``fill_drift_pips`` — the adverse displacement of ``actual`` against
  ``predicted`` (positive = the real market filled worse than the model
  assumed). Both sides share the bar, the ATR and the slippage model, so
  the number isolates real execution cost from the model's estimate.

Would-have P&L then rides the real bars to SL/TP (evaluated against the
bar's high/low, exactly as the engine does) or to an explicit window close.

No network, no broker calls, no LLM, no permission changes: pure state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..backtest.execution import (
    DEFAULT_SPREADS,
    FixedSpreadModel,
    OrderSide,
    OrderType,
)
from ..backtest.orders import ExecutionContext, OrderManager, PositionManager
from ..utils.math_helpers import pip_size_for_symbol
from .base import ALEX_SHADOW_MAGIC, ReadonlyBroker, SymbolSpec
from .reconcile import ReconciliationReport, reconcile

__all__ = [
    "SHADOW_PENDING",
    "SHADOW_OPEN",
    "SHADOW_CLOSED",
    "SHADOW_REJECTED",
    "ShadowError",
    "ShadowEntry",
    "ShadowLedger",
    "ShadowExecutor",
    "engine_atr",
    "session_for",
]

SHADOW_PENDING = "PENDING_FILL"
SHADOW_OPEN = "OPEN"
SHADOW_CLOSED = "CLOSED"
SHADOW_REJECTED = "REJECTED"

# Rejection reasons — each one becomes an ``intent_rejected_*`` flag, the
# D3 change-only flag pattern applied to the execution path.
REJECT_GOVERNOR = "governor_refused"
REJECT_DUPLICATE = "duplicate_key"
REJECT_SPEC_UNAVAILABLE = "symbol_spec_unavailable"
REJECT_TRADE_DISABLED = "trade_disabled"
REJECT_VOLUME_MIN = "volume_min"
REJECT_VOLUME_MAX = "volume_max"
REJECT_VOLUME_STEP = "volume_step"
REJECT_POSITION_CAP = "position_cap"

DEFAULT_ATR_PIPS = 10.0  # engine fallback when an indicator is unavailable


class ShadowError(Exception):
    """Raised when the shadow book is fed something it cannot represent."""


def engine_atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float:
    """ATR exactly as :meth:`BacktestEngine._atr` computes it.

    Duplicated verbatim (rather than imported) because the engine's method
    is private and instance-bound; ``tests/unit/test_broker_shadow.py``
    pins parity against the real implementation so the two cannot drift.
    """
    high_arr = np.asarray(highs, dtype=float)
    low_arr = np.asarray(lows, dtype=float)
    close_arr = np.asarray(closes, dtype=float)
    if len(high_arr) < period + 1:
        return float(high_arr[-1] - low_arr[-1])
    tr = np.maximum(
        high_arr[1:] - low_arr[1:],
        np.maximum(
            np.abs(high_arr[1:] - close_arr[:-1]),
            np.abs(low_arr[1:] - close_arr[:-1]),
        ),
    )
    return float(np.mean(tr[-period:]))


def session_for(ts: datetime) -> str:
    """Trading session label — mirrors ``BacktestEngine._get_session``."""
    hour = ts.hour
    if 0 <= hour < 8:
        return "Asian"
    if 8 <= hour < 13:
        return "London"
    if 13 <= hour < 17:
        return "Overlap"
    return "NewYork"


def _iso(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else value.isoformat()


def _dt(value: Optional[str]) -> Optional[datetime]:
    return None if value is None else datetime.fromisoformat(value)


@dataclass
class ShadowEntry:
    """One would-have trade: intent → fill → exit, all numbers engine-made."""

    key: str
    symbol: str
    side: str
    volume: float
    created_ts: datetime
    sl: Optional[float] = None
    tp: Optional[float] = None
    created_bar: Optional[int] = None
    comment: str = ""
    state: str = SHADOW_PENDING
    predicted_fill_price: Optional[float] = None
    actual_fill_price: Optional[float] = None
    fill_ts: Optional[datetime] = None
    fill_bar: Optional[int] = None
    fill_drift_pips: Optional[float] = None
    commission: float = 0.0
    position_id: Optional[str] = None
    exit_price: Optional[float] = None
    exit_ts: Optional[datetime] = None
    exit_reason: Optional[str] = None
    would_have_pnl: Optional[float] = None

    @property
    def is_open(self) -> bool:
        return self.state == SHADOW_OPEN

    @property
    def is_closed(self) -> bool:
        return self.state == SHADOW_CLOSED

    @property
    def is_filled(self) -> bool:
        return self.state in (SHADOW_OPEN, SHADOW_CLOSED)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        for name in ("created_ts", "fill_ts", "exit_ts"):
            out[name] = _iso(out[name])
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShadowEntry":
        payload = dict(data)
        for name in ("created_ts", "fill_ts", "exit_ts"):
            payload[name] = _dt(payload.get(name))
        return cls(**payload)


class ShadowLedger:
    """Deterministic would-have book: intents in, shadow trades out.

    The ledger owns two engine objects — an :class:`OrderManager` used
    purely as the cost model for a fill, and a :class:`PositionManager`
    that carries the open shadow positions and does the SL/TP and P&L
    arithmetic. Neither ever leaves this process.
    """

    def __init__(self) -> None:
        # Same spread model object the engine's default OrderManager uses.
        self.spread_model = FixedSpreadModel(dict(DEFAULT_SPREADS))
        self._positions = PositionManager()
        self._entries: Dict[str, ShadowEntry] = {}
        self._rejections: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []
        self._last_close: Dict[str, float] = {}

    # -- introspection ---------------------------------------------------

    @property
    def positions(self) -> PositionManager:
        """The engine's own position book holding the shadow positions."""
        return self._positions

    def entries(self) -> List[ShadowEntry]:
        return list(self._entries.values())

    def get(self, key: str) -> Optional[ShadowEntry]:
        return self._entries.get(key)

    def open_entries(self) -> List[ShadowEntry]:
        return [e for e in self._entries.values() if e.state == SHADOW_OPEN]

    def rejections(self) -> List[Dict[str, Any]]:
        return list(self._rejections.values())

    def __len__(self) -> int:
        return len(self._entries)

    # -- events ----------------------------------------------------------

    def _emit(self, kind: str, **payload: Any) -> Dict[str, Any]:
        event = {"kind": kind, **payload}
        self.events.append(event)
        return event

    # -- cost model (OrderManager, never a re-implementation) -------------

    def model_spread_pips(self, symbol: str, ts: datetime) -> float:
        """The E0/backtest model spread for ``symbol`` at ``ts``."""
        return float(self.spread_model.get_spread(symbol, ts, session_for(ts)))

    def _fill_price(
        self,
        *,
        symbol: str,
        side: str,
        volume: float,
        candle: Dict[str, float],
        spread_pips: float,
        atr_pips: float,
        ts: datetime,
        session: str,
    ) -> tuple:
        """Fill price + commission through the engine's own OrderManager."""
        orders = OrderManager()  # stateless cost model: one call, one order
        order = orders.create_order(
            symbol=symbol,
            side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            volume=volume,
            comment="shadow",
        )
        context = ExecutionContext(
            current_candle=candle,
            spread_pips=spread_pips,
            atr_pips=atr_pips,
            session=session,
            timestamp=ts,
        )
        fills = orders.execute_order(order, context)
        if not fills:
            raise ShadowError(
                f"cost model produced no fill for {symbol} {side} {volume}"
            )
        return fills[0].price, fills[0].commission

    # -- intent intake ----------------------------------------------------

    def record_intent(
        self,
        key: str,
        *,
        symbol: str,
        side: str,
        volume: float,
        created_ts: datetime,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        created_bar: Optional[int] = None,
        comment: str = "",
        magic: int = ALEX_SHADOW_MAGIC,
    ) -> ShadowEntry:
        """Write-ahead: record the intent BEFORE it could ever fill.

        Idempotent by ``key`` — a duplicate submission returns the original
        entry instead of creating a second would-have trade.
        """
        existing = self._entries.get(key)
        if existing is not None:
            self._emit("shadow.intent_duplicate", key=key, symbol=symbol)
            return existing

        normalized = "BUY" if str(side).upper() in ("BUY", "LONG") else "SELL"
        entry = ShadowEntry(
            key=key,
            symbol=symbol,
            side=normalized,
            volume=float(volume),
            created_ts=created_ts,
            sl=sl,
            tp=tp,
            created_bar=created_bar,
            comment=str(comment),
            state=SHADOW_PENDING,
        )
        self._entries[key] = entry
        self._emit(
            "shadow.intent_recorded",
            key=key,
            symbol=symbol,
            side=entry.side,
            volume=entry.volume,
            ts=_iso(created_ts),
            magic=magic,
        )
        return entry

    def record_rejection(
        self,
        key: str,
        reason: str,
        *,
        symbol: str = "",
        side: str = "",
        volume: float = 0.0,
        ts: Optional[datetime] = None,
        bar: Optional[int] = None,
        comment: str = "",
    ) -> Dict[str, Any]:
        """Record an intent that never reached the book (and never will).

        Idempotent by key so a re-submitted rejection is counted once.
        """
        existing = self._rejections.get(key)
        if existing is not None:
            return existing
        record = {
            "key": key,
            "reason": reason,
            "symbol": symbol,
            "side": side,
            "volume": float(volume),
            "ts": _iso(ts),
            "bar": bar,
            "comment": comment,
        }
        self._rejections[key] = record
        self._emit("shadow.intent_rejected", **record)
        return record

    # -- bar loop ---------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        ts: datetime,
        *,
        open_: float,
        high: float,
        low: float,
        close: float,
        spread_pips: Optional[float] = None,
        atr_pips: float = DEFAULT_ATR_PIPS,
        price_digits: Optional[int] = None,
        bar_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Advance the shadow book by one real bar.

        The order of operations mirrors ``BacktestEngine._process_bar``:
        positions are marked and stopped against this bar's range FIRST,
        then pending intents fill, then overnight swap is accrued once.
        """
        session = session_for(ts)
        self._last_close[symbol] = float(close)
        produced: List[Dict[str, Any]] = []

        # 1) SL/TP against the bar's high/low (engine semantics).
        close_events = self._positions.update_positions(
            {symbol: float(close)},
            ts,
            bar_highs={symbol: float(high)},
            bar_lows={symbol: float(low)},
        )
        for event in close_events:
            trade = event["trade"]
            entry = self._entries.get(str(trade["id"]))
            if entry is None:
                continue
            self._seal(
                entry,
                trade,
                price=float(trade["close_price"]),
                ts=ts,
                reason=str(trade["reason"]),
            )
            produced.append(self._emit(
                "shadow.shadow_exit",
                key=entry.key,
                reason=entry.exit_reason,
                price=entry.exit_price,
                pnl=entry.would_have_pnl,
                ts=_iso(ts),
            ))

        # 2) Pending intents created on an earlier bar fill on THIS bar —
        #    exactly where a pending market order fills in the base engine.
        live_spread = self.spread_model.get_spread(symbol, ts, session)
        if spread_pips is not None:
            live_spread = float(spread_pips)
        for entry in list(self._entries.values()):
            if entry.state != SHADOW_PENDING or entry.symbol != symbol:
                continue
            if entry.created_ts >= ts:
                continue
            fill_events = self._fill(entry, ts, bar_index, live_spread, atr_pips,
                                     session, open_, high, low, close, price_digits)
            produced.extend(fill_events)

        # 3) Overnight swap, charged once per completed day (engine rule).
        self._positions.calculate_swaps(ts)

        return produced

    def _fill(
        self,
        entry: ShadowEntry,
        ts: datetime,
        bar_index: Optional[int],
        live_spread: float,
        atr_pips: float,
        session: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        price_digits: Optional[int],
    ) -> List[Dict[str, Any]]:
        candle = {"open": open_, "high": high, "low": low, "close": close}
        model_spread = self.model_spread_pips(entry.symbol, ts)

        # Predicted: E0's model spread. Actual: the live broker spread on
        # the same real bar. One bar, one ATR, one slippage model — the
        # drift isolates real execution cost from the estimate.
        predicted, _ = self._fill_price(
            symbol=entry.symbol,
            side=entry.side,
            volume=entry.volume,
            candle=candle,
            spread_pips=model_spread,
            atr_pips=atr_pips,
            ts=ts,
            session=session,
        )
        actual, commission = self._fill_price(
            symbol=entry.symbol,
            side=entry.side,
            volume=entry.volume,
            candle=candle,
            spread_pips=live_spread,
            atr_pips=atr_pips,
            ts=ts,
            session=session,
        )
        if price_digits is not None:
            predicted = round(predicted, int(price_digits))
            actual = round(actual, int(price_digits))

        pip_size = pip_size_for_symbol(entry.symbol)
        raw_drift = (actual - predicted) / pip_size
        if entry.side == "SELL":
            raw_drift = -raw_drift

        position = self._positions.open_position(
            entry.symbol,
            OrderSide.BUY if entry.side == "BUY" else OrderSide.SELL,
            entry.volume,
            actual,
            sl=entry.sl,
            tp=entry.tp,
            commission=commission,
            magic=ALEX_SHADOW_MAGIC,
            comment=entry.comment,
            open_time=ts,
        )
        # Deterministic ledger ids: the engine never reads ``id`` for its
        # arithmetic, only for bookkeeping, so the intent key replaces the
        # uuid4 the position manager minted (the E0 rekey trick).
        self._positions.positions.pop(position.id, None)
        position.id = entry.key
        self._positions.positions[entry.key] = position

        entry.state = SHADOW_OPEN
        entry.predicted_fill_price = predicted
        entry.actual_fill_price = actual
        entry.fill_ts = ts
        entry.fill_bar = bar_index
        entry.fill_drift_pips = round(raw_drift, 6)
        entry.commission = commission
        entry.position_id = entry.key

        return [self._emit(
            "shadow.shadow_fill",
            key=entry.key,
            symbol=entry.symbol,
            side=entry.side,
            volume=entry.volume,
            predicted=predicted,
            actual=actual,
            fill_drift_pips=entry.fill_drift_pips,
            ts=_iso(ts),
        )]

    def _seal(
        self,
        entry: ShadowEntry,
        trade: Dict[str, Any],
        *,
        price: float,
        ts: datetime,
        reason: str,
    ) -> None:
        entry.state = SHADOW_CLOSED
        entry.exit_price = price
        entry.exit_ts = ts
        entry.exit_reason = reason
        entry.would_have_pnl = float(trade.get("profit", 0.0))
        entry.commission = float(trade.get("commission", entry.commission))

    def close_all(self, ts: datetime, reason: str = "WINDOW_END") -> List[Dict[str, Any]]:
        """Close every open shadow position at the last seen close."""
        produced: List[Dict[str, Any]] = []
        for entry in list(self._entries.values()):
            if entry.state != SHADOW_OPEN:
                continue
            price = self._last_close.get(entry.symbol)
            if price is None:
                continue
            trade = self._positions.close_position(
                entry.key, price, commission=0.0, reason=reason, timestamp=ts
            )
            if trade is None:
                continue
            self._seal(entry, trade, price=price, ts=ts, reason=reason)
            produced.append(self._emit(
                "shadow.shadow_exit",
                key=entry.key,
                reason=reason,
                price=price,
                pnl=entry.would_have_pnl,
                ts=_iso(ts),
            ))
        return produced

    # -- reporting --------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Fill-drift + rejection flags + would-have P&L (D3 flag pattern)."""
        entries = self.entries()
        drifts = [
            e.fill_drift_pips for e in entries if e.fill_drift_pips is not None
        ]
        closed = [e for e in entries if e.state == SHADOW_CLOSED]
        open_entries = [e for e in entries if e.state == SHADOW_OPEN]

        realized = sum(e.would_have_pnl or 0.0 for e in closed)
        unrealized = 0.0
        for entry in open_entries:
            position = self._positions.positions.get(entry.position_id or "")
            if position is not None:
                unrealized += float(position.profit)

        rejection_counts: Dict[str, int] = {}
        for record in self._rejections.values():
            reason = str(record["reason"])
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        return {
            "intents": len(entries),
            "filled": sum(1 for e in entries if e.is_filled),
            "open": len(open_entries),
            "closed": len(closed),
            "pending_fill": sum(1 for e in entries if e.state == SHADOW_PENDING),
            "rejected": len(self._rejections),
            "fill_drift": {
                "count": len(drifts),
                "mean_pips": round(sum(drifts) / len(drifts), 6) if drifts else 0.0,
                "max_pips": round(max(drifts), 6) if drifts else 0.0,
                "min_pips": round(min(drifts), 6) if drifts else 0.0,
            },
            "intent_rejected": rejection_counts,
            "would_have_pnl": {
                "realized": realized,
                "unrealized": unrealized,
                "total": realized + unrealized,
            },
        }

    # -- crash image ------------------------------------------------------

    def save_snapshot(self) -> Dict[str, Any]:
        """Byte-exact serializable state — the shape a backend would store."""
        return {
            "entries": [e.to_dict() for e in self.entries()],
            "positions": [
                {
                    "id": p.id,
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "volume": p.volume,
                    "open_price": p.open_price,
                    "current_price": p.current_price,
                    "sl": p.sl,
                    "tp": p.tp,
                    "commission": p.commission,
                    "swap": p.swap,
                    "profit": p.profit,
                    "open_time": _iso(p.open_time),
                    "swap_days": p.swap_days,
                    "magic": p.magic,
                    "comment": p.comment,
                }
                for p in self._positions.positions.values()
            ],
            "closed_trades": [
                {
                    **{k: v for k, v in trade.items() if k not in ("open_time", "close_time")},
                    "open_time": _iso(trade.get("open_time")),
                    "close_time": _iso(trade.get("close_time")),
                }
                for trade in self._positions.closed_trades
            ],
            "rejections": list(self._rejections.values()),
            "events": list(self.events),
            "last_close": dict(self._last_close),
        }

    def load_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self._positions = PositionManager()
        for trade in snapshot.get("closed_trades") or []:
            payload = dict(trade)
            payload["open_time"] = _dt(payload.get("open_time"))
            payload["close_time"] = _dt(payload.get("close_time"))
            self._positions.closed_trades.append(payload)
        for data in snapshot.get("positions") or []:
            payload = dict(data)
            position_id = str(payload.pop("id"))
            side = OrderSide(payload.pop("side"))
            open_time = _dt(payload.pop("open_time"))
            position = self._positions.open_position(
                str(payload.pop("symbol")),
                side,
                float(payload.pop("volume")),
                float(payload.pop("open_price")),
                sl=payload.pop("sl"),
                tp=payload.pop("tp"),
                commission=float(payload.pop("commission") or 0.0),
                magic=int(payload.pop("magic") or 0),
                comment=str(payload.pop("comment") or ""),
                open_time=open_time,
            )
            # Swap the uuid4 for the deterministic intent key.
            self._positions.positions.pop(position.id, None)
            position.id = position_id
            self._positions.positions[position_id] = position
            position.current_price = float(
                payload.pop("current_price", position.open_price)
            )
            position.swap = float(payload.pop("swap") or 0.0)
            position.profit = float(payload.pop("profit") or 0.0)
            position.swap_days = int(payload.pop("swap_days") or 0)
        self._entries = {
            e["key"]: ShadowEntry.from_dict(e)
            for e in (snapshot.get("entries") or [])
        }
        self._rejections = {
            r["key"]: dict(r) for r in (snapshot.get("rejections") or [])
        }
        self.events = list(snapshot.get("events") or [])
        self._last_close = dict(snapshot.get("last_close") or {})


class ShadowExecutor:
    """Runs live signals through the real chain and books them — never sends.

    Validation order, all deterministic:

    1. the E1 governor (armed? clock monotonic? data fresh? duplicate key?)
    2. the broker's own symbol spec (read-only): trade mode and volume
       min/max/step, which is real sizing validation against the terminal
    3. the shadow position cap from ``governor.limits.max_positions``

    Only then is the intent written ahead to the ledger. The executor holds
    a :class:`ReadonlyBroker`, not a write-capable adapter — there is no
    method here that could reach a wire even if one were wired up.
    """

    def __init__(
        self,
        adapter: ReadonlyBroker,
        governor: Any,
        ledger: Optional[ShadowLedger] = None,
        *,
        spec_cache: Optional[Dict[str, SymbolSpec]] = None,
    ) -> None:
        self.adapter = adapter
        self.governor = governor
        self.ledger = ledger or ShadowLedger()
        self._specs: Dict[str, SymbolSpec] = dict(spec_cache or {})
        self.events: List[Dict[str, Any]] = []

    # -- symbol specs (read-only, audited by the adapter) -----------------

    def spec(self, symbol: str) -> SymbolSpec:
        cached = self._specs.get(symbol)
        if cached is not None:
            return cached
        spec = self.adapter.symbol_spec(symbol)
        self._specs[symbol] = spec
        return spec

    # -- intake -----------------------------------------------------------

    def submit(
        self,
        key: str,
        *,
        symbol: str,
        side: str,
        volume: float,
        ts: datetime,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        bar_index: Optional[int] = None,
        comment: str = "",
    ) -> Dict[str, Any]:
        """Validate one intent and record it — or reject it with a reason."""
        reason = self._reject_reason(
            key,
            symbol=symbol,
            side=side,
            volume=volume,
            sl=sl,
            tp=tp,
            ts=ts,
            bar_index=bar_index,
            comment=comment,
        )
        if reason is not None:
            if reason != REJECT_DUPLICATE:
                # A duplicate is the ORIGINAL intent still in flight —
                # cancelling it would drop a live order from the book.
                self.governor.record_cancel(key, reason)
            record = self.ledger.record_rejection(
                key,
                reason,
                symbol=symbol,
                side=side,
                volume=volume,
                ts=ts,
                bar=bar_index,
                comment=comment,
            )
            self.events.append({"kind": "shadow.intent_rejected", **record})
            return {"key": key, "status": "REJECTED", "reason": reason}

        entry = self.ledger.record_intent(
            key,
            symbol=symbol,
            side=side,
            volume=volume,
            created_ts=ts,
            sl=sl,
            tp=tp,
            created_bar=bar_index,
            comment=comment,
        )
        self.events.append(
            {
                "kind": "shadow.intent_accepted",
                "key": entry.key,
                "symbol": entry.symbol,
                "side": entry.side,
                "volume": entry.volume,
                "ts": ts.isoformat(),
            }
        )
        return {"key": key, "status": "RECORDED", "reason": None}

    def _reject_reason(
        self,
        key: str,
        *,
        symbol: str,
        side: str,
        volume: float,
        sl: Optional[float],
        tp: Optional[float],
        ts: datetime,
        bar_index: Optional[int],
        comment: str = "",
    ) -> Optional[str]:
        # 1) Governor: armed, fresh, monotonic clock, not a duplicate.
        record = self.governor.accept_intent(
            key,
            symbol=symbol,
            side=side,
            volume=volume,
            sl=sl,
            tp=tp,
            now=ts,
            current_bar=bar_index,
            comment=comment,
        )
        if record is None:
            if self.ledger.get(key) is not None:
                return REJECT_DUPLICATE
            reason = REJECT_GOVERNOR
            if self.governor.events:
                last = self.governor.events[-1]
                if "reason" in last:
                    reason = str(last["reason"])
            return reason

        # 2) Broker symbol spec: real trade mode + volume bounds/step.
        try:
            spec = self.spec(symbol)
        except Exception as exc:  # noqa: BLE001 — read failure = no send
            return f"{REJECT_SPEC_UNAVAILABLE}:{type(exc).__name__}"
        if not spec.trade_allowed:
            return REJECT_TRADE_DISABLED
        if volume < spec.volume_min:
            return REJECT_VOLUME_MIN
        if volume > spec.volume_max:
            return REJECT_VOLUME_MAX
        step = spec.volume_step or 0.01
        steps = volume / step
        if abs(steps - round(steps)) > 1e-6:
            return REJECT_VOLUME_STEP

        # 3) Shadow position cap (never let the book outrun the governor).
        limit = int(getattr(self.governor.limits, "max_positions", 0) or 0)
        if limit and len(self.ledger.open_entries()) >= limit:
            return REJECT_POSITION_CAP
        return None

    # -- bar loop ---------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        ts: datetime,
        *,
        open_: float,
        high: float,
        low: float,
        close: float,
        spread_pips: Optional[float] = None,
        atr_pips: float = DEFAULT_ATR_PIPS,
        price_digits: Optional[int] = None,
        bar_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        self.governor.record_bar(symbol, ts)
        return self.ledger.on_bar(
            symbol,
            ts,
            open_=open_,
            high=high,
            low=low,
            close=close,
            spread_pips=spread_pips,
            atr_pips=atr_pips,
            price_digits=price_digits,
            bar_index=bar_index,
        )

    def close_window(self, ts: datetime, reason: str = "WINDOW_END") -> List[Dict[str, Any]]:
        """Settle the shadow book at the end of an observation window."""
        return self.ledger.close_all(ts, reason=reason)

    # -- reconciliation ---------------------------------------------------

    def reconcile(self, ts: Optional[datetime] = None) -> ReconciliationReport:
        """Compare the intent book with the broker's read-only positions."""
        report = reconcile(self.governor.journal.pending(), self.adapter.positions(),
                           ts=ts)
        for event in report.events:
            self.events.append(event)
        return report

    # -- report -----------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Everything a shadow run reports: ledger, book, adapter, audit."""
        return {
            "ledger": self.ledger.stats(),
            "governor": {
                "state": self.governor.state.state,
                "tripped": self.governor.state.tripped,
                "pending_intents": len(self.governor.journal.pending()),
                "events": len(self.governor.events),
            },
            "adapter": self.adapter.audit_summary(),
            "sent_writes": len(self.adapter.sent_calls()),
            "shadow_events": len(self.events),
        }
