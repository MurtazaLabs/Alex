"""Stage E2: MetaTrader 5 adapter — read-only now, gated writes for later.

The module imports ``MetaTrader5`` lazily (inside :meth:`MT5Adapter.connect`)
so the whole package imports on a machine with no terminal, which is what
makes ``tests/unit/test_broker_shadow.py`` runnable in CI against a stub.

Reads funnel through :meth:`ReadonlyBroker._invoke`, so every terminal call
lands in the audit trail tagged READ or CONTROL. Writes exist because E3/E4
will need them, but each one starts with
:meth:`BrokerAdapter._authorize_write` — permission level AUTONOMOUS(4)
*and* shadow mode off — so at ``MAX_PERMISSION_LEVEL=0`` they refuse before
the transport is even named. ``MT5Adapter.for_shadow`` is the only
constructor the application builds, and it forces shadow mode on.

Order magic :data:`ALEX_SHADOW_MAGIC` is reserved for Alex: any deal
carrying it in the broker's own history would mean an order escaped the
shadow. ``scripts/e2_live_verify.py`` asserts there are none.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..core.permission_manager import PermissionManager
from .base import (
    ALEX_SHADOW_MAGIC,
    CALL_CONTROL,
    CALL_READ,
    CALL_WRITE,
    AccountSnapshot,
    BrokerAdapter,
    BrokerError,
    BrokerNotConnected,
    OrderAck,
    OrderRequest,
    PositionSnapshot,
    SymbolSpec,
)

__all__ = ["MT5Adapter", "MT5Unavailable"]

#: Market-order slippage tolerance (in points) when the request does not
#: declare one. 20 points equals 2 pips on the 5-digit majors — the same
#: band the E4 caps declare acceptable for fill reconciliation. A request
#: that sets ``deviation`` explicitly always wins.
DEFAULT_MARKET_DEVIATION_POINTS = 20


class MT5Unavailable(BrokerError):
    """The MetaTrader5 python package is not importable on this machine."""


def _import_mt5() -> Any:
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on host
        raise MT5Unavailable(
            "MetaTrader5 package not installed. Install with: pip install MetaTrader5"
        ) from exc
    return mt5


class MT5Adapter(BrokerAdapter):
    """MetaTrader 5 terminal adapter (synchronous IPC)."""

    name = "mt5"

    def __init__(
        self,
        permission: Optional[PermissionManager] = None,
        *,
        shadow_mode: bool = True,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
        timeout_ms: int = 60_000,
    ) -> None:
        super().__init__(permission, shadow_mode=shadow_mode)
        self._login = login
        self._password = password
        self._server = server
        self._path = path
        self._timeout_ms = int(timeout_ms)
        self._mt5: Any = None
        # Brokers suffix instruments (Exness: EURUSD → EURUSDm). Filled by
        # symbol_spec; positions() maps back so the reconciliation loop
        # compares the Alex symbol with the Alex symbol.
        self._resolved: Dict[str, str] = {}
        self._alex_by_broker: Dict[str, str] = {}

    # -- construction ----------------------------------------------------

    @classmethod
    def for_shadow(
        cls, permission: Optional[PermissionManager] = None, **kwargs: Any
    ) -> "MT5Adapter":
        """The only way the application builds an adapter: shadow mode ON.

        ``shadow_mode`` is accepted-but-discarded so no caller can talk the
        factory out of the E2 default.
        """
        kwargs.pop("shadow_mode", None)
        return cls(permission, shadow_mode=True, **kwargs)

    @staticmethod
    def broker_symbol(symbol: str) -> str:
        """``EUR/USD`` → ``EURUSD`` (the broker's own naming)."""
        return symbol.replace("/", "").replace(" ", "")

    def _write_symbol(self, symbol: str) -> str:
        """The broker's own name for a WRITE path (resolved once, cached).

        Reads resolve the suffix via ``symbol_spec`` (Exness: EURUSDm).
        A write that skipped resolution would address ``EURUSD`` — a name
        the broker does not offer — and fail closed as NO_TICK. Both write
        and read paths must address the same instrument.
        """
        cached = self._resolved.get(symbol)
        if cached:
            return cached
        resolved = self._resolve_broker_symbol(symbol) or self.broker_symbol(symbol)
        self._resolved[symbol] = resolved
        self._alex_by_broker.setdefault(resolved, symbol)
        return resolved

    def _resolve_broker_symbol(self, symbol: str) -> Optional[str]:
        """Find the broker's own name when it suffixes the instrument.

        Exact match first; otherwise the feed provider's own rule — base
        name plus the shortest alphanumeric (``.``/``-``/``_``) suffix —
        so ``EUR/USD`` lands on Exness' ``EURUSDm``. The chosen symbol is
        added to the terminal's market watch (CONTROL: no data read, no
        order). Returns ``None`` when nothing matches — the caller fails
        closed.
        """
        symbols_get = getattr(self._mt5, "symbols_get", None)
        if symbols_get is None:
            return None
        try:
            raw = self._invoke("symbols_get", CALL_READ, symbols_get)
        except Exception:  # noqa: BLE001 — a failed scan means "not offered"
            return None
        wanted = self.broker_symbol(symbol).upper()
        chosen: Optional[str] = None
        best: Optional[Tuple[int, str]] = None
        for item in raw or ():
            name = str(getattr(item, "name", item))
            norm = name.upper().replace("/", "")
            if norm == wanted:
                chosen = name
                break
            if not norm.startswith(wanted) or len(norm) <= len(wanted):
                continue
            suffix = norm[len(wanted):]
            if not (suffix.isalnum() or set(suffix) <= {".", "-", "_"}):
                continue
            if best is None or len(suffix) < best[0]:
                best = (len(suffix), name)
        if chosen is None and best is not None:
            chosen = best[1]
        if chosen is None:
            return None
        select = getattr(self._mt5, "symbol_select", None)
        if select is not None:
            self._invoke(
                "symbol_select", CALL_CONTROL, select, chosen, True,
                detail={"symbol": symbol},
            )
        return chosen

    # -- lifecycle -------------------------------------------------------

    def connect(self) -> bool:
        if self._connected:
            return True
        mt5 = _import_mt5()
        self._mt5 = mt5
        kwargs: Dict[str, Any] = {"timeout": self._timeout_ms}
        if self._path:
            exe = self._path if self._path.lower().endswith(".exe") else self._path + ".exe"
            kwargs["path"] = exe
        if self._login and self._password and self._server:
            kwargs.update(
                login=int(self._login), password=self._password, server=self._server
            )
        ok = self._invoke(
            "initialize",
            CALL_CONTROL,
            mt5.initialize,
            **kwargs,
            detail={"shadow_mode": self._shadow_mode},
        )
        if not ok:
            error: Any = None
            try:
                error = mt5.last_error()
            except Exception:  # noqa: BLE001 — last_error is best-effort
                pass
            raise BrokerNotConnected(f"mt5.initialize failed: {error}")
        self._connected = True
        return True

    def disconnect(self) -> None:
        if self._mt5 is not None and self._connected:
            self._invoke("shutdown", CALL_CONTROL, self._mt5.shutdown)
        self._connected = False

    def is_connected(self) -> bool:
        if not self._connected or self._mt5 is None:
            return False
        try:
            info = self._invoke("terminal_info", CALL_READ, self._mt5.terminal_info)
        except Exception:  # noqa: BLE001 — any IPC hiccup means disconnected
            self._connected = False
            return False
        return bool(info and getattr(info, "connected", True))

    # -- reads -----------------------------------------------------------

    def account(self) -> AccountSnapshot:
        self._require_connected()

        def _read_account() -> Any:
            # MT5 convention: account_info() returns None on failure. Raise
            # inside _invoke so the failure lands in the audit trail as a
            # failed READ rather than a silent success.
            info = self._mt5.account_info()
            if info is None:
                raise BrokerNotConnected(
                    f"account_info returned nothing: {self._mt5.last_error()}"
                )
            return info

        info = self._invoke("account_info", CALL_READ, _read_account)
        return AccountSnapshot(
            login=int(getattr(info, "login", 0)),
            server=str(getattr(info, "server", "")),
            currency=str(getattr(info, "currency", "")),
            balance=float(getattr(info, "balance", 0.0)),
            equity=float(getattr(info, "equity", 0.0)),
            margin=float(getattr(info, "margin", 0.0)),
            margin_free=float(getattr(info, "margin_free", 0.0)),
            leverage=int(getattr(info, "leverage", 0)),
            profit=float(getattr(info, "profit", 0.0)),
            timestamp=datetime.utcnow(),
        )

    def positions(self) -> List[PositionSnapshot]:
        self._require_connected()
        raw = self._invoke("positions_get", CALL_READ, self._mt5.positions_get)
        buy_type = int(getattr(self._mt5, "POSITION_TYPE_BUY", 0))
        out: List[PositionSnapshot] = []
        for p in raw or ():
            opened = getattr(p, "time", None)
            raw_symbol = str(p.symbol)
            out.append(
                PositionSnapshot(
                    ticket=int(getattr(p, "ticket", 0)),
                    # Back to the Alex form for symbols we resolved, so
                    # reconciliation compares like with like; unknown
                    # exposures keep the broker's own name.
                    symbol=self._alex_by_broker.get(raw_symbol, raw_symbol),
                    side="BUY" if int(getattr(p, "type", 0)) == buy_type else "SELL",
                    volume=float(getattr(p, "volume", 0.0)),
                    price_open=float(getattr(p, "price_open", 0.0)),
                    profit=float(getattr(p, "profit", 0.0)),
                    swap=float(getattr(p, "swap", 0.0)),
                    sl=float(getattr(p, "sl", 0.0)) or None,
                    tp=float(getattr(p, "tp", 0.0)) or None,
                    magic=int(getattr(p, "magic", 0)),
                    comment=str(getattr(p, "comment", "")),
                    time=datetime.fromtimestamp(opened) if opened else None,
                )
            )
        return out

    def pending_orders(self) -> List[Dict[str, Any]]:
        self._require_connected()
        raw = self._invoke("orders_get", CALL_READ, self._mt5.orders_get)
        return [
            {
                "ticket": int(getattr(o, "ticket", 0)),
                "symbol": str(o.symbol),
                "type": int(getattr(o, "type", 0)),
                "volume": float(getattr(o, "volume_current", 0.0)),
                "price_open": float(getattr(o, "price_open", 0.0)),
                "sl": float(getattr(o, "sl", 0.0)) or None,
                "tp": float(getattr(o, "tp", 0.0)) or None,
                "magic": int(getattr(o, "magic", 0)),
                "comment": str(getattr(o, "comment", "")),
            }
            for o in raw or ()
        ]

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        self._require_connected()
        broker_symbol = self._resolved.get(symbol) or self.broker_symbol(symbol)
        info = self._invoke(
            "symbol_info",
            CALL_READ,
            self._mt5.symbol_info,
            broker_symbol,
            detail={"symbol": symbol},
        )
        if info is None:
            # The broker suffixes the instrument (EURUSDm at Exness):
            # scan the terminal's symbol list, then re-query the chosen
            # name. No match ⇒ fail closed, as before.
            resolved = self._resolve_broker_symbol(symbol)
            if resolved is None:
                raise BrokerError(
                    f"symbol {symbol!r} ({broker_symbol}) not offered by the broker"
                )
            info = self._invoke(
                "symbol_info",
                CALL_READ,
                self._mt5.symbol_info,
                resolved,
                detail={"symbol": symbol},
            )
            if info is None:
                raise BrokerError(
                    f"symbol {symbol!r} ({resolved}) not offered by the broker"
                )
            broker_symbol = resolved
        resolved_name = str(getattr(info, "name", broker_symbol))
        self._resolved[symbol] = resolved_name
        self._alex_by_broker[resolved_name] = symbol
        # Fail closed: without the terminal's own FULL constant we cannot
        # prove the symbol trades, so treat it as not tradable.
        full_mode = getattr(self._mt5, "SYMBOL_TRADE_MODE_FULL", None)
        trade_allowed = (
            False
            if full_mode is None
            else int(getattr(info, "trade_mode", -1)) == int(full_mode)
        )
        point = float(getattr(info, "point", 0.0) or 0.0)
        return SymbolSpec(
            symbol=symbol,
            broker_symbol=resolved_name,
            digits=int(getattr(info, "digits", 0)),
            point=point,
            tick_size=float(getattr(info, "tick_size", 0.0) or point),
            contract_size=float(getattr(info, "trade_contract_size", 0.0) or 0.0),
            volume_min=float(getattr(info, "volume_min", 0.0) or 0.0),
            volume_max=float(getattr(info, "volume_max", 0.0) or 0.0),
            volume_step=float(getattr(info, "volume_step", 0.0) or 0.0),
            spread=float(getattr(info, "spread", 0.0) or 0.0),
            trade_allowed=trade_allowed,
        )

    def history_deals(
        self,
        start: datetime,
        end: datetime,
        magic: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        self._require_connected()
        raw = self._invoke(
            "history_deals_get", CALL_READ, self._mt5.history_deals_get, start, end
        )
        deals = [
            {
                "ticket": int(getattr(d, "ticket", 0)),
                "order": int(getattr(d, "order", 0)),
                "magic": int(getattr(d, "magic", 0)),
                "symbol": str(getattr(d, "symbol", "")),
                "volume": float(getattr(d, "volume", 0.0)),
                "price": float(getattr(d, "price", 0.0)),
                "profit": float(getattr(d, "profit", 0.0)),
                "comment": str(getattr(d, "comment", "")),
                "time": int(getattr(d, "time", 0)),
                # Statement-grade fields (E4 daily report): additive keys —
                # existing consumers only read the keys above.
                "type": int(getattr(d, "type", 0)),
                "entry": int(getattr(d, "entry", 0)),
                "commission": float(getattr(d, "commission", 0.0)),
                "swap": float(getattr(d, "swap", 0.0)),
                "position_id": int(getattr(d, "position_id", 0)),
            }
            for d in raw or ()
        ]
        if magic is not None:
            wanted = int(magic)
            deals = [d for d in deals if d["magic"] == wanted]
        return deals

    # -- writes (unreachable at MAX_PERMISSION_LEVEL=0) -------------------

    def _terminal_write_blocked(self) -> Optional[OrderAck]:
        """Refuse writes before the wire when terminal Algo Trading is off.

        In that state the terminal's send call returns ``None`` (no
        retcode), an ambiguous outcome that forces fail-closed
        reconciliation. Refuse deterministically instead, with an
        actionable reason.
        """
        info = self._invoke("terminal_info", CALL_READ, self._mt5.terminal_info)
        if info is None or bool(getattr(info, "trade_allowed", True)):
            return None
        return OrderAck(
            sent=False,
            status="TERMINAL_BLOCKED",
            reason=(
                "AutoTrading is disabled in the MT5 terminal "
                "(toolbar Algo Trading button is off) - nothing was sent"
            ),
        )

    def submit_order(self, request: OrderRequest) -> OrderAck:
        """Send an order. Refused at level 0 and in shadow mode — by design."""
        self._authorize_write("submit_order")
        self._require_connected()
        blocked = self._terminal_write_blocked()
        if blocked is not None:
            return blocked
        mt5 = self._mt5
        broker_symbol = self._write_symbol(request.symbol)
        is_buy = str(request.side).upper() in ("BUY", "LONG")
        tick = self._invoke(
            "symbol_info_tick", CALL_READ, mt5.symbol_info_tick, broker_symbol
        )
        if tick is None:
            return OrderAck(
                sent=False,
                status="NO_TICK",
                reason=f"no tick for {broker_symbol}",
                request=request.to_dict(),
            )
        price = float(tick.ask if is_buy else tick.bid)
        payload = {
            "action": getattr(mt5, "TRADE_ACTION_DEAL", 1),
            "symbol": broker_symbol,
            "type": getattr(mt5, "ORDER_TYPE_BUY", 0)
            if is_buy
            else getattr(mt5, "ORDER_TYPE_SELL", 1),
            "volume": float(request.volume),
            "price": price,
            "sl": float(request.sl or 0.0),
            "tp": float(request.tp or 0.0),
            "deviation": int(request.deviation)
            if int(request.deviation) > 0
            else DEFAULT_MARKET_DEVIATION_POINTS,
            "magic": int(request.magic or ALEX_SHADOW_MAGIC),
            "comment": request.comment or "",
            "type_time": getattr(mt5, "ORDER_TIME_GTC", 0),
            "type_filling": getattr(mt5, "ORDER_FILLING_IOC", 0),
        }
        result = self._invoke(
            "order_send",
            CALL_WRITE,
            mt5.order_send,
            payload,
            detail={"key": request.key, "symbol": request.symbol},
        )
        if result is None:
            return OrderAck(
                sent=True,
                status="FAILED",
                reason=str(mt5.last_error()),
                request=payload,
            )
        retcode = int(getattr(result, "retcode", 0))
        done = int(getattr(mt5, "TRADE_RETCODE_DONE", 10009))
        return OrderAck(
            sent=True,
            status="FILLED" if retcode == done else "REJECTED",
            retcode=retcode,
            ticket=int(getattr(result, "order", 0) or 0) or None,
            reason=str(getattr(result, "comment", "")),
            request=payload,
        )

    def modify_order(
        self, ticket: int, *, sl: Optional[float] = None, tp: Optional[float] = None
    ) -> OrderAck:
        self._authorize_write("modify_order")
        self._require_connected()
        blocked = self._terminal_write_blocked()
        if blocked is not None:
            return blocked
        mt5 = self._mt5
        payload = {
            "action": getattr(mt5, "TRADE_ACTION_SLTP", 6),
            "position": int(ticket),
            "sl": float(sl or 0.0),
            "tp": float(tp or 0.0),
        }
        result = self._invoke(
            "order_send",
            CALL_WRITE,
            mt5.order_send,
            payload,
            detail={"ticket": int(ticket)},
        )
        if result is None:
            return OrderAck(sent=True, status="FAILED", reason=str(mt5.last_error()))
        retcode = int(getattr(result, "retcode", 0))
        done = int(getattr(mt5, "TRADE_RETCODE_DONE", 10009))
        return OrderAck(
            sent=True,
            status="MODIFIED" if retcode == done else "REJECTED",
            retcode=retcode,
            ticket=int(ticket),
            reason=str(getattr(result, "comment", "")),
        )

    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
        *,
        magic: Optional[int] = None,
    ) -> OrderAck:
        self._authorize_write("close_position")
        self._require_connected()
        blocked = self._terminal_write_blocked()
        if blocked is not None:
            return blocked
        mt5 = self._mt5
        found = self._invoke(
            "positions_get", CALL_READ, self._mt5.positions_get, ticket=int(ticket)
        )
        position = found[0] if found else None
        if position is None:
            return OrderAck(
                sent=False, status="NOT_FOUND", reason=f"no position {ticket}"
            )
        broker_symbol = str(position.symbol)
        is_buy = int(getattr(position, "type", 0)) == int(
            getattr(mt5, "POSITION_TYPE_BUY", 0)
        )
        tick = self._invoke(
            "symbol_info_tick", CALL_READ, mt5.symbol_info_tick, broker_symbol
        )
        if tick is None:
            return OrderAck(
                sent=False, status="NO_TICK", reason=f"no tick for {broker_symbol}"
            )
        price = float(tick.bid if is_buy else tick.ask)
        # Attribution: the closing order carries the position's own magic
        # unless the caller overrides it. A live close must NOT stamp the
        # shadow magic — the daily E2 check reads shadow-magic deals as
        # order evidence and would false-positive.
        close_magic = (
            int(magic)
            if magic is not None
            else int(getattr(position, "magic", 0) or 0)
        )
        payload = {
            "action": getattr(mt5, "TRADE_ACTION_DEAL", 1),
            "position": int(ticket),
            "symbol": broker_symbol,
            "type": getattr(mt5, "ORDER_TYPE_SELL", 1)
            if is_buy
            else getattr(mt5, "ORDER_TYPE_BUY", 0),
            "volume": float(volume if volume is not None else position.volume),
            "price": price,
            "magic": close_magic,
            "comment": "alex close",
            "type_time": getattr(mt5, "ORDER_TIME_GTC", 0),
            "type_filling": getattr(mt5, "ORDER_FILLING_IOC", 0),
        }
        result = self._invoke(
            "order_send", CALL_WRITE, mt5.order_send, payload, detail={"ticket": int(ticket)}
        )
        if result is None:
            return OrderAck(sent=True, status="FAILED", reason=str(mt5.last_error()))
        retcode = int(getattr(result, "retcode", 0))
        done = int(getattr(mt5, "TRADE_RETCODE_DONE", 10009))
        return OrderAck(
            sent=True,
            status="CLOSED" if retcode == done else "REJECTED",
            retcode=retcode,
            ticket=int(ticket),
            reason=str(getattr(result, "comment", "")),
        )
