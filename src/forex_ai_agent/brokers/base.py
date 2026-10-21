"""Stage E2: broker adapter interface (shadow mode).

Two surfaces, hard-separated:

* **READ** — account snapshot, open positions, pending orders, symbol
  specs, deal history. Callable at any permission level; this is the whole
  surface a shadow run ever touches.
* **WRITE** — submit / modify / close. The methods exist because E3/E4
  will need them, but they are unreachable today. Every write method calls
  :meth:`BrokerAdapter._authorize_write` *before* it can touch a transport
  and that helper refuses unless **both** guards pass:

  1. the permission manager reports AUTONOMOUS(4) — impossible while
     ``MAX_PERMISSION_LEVEL=0`` is the configured ceiling, and
  2. the adapter is not in shadow mode, which is the default and the only
     mode this application ever builds (``MT5Adapter.for_shadow``).

Permission self-promotion does not exist here: nothing in this package may
promote its own permission level. Level changes are human sign-offs
elsewhere.

Every terminal call is recorded in an auditable trail tagged READ / WRITE /
CONTROL, so "zero real orders were sent" becomes a check over a log
(:meth:`ReadonlyBroker.sent_calls`) instead of a promise. The transport is
whatever a subclass injects, so the whole interface is testable against a
stub with no MetaTrader terminal present.

No network, no LLM, no numbers of its own: this module only declares types
and enforces gates.
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..core.permission_manager import PermissionManager, PermissionLevel

__all__ = [
    "CALL_READ",
    "CALL_WRITE",
    "CALL_CONTROL",
    "WRITE_PERMISSION_LEVEL",
    "ALEX_SHADOW_MAGIC",
    "BrokerError",
    "BrokerNotConnected",
    "BrokerPermissionDenied",
    "ShadowModeRefusal",
    "AccountSnapshot",
    "PositionSnapshot",
    "SymbolSpec",
    "BrokerCall",
    "OrderRequest",
    "OrderAck",
    "ReadonlyBroker",
    "BrokerAdapter",
    "default_permission_manager",
]

# Audit classes. READ/WRITE are the load-bearing distinction; CONTROL is
# connection lifecycle (initialize/shutdown) which neither reads data nor
# places anything.
CALL_READ = "READ"
CALL_WRITE = "WRITE"
CALL_CONTROL = "CONTROL"

# Sending an order is AUTONOMOUS(4) work. Checked at runtime by every write
# method before it reaches a transport — see BrokerAdapter._authorize_write.
WRITE_PERMISSION_LEVEL = PermissionLevel.AUTONOMOUS

# Reserved order magic. Nothing in this package sends an order, so no deal
# carrying this magic can exist; the live verify asserts exactly that
# against the broker's own deal history (an independent, terminal-side
# proof rather than a self-report).
ALEX_SHADOW_MAGIC = 314159


class BrokerError(Exception):
    """Base class for adapter failures."""


class BrokerNotConnected(BrokerError):
    """Raised when a read/write is attempted without a live connection."""


class BrokerPermissionDenied(BrokerError):
    """Raised when a write is attempted below WRITE_PERMISSION_LEVEL."""


class ShadowModeRefusal(BrokerError):
    """Raised when a write is attempted on a shadow-mode adapter."""


def default_permission_manager() -> PermissionManager:
    """Permission manager seeded from ``MAX_PERMISSION_LEVEL``.

    Fail-closed: if settings cannot be read for any reason the level falls
    back to OBSERVE(0) — the least privileged level — never upward.
    """
    try:
        from ..config import get_settings

        level = int(get_settings().max_permission_level)
    except Exception:  # noqa: BLE001 — any settings failure means level 0
        level = PermissionLevel.OBSERVE
    return PermissionManager(level)


def _dump(instance: Any) -> Dict[str, Any]:
    """JSON-friendly dict for the snapshot dataclasses (datetimes ISO)."""
    out = asdict(instance)
    for name, value in out.items():
        if isinstance(value, datetime):
            out[name] = value.isoformat()
    return out


# ---------------------------------------------------------------------------
# Read-only snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountSnapshot:
    """Point-in-time account state exactly as the broker reports it."""

    login: int
    server: str
    currency: str
    balance: float
    equity: float
    margin: float
    margin_free: float
    leverage: int
    profit: float = 0.0
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return _dump(self)


@dataclass(frozen=True)
class PositionSnapshot:
    """One open broker position (read-only; ``side`` is BUY or SELL)."""

    ticket: int
    symbol: str
    side: str
    volume: float
    price_open: float
    profit: float = 0.0
    swap: float = 0.0
    sl: Optional[float] = None
    tp: Optional[float] = None
    magic: int = 0
    comment: str = ""
    time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return _dump(self)


@dataclass(frozen=True)
class SymbolSpec:
    """Trading specification used to validate an intent before it is armed.

    ``spread`` is the broker's live spread in POINTS (MT5's own units);
    :func:`broker_spread_pips` converts it to pips with ``point``.
    """

    symbol: str
    broker_symbol: str
    digits: int
    point: float
    tick_size: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    spread: float = 0.0
    trade_allowed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return _dump(self)


@dataclass(frozen=True)
class BrokerCall:
    """One recorded terminal call — the zero-orders audit trail."""

    method: str
    mode: str  # READ / WRITE / CONTROL
    ok: bool
    error: Optional[str]
    timestamp: datetime
    detail: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return _dump(self)


@dataclass(frozen=True)
class OrderRequest:
    """A fully specified order intent handed to :meth:`BrokerAdapter.submit_order`."""

    key: str
    symbol: str
    side: str  # BUY / SELL
    volume: float
    price: float = 0.0
    sl: Optional[float] = None
    tp: Optional[float] = None
    magic: int = ALEX_SHADOW_MAGIC
    comment: str = ""
    deviation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return _dump(self)


@dataclass(frozen=True)
class OrderAck:
    """Outcome of a write attempt.

    ``sent`` is False whenever authorization refused — the refusal is the
    success case at permission level 0. ``would_have_been_sent`` preserves
    the request for the audit log without ever touching the wire.
    """

    sent: bool
    status: str
    reason: str = ""
    ticket: Optional[int] = None
    retcode: Optional[int] = None
    request: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return _dump(self)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class ReadonlyBroker(abc.ABC):
    """Read-only broker surface plus the call audit trail.

    Subclasses must funnel every terminal call through :meth:`_invoke` so
    the audit trail is complete by construction — a method that bypasses it
    cannot be proven read-only, so don't write one.
    """

    name = "readonly"

    def __init__(self, permission: Optional[PermissionManager] = None) -> None:
        self._permission = permission or default_permission_manager()
        self._connected = False
        self._calls: List[BrokerCall] = []

    # -- observable state ------------------------------------------------

    @property
    def permission(self) -> PermissionManager:
        return self._permission

    @property
    def permission_level(self) -> int:
        return self._permission.current_level

    @property
    def connected(self) -> bool:
        return self._connected

    # -- audit trail -----------------------------------------------------

    def audit_trail(self) -> List[BrokerCall]:
        """Every recorded call, oldest first."""
        return list(self._calls)

    def calls(self, mode: Optional[str] = None) -> List[BrokerCall]:
        return [c for c in self._calls if mode is None or c.mode == mode]

    def write_calls(self) -> List[BrokerCall]:
        """All WRITE-class records: refused attempts and real sends alike."""
        return self.calls(CALL_WRITE)

    def sent_calls(self) -> List[BrokerCall]:
        """WRITE-class calls that PASSED authorization and reached the transport.

        This is the zero-orders assertion: it must be empty for a shadow run.
        """
        return [c for c in self.write_calls() if c.ok]

    def read_calls(self) -> List[BrokerCall]:
        return self.calls(CALL_READ)

    def audit_summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for call in self._calls:
            counts[call.mode] = counts.get(call.mode, 0) + 1
        return {
            "adapter": self.name,
            "connected": self._connected,
            "permission_level": self._permission.current_level,
            "shadow_mode": getattr(self, "shadow_mode", None),
            "calls": counts,
            "sent_writes": len(self.sent_calls()),
        }

    # -- audit helper ----------------------------------------------------

    def _record_call(
        self,
        method: str,
        mode: str,
        *,
        ok: bool = True,
        error: Optional[str] = None,
        **detail: Any,
    ) -> BrokerCall:
        call = BrokerCall(
            method=method,
            mode=mode,
            ok=bool(ok),
            error=error,
            timestamp=datetime.utcnow(),
            detail=detail,
        )
        self._calls.append(call)
        return call

    def _invoke(
        self,
        method: str,
        mode: str,
        fn: Callable[..., Any],
        *args: Any,
        detail: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run one terminal call and record it (success or exception)."""
        payload = detail or {}
        try:
            # CPython passes an empty keyword dict when the call uses
            # ``**kwargs`` syntax even when it is empty, and the terminal
            # C parser rejects ANY keyword dict on the send call (returns
            # None + (-2, 'Unnamed arguments not allowed')). Call without
            # keyword spreading when there are none.
            if kwargs:
                result = fn(*args, **kwargs)
            else:
                result = fn(*args)
        except Exception as exc:  # noqa: BLE001 — IPC errors are data here
            self._record_call(
                method, mode, ok=False, error=f"{type(exc).__name__}: {exc}", **payload
            )
            raise
        self._record_call(method, mode, ok=True, **payload)
        return result

    def _require_connected(self) -> None:
        if not self._connected:
            raise BrokerNotConnected(f"{self.name} adapter is not connected")

    # -- lifecycle -------------------------------------------------------

    @abc.abstractmethod
    def connect(self) -> bool:
        """Open the terminal connection. Returns True when usable."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close the terminal connection (idempotent)."""

    # -- reads -----------------------------------------------------------

    @abc.abstractmethod
    def account(self) -> AccountSnapshot:
        """Account snapshot: balance, equity, margin, leverage."""

    @abc.abstractmethod
    def positions(self) -> List[PositionSnapshot]:
        """Open positions."""

    @abc.abstractmethod
    def pending_orders(self) -> List[Dict[str, Any]]:
        """Pending (not yet executed) broker orders."""

    @abc.abstractmethod
    def symbol_spec(self, symbol: str) -> SymbolSpec:
        """Trading spec for ``symbol`` (Alex form, e.g. ``EUR/USD``)."""

    @abc.abstractmethod
    def history_deals(
        self,
        start: datetime,
        end: datetime,
        magic: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Deal history for the window (optionally filtered by magic)."""


class BrokerAdapter(ReadonlyBroker):
    """Read-only surface plus permission-gated write operations.

    Subclasses implement the writes, but the gate lives here so that every
    adapter — present and future — inherits the same two-refusal rule.
    """

    def __init__(
        self,
        permission: Optional[PermissionManager] = None,
        *,
        shadow_mode: bool = True,
    ) -> None:
        super().__init__(permission)
        self._shadow_mode = bool(shadow_mode)

    @property
    def shadow_mode(self) -> bool:
        return self._shadow_mode

    # -- the gate --------------------------------------------------------

    def _authorize_write(self, operation: str) -> None:
        """Guard 1 (permission) then Guard 2 (shadow mode).

        Both must pass before a write method may reach a transport. Refusal
        records an unsuccessful WRITE call so the attempt is auditable, and
        raises — callers never see a "maybe it went out".
        """
        if not self._permission.check_permission(WRITE_PERMISSION_LEVEL):
            self._record_call(
                operation,
                CALL_WRITE,
                ok=False,
                error="permission_denied",
                required_level=WRITE_PERMISSION_LEVEL,
                current_level=self._permission.current_level,
            )
            raise BrokerPermissionDenied(
                f"{operation} refused: requires permission level "
                f"{WRITE_PERMISSION_LEVEL} ({PermissionLevel.label(WRITE_PERMISSION_LEVEL)}), "
                f"current level is {self._permission.current_level} "
                f"({PermissionLevel.label(self._permission.current_level)}). "
                "MAX_PERMISSION_LEVEL=0 keeps this adapter read-only."
            )
        if self._shadow_mode:
            self._record_call(
                operation,
                CALL_WRITE,
                ok=False,
                error="shadow_mode",
                current_level=self._permission.current_level,
            )
            raise ShadowModeRefusal(
                f"{operation} refused: adapter is in shadow mode (Stage E2). "
                "Disabling shadow mode is a human sign-off, not a default."
            )
        # Both guards passed: a human granted AUTONOMOUS(4) AND shadow mode
        # was explicitly switched off. The subclass may now reach the wire.

    # -- write surface (unreachable at MAX_PERMISSION_LEVEL=0) -----------

    def submit_order(self, request: OrderRequest) -> OrderAck:
        raise NotImplementedError

    def modify_order(
        self, ticket: int, *, sl: Optional[float] = None, tp: Optional[float] = None
    ) -> OrderAck:
        raise NotImplementedError

    def close_position(self, ticket: int, volume: Optional[float] = None) -> OrderAck:
        raise NotImplementedError


def broker_spread_pips(spec: SymbolSpec, *, pip_size: float) -> float:
    """Broker spread in points → pips (uses the spec's own ``point``)."""
    if pip_size <= 0:
        raise ValueError(f"pip_size must be positive, got {pip_size!r}")
    return (float(spec.spread) * float(spec.point)) / pip_size


def snapshot_fields(cls: type) -> List[str]:
    """Field names of a snapshot dataclass (used by tests and reports)."""
    return [f.name for f in fields(cls)]
