"""Stage E2: broker adapter layer (shadow mode).

An adapter interface, a MetaTrader 5 implementation, a reconciliation loop
and a would-have-filled shadow ledger. The layer reads the account and
positions and records what an order *would* have done; it never places one.

Constitution, enforced by ``tests/unit/test_broker_shadow.py``:

* every number comes from the existing execution engine (the same
  ``OrderManager``/``PositionManager`` the backtest and paper layers use) —
  there is no second cost model;
* writes are double-gated (AUTONOMOUS(4) permission **and** shadow mode
  off) and unreachable while ``MAX_PERMISSION_LEVEL=0``;
* no permission self-promotion, no network, no LLM in this package.

Use :meth:`forex_ai_agent.brokers.MT5Adapter.for_shadow` to build an
adapter — it forces shadow mode on regardless of what a caller passes.
"""

from .base import (
    ALEX_SHADOW_MAGIC,
    CALL_CONTROL,
    CALL_READ,
    CALL_WRITE,
    WRITE_PERMISSION_LEVEL,
    AccountSnapshot,
    BrokerAdapter,
    BrokerCall,
    BrokerError,
    BrokerNotConnected,
    BrokerPermissionDenied,
    OrderAck,
    OrderRequest,
    PositionSnapshot,
    ReadonlyBroker,
    ShadowModeRefusal,
    SymbolSpec,
    broker_spread_pips,
    default_permission_manager,
)
from .mt5_adapter import MT5Adapter, MT5Unavailable
from .reconcile import (
    RECONCILE_ADOPT,
    RECONCILE_POLICY,
    RECONCILE_REVIEW,
    ReconciliationReport,
    adopt_into,
    reconcile,
)
from .shadow import (
    ShadowEntry,
    ShadowExecutor,
    ShadowLedger,
    engine_atr,
    session_for,
)

__all__ = [
    "ALEX_SHADOW_MAGIC",
    "CALL_CONTROL",
    "CALL_READ",
    "CALL_WRITE",
    "WRITE_PERMISSION_LEVEL",
    "AccountSnapshot",
    "BrokerAdapter",
    "BrokerCall",
    "BrokerError",
    "BrokerNotConnected",
    "BrokerPermissionDenied",
    "OrderAck",
    "OrderRequest",
    "PositionSnapshot",
    "ReadonlyBroker",
    "ShadowModeRefusal",
    "SymbolSpec",
    "broker_spread_pips",
    "default_permission_manager",
    "MT5Adapter",
    "MT5Unavailable",
    "RECONCILE_ADOPT",
    "RECONCILE_POLICY",
    "RECONCILE_REVIEW",
    "ReconciliationReport",
    "adopt_into",
    "reconcile",
    "ShadowEntry",
    "ShadowExecutor",
    "ShadowLedger",
    "engine_atr",
    "session_for",
]
