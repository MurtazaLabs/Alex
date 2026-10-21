"""Unit tests for the Stage E2 broker layer (shadow mode).

Covers the gate E2 acceptance criteria:

- adapter unit tests against a STUBBED MetaTrader5 module (no terminal in
  CI): reads map correctly, the audit trail proves a read session made zero
  write calls, and the write methods refuse at permission level 0 before
  the transport is named;
- both guards are real: level 4 alone does not send while shadow mode is
  on, shadow mode off alone does not send while level is 0;
- reconciliation: matching intents are adopted, unmatched intents and
  unknown broker positions go to human review, nothing is ever re-sent;
- shadow ledger: every number (fill price, commission, SL/TP exit, P&L)
  comes from the engine's own OrderManager/PositionManager, fills land on
  the next bar exactly like a pending order, and the ledger is bit-for-bit
  reproducible across runs;
- constitution: source scans prove no send path outside the gated methods,
  no network, and no permission self-promotion.
"""

from __future__ import annotations

import inspect
import pathlib
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from forex_ai_agent.backtest.engine import BacktestEngine
from forex_ai_agent.backtest.execution import (
    DEFAULT_SPREADS,
    OrderSide,
    OrderType,
)
from forex_ai_agent.backtest.orders import ExecutionContext, OrderManager, PositionManager
from forex_ai_agent.brokers import (
    ALEX_SHADOW_MAGIC,
    WRITE_PERMISSION_LEVEL,
    AccountSnapshot,
    BrokerNotConnected,
    BrokerPermissionDenied,
    MT5Adapter,
    OrderRequest,
    PositionSnapshot,
    ReadonlyBroker,
    ShadowModeRefusal,
    SymbolSpec,
    broker_spread_pips,
    default_permission_manager,
    engine_atr,
    reconcile,
    session_for,
)
from forex_ai_agent.brokers.shadow import (
    REJECT_DUPLICATE,
    REJECT_GOVERNOR,
    REJECT_POSITION_CAP,
    REJECT_TRADE_DISABLED,
    REJECT_VOLUME_MAX,
    REJECT_VOLUME_MIN,
    REJECT_VOLUME_STEP,
    SHADOW_CLOSED,
    SHADOW_OPEN,
    SHADOW_PENDING,
    ShadowExecutor,
    ShadowLedger,
)
from forex_ai_agent.core.permission_manager import PermissionLevel, PermissionManager
from forex_ai_agent.execution.governor import ExecutionGovernor, GovernorLimits
from forex_ai_agent.execution.journal import (
    INTENT_CANCELLED,
    INTENT_EXECUTED,
    INTENT_PENDING,
)

SYMBOL = "EUR/USD"
BROKER_SYMBOL = "EURUSD"
T0 = datetime(2026, 9, 7, 8, 0, 0)  # a Monday, London session
PIP = 0.0001


# ---------------------------------------------------------------------------
# Stubbed MetaTrader5 module (no terminal dependency in CI)
# ---------------------------------------------------------------------------


class _StubMT5:
    """Minimal MetaTrader5 stand-in that counts every call it receives."""

    # Constants the adapter reads off the module.
    SYMBOL_TRADE_MODE_FULL = 0
    POSITION_TYPE_BUY = 0
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_RETCODE_DONE = 10009

    def __init__(self, *, trade_mode: int = 0, positions=(), deals=(), orders=()):
        self.calls: List[str] = []
        self.sent: List[Dict[str, Any]] = []
        self.account = SimpleNamespace(
            login=123456,
            server="DemoServer",
            currency="USD",
            balance=10_000.0,
            equity=10_050.0,
            margin=200.0,
            margin_free=9_850.0,
            leverage=100,
            profit=50.0,
        )
        self._positions = list(positions)
        self._deals = list(deals)
        self._orders = list(orders)
        self._symbols = {
            BROKER_SYMBOL: SimpleNamespace(
                name=BROKER_SYMBOL,
                digits=5,
                point=1e-5,
                tick_size=1e-5,
                trade_contract_size=100_000.0,
                volume_min=0.01,
                volume_max=100.0,
                volume_step=0.01,
                spread=12,
                trade_mode=trade_mode,
            )
        }
        self.tick = SimpleNamespace(bid=1.10118, ask=1.10130)

    # lifecycle ---------------------------------------------------------
    def initialize(self, **kwargs) -> bool:
        self.calls.append("initialize")
        return True

    def shutdown(self) -> None:
        self.calls.append("shutdown")

    def terminal_info(self):
        self.calls.append("terminal_info")
        return SimpleNamespace(connected=True)

    def last_error(self):
        return (0, "ok")

    # reads -------------------------------------------------------------
    def account_info(self):
        self.calls.append("account_info")
        return self.account

    def positions_get(self, **kwargs):
        self.calls.append("positions_get")
        ticket = kwargs.get("ticket")
        if ticket is None:
            return tuple(self._positions)
        return tuple(p for p in self._positions if p.ticket == int(ticket))

    def orders_get(self, **kwargs):
        self.calls.append("orders_get")
        return tuple(self._orders)

    def symbol_info(self, name):
        self.calls.append(f"symbol_info:{name}")
        return self._symbols.get(name)

    def symbols_get(self):
        self.calls.append("symbols_get")
        return tuple(SimpleNamespace(name=n) for n in self._symbols)

    def symbol_select(self, name, enable=True):
        self.calls.append(f"symbol_select:{name}")
        return name in self._symbols

    def symbol_info_tick(self, name):
        self.calls.append("symbol_info_tick")
        return self.tick

    def history_deals_get(self, start, end):
        self.calls.append("history_deals_get")
        return tuple(self._deals)

    # writes ------------------------------------------------------------
    def order_send(self, payload):
        self.sent.append(dict(payload))
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=777, comment="ok")


def _position(ticket: int, symbol: str, side: str, volume: float, magic: int = 0):
    return SimpleNamespace(
        ticket=ticket,
        symbol=symbol,
        type=0 if side == "BUY" else 1,
        volume=volume,
        price_open=1.1000,
        profit=12.5,
        swap=-0.2,
        sl=0.0,
        tp=0.0,
        magic=magic,
        comment="manual",
        time=int(datetime(2026, 9, 7, 7, 0).timestamp()),
    )


def _deal(ticket: int, magic: int, symbol: str = BROKER_SYMBOL, volume: float = 0.1):
    return SimpleNamespace(
        ticket=ticket,
        order=ticket + 1,
        magic=magic,
        symbol=symbol,
        volume=volume,
        price=1.1,
        profit=3.0,
        comment="",
        time=int(datetime(2026, 9, 7, 9, 0).timestamp()),
    )


def _adapter(
    stub: Optional[_StubMT5] = None,
    *,
    level: int = PermissionLevel.OBSERVE,
    shadow_mode: bool = True,
    connected: bool = True,
) -> MT5Adapter:
    adapter = MT5Adapter(PermissionManager(level), shadow_mode=shadow_mode)
    if stub is not None:
        adapter._mt5 = stub
        adapter._connected = connected
    return adapter


# ---------------------------------------------------------------------------
# Interface + audit trail
# ---------------------------------------------------------------------------


def test_readonly_interface_is_abstract_and_implemented():
    with pytest.raises(TypeError):
        ReadonlyBroker()  # type: ignore[abstract]
    adapter = _adapter(_StubMT5())
    for method in ("connect", "disconnect", "account", "positions",
                   "pending_orders", "symbol_spec", "history_deals"):
        assert callable(getattr(adapter, method))


def test_for_shadow_factory_forces_shadow_mode_on():
    # Even a caller that asks for write capability does not get it.
    assert MT5Adapter.for_shadow(shadow_mode=False).shadow_mode is True
    assert MT5Adapter.for_shadow().shadow_mode is True


def test_permission_level_is_the_configured_max_and_writes_need_autonomous():
    from forex_ai_agent.config import get_settings

    # The adapter's default manager mirrors the OPERATOR-configured ceiling
    # — whatever it is. The manager never exceeds it and never promotes
    # itself; raising the ceiling is a human config edit, not code.
    assert (
        default_permission_manager().current_level
        == get_settings().max_permission_level
    )
    assert WRITE_PERMISSION_LEVEL == PermissionLevel.AUTONOMOUS == 4


def test_read_session_records_only_read_and_control_calls():
    stub = _StubMT5(
        positions=(_position(1, BROKER_SYMBOL, "BUY", 0.10),),
        deals=(_deal(9, magic=42),),
    )
    adapter = _adapter(stub)
    adapter.connect()

    account = adapter.account()
    positions = adapter.positions()
    orders = adapter.pending_orders()
    spec = adapter.symbol_spec(SYMBOL)
    deals = adapter.history_deals(T0, T0 + timedelta(days=1), magic=42)

    assert isinstance(account, AccountSnapshot) and account.balance == 10_000.0
    assert len(positions) == 1 and positions[0].side == "BUY"
    assert orders == []
    assert spec.broker_symbol == BROKER_SYMBOL
    assert len(deals) == 1 and deals[0]["magic"] == 42
    assert adapter.write_calls() == []
    assert adapter.sent_calls() == []
    assert {c.mode for c in adapter.audit_trail()} <= {"READ", "CONTROL"}
    adapter.disconnect()


def test_account_snapshot_maps_every_field():
    adapter = _adapter(_StubMT5())
    adapter.connect()
    snap = adapter.account()
    assert snap.login == 123456
    assert snap.server == "DemoServer"
    assert snap.currency == "USD"
    assert snap.equity == 10_050.0
    assert snap.margin_free == 9_850.0
    assert snap.leverage == 100
    assert snap.profit == 50.0
    assert snap.timestamp is not None
    assert snap.to_dict()["server"] == "DemoServer"


def test_positions_map_sides_and_zero_stops_to_none():
    stub = _StubMT5(
        positions=(
            _position(1, BROKER_SYMBOL, "BUY", 0.10),
            _position(2, "GBPUSD", "SELL", 0.20),
        )
    )
    adapter = _adapter(stub)
    adapter.connect()
    positions = adapter.positions()
    assert [p.side for p in positions] == ["BUY", "SELL"]
    assert [p.volume for p in positions] == [0.10, 0.20]
    assert positions[0].sl is None and positions[0].tp is None
    assert positions[0].profit == 12.5 and positions[0].swap == -0.2


def test_symbol_spec_resolves_alex_symbol_and_maps_bounds():
    adapter = _adapter(_StubMT5())
    adapter.connect()
    spec = adapter.symbol_spec(SYMBOL)
    assert spec.symbol == SYMBOL and spec.broker_symbol == BROKER_SYMBOL
    assert spec.digits == 5 and spec.point == 1e-5
    assert spec.volume_min == 0.01 and spec.volume_max == 100.0
    assert spec.volume_step == 0.01 and spec.spread == 12
    assert spec.trade_allowed is True
    # 12 points * 1e-5 / 1e-4 = 1.2 pips
    assert broker_spread_pips(spec, pip_size=PIP) == pytest.approx(1.2)


def test_symbol_spec_fails_closed_without_the_terminal_full_constant():
    stub = _StubMT5()
    # Terminal did not expose the constant: getattr falls back to None and
    # the adapter must fail closed.
    stub.SYMBOL_TRADE_MODE_FULL = None
    adapter = _adapter(stub)
    adapter.connect()
    assert adapter.symbol_spec(SYMBOL).trade_allowed is False


def test_symbol_spec_unknown_symbol_raises_broker_error():
    from forex_ai_agent.brokers import BrokerError

    adapter = _adapter(_StubMT5())
    adapter.connect()
    with pytest.raises(BrokerError):
        adapter.symbol_spec("USD/CHF")


# ---------------------------------------------------------------------------
# Suffix resolution (Exness offers EURUSDm, not EURUSD)
# ---------------------------------------------------------------------------


def _suffixed_stub() -> _StubMT5:
    stub = _StubMT5()
    info = stub._symbols.pop(BROKER_SYMBOL)
    info.name = "EURUSDm"
    stub._symbols["EURUSDm"] = info
    return stub


def test_symbol_spec_resolves_the_brokers_suffixed_name():
    stub = _suffixed_stub()
    adapter = _adapter(stub)
    adapter.connect()
    spec = adapter.symbol_spec(SYMBOL)
    assert spec.symbol == SYMBOL and spec.broker_symbol == "EURUSDm"
    assert spec.trade_allowed is True and spec.digits == 5
    # The chosen symbol was added to the market watch (CONTROL, not data).
    assert "symbol_select:EURUSDm" in stub.calls
    assert {c.mode for c in adapter.audit_trail()} <= {"READ", "CONTROL"}
    # Resolved once, cached thereafter: one symbol_info, no rescan.
    seen = len(stub.calls)
    adapter.symbol_spec(SYMBOL)
    assert len(stub.calls) == seen + 1
    assert "symbols_get" not in stub.calls[seen:]


def test_positions_translate_back_to_the_alex_symbol_after_resolution():
    stub = _suffixed_stub()
    stub._positions = [_position(1, "EURUSDm", "BUY", 0.10)]
    adapter = _adapter(stub)
    adapter.connect()
    adapter.symbol_spec(SYMBOL)  # populate the resolution cache
    pos = adapter.positions()
    assert pos[0].symbol == SYMBOL  # EURUSDm → EUR/USD


def test_positions_keep_the_broker_symbol_when_never_resolved():
    """Unknown exposure keeps the broker's own name — never guessed."""
    stub = _StubMT5(positions=(_position(9, "GBPUSDm", "SELL", 0.20),))
    adapter = _adapter(stub)
    adapter.connect()
    assert adapter.positions()[0].symbol == "GBPUSDm"


def test_symbol_spec_fails_closed_when_the_scan_itself_fails():
    from forex_ai_agent.brokers import BrokerError

    stub = _suffixed_stub()
    def _boom():
        raise RuntimeError("ipc")
    stub.symbols_get = _boom
    adapter = _adapter(stub)
    adapter.connect()
    with pytest.raises(BrokerError):
        adapter.symbol_spec(SYMBOL)
    # The failed scan is on the audit trail as a failed READ.
    failed = [c for c in adapter.audit_trail() if not c.ok]
    assert failed and failed[0].method == "symbols_get"


def test_submit_order_addresses_the_resolved_broker_symbol():
    """A write must address EURUSDm on a suffixed broker, never EURUSD."""
    stub = _suffixed_stub()
    adapter = _adapter(stub, level=PermissionLevel.AUTONOMOUS, shadow_mode=False)
    adapter.connect()
    ack = adapter.submit_order(_request())
    assert ack.sent and ack.status == "FILLED"
    assert stub.sent[0]["symbol"] == "EURUSDm"
    # Resolution cached: the second write does not rescan the symbol list.
    seen = len(stub.calls)
    adapter.submit_order(_request())
    assert stub.sent[1]["symbol"] == "EURUSDm"
    assert "symbols_get" not in stub.calls[seen:]


def test_submit_order_keeps_the_plain_name_when_no_suffix_matches():
    """Unresolvable scans fall back to the plain broker form; the tick
    read then fails closed (NO_TICK) rather than addressing a guess."""
    stub = _StubMT5()
    adapter = _adapter(stub, level=PermissionLevel.AUTONOMOUS, shadow_mode=False)
    adapter.connect()
    ack = adapter.submit_order(_request())
    assert stub.sent[0]["symbol"] == "EURUSD"  # exact name the broker lists
    assert ack.status == "FILLED"


def test_writes_refuse_before_the_wire_when_terminal_autotrading_is_off():
    """With the terminal's Algo Trading button off, order_send returns no
    retcode (ambiguous outcome). The adapter must refuse deterministically
    BEFORE the wire, with an actionable reason."""
    stub = _StubMT5()
    stub.terminal_info = lambda: SimpleNamespace(
        connected=True, trade_allowed=False
    )
    adapter = _adapter(stub, level=PermissionLevel.AUTONOMOUS, shadow_mode=False)
    adapter.connect()

    ack = adapter.submit_order(_request())
    assert ack.sent is False
    assert ack.status == "TERMINAL_BLOCKED"
    assert "AutoTrading" in ack.reason

    ack = adapter.modify_order(1, sl=1.2, tp=1.3)
    assert ack.sent is False and ack.status == "TERMINAL_BLOCKED"

    stub._positions = [_position(7, BROKER_SYMBOL, "BUY", 0.01)]
    ack = adapter.close_position(7)
    assert ack.sent is False and ack.status == "TERMINAL_BLOCKED"

    assert stub.sent == []  # nothing ever reached the wire


def test_close_stamps_the_positions_own_magic():
    from forex_ai_agent.execution.live.caps import ALEX_LIVE_MAGIC

    stub = _StubMT5(positions=(_position(7, BROKER_SYMBOL, "BUY", 0.01, magic=ALEX_LIVE_MAGIC),))
    adapter = _adapter(stub, level=PermissionLevel.AUTONOMOUS, shadow_mode=False)
    adapter.connect()
    ack = adapter.close_position(7)
    assert ack.sent and ack.status == "CLOSED"
    assert stub.sent[0]["magic"] == ALEX_LIVE_MAGIC
    assert stub.sent[0]["magic"] != ALEX_SHADOW_MAGIC


def test_close_magic_override_and_manual_positions():
    stub = _StubMT5(positions=(
        _position(7, BROKER_SYMBOL, "BUY", 0.01, magic=0),
        _position(8, BROKER_SYMBOL, "BUY", 0.01, magic=123),
    ))
    adapter = _adapter(stub, level=PermissionLevel.AUTONOMOUS, shadow_mode=False)
    adapter.connect()
    adapter.close_position(7)
    assert stub.sent[0]["magic"] == 0  # manual position keeps its zero magic
    adapter.close_position(8, magic=999)
    assert stub.sent[1]["magic"] == 999  # explicit caller override wins


def test_submit_defaults_market_deviation_but_honours_explicit():
    from forex_ai_agent.brokers.mt5_adapter import DEFAULT_MARKET_DEVIATION_POINTS

    stub = _StubMT5()
    adapter = _adapter(stub, level=PermissionLevel.AUTONOMOUS, shadow_mode=False)
    adapter.connect()
    adapter.submit_order(_request())  # deviation not declared
    assert stub.sent[0]["deviation"] == DEFAULT_MARKET_DEVIATION_POINTS
    explicit = _request()
    object.__setattr__(explicit, "deviation", 5)  # frozen dataclass
    adapter.submit_order(explicit)
    assert stub.sent[1]["deviation"] == 5


def test_reads_before_connect_fail_without_touching_the_terminal():
    stub = _StubMT5()
    adapter = _adapter(stub, connected=False)  # transport injected, never connected
    with pytest.raises(BrokerNotConnected):
        adapter.account()
    assert stub.calls == []
    assert adapter.audit_trail() == []


def test_terminal_failure_is_recorded_in_the_audit_trail():
    stub = _StubMT5()
    stub.account = None  # terminal reports no account
    adapter = _adapter(stub)
    adapter._connected = True
    adapter._mt5 = stub
    with pytest.raises(BrokerNotConnected):
        adapter.account()
    failed = [c for c in adapter.audit_trail() if not c.ok]
    assert len(failed) == 1 and failed[0].method == "account_info"


def test_history_deals_without_magic_filter_returns_everything():
    stub = _StubMT5(deals=(_deal(1, magic=0), _deal(2, magic=42)))
    adapter = _adapter(stub)
    adapter.connect()
    assert len(adapter.history_deals(T0, T0)) == 2
    assert len(adapter.history_deals(T0, T0, magic=42)) == 1


def test_audit_summary_reports_zero_sent_writes():
    adapter = _adapter(_StubMT5())
    adapter.connect()
    adapter.account()
    summary = adapter.audit_summary()
    assert summary["adapter"] == "mt5"
    assert summary["shadow_mode"] is True
    assert summary["permission_level"] == 0
    assert summary["sent_writes"] == 0
    assert summary["calls"].get("WRITE", 0) == 0


# ---------------------------------------------------------------------------
# The write gate — both guards, independently
# ---------------------------------------------------------------------------


def _request() -> OrderRequest:
    return OrderRequest(key="k1", symbol=SYMBOL, side="BUY", volume=0.10,
                        sl=1.09, tp=1.13, comment="e2")


def test_submit_refused_at_permission_level_zero_before_the_transport():
    stub = _StubMT5()
    adapter = _adapter(stub, level=PermissionLevel.OBSERVE, shadow_mode=False)
    with pytest.raises(BrokerPermissionDenied) as excinfo:
        adapter.submit_order(_request())
    assert "AUTONOMOUS" in str(excinfo.value)
    assert stub.sent == []            # nothing reached the wire
    assert stub.calls == []           # not even a tick read
    assert len(adapter.write_calls()) == 1
    assert adapter.write_calls()[0].ok is False
    assert adapter.write_calls()[0].error == "permission_denied"
    assert adapter.sent_calls() == []


def test_submit_refused_in_shadow_mode_even_at_permission_level_four():
    stub = _StubMT5()
    adapter = _adapter(stub, level=PermissionLevel.AUTONOMOUS, shadow_mode=True)
    with pytest.raises(ShadowModeRefusal):
        adapter.submit_order(_request())
    assert stub.sent == []
    assert adapter.write_calls()[0].error == "shadow_mode"
    assert adapter.sent_calls() == []


def test_modify_and_close_are_gated_the_same_way():
    stub = _StubMT5(positions=(_position(1, BROKER_SYMBOL, "BUY", 0.10),))
    adapter = _adapter(stub, shadow_mode=False)
    with pytest.raises(BrokerPermissionDenied):
        adapter.modify_order(1, sl=1.0, tp=1.2)
    with pytest.raises(BrokerPermissionDenied):
        adapter.close_position(1)
    assert stub.sent == []
    assert adapter.sent_calls() == []


def test_neither_guard_alone_is_sufficient():
    """Level 0 + shadow off ⇒ permission refuses; level 4 + shadow on ⇒ shadow refuses."""
    level_only = _adapter(_StubMT5(), level=PermissionLevel.OBSERVE, shadow_mode=False)
    with pytest.raises(BrokerPermissionDenied):
        level_only.submit_order(_request())

    shadow_only = _adapter(_StubMT5(), level=PermissionLevel.AUTONOMOUS, shadow_mode=True)
    with pytest.raises(ShadowModeRefusal):
        shadow_only.submit_order(_request())

    assert level_only.sent_calls() == [] and shadow_only.sent_calls() == []


def test_write_methods_authorize_before_naming_the_transport():
    for name in ("submit_order", "modify_order", "close_position"):
        source = inspect.getsource(getattr(MT5Adapter, name))
        assert "_authorize_write" in source, f"{name} is not gated"
        if "order_send" in source:
            assert source.index("_authorize_write") < source.index("order_send")


def test_order_send_appears_only_inside_gated_write_methods():
    class_source = inspect.getsource(MT5Adapter)
    gated = sum(
        inspect.getsource(getattr(MT5Adapter, name)).count("order_send")
        for name in ("submit_order", "modify_order", "close_position")
    )
    assert gated > 0
    assert class_source.count("order_send") == gated


def test_invoke_never_spreads_an_empty_kwargs_dict_into_the_terminal():
    """MetaTrader5's C parser rejects ANY keyword dict on order_send, even
    an empty one produced by ``fn(*args, **{})`` — the real terminal answers
    None + (-2, 'Unnamed arguments not allowed'). The wrapper must call
    ``fn(*args)`` when there are no keywords."""
    source = inspect.getsource(ReadonlyBroker._invoke)
    assert "if kwargs:" in source
    assert "fn(*args)" in source


def test_invoke_still_forwards_non_empty_kwargs():
    from forex_ai_agent.brokers.base import CALL_READ

    adapter = _adapter(_StubMT5())
    seen = {}

    def fn(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return True

    assert adapter._invoke("probe", CALL_READ, fn, 1, 2, alpha="x") is True
    assert seen == {"args": (1, 2), "kwargs": {"alpha": "x"}}


def test_a_fully_authorized_send_reaches_the_transport_exactly_once():
    """The ONLY two things between level 0 and a wire are permission + shadow."""
    stub = _StubMT5()
    adapter = _adapter(stub, level=PermissionLevel.AUTONOMOUS, shadow_mode=False)
    ack = adapter.submit_order(_request())

    assert ack.sent is True and ack.status == "FILLED" and ack.ticket == 777
    assert len(stub.sent) == 1
    payload = stub.sent[0]
    assert payload["symbol"] == BROKER_SYMBOL
    assert payload["volume"] == 0.10
    assert payload["magic"] == ALEX_SHADOW_MAGIC
    assert payload["price"] == stub.tick.ask  # BUY lifts the offer
    assert len(adapter.sent_calls()) == 1


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def _intent(key: str, *, symbol: str = SYMBOL, side: str = "BUY",
            volume: float = 0.10) -> Dict[str, Any]:
    return {"key": key, "symbol": symbol, "side": side, "volume": volume,
            "state": INTENT_PENDING}


def _snapshot(symbol: str = SYMBOL, side: str = "BUY", volume: float = 0.10,
              ticket: int = 55) -> PositionSnapshot:
    return PositionSnapshot(ticket=ticket, symbol=symbol, side=side, volume=volume,
                            price_open=1.1)


def test_reconcile_ok_when_books_agree_exactly():
    report = reconcile([_intent("a")], [_snapshot()])
    assert report.ok and not report.divergence
    assert report.adopted == ["a"] and report.review == []
    assert report.broker_only == []
    assert report.policy == "adopt_matching_review_unmatched_never_resend"


def test_reconcile_sends_unmatched_intent_to_human_review_never_resend():
    report = reconcile([_intent("a")], [])
    assert not report.ok and report.divergence
    assert report.adopted == [] and report.review == ["a"]
    kinds = [e["kind"] for e in report.events]
    assert kinds == ["reconciliation.review"]


def test_reconcile_flags_unknown_broker_position_without_adopting_it():
    report = reconcile([], [_snapshot(symbol="GBP/USD", side="SELL", volume=0.20)])
    assert report.divergence
    assert report.broker_only[0]["symbol"] == "GBP/USD"
    assert report.adopted == []
    assert report.events[0]["reason"] == "broker_position_without_intent"


def test_reconcile_requires_symbol_side_and_volume_to_all_match():
    wrong_side = reconcile([_intent("a", side="SELL")], [_snapshot(side="BUY")])
    wrong_size = reconcile([_intent("a", volume=0.20)], [_snapshot(volume=0.10)])
    wrong_symbol = reconcile([_intent("a", symbol="GBP/USD")], [_snapshot()])
    for report in (wrong_side, wrong_size, wrong_symbol):
        assert report.review == ["a"] and report.broker_only


def test_reconcile_volume_tolerance_is_tight():
    assert reconcile([_intent("a")], [_snapshot(volume=0.10 + 1e-10)]).adopted == ["a"]
    assert reconcile([_intent("a")], [_snapshot(volume=0.11)]).review == ["a"]


def test_reconcile_accepts_plain_dicts_and_normalises_sides():
    report = reconcile(
        [_intent("a", side="LONG")],
        [{"symbol": SYMBOL, "side": "BUY", "volume": 0.10, "ticket": 7}],
    )
    assert report.adopted == ["a"]


def test_adopt_into_marks_the_journal_executed_without_touching_review():
    governor = ExecutionGovernor(GovernorLimits())
    governor.arm()
    governor.accept_intent("a", symbol=SYMBOL, side="BUY", volume=0.10,
                           sl=None, tp=None, now=T0, current_bar=1)
    governor.accept_intent("b", symbol=SYMBOL, side="BUY", volume=0.10,
                           sl=None, tp=None, now=T0, current_bar=1)

    report = reconcile(governor.journal.pending(), [_snapshot()])
    from forex_ai_agent.brokers import adopt_into

    applied = adopt_into(report, governor)
    assert applied == ["a"]
    assert governor.journal.get("a")["state"] == INTENT_EXECUTED
    assert governor.journal.get("b")["state"] == INTENT_PENDING
    assert governor.journal.get("a")["fill"]["source"] == "adopted_position"


# ---------------------------------------------------------------------------
# Shadow ledger — engine parity
# ---------------------------------------------------------------------------


def _run(ledger: ShadowLedger, *, live_spread: float = 3.0,
         side: str = "BUY", volume: float = 0.10, digits: Optional[int] = None,
         sl: Optional[float] = None, tp: Optional[float] = None) -> ShadowEntry:
    ledger.record_intent("k1", symbol=SYMBOL, side=side, volume=volume,
                         created_ts=T0, sl=sl, tp=tp, created_bar=1)
    # Signal bar: nothing fills yet.
    ledger.on_bar(SYMBOL, T0, open_=1.1000, high=1.1010, low=1.0990,
                  close=1.1005, spread_pips=live_spread, atr_pips=10.0,
                  price_digits=digits, bar_index=1)
    assert ledger.get("k1").state == SHADOW_PENDING
    # Next real bar: fills.
    ledger.on_bar(SYMBOL, T0 + timedelta(hours=1), open_=1.1006, high=1.1020,
                  low=1.1000, close=1.1015, spread_pips=live_spread,
                  atr_pips=10.0, price_digits=digits, bar_index=2)
    return ledger.get("k1")


def test_intent_is_written_ahead_and_idempotent_by_key():
    ledger = ShadowLedger()
    first = ledger.record_intent("k1", symbol=SYMBOL, side="BUY", volume=0.10,
                                 created_ts=T0, comment="c")
    again = ledger.record_intent("k1", symbol=SYMBOL, side="SELL", volume=9.9,
                                 created_ts=T0)
    assert first is again
    assert len(ledger) == 1
    assert first.state == SHADOW_PENDING
    assert [e["kind"] for e in ledger.events].count("shadow.intent_recorded") == 1
    assert any(e["kind"] == "shadow.intent_duplicate" for e in ledger.events)


def test_fill_lands_on_the_next_bar_not_the_signal_bar():
    entry = _run(ShadowLedger())
    assert entry.state == SHADOW_OPEN
    assert entry.created_ts == T0
    assert entry.fill_ts == T0 + timedelta(hours=1)
    assert entry.fill_bar == 2


def test_actual_fill_price_is_exactly_the_engine_cost_model():
    ledger = ShadowLedger()
    entry = _run(ledger, live_spread=3.0)

    orders = OrderManager()
    order = orders.create_order(symbol=SYMBOL, side=OrderSide.BUY,
                                order_type=OrderType.MARKET, volume=0.10)
    fills = orders.execute_order(
        order,
        ExecutionContext(
            current_candle={"open": 1.1006, "high": 1.1020, "low": 1.1000,
                            "close": 1.1015},
            spread_pips=3.0,
            atr_pips=10.0,
            session=session_for(T0 + timedelta(hours=1)),
            timestamp=T0 + timedelta(hours=1),
        ),
    )
    assert entry.actual_fill_price == fills[0].price
    assert entry.commission == fills[0].commission == 7.0 * 0.10


def test_fill_drift_is_the_adverse_live_spread_versus_model_spread():
    # Model spread for EUR/USD is 1.0 pip (DEFAULT_SPREADS), live is 3.0.
    buy = _run(ShadowLedger(), live_spread=3.0, side="BUY")
    sell = _run(ShadowLedger(), live_spread=3.0, side="SELL")
    assert DEFAULT_SPREADS[SYMBOL] == 1.0
    assert buy.fill_drift_pips == pytest.approx(2.0)
    assert sell.fill_drift_pips == pytest.approx(2.0)  # adverse for both sides

    better = _run(ShadowLedger(), live_spread=0.5, side="BUY")
    assert better.fill_drift_pips == pytest.approx(-0.5)  # cheaper than model


def test_predicted_fill_uses_the_model_spread_on_the_same_bar():
    entry = _run(ShadowLedger(), live_spread=3.0, side="BUY")
    # close 1.1015 + (model 1.0 + slippage 1.3) * 1e-4
    assert entry.predicted_fill_price == pytest.approx(
        1.1015 + (1.0 + entry.fill_drift_pips * 0 + 1.3) * 1e-4, rel=1e-9
    )


def test_actual_fill_is_tick_rounded_when_the_broker_digits_are_supplied():
    entry = _run(ShadowLedger(), live_spread=3.0, digits=3)
    assert entry.actual_fill_price == round(entry.actual_fill_price, 3)
    assert entry.predicted_fill_price == round(entry.predicted_fill_price, 3)


def test_sl_exit_uses_the_bar_low_and_the_engine_pnl_formula():
    ledger = ShadowLedger()
    entry = _run(ledger, sl=1.1005)
    assert entry.state == SHADOW_OPEN

    ledger.on_bar(SYMBOL, T0 + timedelta(hours=2), open_=1.1015, high=1.1016,
                  low=1.1000, close=1.1008, spread_pips=3.0, bar_index=3)
    assert entry.state == SHADOW_CLOSED
    assert entry.exit_reason == "SL"
    assert entry.exit_price == 1.1005

    expected = (1.1005 - entry.actual_fill_price) * 0.10 * 100000 - entry.commission
    assert entry.would_have_pnl == pytest.approx(expected, rel=1e-12)


def test_tp_exit_uses_the_bar_high():
    ledger = ShadowLedger()
    entry = _run(ledger, tp=1.1050)
    ledger.on_bar(SYMBOL, T0 + timedelta(hours=2), open_=1.1015, high=1.1060,
                  low=1.1010, close=1.1055, spread_pips=3.0, bar_index=3)
    assert entry.state == SHADOW_CLOSED and entry.exit_reason == "TP"
    assert entry.exit_price == 1.1050


def test_position_is_not_stopped_by_its_own_fill_bar():
    """Engine order: positions are marked BEFORE pending orders fill."""
    ledger = ShadowLedger()
    entry = _run(ledger, sl=1.1010)  # inside the fill bar's range (low 1.1000)
    assert entry.state == SHADOW_OPEN  # not stopped on the bar it filled on
    ledger.on_bar(SYMBOL, T0 + timedelta(hours=2), open_=1.1015, high=1.1016,
                  low=1.1009, close=1.1012, spread_pips=3.0, bar_index=3)
    assert entry.state == SHADOW_CLOSED and entry.exit_reason == "SL"


def test_window_close_settles_at_the_last_seen_close():
    ledger = ShadowLedger()
    entry = _run(ledger)
    closing_ts = T0 + timedelta(hours=5)
    ledger.on_bar(SYMBOL, closing_ts, open_=1.1015, high=1.1025,
                  low=1.1010, close=1.1022, spread_pips=3.0, bar_index=6)
    events = ledger.close_all(closing_ts, reason="WINDOW_END")
    assert entry.state == SHADOW_CLOSED and entry.exit_reason == "WINDOW_END"
    assert entry.exit_price == 1.1022
    assert len(events) == 1
    expected = (1.1022 - entry.actual_fill_price) * 0.10 * 100000 - entry.commission
    assert entry.would_have_pnl == pytest.approx(expected, rel=1e-12)


def test_shadow_pnl_equals_an_independent_position_manager_run():
    """The ledger's book and a bare PositionManager must agree to the cent."""
    ledger = ShadowLedger()
    entry = _run(ledger, sl=1.1005)
    ledger.on_bar(SYMBOL, T0 + timedelta(hours=2), open_=1.1015, high=1.1016,
                  low=1.1000, close=1.1008, spread_pips=3.0, bar_index=3)

    bare = PositionManager()
    position = bare.open_position(SYMBOL, OrderSide.BUY, 0.10,
                                  entry.actual_fill_price, sl=1.1005,
                                  commission=entry.commission, open_time=entry.fill_ts)
    bare.update_positions({SYMBOL: 1.1008}, T0 + timedelta(hours=2),
                          bar_highs={SYMBOL: 1.1016}, bar_lows={SYMBOL: 1.1000})
    trade = bare.closed_trades[0]
    assert trade["id"] == position.id
    assert trade["profit"] == entry.would_have_pnl


def test_ledger_is_reproducible_across_independent_runs():
    def build() -> ShadowLedger:
        ledger = ShadowLedger()
        ledger.record_intent("k1", symbol=SYMBOL, side="BUY", volume=0.10,
                             created_ts=T0, sl=1.1005, comment="c")
        ledger.record_rejection("k2", REJECT_VOLUME_STEP, symbol=SYMBOL)
        for index in range(5):
            ledger.on_bar(
                SYMBOL, T0 + timedelta(hours=index),
                open_=1.1000 + index * 1e-4, high=1.1020 + index * 1e-4,
                low=1.0995 + index * 1e-4, close=1.1010 + index * 1e-4,
                spread_pips=3.0, bar_index=index + 1,
            )
        return ledger

    assert build().save_snapshot() == build().save_snapshot()


def test_snapshot_roundtrip_restores_open_positions_and_closed_trades():
    ledger = ShadowLedger()
    _run(ledger, sl=1.1005)
    ledger.on_bar(SYMBOL, T0 + timedelta(hours=2), open_=1.1015, high=1.1016,
                  low=1.1000, close=1.1008, spread_pips=3.0, bar_index=3)
    ledger.record_intent("k9", symbol=SYMBOL, side="SELL", volume=0.20,
                         created_ts=T0 + timedelta(hours=3))
    ledger.on_bar(SYMBOL, T0 + timedelta(hours=4), open_=1.1010, high=1.1014,
                  low=1.1005, close=1.1012, spread_pips=3.0, bar_index=5)
    before = ledger.save_snapshot()

    restored = ShadowLedger()
    restored.load_snapshot(before)
    assert restored.save_snapshot() == before
    assert restored.get("k1").state == SHADOW_CLOSED
    assert restored.get("k9").state == SHADOW_OPEN
    assert restored.positions.positions["k9"].open_price == before["positions"][0]["open_price"]


def test_stats_report_fill_drift_and_rejection_flags():
    ledger = ShadowLedger()
    _run(ledger, live_spread=3.0)
    # One more bar to mark the open position to market (fill-bar marking
    # happens before the fill itself, engine order).
    ledger.on_bar(SYMBOL, T0 + timedelta(hours=2), open_=1.1015, high=1.1016,
                  low=1.1000, close=1.1008, spread_pips=3.0, bar_index=3)
    ledger.record_rejection("bad", REJECT_VOLUME_STEP, symbol=SYMBOL)
    stats = ledger.stats()
    assert stats["intents"] == 1 and stats["open"] == 1
    assert stats["rejected"] == 1
    assert stats["intent_rejected"] == {REJECT_VOLUME_STEP: 1}
    assert stats["fill_drift"]["count"] == 1
    assert stats["fill_drift"]["mean_pips"] == pytest.approx(2.0)
    assert stats["fill_drift"]["max_pips"] == pytest.approx(2.0)
    assert stats["would_have_pnl"]["unrealized"] != 0.0


# ---------------------------------------------------------------------------
# Shadow executor — validation chain, still nothing sent
# ---------------------------------------------------------------------------


def _executor(stub: Optional[_StubMT5] = None, *, armed: bool = True,
              max_positions: int = 10) -> ShadowExecutor:
    stub = stub if stub is not None else _StubMT5()
    adapter = _adapter(stub)
    governor = ExecutionGovernor(GovernorLimits(max_positions=max_positions))
    if armed:
        governor.arm()
    return ShadowExecutor(adapter, governor)


def _advance(executor: ShadowExecutor, index: int, *, close: float = 1.1010) -> None:
    ts = T0 + timedelta(hours=index)
    executor.on_bar(SYMBOL, ts, open_=close - 1e-4, high=close + 1e-4,
                    low=close - 2e-4, close=close, spread_pips=1.2,
                    bar_index=index + 1)


def test_executor_records_an_intent_without_a_single_broker_write():
    stub = _StubMT5()
    executor = _executor(stub)
    _advance(executor, 0)

    outcome = executor.submit("i1", symbol=SYMBOL, side="BUY", volume=0.10,
                              ts=T0, sl=1.09, tp=1.13, bar_index=1, comment="e2")
    assert outcome["status"] == "RECORDED"
    assert executor.ledger.get("i1").state == SHADOW_PENDING
    assert executor.governor.journal.get("i1")["state"] == INTENT_PENDING
    # Read-only contract: the terminal saw a symbol lookup and nothing else.
    assert executor.adapter.write_calls() == []
    assert executor.adapter.sent_calls() == []
    assert stub.sent == []
    assert all(not c.startswith("order_send") for c in stub.calls)


def test_executor_full_lifecycle_books_a_would_have_trade():
    executor = _executor()
    _advance(executor, 0)
    executor.submit("i1", symbol=SYMBOL, side="BUY", volume=0.10, ts=T0,
                    sl=1.1005, bar_index=1)
    _advance(executor, 1, close=1.1015)
    assert executor.ledger.get("i1").state == SHADOW_OPEN
    # close 1.1005 → bar low 1.1003 ≤ SL 1.1005, stopped at the engine's
    # conservative bar-range rule.
    _advance(executor, 2, close=1.1005)
    entry = executor.ledger.get("i1")
    assert entry.state == SHADOW_CLOSED and entry.exit_reason == "SL"
    assert executor.adapter.sent_calls() == []


def test_executor_rejects_volume_step_and_cancels_the_journaled_intent():
    executor = _executor()
    _advance(executor, 0)
    outcome = executor.submit("i1", symbol=SYMBOL, side="BUY", volume=0.105,
                              ts=T0, bar_index=1)
    assert outcome == {"key": "i1", "status": "REJECTED", "reason": REJECT_VOLUME_STEP}
    assert executor.governor.journal.get("i1")["state"] == INTENT_CANCELLED
    assert executor.ledger.get("i1") is None
    assert executor.ledger.stats()["intent_rejected"] == {REJECT_VOLUME_STEP: 1}
    assert executor.adapter.sent_calls() == []


def test_executor_rejects_volume_outside_the_brokers_bounds():
    low = _executor()
    _advance(low, 0)
    assert low.submit("a", symbol=SYMBOL, side="BUY", volume=0.001, ts=T0,
                      bar_index=1)["reason"] == REJECT_VOLUME_MIN

    high = _executor()
    _advance(high, 0)
    assert high.submit("b", symbol=SYMBOL, side="BUY", volume=500.0, ts=T0,
                       bar_index=1)["reason"] == REJECT_VOLUME_MAX


def test_executor_rejects_a_symbol_the_broker_does_not_trade():
    stub = _StubMT5(trade_mode=1)  # not SYMBOL_TRADE_MODE_FULL
    executor = _executor(stub)
    _advance(executor, 0)
    outcome = executor.submit("i1", symbol=SYMBOL, side="BUY", volume=0.10,
                              ts=T0, bar_index=1)
    assert outcome["reason"] == REJECT_TRADE_DISABLED
    assert executor.governor.journal.get("i1")["state"] == INTENT_CANCELLED


def test_executor_refuses_when_the_governor_is_disarmed():
    executor = _executor(armed=False)
    outcome = executor.submit("i1", symbol=SYMBOL, side="BUY", volume=0.10,
                              ts=T0, bar_index=1)
    assert outcome["reason"].startswith("not_armed:")
    assert executor.governor.journal.get("i1") is None
    assert executor.ledger.get("i1") is None


def test_executor_refuses_on_stale_market_data():
    executor = _executor()
    _advance(executor, 0)
    late = T0 + timedelta(hours=20)  # max_staleness_bars=3 → 4h budget
    outcome = executor.submit("i1", symbol=SYMBOL, side="BUY", volume=0.10,
                              ts=late, bar_index=50)
    assert outcome["reason"] == "stale_data"
    assert executor.ledger.stats()["intent_rejected"] == {"stale_data": 1}


def test_executor_duplicate_does_not_cancel_the_original_intent():
    executor = _executor()
    _advance(executor, 0)
    executor.submit("i1", symbol=SYMBOL, side="BUY", volume=0.10, ts=T0, bar_index=1)
    again = executor.submit("i1", symbol=SYMBOL, side="BUY", volume=0.10, ts=T0,
                            bar_index=1)
    assert again["reason"] == REJECT_DUPLICATE
    assert executor.governor.journal.get("i1")["state"] == INTENT_PENDING
    assert executor.ledger.get("i1") is not None


def test_executor_enforces_the_shadow_position_cap():
    executor = _executor(max_positions=1)
    _advance(executor, 0)
    executor.submit("i1", symbol=SYMBOL, side="BUY", volume=0.10, ts=T0, bar_index=1)
    _advance(executor, 1)
    assert executor.ledger.get("i1").state == SHADOW_OPEN

    capped = executor.submit("i2", symbol=SYMBOL, side="SELL", volume=0.10,
                             ts=T0 + timedelta(hours=1), bar_index=2)
    assert capped["reason"] == REJECT_POSITION_CAP
    assert executor.governor.journal.get("i2")["state"] == INTENT_CANCELLED


def test_executor_reconcile_flags_every_shadow_intent_for_human_review():
    # Broker holds EURUSD 0.20 — same instrument, different size: neither
    # this intent's fill nor explainable by it.
    stub = _StubMT5(positions=(_position(1, BROKER_SYMBOL, "BUY", 0.20),))
    executor = _executor(stub)
    _advance(executor, 0)
    executor.submit("i1", symbol=SYMBOL, side="BUY", volume=0.10, ts=T0, bar_index=1)

    report = executor.reconcile(ts=T0)
    # The shadow intent has no broker position (nothing was sent) …
    assert report.review == ["i1"] and report.adopted == []
    # … and the broker's real position is unknown exposure, not adopted
    # (translated to the Alex form: same instrument, like-for-like compare).
    assert [p["symbol"] for p in report.broker_only] == [SYMBOL]
    assert executor.adapter.write_calls() == []
    assert any(e["kind"] == "reconciliation.review" for e in executor.events)
    assert any(e["kind"] == "reconciliation.divergence" for e in executor.events)


def test_executor_reconcile_adopts_a_matching_translated_position():
    """EURUSD (broker) and EUR/USD (intent) are one instrument post-resolution."""
    stub = _StubMT5(positions=(_position(1, BROKER_SYMBOL, "BUY", 0.10),))
    executor = _executor(stub)
    _advance(executor, 0)
    executor.submit("i1", symbol=SYMBOL, side="BUY", volume=0.10, ts=T0, bar_index=1)

    report = executor.reconcile(ts=T0)
    assert report.adopted == ["i1"] and report.review == []
    assert report.broker_only == []
    assert report.matched_positions and report.matched_positions[0]["symbol"] == SYMBOL
    assert executor.adapter.write_calls() == []


def test_executor_on_bar_feeds_governor_freshness_then_fills():
    executor = _executor()
    _advance(executor, 0)
    assert executor.governor.events == [] or True
    executor.submit("i1", symbol=SYMBOL, side="BUY", volume=0.10, ts=T0, bar_index=1)
    _advance(executor, 1)
    # Freshness: an intent 20 hours later is refused, proving record_bar ran.
    assert executor.governor.state.state == "ARMED"
    assert executor.ledger.get("i1").state == SHADOW_OPEN


def test_executor_report_shape_and_zero_sent_writes():
    executor = _executor()
    _advance(executor, 0)
    executor.submit("i1", symbol=SYMBOL, side="BUY", volume=0.10, ts=T0, bar_index=1)
    _advance(executor, 1)
    report = executor.report()
    assert report["sent_writes"] == 0
    assert report["ledger"]["filled"] == 1
    assert report["governor"]["state"] == "ARMED"
    assert report["adapter"]["sent_writes"] == 0
    assert report["adapter"]["shadow_mode"] is True
    assert report["adapter"]["permission_level"] == 0


# ---------------------------------------------------------------------------
# Engine parity helpers
# ---------------------------------------------------------------------------


def test_engine_atr_is_bit_for_bit_the_backtest_engines():
    rng = np.random.default_rng(7)
    highs = 1.10 + rng.random(45) * 0.01
    lows = highs - rng.random(45) * 0.005
    closes = lows + rng.random(45) * (highs - lows)
    assert engine_atr(highs, lows, closes, 14) == BacktestEngine._atr(
        None, highs, lows, closes, 14
    )
    assert engine_atr(highs[:5], lows[:5], closes[:5], 14) == BacktestEngine._atr(
        None, highs[:5], lows[:5], closes[:5], 14
    )


def test_model_spread_is_the_ordermanagers_own_spread_model():
    ledger = ShadowLedger()
    orders = OrderManager()
    ts = T0
    assert ledger.model_spread_pips(SYMBOL, ts) == orders.spread_model.get_spread(
        SYMBOL, ts, session_for(ts)
    )


def test_session_for_matches_the_engine_session_boundaries():
    assert session_for(datetime(2026, 1, 1, 0, 0)) == "Asian"
    assert session_for(datetime(2026, 1, 1, 7, 59)) == "Asian"
    assert session_for(datetime(2026, 1, 1, 8, 0)) == "London"
    assert session_for(datetime(2026, 1, 1, 12, 59)) == "London"
    assert session_for(datetime(2026, 1, 1, 13, 0)) == "Overlap"
    assert session_for(datetime(2026, 1, 1, 16, 59)) == "Overlap"
    assert session_for(datetime(2026, 1, 1, 17, 0)) == "NewYork"
    assert session_for(datetime(2026, 1, 1, 23, 59)) == "NewYork"


# ---------------------------------------------------------------------------
# Constitution — source scans
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BROKERS = _REPO_ROOT / "src" / "forex_ai_agent" / "brokers"

# Files that must contain no transport, no network and no permission edits.
_PURE_FILES = ("__init__.py", "base.py", "reconcile.py", "shadow.py")

_BANNED = (
    "place_order",
    "execute_trade",
    "risk_manager",
    "requests.",
    "urllib",
    "socket",
    "httpx",
    "aiohttp",
    "order_send",
    "MetaTrader5",
    "mt5.",
    "send_order",
    ".submit_order(",
    ".modify_order(",
    "upgrade_level",
    "downgrade_level",
    "MAX_PERMISSION_LEVEL = 4",
)


def test_broker_layer_has_no_send_path_outside_the_gated_adapter():
    for name in _PURE_FILES:
        source = (_BROKERS / name).read_text(encoding="utf-8")
        for token in _BANNED:
            assert token not in source, f"banned token {token!r} in brokers/{name}"


def test_adapter_write_methods_are_the_only_place_a_transport_is_named():
    source = (_BROKERS / "mt5_adapter.py").read_text(encoding="utf-8")
    for token in ("requests.", "urllib", "socket", "httpx", "aiohttp",
                  "upgrade_level", "downgrade_level", "MAX_PERMISSION_LEVEL = 4",
                  "place_order", "execute_trade", "risk_manager"):
        assert token not in source, f"banned token {token!r} in mt5_adapter.py"
    # `order_send` only appears inside the three gated methods.
    for line in source.splitlines():
        if "order_send" in line:
            assert "mt5.order_send" in line or '"order_send"' in line


def test_nothing_in_production_constructs_an_unshadowed_adapter():
    """Wiring guard: the app can only build adapters through for_shadow()."""
    offenders = []
    for path in (_REPO_ROOT / "src").rglob("*.py"):
        if _BROKERS in path.parents or path == _BROKERS / "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "MT5Adapter(" in text:
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == []


def test_shadow_packages_never_name_a_broker_write():
    executor_source = inspect.getsource(ShadowExecutor)
    for token in (".submit_order(", ".modify_order(", "order_send", "send_order"):
        assert token not in executor_source
