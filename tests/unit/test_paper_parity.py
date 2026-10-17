"""Stage D1 unit tests: paper↔backtest fill parity + lifecycle accuracy.

The Stage D accept line — "paper fills statistically match backtest cost
assumptions; probation records reflect lifecycle-accurate trades" — is pinned
here at the unit level:

- A paper entry fill is byte-identical to the backtest OrderManager's market
  fill for the same inputs (same spread model, slippage model, commission).
- Paper trade lifecycle fields (open/close/hold) come from observed bar time,
  never wall-clock.
- Swap accrues once per completed day held on BOTH engines (the backtest's
  old quadratic overcharge is pinned as fixed), and is credited identically.
- The backtest records real open times on positions (swap/hold lifecycle).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from forex_ai_agent.backtest.engine import BacktestEngine  # noqa: F401 — session helper parity
from forex_ai_agent.backtest.execution import (
    DEFAULT_COMMISSION,
    DEFAULT_SPREADS,
    DEFAULT_SWAP,
    FixedSlippageModel,
    FixedSpreadModel,
    FixedSwapModel,
    OrderSide,
    OrderType,
    PerLotCommissionModel,
)
from forex_ai_agent.backtest.orders import ExecutionContext, OrderManager, PositionManager
from forex_ai_agent.paper.engine import PaperTradingEngine, PaperTradingMode


# ---------------------------------------------------------------------------
# Shared fixtures/models — ONE set of models for both engines.
# ---------------------------------------------------------------------------

SYMBOL = "EUR/USD"
_SPREADS = {SYMBOL: 1.0}
_SLIPPAGE = FixedSlippageModel({SYMBOL: 0.5})
_COMMISSION = PerLotCommissionModel(7.0)
_SWAP = FixedSwapModel({SYMBOL: -0.5}, {SYMBOL: 0.2})
_T0 = datetime(2026, 9, 28, 10, 0, 0)


def _bt_fill(side: OrderSide, mid: float, ts: datetime, atr_pips: float = 10.0):
    """The backtest's exact fill for the given inputs."""
    om = OrderManager(
        spread_model=FixedSpreadModel(_SPREADS),
        slippage_model=_SLIPPAGE,
        commission_model=_COMMISSION,
        swap_model=_SWAP,
    )
    order = om.create_order(symbol=SYMBOL, side=side, order_type=OrderType.MARKET, volume=1.0)
    context = ExecutionContext(
        current_candle={"open": mid, "high": mid, "low": mid, "close": mid},
        spread_pips=FixedSpreadModel(_SPREADS).get_spread(SYMBOL, ts, "London"),
        atr_pips=atr_pips,
        session="London",
        timestamp=ts,
    )
    fills = om.execute_order(order, context)
    assert fills, "backtest order did not fill"
    return fills[0]


def _paper_engine() -> PaperTradingEngine:
    eng = PaperTradingEngine(
        mode=PaperTradingMode.REALISTIC,
        initial_balance=100000.0,
        spread_model=FixedSpreadModel(_SPREADS),
        slippage_model=_SLIPPAGE,
        commission_model=_COMMISSION,
        swap_model=_SWAP,
    )
    return eng


def _signal(side: str, mid: float) -> Dict[str, Any]:
    return {
        "symbol": SYMBOL,
        "direction": side,
        "entry_price": mid,
        "sl": mid - 0.0020 if side == "LONG" else mid + 0.0020,
        "tp": mid + 0.0040 if side == "LONG" else mid - 0.0040,
        "volume": 1.0,
        "id": "sig-1",
        "strategy_version": "deadbeef",
    }


# ---------------------------------------------------------------------------
# 1. Fill parity — same inputs, identical fill.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paper_entry_fill_matches_backtest_fill_exactly():
    eng = _paper_engine()
    ts = _T0
    mid = 1.10000
    eng.clock = ts

    fill = _bt_fill(OrderSide.BUY, mid, ts)
    pid = await eng.process_signal(_signal("LONG", mid))
    assert pid is not None
    pos = eng.positions[pid]

    # Identical price AND commission from the shared model chain.
    assert pos.entry_price == pytest.approx(fill.price, abs=1e-12)
    assert pos.commission == pytest.approx(fill.commission, abs=1e-12)
    # The fill is worse than mid by (spread + slippage) pips for a BUY.
    assert pos.entry_price == pytest.approx(mid + 1.5e-4, abs=1e-12)


@pytest.mark.asyncio
async def test_paper_sell_fill_matches_backtest_fill_exactly():
    eng = _paper_engine()
    ts = _T0
    mid = 1.10000
    eng.clock = ts

    fill = _bt_fill(OrderSide.SELL, mid, ts)
    pid = await eng.process_signal(_signal("SELL", mid))
    pos = eng.positions[pid]
    assert pos.entry_price == pytest.approx(fill.price, abs=1e-12)
    assert pos.commission == pytest.approx(fill.commission, abs=1e-12)
    assert pos.entry_price == pytest.approx(mid - 1.5e-4, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Lifecycle — bar time, not wall-clock.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paper_trade_lifecycle_uses_bar_time_not_wall_clock():
    eng = _paper_engine()
    open_ts = _T0
    eng.clock = open_ts
    pid = await eng.process_signal(_signal("LONG", 1.10000))

    close_ts = open_ts + timedelta(hours=5)
    trade = await eng.close_position(pid, 1.10400, reason="TP", timestamp=close_ts)
    assert trade is not None
    assert trade.open_time == open_ts
    assert trade.close_time == close_ts
    assert trade.hold_time_seconds == pytest.approx(5 * 3600.0)


# ---------------------------------------------------------------------------
# 3. Swap parity — once per completed day, credited at close on both engines.
# ---------------------------------------------------------------------------


def test_backtest_swap_accrues_once_per_day_not_quadratically():
    pm = PositionManager(swap_model=_SWAP)
    pos = pm.open_position(
        symbol=SYMBOL, side=OrderSide.BUY, volume=1.0, price=1.10,
        open_time=_T0,
    )
    day1 = _T0 + timedelta(days=1, hours=1)
    day2 = _T0 + timedelta(days=2, hours=2)
    s1 = pm.calculate_swaps(day1)
    s2 = pm.calculate_swaps(day2)
    # Each new day charged exactly once: swap(1d) then swap(1d) again.
    assert s1 == pytest.approx(_SWAP.calculate_swap(SYMBOL, OrderSide.BUY, 1.0, 1.10, 1))
    assert s2 == pytest.approx(_SWAP.calculate_swap(SYMBOL, OrderSide.BUY, 1.0, 1.10, 1))
    assert pos.swap == pytest.approx(s1 + s2)


def test_backtest_position_records_real_open_time():
    pm = PositionManager(swap_model=_SWAP)
    pos = pm.open_position(symbol=SYMBOL, side=OrderSide.BUY, volume=1.0, price=1.10, open_time=_T0)
    closed = pm.close_position(pos.id, 1.11, timestamp=_T0 + timedelta(hours=3))
    assert closed is not None
    assert closed["open_time"] == _T0
    assert closed["close_time"] == _T0 + timedelta(hours=3)
    assert closed["hold_time_hours"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_paper_swap_matches_backtest_swap_accrual():
    eng = _paper_engine()
    open_ts = _T0
    eng.clock = open_ts
    pid = await eng.process_signal(_signal("LONG", 1.10000))

    # Two full days pass via market data ticks.
    for hours in (25, 49):
        await eng.on_market_data(SYMBOL, 1.1000, 1.1001, open_ts + timedelta(hours=hours))

    pos = eng.positions[pid]
    expected = 2 * _SWAP.calculate_swap(SYMBOL, OrderSide.BUY, 1.0, 1.10, 1)
    assert pos.swap == pytest.approx(expected, rel=1e-9)

    # Closing credits the same swap the backtest would have accrued.
    trade = await eng.close_position(pid, 1.10400, reason="TP",
                                      timestamp=open_ts + timedelta(hours=49, minutes=30))
    assert trade.swap == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. Paper close accounting matches the backtest close math exactly.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paper_close_profit_matches_backtest_close_math():
    eng = _paper_engine()
    ts = _T0
    eng.clock = ts
    pid = await eng.process_signal(_signal("LONG", 1.10000))
    pos_before = eng.positions[pid]
    entry, commission = pos_before.entry_price, pos_before.commission

    exit_price = 1.10400
    trade = await eng.close_position(pid, exit_price, reason="TP", timestamp=ts + timedelta(hours=2))

    # Backtest math: profit = move × volume × 100000 − commission (fill-time
    # only) + swap (0 here — no days held).
    expected = (exit_price - entry) * 1.0 * 100000 - commission
    assert trade.profit == pytest.approx(expected, abs=1e-9)
    assert trade.commission == pytest.approx(commission, abs=1e-12)
