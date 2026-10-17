"""Order and Position Management for Backtesting."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4

from forex_ai_agent.backtest.execution import (
    Order,
    Position,
    OrderType,
    OrderSide,
    OrderStatus,
    SpreadModel,
    SlippageModel,
    CommissionModel,
    SwapModel,
    FixedSpreadModel,
    VolatilitySizeSlippageModel,
    PerLotCommissionModel,
    FixedSwapModel,
    DEFAULT_SPREADS,
    DEFAULT_COMMISSION,
    DEFAULT_SWAP,
)


class FillPolicy(Enum):
    """How to handle fills when price gaps."""
    NEXT_OPEN = "NEXT_OPEN"      # Fill at next candle open
    WORST_CASE = "WORST_CASE"    # Fill at worst price in gap
    REJECT = "REJECT"            # Reject order if gapped


@dataclass
class Fill:
    """Order fill record."""
    order_id: str
    symbol: str
    side: OrderSide
    volume: float
    price: float
    commission: float
    timestamp: datetime
    fill_type: str = "MARKET"


@dataclass
class ExecutionContext:
    """Context for order execution."""
    current_candle: Dict[str, Any]  # {open, high, low, close, time}
    spread_pips: float
    atr_pips: float
    session: str
    timestamp: datetime


class OrderManager:
    """Manages order lifecycle and execution."""

    def __init__(
        self,
        spread_model: SpreadModel = None,
        slippage_model: SlippageModel = None,
        commission_model: CommissionModel = None,
        swap_model: SwapModel = None,
        fill_policy: FillPolicy = FillPolicy.NEXT_OPEN,
    ):
        self.spread_model = spread_model or FixedSpreadModel(DEFAULT_SPREADS)
        self.slippage_model = slippage_model or VolatilitySizeSlippageModel()
        self.commission_model = commission_model or DEFAULT_COMMISSION
        self.swap_model = swap_model or DEFAULT_SWAP
        self.fill_policy = fill_policy

        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.order_history: List[Order] = []

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        volume: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        magic: int = 0,
        comment: str = "",
    ) -> Order:
        """Create a new order."""
        order = Order(
            id=str(uuid4())[:8],
            symbol=symbol,
            side=side,
            type=order_type,
            volume=volume,
            price=price,
            stop_price=stop_price,
            sl=sl,
            tp=tp,
            magic=magic,
            comment=comment,
        )
        self.orders[order.id] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        orders = [o for o in self.orders.values() if o.status == OrderStatus.PENDING]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def execute_order(
        self,
        order: Order,
        context: ExecutionContext,
    ) -> List[Fill]:
        """Execute an order against current market conditions."""
        if order.status != OrderStatus.PENDING:
            return []

        fills = []

        if order.type == OrderType.MARKET:
            fills = self._execute_market(order, context)
        elif order.type == OrderType.LIMIT:
            fills = self._execute_limit(order, context)
        elif order.type == OrderType.STOP:
            fills = self._execute_stop(order, context)

        for fill in fills:
            order.filled_volume += fill.volume
            order.avg_fill_price = (
                (order.avg_fill_price * (order.filled_volume - fill.volume) + fill.price * fill.volume)
                / order.filled_volume
            )
            order.commission += fill.commission
            order.status = OrderStatus.FILLED if order.filled_volume >= order.volume else OrderStatus.PARTIAL
            order.filled_time = fill.timestamp

        return fills

    def _execute_market(self, order: Order, context: ExecutionContext) -> List[Fill]:
        """Execute market order."""
        spread = context.spread_pips
        slippage = self.slippage_model.get_slippage(
            order.symbol, order.side, order.volume,
            spread, context.atr_pips, context.timestamp
        )

        if order.side == OrderSide.BUY:
            fill_price = context.current_candle["close"] + (spread + slippage) * 0.0001
        else:
            fill_price = context.current_candle["close"] - (spread + slippage) * 0.0001

        commission = self.commission_model.calculate_commission(
            order.symbol, order.volume, fill_price, order.side
        )

        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            volume=order.volume,
            price=fill_price,
            commission=commission,
            timestamp=context.timestamp,
            fill_type="MARKET",
        )
        return [fill]

    def _execute_limit(self, order: Order, context: ExecutionContext) -> List[Fill]:
        """Execute limit order."""
        if order.price is None:
            return []

        candle = context.current_candle
        filled = False
        fill_price = order.price

        if order.side == OrderSide.BUY:
            # Buy limit: fill if low <= price
            if candle["low"] <= order.price:
                filled = True
                fill_price = min(order.price, candle["open"])  # conservative
        else:
            # Sell limit: fill if high >= price
            if candle["high"] >= order.price:
                filled = True
                fill_price = max(order.price, candle["open"])

        if not filled:
            return []

        commission = self.commission_model.calculate_commission(
            order.symbol, order.volume, fill_price, order.side
        )

        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            volume=order.volume,
            price=fill_price,
            commission=commission,
            timestamp=context.timestamp,
            fill_type="LIMIT",
        )
        return [fill]

    def _execute_stop(self, order: Order, context: ExecutionContext) -> List[Fill]:
        """Execute stop order (becomes market when triggered)."""
        if order.stop_price is None:
            return []

        candle = context.current_candle
        triggered = False

        if order.side == OrderSide.BUY:
            # Buy stop: trigger if high >= stop_price
            if candle["high"] >= order.stop_price:
                triggered = True
        else:
            # Sell stop: trigger if low <= stop_price
            if candle["low"] <= order.stop_price:
                triggered = True

        if not triggered:
            return []

        # Execute as market with slippage
        spread = context.spread_pips
        slippage = self.slippage_model.get_slippage(
            order.symbol, order.side, order.volume,
            spread, context.atr_pips, context.timestamp
        )

        if order.side == OrderSide.BUY:
            fill_price = max(order.stop_price, candle["open"]) + (spread + slippage) * 0.0001
        else:
            fill_price = min(order.stop_price, candle["open"]) - (spread + slippage) * 0.0001

        commission = self.commission_model.calculate_commission(
            order.symbol, order.volume, fill_price, order.side
        )

        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            volume=order.volume,
            price=fill_price,
            commission=commission,
            timestamp=context.timestamp,
            fill_type="STOP",
        )
        return [fill]


class PositionManager:
    """Manages open positions and calculates P&L."""

    def __init__(
        self,
        swap_model: SwapModel = None,
    ):
        self.swap_model = swap_model or DEFAULT_SWAP
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict] = []

    def open_position(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        price: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        commission: float = 0.0,
        magic: int = 0,
        comment: str = "",
        open_time: Optional[datetime] = None,
    ) -> Position:
        """Open a new position at ``open_time`` (bar time; UTC now if absent)."""
        position = Position(
            id=str(uuid4())[:8],
            symbol=symbol,
            side=side,
            volume=volume,
            open_price=price,
            current_price=price,
            sl=sl,
            tp=tp,
            commission=commission,
            magic=magic,
            comment=comment,
            open_time=open_time or datetime.utcnow(),
        )
        self.positions[position.id] = position
        return position

    def close_position(
        self,
        position_id: str,
        price: float,
        commission: float = 0.0,
        reason: str = "MANUAL",
        timestamp: Optional[datetime] = None,
    ) -> Optional[Dict]:
        """Close a position and record trade."""
        if position_id not in self.positions:
            return None

        position = self.positions.pop(position_id)
        _ts = timestamp or datetime.utcnow()
        _open_ts = position.open_time or _ts  # Stage D: never crash on missing open_time

        # Calculate profit
        if position.side == OrderSide.BUY:
            profit = (price - position.open_price) * position.volume * 100000
        else:
            profit = (position.open_price - price) * position.volume * 100000

        profit -= position.commission + commission
        profit += position.swap

        trade_record = {
            "id": position.id,
            "symbol": position.symbol,
            "side": position.side.value,
            "volume": position.volume,
            "open_price": position.open_price,
            "close_price": price,
            "profit": profit,
            "commission": position.commission + commission,
            "swap": position.swap,
            "open_time": _open_ts,
            "close_time": _ts,
            "hold_time_hours": max(
                0.0,
                (_ts - _open_ts).total_seconds() / 3600,
            ),
            "reason": reason,
            "mae": 0.0,  # Will be calculated during updates
            "mfe": 0.0,
            "magic": position.magic,
            "comment": position.comment,
        }

        self.closed_trades.append(trade_record)
        return trade_record

    def update_positions(
        self,
        prices: Dict[str, float],
        timestamp: datetime,
        atr_values: Optional[Dict[str, float]] = None,
        bar_highs: Optional[Dict[str, float]] = None,
        bar_lows: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """Update all positions with current prices, check SL/TP.

        SL/TP are evaluated against the bar's high/low range (conservative
        worst-case within the bar) rather than the close only.
        """
        events = []
        bar_highs = bar_highs or {}
        bar_lows = bar_lows or {}

        for pos_id, position in list(self.positions.items()):
            current_price = prices.get(position.symbol)
            if current_price is None:
                continue
            bar_high = bar_highs.get(position.symbol, current_price)
            bar_low = bar_lows.get(position.symbol, current_price)
            if current_price is None:
                continue

            position.current_price = current_price

            # Calculate unrealized P&L
            if position.side == OrderSide.BUY:
                position.profit = (current_price - position.open_price) * position.volume * 100000
            else:
                position.profit = (position.open_price - current_price) * position.volume * 100000

            # Check SL (conservative: assume the adverse extreme was hit)
            if position.sl is not None:
                sl_hit = False
                if position.side == OrderSide.BUY and bar_low <= position.sl:
                    sl_hit = True
                elif position.side == OrderSide.SELL and bar_high >= position.sl:
                    sl_hit = True

                if sl_hit:
                    trade = self.close_position(
                        pos_id, position.sl, reason="SL", timestamp=timestamp
                    )
                    if trade:
                        events.append({"type": "SL_HIT", "trade": trade})
                    continue

            # Check TP
            if position.tp is not None:
                tp_hit = False
                if position.side == OrderSide.BUY and bar_high >= position.tp:
                    tp_hit = True
                elif position.side == OrderSide.SELL and bar_low <= position.tp:
                    tp_hit = True

                if tp_hit:
                    trade = self.close_position(
                        pos_id, position.tp, reason="TP", timestamp=timestamp
                    )
                    if trade:
                        events.append({"type": "TP_HIT", "trade": trade})
                    continue

        return events

    def calculate_swaps(self, timestamp: datetime) -> float:
        """Calculate and apply swap for all positions held overnight.

        Stage D lifecycle fix: each newly completed day held is charged
        EXACTLY ONCE (the old code re-charged the full day count every bar,
        so a 3-day hold paid 1+2+3 = 6 days of swap — quadratic overcharge).
        """
        total_swap = 0.0
        for position in self.positions.values():
            days_held = (
                (timestamp - position.open_time).days
                if position.open_time is not None
                else 0
            )
            if days_held > position.swap_days:
                new_days = days_held - position.swap_days
                swap = self.swap_model.calculate_swap(
                    position.symbol,
                    position.side,
                    position.volume,
                    position.current_price,
                    new_days,
                )
                position.swap += swap
                position.swap_days = days_held
                total_swap += swap
        return total_swap

    def get_position(self, position_id: str) -> Optional[Position]:
        return self.positions.get(position_id)

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        positions = list(self.positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    def get_total_exposure(self) -> float:
        """Get total notional exposure."""
        total = 0.0
        for pos in self.positions.values():
            total += pos.volume * pos.current_price * 100000
        return total

    def get_unrealized_pnl(self) -> float:
        return sum(p.profit for p in self.positions.values())


class AccountState:
    """Simulated account state."""

    def __init__(
        self,
        initial_balance: float = 100000.0,
        leverage: int = 100,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.margin_used = 0.0
        self.free_margin = initial_balance
        self.leverage = leverage
        self.daily_pnl = 0.0
        self.max_equity = initial_balance
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0

    def update(self, position_manager: PositionManager):
        """Update equity and margin."""
        unrealized = position_manager.get_unrealized_pnl()
        self.equity = self.balance + unrealized

        if self.equity > self.max_equity:
            self.max_equity = self.equity

        dd = self.max_equity - self.equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd
            self.max_drawdown_pct = (dd / self.max_equity) * 100 if self.max_equity > 0 else 0

    def can_open_position(self, volume: float, price: float, margin_rate: float = 0.01) -> bool:
        """Check if enough free margin."""
        required_margin = volume * price * 100000 * margin_rate / self.leverage
        return self.free_margin >= required_margin