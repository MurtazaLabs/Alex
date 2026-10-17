"""Paper Trading Engine - Real-time paper trading with realistic execution."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
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
from forex_ai_agent.backtest.orders import OrderManager as BacktestOrderManager
from forex_ai_agent.backtest.engine import BacktestEngine  # noqa: F401 — session helper parity
from forex_ai_agent.backtest.orders import ExecutionContext
from forex_ai_agent.backtest.execution import SessionSpreadModel, FixedSlippageModel
from forex_ai_agent.core.event_bus import Event, EventBus, InMemoryEventBus
from forex_ai_agent.news import NewsAgent, NewsImpact


class PaperTradingMode(Enum):
    REALISTIC = "REALISTIC"      # Full simulation with slippage, spread, latency
    IDEAL = "IDEAL"              # Fill at exact signal price
    CONSERVATIVE = "CONSERVATIVE" # Worse fills, wider spreads


@dataclass
class PaperTrade:
    """Completed paper trade record."""
    id: str
    symbol: str
    side: OrderSide
    volume: float
    entry_price: float
    exit_price: float
    sl: Optional[float]
    tp: Optional[float]
    profit: float
    commission: float
    swap: float
    open_time: datetime
    close_time: datetime
    hold_time_seconds: float
    reason: str  # SL, TP, MANUAL, SIGNAL_REVERSAL
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion
    signal_id: Optional[str] = None
    strategy_version: Optional[str] = None
    r_multiple: float = 0.0


@dataclass
class PaperPosition:
    """Open paper position with tracking."""
    id: str
    symbol: str
    side: OrderSide
    volume: float
    entry_price: float
    current_price: float
    sl: Optional[float]
    tp: Optional[float]
    trailing_sl: Optional[float] = None
    commission: float = 0.0
    swap: float = 0.0
    swap_days: int = 0   # days of swap already charged (Stage D: once per day)
    unrealized_pnl: float = 0.0
    mae: float = 0.0
    mfe: float = 0.0
    open_time: Optional[datetime] = None
    signal_id: Optional[str] = None
    strategy_version: Optional[str] = None
    max_favorable: float = 0.0
    max_adverse: float = 0.0


class PaperAccount:
    """Paper trading account state."""

    def __init__(
        self,
        initial_balance: float = 100000.0,
        leverage: int = 100,
        margin_rate: float = 0.01,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.used_margin = 0.0
        self.free_margin = initial_balance
        self.leverage = leverage
        self.margin_rate = margin_rate

        # Daily tracking
        self.daily_start_equity = initial_balance
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_loss_pct = 0.02  # 2%

        # Overall stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        self.peak_equity = initial_balance

    def update_equity(self, unrealized_pnl: float):
        self.equity = self.balance + unrealized_pnl

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        dd = self.peak_equity - self.equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd
            self.max_drawdown_pct = (dd / self.peak_equity) * 100

        self.daily_pnl = self.equity - self.daily_start_equity

    def reset_daily(self):
        self.daily_start_equity = self.equity
        self.daily_pnl = 0.0
        self.daily_trades = 0

    def can_trade(self) -> bool:
        if self.daily_pnl <= -self.daily_start_equity * self.max_daily_loss_pct:
            return False
        return True

    def record_trade(self, profit: float):
        self.total_trades += 1
        self.daily_trades += 1
        if profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1


class PaperTradingEngine:
    """
    Real-time paper trading engine.
    
    Converts signals to paper trades, manages positions in real-time,
    simulates realistic execution with slippage/spread/commission,
    and provides performance tracking.
    """

    def __init__(
        self,
        mode: PaperTradingMode = PaperTradingMode.REALISTIC,
        initial_balance: float = 100000.0,
        leverage: int = 100,
        margin_rate: float = 0.01,
        spread_model: SpreadModel = None,
        slippage_model: SlippageModel = None,
        commission_model: CommissionModel = None,
        swap_model: SwapModel = None,
        event_bus: Optional[EventBus] = None,
        news_agent: Optional[NewsAgent] = None,
        max_positions: int = 10,
        max_correlated_exposure_pct: float = 5.0,
    ):
        self.mode = mode
        self.event_bus = event_bus or InMemoryEventBus()
        self.news_agent = news_agent

        # Account
        self.account = PaperAccount(initial_balance, leverage)

        # Execution models (adjusted by mode)
        self._setup_execution_models(spread_model, slippage_model, commission_model, swap_model)

        # Stage D parity: entries fill through the SAME OrderManager the
        # backtest gate uses (spread + slippage + commission identical by
        # construction), driven by the engine's bar-time clock.
        self.clock: Optional[datetime] = None   # last observed market time
        self._order_manager = BacktestOrderManager(
            spread_model=self.spread_model,
            slippage_model=self.slippage_model,
            commission_model=self.commission_model,
            swap_model=self.swap_model,
        )

        # State
        self.positions: Dict[str, PaperPosition] = {}
        self.closed_trades: List[PaperTrade] = []
        self.pending_signals: Dict[str, Dict] = {}
        self.max_positions = max_positions
        self.max_correlated_exposure_pct = max_correlated_exposure_pct

        # Risk management
        self.risk_per_trade_pct = 0.005  # 0.5%
        self.min_risk_reward = 2.5
        self.max_total_exposure_pct = 10.0
        self.daily_loss_limit_pct = 0.02

        # Callbacks
        self.on_position_open: Optional[Callable] = None
        self.on_position_close: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None
        self.on_margin_call: Optional[Callable] = None

        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

    def _setup_execution_models(
        self,
        spread_model: SpreadModel,
        slippage_model: SlippageModel,
        commission_model: CommissionModel,
        swap_model: SwapModel,
    ):
        if self.mode == PaperTradingMode.IDEAL:
            self.spread_model = FixedSpreadModel({s: 0.0 for s in DEFAULT_SPREADS})
            self.slippage_model = FixedSlippageModel({s: 0.0 for s in DEFAULT_SPREADS})
            self.commission_model = PerLotCommissionModel(0.0)
            self.swap_model = FixedSwapModel({}, {})
        elif self.mode == PaperTradingMode.CONSERVATIVE:
            conservative_spreads = {k: v * 2 for k, v in DEFAULT_SPREADS.items()}
            self.spread_model = spread_model or FixedSpreadModel(conservative_spreads)
            self.slippage_model = slippage_model or VolatilitySizeSlippageModel()
            self.commission_model = commission_model or PerLotCommissionModel(14.0)  # double
            self.swap_model = swap_model or DEFAULT_SWAP
        else:  # REALISTIC
            self.spread_model = spread_model or FixedSpreadModel(DEFAULT_SPREADS)
            self.slippage_model = slippage_model or VolatilitySizeSlippageModel()
            self.commission_model = commission_model or DEFAULT_COMMISSION
            self.swap_model = swap_model or DEFAULT_SWAP

    async def start(self):
        """Start the paper trading engine."""
        if self._running:
            return
        self._running = True
        self.account.daily_start_equity = self.account.equity

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._position_monitor_loop()),
            asyncio.create_task(self._risk_monitor_loop()),
            asyncio.create_task(self._daily_reset_loop()),
        ]

        await self._emit_event("paper_engine.started", {
            "mode": self.mode.value,
            "initial_balance": self.account.initial_balance,
        })

    async def stop(self):
        """Stop the paper trading engine."""
        self._running = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Close all positions at market
        await self.close_all_positions("ENGINE_STOP")

        await self._emit_event("paper_engine.stopped", {
            "final_equity": self.account.equity,
            "total_trades": len(self.closed_trades),
        })

    # Signal processing

    async def process_signal(self, signal: Dict) -> Optional[str]:
        """Process a trading signal and create paper position if valid."""
        if not self.account.can_trade():
            await self._emit_event("signal.rejected", {"signal": signal, "reason": "daily_loss_limit"})
            return None

        if len(self.positions) >= self.max_positions:
            await self._emit_event("signal.rejected", {"signal": signal, "reason": "max_positions"})
            return None

        # Validate signal
        if not self._validate_signal(signal):
            return None

        # Check correlation limits
        if not self._check_correlation(signal):
            await self._emit_event("signal.rejected", {"signal": signal, "reason": "correlation_limit"})
            return None

        # Create position
        position_id = await self._open_position(signal)
        return position_id

    def _validate_signal(self, signal: Dict) -> bool:
        required = ["symbol", "direction", "entry_price", "sl", "tp", "volume"]
        return all(k in signal for k in required)

    def _check_correlation(self, signal: Dict) -> bool:
        """Check if adding this position would exceed correlation limits.
        
        Uses risk-based exposure (position risk in USD) rather than notional exposure.
        """
        symbol = signal["symbol"]
        base_currency = symbol.split("/")[0]

        # Calculate risk-based exposure for existing positions
        correlated_risk = 0.0
        for pos in self.positions.values():
            if pos.symbol.split("/")[0] == base_currency:
                # Risk = distance to SL * volume * pip_value
                if pos.sl:
                    sl_distance_pips = abs(pos.entry_price - pos.sl) * 10000
                    pip_value = 10.0 if "JPY" not in pos.symbol else 1000.0
                    risk_usd = sl_distance_pips * pos.volume * pip_value
                    correlated_risk += risk_usd

        # Calculate risk for new signal
        if "sl" in signal:
            sl_distance_pips = abs(signal["entry_price"] - signal["sl"]) * 10000
            pip_value = 10.0 if "JPY" not in signal["symbol"] else 1000.0
            new_risk_usd = sl_distance_pips * signal["volume"] * pip_value
        else:
            # Conservative estimate if no SL provided
            new_risk_usd = signal["volume"] * 100000 * 0.01  # 1% of notional

        total_risk = correlated_risk + new_risk_usd
        total_equity = self.account.equity

        if total_equity > 0:
            pct = (total_risk / total_equity) * 100
            return pct <= self.max_correlated_exposure_pct
        return True

    async def _open_position(self, signal: Dict) -> str:
        """Open a new paper position with a PARITY fill.

        Stage D: the fill goes through the same OrderManager._execute_market
        the backtest gate uses — entry price carries spread + slippage
        (±(spread+slippage)·1e-4 around mid), commission comes from the same
        model, and the open time is the observed bar time (lifecycle), not
        wall-clock.
        """
        position_id = str(uuid4())[:8]
        symbol = signal["symbol"]
        side = OrderSide.BUY if signal["direction"] == "LONG" else OrderSide.SELL
        ts = self.clock or datetime.utcnow()

        order = self._order_manager.create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            volume=signal["volume"],
            sl=signal.get("sl"),
            tp=signal.get("tp"),
            comment=str(signal.get("id") or ""),
        )
        session = self._get_session(ts)
        context = ExecutionContext(
            current_candle={
                "open": signal["entry_price"],
                "high": signal["entry_price"],
                "low": signal["entry_price"],
                "close": signal["entry_price"],
            },
            spread_pips=self.spread_model.get_spread(symbol, ts, session),
            atr_pips=float(signal.get("atr_pips") or 10.0),
            session=session,
            timestamp=ts,
        )
        fills = self._order_manager.execute_order(order, context)
        fill = fills[0] if fills else None
        entry_price = fill.price if fill is not None else signal["entry_price"]
        commission = fill.commission if fill is not None else 0.0

        position = PaperPosition(
            id=position_id,
            symbol=symbol,
            side=side,
            volume=signal["volume"],
            entry_price=entry_price,
            current_price=entry_price,
            sl=signal.get("sl"),
            tp=signal.get("tp"),
            commission=commission,
            signal_id=signal.get("id"),
            strategy_version=signal.get("strategy_version"),
            open_time=ts,
        )

        self.positions[position_id] = position
        self.account.used_margin += position.volume * position.entry_price * 100000 * self.account.margin_rate / self.account.leverage
        self.account.free_margin = self.account.equity - self.account.used_margin

        await self._emit_event("position.opened", {
            "position_id": position_id,
            "symbol": signal["symbol"],
            "direction": signal["direction"],
            "volume": signal["volume"],
            "entry_price": signal["entry_price"],
            "sl": signal.get("sl"),
            "tp": signal.get("tp"),
        })

        if self.on_position_open:
            await self.on_position_open(position)

        return position_id

    # Position management

    async def close_position(
        self,
        position_id: str,
        price: float,
        reason: str = "MANUAL",
        timestamp: Optional[datetime] = None,
    ) -> Optional[PaperTrade]:
        """Close a paper position with backtest-parity accounting.

        Stage D: profit, commission and swap follow the backtest's
        ``PositionManager.close_position`` math exactly — mid-to-SL/TP price
        move, fill-time commission only (no closing commission), accrued
        swap credited at close — and lifecycle fields use the observed bar
        time, never wall-clock.
        """
        if position_id not in self.positions:
            return None

        position = self.positions.pop(position_id)
        ts = timestamp or self.clock or datetime.utcnow()

        # Profit: raw price move × volume (same as backtest).
        if position.side == OrderSide.BUY:
            profit = (price - position.entry_price) * position.volume * 100000
        else:
            profit = (position.entry_price - price) * position.volume * 100000

        # Parity: the backtest charges commission at FILL TIME only (SL/TP
        # closes pass commission=0), so the paper engine does the same.
        total_commission = position.commission
        net_profit = profit - total_commission + position.swap

        # R-multiple (gross, same as before)
        risk_per_unit = abs(position.entry_price - position.sl) if position.sl else 0
        r_multiple = profit / (risk_per_unit * position.volume * 100000) if risk_per_unit > 0 else 0

        open_time = position.open_time or ts
        trade = PaperTrade(
            id=str(uuid4())[:8],
            symbol=position.symbol,
            side=position.side,
            volume=position.volume,
            entry_price=position.entry_price,
            exit_price=price,
            sl=position.sl,
            tp=position.tp,
            profit=net_profit,
            commission=total_commission,
            swap=position.swap,
            open_time=open_time,
            close_time=ts,
            hold_time_seconds=max(0.0, (ts - open_time).total_seconds()),
            reason=reason,
            mae=position.max_adverse,
            mfe=position.max_favorable,
            signal_id=position.signal_id,
            strategy_version=position.strategy_version,
            r_multiple=r_multiple,
        )

        self.closed_trades.append(trade)
        self.account.balance += net_profit
        self.account.record_trade(net_profit)

        # Release margin
        self.account.used_margin -= position.volume * position.entry_price * 100000 * self.account.margin_rate / self.account.leverage
        self.account.free_margin = self.account.equity - self.account.used_margin

        await self._emit_event("position.closed", {
            "trade": trade.__dict__,
        })

        if self.on_position_close:
            await self.on_position_close(trade)

        return trade

    async def close_all_positions(self, reason: str = "MANUAL"):
        """Close all open positions."""
        for position_id in list(self.positions.keys()):
            position = self.positions[position_id]
            await self.close_position(position_id, position.current_price, reason)

    async def modify_position(
        self,
        position_id: str,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        trailing_sl: Optional[float] = None,
    ) -> bool:
        """Modify position SL/TP/trailing stop."""
        if position_id not in self.positions:
            return False

        position = self.positions[position_id]
        if sl is not None:
            position.sl = sl
        if tp is not None:
            position.tp = tp
        if trailing_sl is not None:
            position.trailing_sl = trailing_sl

        await self._emit_event("position.modified", {
            "position_id": position_id,
            "sl": sl,
            "tp": tp,
            "trailing_sl": trailing_sl,
        })
        return True

    # Market data processing

    def _get_session(self, ts: Optional[datetime] = None) -> str:
        """Session name for a timestamp — parity with BacktestEngine._get_session."""
        t = ts or self.clock
        if t is None:
            return "Unknown"
        hour = t.hour
        if 0 <= hour < 8:
            return "Asian"
        elif 8 <= hour < 13:
            return "London"
        elif 13 <= hour < 17:
            return "Overlap"
        return "NewYork"

    async def on_market_data(self, symbol: str, bid: float, ask: float, timestamp: datetime):
        """Process market data update."""
        # Stage D: keep the engine's clock on observed market time so fills,
        # lifecycle fields and swap accrual all use bar time (never wall-clock).
        self.clock = timestamp
        mid = (bid + ask) / 2
        spread = (ask - bid) * 10000  # pips

        # Update positions
        for pos_id in list(self.positions.keys()):
            position = self.positions.get(pos_id)
            if position is None:
                continue
            if position.symbol != symbol:
                continue

            position.current_price = mid

            # Calculate unrealized P&L
            if position.side == OrderSide.BUY:
                position.unrealized_pnl = (mid - position.entry_price) * position.volume * 100000
                adverse = (position.entry_price - mid) * position.volume * 100000
                favorable = (mid - position.entry_price) * position.volume * 100000
            else:
                position.unrealized_pnl = (position.entry_price - mid) * position.volume * 100000
                adverse = (mid - position.entry_price) * position.volume * 100000
                favorable = (position.entry_price - mid) * position.volume * 100000

            # Track MAE/MFE
            position.max_adverse = max(position.max_adverse, max(0, adverse))
            position.max_favorable = max(position.max_favorable, max(0, favorable))

            # Stage D: accrue swap for each newly completed day held — the
            # same incremental scheme as the backtest's calculate_swaps.
            if position.open_time is not None:
                days_held = (timestamp - position.open_time).days
                if days_held > position.swap_days:
                    new_swap = self.swap_model.calculate_swap(
                        position.symbol,
                        position.side,
                        position.volume,
                        mid,
                        days_held - position.swap_days,
                    )
                    position.swap += new_swap
                    position.swap_days = days_held

            # Check SL
            if position.sl:
                sl_hit = False
                if position.side == OrderSide.BUY and mid <= position.sl:
                    sl_hit = True
                elif position.side == OrderSide.SELL and mid >= position.sl:
                    sl_hit = True

                if sl_hit:
                    await self.close_position(position.id, position.sl, "SL")
                    continue

            # Check TP
            if position.tp:
                tp_hit = False
                if position.side == OrderSide.BUY and mid >= position.tp:
                    tp_hit = True
                elif position.side == OrderSide.SELL and mid <= position.tp:
                    tp_hit = True

                if tp_hit:
                    await self.close_position(position.id, position.tp, "TP")
                    continue

            # Trailing stop
            if position.trailing_sl:
                if position.side == OrderSide.BUY:
                    new_sl = mid - position.trailing_sl * 0.0001
                    if new_sl > (position.sl or 0):
                        position.sl = new_sl
                else:
                    new_sl = mid + position.trailing_sl * 0.0001
                    if new_sl < (position.sl or float('inf')):
                        position.sl = new_sl

        # Update account equity
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        self.account.update_equity(total_unrealized)

        if self.on_position_update:
            await self.on_position_update(self.positions)

    # Risk monitoring

    async def _position_monitor_loop(self):
        """Background position monitoring."""
        while self._running:
            try:
                await asyncio.sleep(1)
                # Check for stale positions, margin, etc.
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._emit_event("engine.error", {"error": str(e)})

    async def _risk_monitor_loop(self):
        """Background risk monitoring."""
        while self._running:
            try:
                await asyncio.sleep(5)
                await self._check_risk_limits()
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._emit_event("engine.error", {"error": str(e)})

    async def _check_risk_limits(self):
        """Check portfolio risk limits."""
        # Daily loss limit
        if self.account.daily_pnl <= -self.account.daily_start_equity * self.daily_loss_limit_pct:
            await self._emit_event("risk.daily_loss_limit", {
                "daily_pnl": self.account.daily_pnl,
                "limit": self.account.daily_start_equity * self.daily_loss_limit_pct,
            })
            if self.on_margin_call:
                await self.on_margin_call()

        # Margin check
        if self.account.free_margin < 0:
            await self._emit_event("risk.margin_call", {
                "free_margin": self.account.free_margin,
            })
            if self.on_margin_call:
                await self.on_margin_call()

        # Max drawdown
        if self.account.max_drawdown_pct >= 10.0:
            await self._emit_event("risk.max_drawdown", {
                "drawdown_pct": self.account.max_drawdown_pct,
            })

    async def _daily_reset_loop(self):
        """Daily statistics reset."""
        while self._running:
            try:
                now = datetime.utcnow()
                next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                wait_seconds = (next_midnight - now).total_seconds()
                await asyncio.sleep(wait_seconds)
                self.account.reset_daily()
                await self._emit_event("daily.reset", {"new_day_equity": self.account.equity})
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._emit_event("engine.error", {"error": str(e)})

    # Event emission

    async def _emit_event(self, event_type: str, payload: Dict):
        if self.event_bus:
            event = Event(
                event_type=event_type,
                aggregate_id="paper_engine",
                aggregate_type="paper_trading",
                payload=payload,
                source_agent="paper_engine",
            )
            await self.event_bus.publish(event)

    # Reporting

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.closed_trades:
            return {}

        profits = [t.profit for t in self.closed_trades]
        winning = [p for p in profits if p > 0]
        losing = [p for p in profits if p <= 0]

        return {
            "account": {
                "balance": self.account.balance,
                "equity": self.account.equity,
                "used_margin": self.account.used_margin,
                "free_margin": self.account.free_margin,
                "leverage": self.account.leverage,
            },
            "trades": {
                "total": len(self.closed_trades),
                "winning": len(winning),
                "losing": len(losing),
                "win_rate": len(winning) / len(self.closed_trades) if self.closed_trades else 0,
                "total_profit": sum(profits),
                "avg_win": sum(winning) / len(winning) if winning else 0,
                "avg_loss": sum(losing) / len(losing) if losing else 0,
                "profit_factor": sum(winning) / abs(sum(losing)) if losing else 0,
                "avg_r_multiple": sum(t.r_multiple for t in self.closed_trades) / len(self.closed_trades) if self.closed_trades else 0,
            },
            "risk": {
                "max_drawdown": self.account.max_drawdown,
                "max_drawdown_pct": self.account.max_drawdown_pct,
                "daily_pnl": self.account.daily_pnl,
                "daily_pnl_pct": (self.account.daily_pnl / self.account.daily_start_equity) * 100 if self.account.daily_start_equity else 0,
            },
            "open_positions": len(self.positions),
        }

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get recent trade history."""
        return [t.__dict__ for t in self.closed_trades[-limit:]]

    def get_open_positions(self) -> List[Dict]:
        """Get open positions summary."""
        return [
            {
                "id": p.id,
                "symbol": p.symbol,
                "side": p.side.value,
                "volume": p.volume,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "unrealized_pnl": p.unrealized_pnl,
                "sl": p.sl,
                "tp": p.tp,
                "trailing_sl": p.trailing_sl,
                "mae": p.max_adverse,
                "mfe": p.max_favorable,
                "hold_time_hours": (datetime.utcnow() - p.open_time).total_seconds() / 3600,
            }
            for p in self.positions.values()
        ]


# Factory
async def create_paper_engine(
    mode: PaperTradingMode = PaperTradingMode.REALISTIC,
    initial_balance: float = 100000.0,
    event_bus: Optional[EventBus] = None,
    news_agent: Optional[NewsAgent] = None,
) -> PaperTradingEngine:
    """Create and start paper trading engine."""
    engine = PaperTradingEngine(
        mode=mode,
        initial_balance=initial_balance,
        event_bus=event_bus,
        news_agent=news_agent,
    )
    await engine.start()
    return engine