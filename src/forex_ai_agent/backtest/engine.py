"""Core Backtest Engine - Event-driven, candle-by-candle."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
    DEFAULT_SPREADS,
)
from forex_ai_agent.backtest.orders import (
    OrderManager,
    PositionManager,
    AccountState,
    ExecutionContext,
    FillPolicy,
)
from forex_ai_agent.core.event_bus import Event, EventBus, InMemoryEventBus


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    start_date: datetime
    end_date: datetime
    initial_balance: float = 100000.0
    leverage: int = 100
    margin_rate: float = 0.01  # 1% margin
    spread_model: SpreadModel = None
    fill_policy: FillPolicy = FillPolicy.NEXT_OPEN
    commission_model: Any = None
    swap_model: Any = None
    slippage_model: Any = None
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "H1"
    warmup_bars: int = 100


@dataclass
class BarData:
    """Single price bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest result."""
    config: BacktestConfig
    trades: List[Dict]
    equity_curve: List[Tuple[datetime, float]]
    daily_returns: List[Tuple[datetime, float]]
    metrics: Dict[str, float]
    position_history: List[Dict]
    order_history: List[Dict]
    stats: Dict[str, Any]


class DataFeed:
    """Historical data feed for backtesting."""

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self.symbols = list(data.keys())
        self._current_idx: Dict[str, int] = {s: 0 for s in self.symbols}

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> List[BarData]:
        df = self.data.get(symbol)
        if df is None:
            return []
        mask = (df.index >= start) & (df.index <= end)
        df = df[mask]
        return [
            BarData(
                timestamp=idx,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row.get("volume", 0),
            )
            for idx, row in df.iterrows()
        ]

    def get_current_bar(self, symbol: str) -> Optional[BarData]:
        idx = self._current_idx.get(symbol, 0)
        df = self.data.get(symbol)
        if df is None or idx >= len(df):
            return None
        row = df.iloc[idx]
        return BarData(
            timestamp=df.index[idx],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row.get("volume", 0),
        )

    def advance(self) -> bool:
        """Advance all symbols to next bar. Returns True if ALL are exhausted (no more data)."""
        all_done = True
        for symbol in self.symbols:
            idx = self._current_idx.get(symbol, 0)
            df = self.data.get(symbol)
            if df is not None and idx + 1 < len(df):
                self._current_idx[symbol] = idx + 1
                all_done = False
        return all_done

    def reset(self):
        self._current_idx = {s: 0 for s in self.symbols}

    def append_bar(
        self,
        symbol: str,
        timestamp: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
    ) -> None:
        """Append one live bar and position the cursor on it (Stage D2).

        Lets event-driven consumers stream bars into a persistent engine
        instead of pre-loading a full DataFrame; indicators computed by
        ``_calculate_indicators`` then see exactly the same trailing window
        as a batch replay over the same bars.
        """
        df = self.data.get(symbol)
        if df is None:
            df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=pd.DatetimeIndex([], name="timestamp"),
            ).astype(
                {
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "float64",
                }
            )
            self.data[symbol] = df
            if symbol not in self.symbols:
                self.symbols.append(symbol)
            self._current_idx[symbol] = 0
        df.loc[timestamp] = [float(open), float(high), float(low), float(close), float(volume)]
        self._current_idx[symbol] = len(df) - 1

    def get_current_timestamp(self) -> Optional[datetime]:
        timestamps = []
        for symbol in self.symbols:
            bar = self.get_current_bar(symbol)
            if bar:
                timestamps.append(bar.timestamp)
        return min(timestamps) if timestamps else None


class SignalGenerator(ABC):
    """Abstract signal generator for backtesting."""

    @abstractmethod
    def generate_signals(
        self,
        timestamp: datetime,
        data: Dict[str, BarData],
        indicators: Dict[str, Dict[str, float]],
    ) -> List[Dict]:
        """Generate trading signals for current bar."""
        pass


class BacktestEngine:
    """
    Event-driven backtest engine.
    
    Processes candles sequentially, executes orders, manages positions,
    and calculates comprehensive metrics.
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_feed: DataFeed,
        signal_generator: SignalGenerator,
        event_bus: Optional[EventBus] = None,
    ):
        self.config = config
        self.data_feed = data_feed
        self.signal_generator = signal_generator
        self.event_bus = event_bus or InMemoryEventBus()

        # Components
        self.order_manager = OrderManager(
            spread_model=config.spread_model,
            fill_policy=config.fill_policy,
        )
        self.position_manager = PositionManager()
        self.account = AccountState(config.initial_balance, config.leverage)

        # State
        self.current_time: Optional[datetime] = None
        self.current_bars: Dict[str, BarData] = {}
        self.indicators: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[Tuple[datetime, float]] = []
        self.session_stats: Dict[str, Any] = defaultdict(list)
        self._last_bar_date: Optional[Any] = None  # calendar date of last processed bar (daily tracking)

        # Callbacks
        self.on_bar: Optional[Callable] = None
        self.on_signal: Optional[Callable] = None
        self.on_fill: Optional[Callable] = None
        self.on_trade_close: Optional[Callable] = None

    def run(self) -> BacktestResult:
        """Run the complete backtest."""
        print(f"Starting backtest: {self.config.start_date} to {self.config.end_date}")
        print(f"Symbols: {self.config.symbols}, Timeframe: {self.config.timeframe}")

        # Ensure config dates are datetime
        start_date = self.config.start_date
        end_date = self.config.end_date
        if hasattr(start_date, 'to_pydatetime'):
            start_date = start_date.to_pydatetime()
        if hasattr(end_date, 'to_pydatetime'):
            end_date = end_date.to_pydatetime()

        # Warmup period
        self._warmup()

        # Advance to start_date
        self._advance_to_date(start_date)

        # Main loop
        self._last_bar_date = None
        bar_count = 0

        while True:
            timestamp = self.data_feed.get_current_timestamp()
            if timestamp is None:
                break
            if timestamp > end_date:
                break

            # One bar of the loop body, extracted verbatim into _process_bar
            # so event-driven consumers (Stage D2 probation accrual) execute
            # the IDENTICAL per-bar code path as this batch replay.
            if not self._process_bar(timestamp):
                break

            # Advance to next bar
            if self.data_feed.advance():
                break

            bar_count += 1
            if bar_count % 1000 == 0:
                print(f"  Processed {bar_count} bars, equity: ${self.account.equity:,.2f}")

        print(f"Backtest complete: {bar_count} bars processed")
        return self._generate_result()

    def _process_bar(self, timestamp: datetime) -> bool:
        """Process a single bar: indicators → positions → orders → signals.

        Extracted verbatim from the former inline body of ``run`` (Stage D2)
        so the event-driven probation accrual streams bars through the same
        code the batch gate runs — parity by construction. The bar must
        already be positioned in the DataFeed; returns False when no bar data
        was available (the batch loop treats that as end-of-data).
        """
        self.current_time = timestamp
        self.current_bars = {}

        # Get current bars for all symbols
        for symbol in self.config.symbols:
            bar = self.data_feed.get_current_bar(symbol)
            if bar:
                self.current_bars[symbol] = bar

        if not self.current_bars:
            return False

        # Calculate indicators
        self._calculate_indicators()

        # Update positions with current prices; SL/TP evaluated against
        # the bar's high/low range (intra-bar), not just the close.
        prices = {s: b.close for s, b in self.current_bars.items()}
        self.position_manager.update_positions(
            prices,
            timestamp,
            bar_highs={s: b.high for s, b in self.current_bars.items()},
            bar_lows={s: b.low for s, b in self.current_bars.items()},
        )
        # Book realized P&L from any positions closed this bar.
        self.account.balance = self.account.initial_balance + sum(
            t.get("profit", 0.0) for t in self.position_manager.closed_trades
        )
        self.account.free_margin = self.account.balance

        # Process pending orders
        self._process_orders()

        # Generate signals
        signals = self.signal_generator.generate_signals(
            timestamp, self.current_bars, self.indicators
        )
        for signal in signals:
            self._execute_signal(signal)

        # Calculate swaps
        self.position_manager.calculate_swaps(timestamp)

        # Update account
        self.account.update(self.position_manager)

        # Record equity
        self.equity_curve.append((timestamp, self.account.equity))

        # Daily tracking
        last_date = self._last_bar_date
        if last_date is None or timestamp.date() != last_date:
            if last_date is not None:
                prev_equity = self.daily_returns[-1][1] if self.daily_returns else self.account.initial_balance
                daily_ret = (self.account.equity - prev_equity) / prev_equity if prev_equity != 0 else 0
                self.daily_returns.append((timestamp, daily_ret))
            else:
                # First day - record initial equity as baseline
                self.daily_returns.append((timestamp, 0.0))
            self._last_bar_date = timestamp.date()

        # Callbacks
        if self.on_bar:
            self.on_bar(timestamp, self.current_bars, self.account)

        return True

    def _advance_to_date(self, target_date: datetime):
        """Advance data feed to target date."""
        while True:
            timestamp = self.data_feed.get_current_timestamp()
            if timestamp is None or timestamp >= target_date:
                break
            if self.data_feed.advance():
                break

    def _warmup(self):
        """Run warmup bars to initialize indicators."""
        for _ in range(self.config.warmup_bars):
            if self.data_feed.advance():
                break
            for symbol in self.config.symbols:
                bar = self.data_feed.get_current_bar(symbol)
                if bar:
                    self.current_bars[symbol] = bar
            self._calculate_indicators()

        # Reset to start date
        self.data_feed.reset()
        self._advance_to_date(self.config.start_date)

    def _calculate_indicators(self):
        """Calculate technical indicators for current bars."""
        for symbol, bar in self.current_bars.items():
            # Get historical data for indicator calculation
            df = self.data_feed.data.get(symbol)
            if df is None:
                continue

            idx = self.data_feed._current_idx.get(symbol, 0)
            if idx < 50:
                continue

            # Get lookback window
            lookback = df.iloc[max(0, idx-200):idx+1]
            closes = lookback["close"].values
            highs = lookback["high"].values
            lows = lookback["low"].values

            # Simple indicators
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
            # Bollinger Bands: FIXED 20-period SMA / 2.0 sigma (population),
            # matching analysis.indicators.volatility.bollinger_bands — the
            # human-confirmed mean-reversion parameters (2026-10-16).
            bb_window = closes[-20:] if len(closes) >= 20 else closes
            bb_middle = float(np.mean(bb_window))
            bb_std = float(np.std(bb_window))
            self.indicators[symbol] = {
                "sma_20": sma_20,
                "sma_50": np.mean(closes[-50:]) if len(closes) >= 50 else closes[-1],
                "ema_20": self._ema(closes, 20),
                "atr_14": self._atr(highs, lows, closes, 14),
                "rsi_14": self._rsi(closes, 14),
                "bb_middle": bb_middle,
                "bb_upper": bb_middle + 2.0 * bb_std,
                "bb_lower": bb_middle - 2.0 * bb_std,
            }

    def _ema(self, data: np.ndarray, period: int) -> float:
        if len(data) < period:
            return data[-1]
        alpha = 2 / (period + 1)
        ema = data[-period]
        for v in data[-period+1:]:
            ema = alpha * v + (1 - alpha) * ema
        return ema

    def _atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
        if len(highs) < period + 1:
            return highs[-1] - lows[-1]
        tr = np.maximum(highs[1:] - lows[1:],
                       np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        return np.mean(tr[-period:])

    def _rsi(self, closes: np.ndarray, period: int) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _process_orders(self):
        """Process all pending orders against current market."""
        for symbol, bar in self.current_bars.items():
            context = ExecutionContext(
                current_candle={"open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close},
                spread_pips=self.order_manager.spread_model.get_spread(symbol, self.current_time, self._get_session()),
                atr_pips=self.indicators.get(symbol, {}).get("atr_14", 10.0),
                session=self._get_session(),
                timestamp=self.current_time,
            )

            pending = self.order_manager.get_pending_orders(symbol)
            for order in pending:
                fills = self.order_manager.execute_order(order, context)
                for fill in fills:
                    if self.on_fill:
                        self.on_fill(fill)

                    # Open position on fill
                    if order.type in (OrderType.MARKET, OrderType.STOP):
                        self.position_manager.open_position(
                            symbol=fill.symbol,
                            side=fill.side,
                            volume=fill.volume,
                            price=fill.price,
                            sl=order.sl,
                            tp=order.tp,
                            commission=fill.commission,
                            magic=order.magic,
                            comment=order.comment,
                            open_time=self.current_time,  # Stage D: bar-time lifecycle
                        )

    def _execute_signal(self, signal: Dict):
        """Convert signal to orders."""
        if self.on_signal:
            self.on_signal(signal)

        symbol = signal.get("symbol")
        direction = signal.get("direction")
        bar = self.current_bars.get(symbol)
        entry_price = signal.get("entry_price", bar.close if bar else 0)
        sl = signal.get("sl")
        tp = signal.get("tp")
        volume = signal.get("volume", 1.0)

        if not symbol or direction is None:
            return

        side = OrderSide.BUY if direction == "LONG" else OrderSide.SELL

        order = self.order_manager.create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            volume=volume,
            sl=sl,
            tp=tp,
            magic=signal.get("magic", 0),
            comment=signal.get("comment", ""),
        )

    def _get_session(self) -> str:
        """Determine current trading session."""
        if self.current_time is None:
            return "Unknown"
        hour = self.current_time.hour
        if 0 <= hour < 8:
            return "Asian"
        elif 8 <= hour < 13:
            return "London"
        elif 13 <= hour < 17:
            return "Overlap"
        else:
            return "NewYork"

    def _generate_result(self) -> BacktestResult:
        """Generate final backtest result."""
        from forex_ai_agent.backtest.metrics import calculate_metrics

        metrics = calculate_metrics(
            self.equity_curve,
            self.position_manager.closed_trades,
            self.account.initial_balance,
        )

        return BacktestResult(
            config=self.config,
            trades=self.position_manager.closed_trades,
            equity_curve=self.equity_curve,
            daily_returns=self.daily_returns,
            metrics=metrics,
            position_history=[
                {
                    "id": p.id,
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "volume": p.volume,
                    "open_price": p.open_price,
                    "open_time": p.open_time,
                }
                for p in self.position_manager.positions.values()
            ],
            order_history=[
                {
                    "id": o.id,
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "type": o.type.value,
                    "volume": o.volume,
                    "price": o.price,
                    "status": o.status.value,
                }
                for o in self.order_manager.order_history
            ],
            stats={
                "total_bars": len(self.equity_curve),
                "final_equity": self.account.equity,
                "total_return_pct": (self.account.equity - self.account.initial_balance) / self.account.initial_balance * 100,
                "max_drawdown": self.account.max_drawdown,
                "max_drawdown_pct": self.account.max_drawdown_pct,
            },
        )


# Convenience function
def run_backtest(
    data: Dict[str, pd.DataFrame],
    signal_generator: SignalGenerator,
    config: BacktestConfig,
) -> BacktestResult:
    """Run a backtest with given data and signal generator."""
    feed = DataFeed(data)
    engine = BacktestEngine(config, feed, signal_generator)
    return engine.run()