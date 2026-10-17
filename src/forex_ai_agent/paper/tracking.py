"""Paper Trading Performance Tracking & Analytics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forex_ai_agent.paper.engine import PaperTrade, PaperPosition, PaperAccount


@dataclass
class TradeMetrics:
    """Per-trade detailed metrics."""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    volume: float
    profit: float
    r_multiple: float
    hold_time_hours: float
    mae: float
    mfe: float
    mae_pct: float  # MAE as % of risk
    mfe_pct: float  # MFE as % of risk
    efficiency: float  # profit / MFE
    adverse_excursion_pct: float  # MAE / entry
    favorable_excursion_pct: float  # MFE / entry


@dataclass
class SessionMetrics:
    """Metrics for a trading session."""
    session: str  # Asian, London, NY, Overlap
    trades: int
    win_rate: float
    total_profit: float
    avg_r_multiple: float
    best_trade: float
    worst_trade: float


@dataclass
class StrategyPerformance:
    """Performance by strategy version."""
    version: str
    trades: int
    win_rate: float
    total_profit: float
    sharpe_ratio: float
    max_drawdown_pct: float
    avg_r_multiple: float
    profit_factor: float


class PerformanceTracker:
    """
    Comprehensive performance tracking for paper trading.
    
    Tracks per-trade, aggregate, and comparative metrics in real-time.
    """

    def __init__(self, engine):
        self.engine = engine
        self.account = engine.account
        self.trades = engine.closed_trades
        self.positions = engine.positions

        # Cached calculations
        self._trade_metrics: Dict[str, Any] = {}
        self._session_metrics: Dict[str, SessionMetrics] = {}
        self._strategy_metrics: Dict[str, StrategyPerformance] = {}
        self._daily_series: List[Tuple[datetime, float]] = []
        self._equity_curve: List[Tuple[datetime, float]] = []

        # Real-time tracking
        self._last_update = datetime.utcnow()

    def record_equity_point(self, timestamp: datetime, equity: float):
        """Record equity point for curve."""
        self._equity_curve.append((timestamp, equity))

    def update(self):
        """Update all metrics."""
        self._update_equity_curve()
        self._update_trade_metrics()
        self._update_session_metrics()
        self._update_strategy_metrics()
        self._update_daily_series()
        self._last_update = datetime.utcnow()

    def _update_equity_curve(self):
        """Update equity curve from account."""
        if self._equity_curve:
            last_time, _ = self._equity_curve[-1]
            if (datetime.utcnow() - last_time).total_seconds() < 60:
                return

        # In real-time, this would be called from engine
        pass

    def _update_trade_metrics(self):
        """Calculate per-trade metrics."""
        self._trade_metrics = {}
        for trade in self.engine.closed_trades:
            self._trade_metrics[trade.id] = self._calculate_trade_metrics(trade)

    def _calculate_trade_metrics(self, trade: "PaperTrade") -> Dict[str, Any]:
        """Calculate detailed metrics for a single trade."""
        risk_per_unit = abs(trade.entry_price - trade.sl) if trade.sl else 0
        risk_total = risk_per_unit * trade.volume * 100000 if risk_per_unit > 0 else 1

        mae_pct = (trade.mae / risk_total) * 100 if risk_total > 0 else 0
        mfe_pct = (trade.mfe / risk_total) * 100 if risk_total > 0 else 0
        efficiency = trade.profit / trade.mfe if trade.mfe > 0 else 0

        return {
            "trade_id": trade.id,
            "symbol": trade.symbol,
            "side": trade.side.value,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "volume": trade.volume,
            "profit": trade.profit,
            "r_multiple": trade.r_multiple,
            "hold_time_hours": trade.hold_time_seconds / 3600,
            "mae": trade.mae,
            "mfe": trade.mfe,
            "mae_pct": mae_pct,
            "mfe_pct": mfe_pct,
            "efficiency": efficiency,
            "adverse_excursion_pct": trade.mae / trade.entry_price * 100 if trade.entry_price else 0,
            "favorable_excursion_pct": trade.mfe / trade.entry_price * 100 if trade.entry_price else 0,
            "commission": trade.commission,
            "swap": trade.swap,
            "reason": trade.reason,
        }

    def _update_session_metrics(self):
        """Calculate metrics by trading session."""
        self._session_metrics = {}
        sessions = {
            "Asian": (0, 8),
            "London": (8, 13),
            "Overlap": (13, 17),
            "NewYork": (17, 24),
        }

        for session_name, (start, end) in sessions.items():
            session_trades = []
            for trade in self.engine.closed_trades:
                hour = trade.open_time.hour
                if start <= hour < end:
                    session_trades.append(trade)

            if session_trades:
                profits = [t.profit for t in session_trades]
                winning = [p for p in profits if p > 0]

                self._session_metrics[session_name] = SessionMetrics(
                    session=session_name,
                    trades=len(session_trades),
                    win_rate=len(winning) / len(session_trades),
                    total_profit=sum(profits),
                    avg_r_multiple=np.mean([t.r_multiple for t in session_trades]),
                    best_trade=max(profits),
                    worst_trade=min(profits),
                )

    def _update_strategy_metrics(self):
        """Calculate metrics by strategy version."""
        self._strategy_metrics = {}
        by_version = defaultdict(list)

        for trade in self.engine.closed_trades:
            version = trade.strategy_version or "unknown"
            by_version[version].append(trade)

        for version, trades in by_version.items():
            profits = [t.profit for t in trades]
            winning = [p for p in profits if p > 0]
            losing = [p for p in profits if p <= 0]

            # Equity curve for Sharpe
            equity = 100000.0
            equity_curve = [equity]
            for t in trades:
                equity += t.profit
                equity_curve.append(equity)

            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

            self._strategy_metrics[version] = StrategyPerformance(
                version=version,
                trades=len(trades),
                win_rate=len(winning) / len(trades),
                total_profit=sum(profits),
                sharpe_ratio=sharpe,
                max_drawdown_pct=0,  # Would need equity curve
                avg_r_multiple=np.mean([t.r_multiple for t in trades]),
                profit_factor=sum(winning) / abs(sum(losing)) if losing else 0,
            )

    def _update_daily_series(self):
        """Update daily P&L series."""
        # In real implementation, this would track daily snapshots
        pass

    # Query methods

    def get_trade_metrics(self, trade_id: str) -> Optional[Dict]:
        """Get metrics for a specific trade."""
        return self._trade_metrics.get(trade_id)

    def get_all_trade_metrics(self) -> List[Dict]:
        """Get all trade metrics as list."""
        return list(self._trade_metrics.values())

    def get_session_metrics(self) -> Dict[str, SessionMetrics]:
        """Get metrics by session."""
        return self._session_metrics

    def get_strategy_metrics(self) -> Dict[str, StrategyPerformance]:
        """Get metrics by strategy version."""
        return self._strategy_metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        if not self.trades:
            return {}

        profits = [t.profit for t in self.trades]
        winning = [p for p in profits if p > 0]
        losing = [p for p in profits if p <= 0]

        returns = np.array(profits) / 100000  # approximate returns
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Equity curve for drawdown
        equity = 100000.0
        equity_curve = [equity]
        for t in self.trades:
            equity += t.profit
            equity_curve.append(equity)

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = np.max(drawdown) * 100

        return {
            "overview": {
                "total_trades": len(self.trades),
                "winning_trades": len(winning),
                "losing_trades": len(losing),
                "win_rate": len(winning) / len(self.trades),
                "total_profit": sum(profits),
                "avg_profit": np.mean(profits),
                "avg_win": np.mean(winning) if winning else 0,
                "avg_loss": np.mean(losing) if losing else 0,
                "profit_factor": sum(winning) / abs(sum(losing)) if losing else 0,
                "expectancy": np.mean(profits),
                "sharpe_ratio": sharpe,
                "max_drawdown_pct": max_dd,
            },
            "r_multiples": {
                "avg": np.mean([t.r_multiple for t in self.trades]),
                "median": np.median([t.r_multiple for t in self.trades]),
                "best": max(t.r_multiple for t in self.trades),
                "worst": min(t.r_multiple for t in self.trades),
            },
            "hold_time": {
                "avg_hours": np.mean([t.hold_time_seconds / 3600 for t in self.trades]),
                "median_hours": np.median([t.hold_time_seconds / 3600 for t in self.trades]),
            },
            "mae_mfe": {
                "avg_mae_pct": np.mean([t.mae / abs(t.entry_price - t.sl) * 100 for t in self.trades if t.sl]),
                "avg_mfe_pct": np.mean([t.mfe / abs(t.entry_price - t.sl) * 100 for t in self.trades if t.sl]),
                "efficiency": np.mean([t.profit / t.mfe for t in self.trades if t.mfe > 0]),
            },
            "by_session": {
                s: {
                    "trades": m.trades,
                    "win_rate": m.win_rate,
                    "profit": m.total_profit,
                    "avg_r": m.avg_r_multiple,
                }
                for s, m in self._session_metrics.items()
            },
            "by_strategy": {
                v: {
                    "trades": m.trades,
                    "win_rate": m.win_rate,
                    "profit": m.total_profit,
                    "sharpe": m.sharpe_ratio,
                    "pf": m.profit_factor,
                }
                for v, m in self._strategy_metrics.items()
            },
        }

    def get_equity_curve(self) -> List[Tuple[datetime, float]]:
        """Get equity curve for plotting."""
        equity = self.engine.account.initial_balance
        curve = [(datetime.utcnow() - timedelta(days=30), equity)]  # placeholder

        for trade in sorted(self.trades, key=lambda t: t.open_time):
            equity += trade.profit
            curve.append((trade.close_time, equity))

        return curve

    def get_drawdown_curve(self) -> List[Tuple[datetime, float]]:
        """Get drawdown curve."""
        equity_curve = self.get_equity_curve()
        if not equity_curve:
            return []

        values = [v for _, v in equity_curve]
        peak = np.maximum.accumulate(values)
        drawdown = (np.array(values) - peak) / peak * 100

        return [(equity_curve[i][0], drawdown[i]) for i in range(len(drawdown))]

    def get_monthly_returns(self) -> Dict[str, float]:
        """Get monthly returns."""
        if not self.trades:
            return {}

        df = pd.DataFrame([{
            "date": t.close_time,
            "profit": t.profit,
        } for t in self.trades])

        df["month"] = df["date"].dt.to_period("M")
        monthly = df.groupby("month")["profit"].sum() / 100000 * 100  # % return

        return {str(k): v for k, v in monthly.items()}

    def get_rolling_metrics(self, window: int = 30) -> Dict[str, List[float]]:
        """Get rolling window metrics."""
        if len(self.trades) < window:
            return {}

        metrics = defaultdict(list)
        for i in range(window, len(self.trades) + 1):
            window_trades = self.trades[i-window:i]
            profits = [t.profit for t in window_trades]
            winning = [p for p in profits if p > 0]
            losing = [p for p in profits if p <= 0]

            metrics["win_rate"].append(len(winning) / window)
            metrics["profit_factor"].append(sum(winning) / abs(sum(losing)) if losing else 0)
            metrics["avg_r"].append(np.mean([t.r_multiple for t in window_trades]))
            metrics["sharpe"].append(
                np.mean(profits) / np.std(profits) * np.sqrt(252) if np.std(profits) > 0 else 0
            )

        return dict(metrics)

    def export_trades_csv(self) -> str:
        """Export trades as CSV string."""
        if not self.trades:
            return ""

        rows = ["trade_id,symbol,side,volume,entry,exit,sl,tp,profit,r_multiple,hold_hours,reason,session,strategy"]
        for t in self.trades:
            session = "Asian" if t.open_time.hour < 8 else \
                      "London" if t.open_time.hour < 13 else \
                      "Overlap" if t.open_time.hour < 17 else "NewYork"

            rows.append(f"{t.id},{t.symbol},{t.side.value},{t.volume},{t.entry_price:.5f},"
                        f"{t.exit_price:.5f},{t.sl:.5f},{t.tp:.5f},{t.profit:.2f},"
                        f"{t.r_multiple:.2f},{t.hold_time_seconds/3600:.2f},{t.reason},"
                        f"{session},{t.strategy_version or 'unknown'}")

        return "\n".join(rows)