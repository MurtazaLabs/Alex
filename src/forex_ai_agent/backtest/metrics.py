"""Metrics Calculation for Backtesting.

Comprehensive metrics suite including standard and advanced metrics.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class TradeMetrics:
    """Individual trade metrics."""
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    avg_r_multiple: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_hold_time_hours: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    gross_profit: float
    gross_loss: float


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    avg_drawdown: float
    recovery_factor: float
    ulcer_index: float


@dataclass
class RiskMetrics:
    """Risk-adjusted metrics."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    tail_ratio: float
    omega_ratio: float
    gain_to_pain_ratio: float


def calculate_metrics(
    equity_curve: List[Tuple[datetime, float]],
    trades: List[Dict],
    initial_balance: float,
) -> Dict[str, Any]:
    """Calculate comprehensive backtest metrics."""

    if not equity_curve:
        return {}

    # Equity curve arrays
    times = [t for t, _ in equity_curve]
    values = np.array([v for _, v in equity_curve])

    # Returns
    returns = np.diff(values) / values[:-1]
    daily_returns = _resample_returns(times, values, "D")

    # Trade metrics
    trade_metrics = _calculate_trade_metrics(trades)

    # Portfolio metrics
    portfolio_metrics = _calculate_portfolio_metrics(values, times, daily_returns, initial_balance)

    # Risk metrics
    risk_metrics = _calculate_risk_metrics(returns, daily_returns)

    # Breakdown metrics
    breakdowns = _calculate_breakdowns(trades)

    return {
        "trades": trade_metrics.__dict__,
        "portfolio": portfolio_metrics.__dict__,
        "risk": risk_metrics.__dict__,
        "breakdowns": breakdowns,
        "summary": {
            "total_trades": trade_metrics.total_trades,
            "win_rate": trade_metrics.win_rate,
            "profit_factor": trade_metrics.profit_factor,
            "expectancy": trade_metrics.expectancy,
            "sharpe_ratio": portfolio_metrics.sharpe_ratio,
            "sortino_ratio": portfolio_metrics.sortino_ratio,
            "calmar_ratio": portfolio_metrics.calmar_ratio,
            "max_drawdown_pct": portfolio_metrics.max_drawdown_pct,
            "total_return_pct": portfolio_metrics.total_return,
            "annualized_return": portfolio_metrics.annualized_return,
        },
    }


def _calculate_trade_metrics(trades: List[Dict]) -> TradeMetrics:
    """Calculate per-trade statistics."""
    if not trades:
        return TradeMetrics(
            win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
            expectancy=0, avg_r_multiple=0, max_consecutive_wins=0,
            max_consecutive_losses=0, avg_hold_time_hours=0,
            total_trades=0, winning_trades=0, losing_trades=0,
            gross_profit=0, gross_loss=0
        )

    profits = np.array([t.get("profit", 0) for t in trades])
    winning = profits[profits > 0]
    losing = profits[profits <= 0]

    total_trades = len(trades)
    winning_trades = len(winning)
    losing_trades = len(losing)

    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_win = np.mean(winning) if len(winning) > 0 else 0
    avg_loss = np.mean(losing) if len(losing) > 0 else 0

    gross_profit = np.sum(winning) if len(winning) > 0 else 0
    gross_loss = abs(np.sum(losing)) if len(losing) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

    # R-multiples (profit / risk) - simplified
    r_multiples = []
    for t in trades:
        if t.get("sl") and t.get("entry_price"):
            risk = abs(t["entry_price"] - t["sl"]) * t.get("volume", 1) * 100000
            if risk > 0:
                r_multiples.append(t.get("profit", 0) / risk)

    avg_r_multiple = np.mean(r_multiples) if r_multiples else 0

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0

    for p in profits:
        if p > 0:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    avg_hold_time = np.mean([t.get("hold_time_hours", 0) for t in trades])

    return TradeMetrics(
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_r_multiple=avg_r_multiple,
        max_consecutive_wins=max_consec_wins,
        max_consecutive_losses=max_consec_losses,
        avg_hold_time_hours=avg_hold_time,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
    )


def _calculate_portfolio_metrics(
    values: np.ndarray,
    times: List[datetime],
    daily_returns: np.ndarray,
    initial_balance: float,
) -> PortfolioMetrics:
    """Calculate portfolio-level metrics."""

    total_return = (values[-1] - values[0]) / values[0]

    # Annualized return
    days = (times[-1] - times[0]).days
    years = max(days / 365.25, 1/365.25)
    annualized_return = (values[-1] / values[0]) ** (1 / years) - 1

    # Daily returns for Sharpe/Sortino
    if len(daily_returns) > 1:
        daily_mean = np.mean(daily_returns)
        daily_std = np.std(daily_returns, ddof=1)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.01

        # Sharpe (assuming 0% risk-free)
        sharpe_ratio = daily_mean / daily_std * np.sqrt(252) if daily_std > 0 else 0

        # Sortino
        sortino_ratio = daily_mean / downside_std * np.sqrt(252) if downside_std > 0 else 0
    else:
        sharpe_ratio = 0
        sortino_ratio = 0

    # Drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    max_drawdown_pct = np.max(drawdown) * 100
    max_drawdown = np.max(peak - values)

    # Max drawdown duration
    dd_duration = 0
    max_dd_duration = 0
    in_drawdown = False
    for i, dd in enumerate(drawdown):
        if dd > 0:
            if not in_drawdown:
                in_drawdown = True
                dd_duration = 1
            else:
                dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            in_drawdown = False
            dd_duration = 0

    # Calmar ratio
    calmar_ratio = annualized_return / (max_drawdown_pct / 100) if max_drawdown_pct > 0 else 0

    # Recovery factor
    recovery_factor = abs(np.sum(values[-1] - values[0])) / max_drawdown if max_drawdown > 0 else 0

    # Ulcer Index
    ulcer_index = np.sqrt(np.mean(drawdown ** 2)) * 100

    return PortfolioMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=values[0] - np.min(values),
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_duration_days=max_dd_duration,
        avg_drawdown=np.mean(drawdown) * 100,
        recovery_factor=recovery_factor,
        ulcer_index=ulcer_index,
    )


def _calculate_risk_metrics(returns: np.ndarray, daily_returns: np.ndarray) -> RiskMetrics:
    """Calculate risk metrics (VaR, CVaR, tail ratios)."""

    if len(daily_returns) < 30:
        return RiskMetrics(0, 0, 0, 0, 0, 0, 0)

    # VaR
    var_95 = np.percentile(daily_returns, 5)
    var_99 = np.percentile(daily_returns, 1)

    # CVaR (Expected Shortfall)
    cvar_95 = np.mean(daily_returns[daily_returns <= var_95]) if np.any(daily_returns <= var_95) else var_95
    cvar_99 = np.mean(daily_returns[daily_returns <= var_99]) if np.any(daily_returns <= var_99) else var_99

    # Tail Ratio (95th percentile / 5th percentile)
    p95 = np.percentile(daily_returns, 95)
    p05 = np.percentile(daily_returns, 5)
    tail_ratio = abs(p95 / p05) if p05 != 0 else 0

    # Omega Ratio (threshold = 0)
    threshold = 0
    gains = daily_returns[daily_returns > threshold]
    losses = daily_returns[daily_returns <= threshold]
    omega_ratio = np.sum(gains - threshold) / np.sum(threshold - losses) if np.sum(threshold - losses) > 0 else 0

    # Gain to Pain Ratio
    gross_profit = np.sum(daily_returns[daily_returns > 0])
    gross_loss = abs(np.sum(daily_returns[daily_returns < 0]))
    gain_to_pain = gross_profit / gross_loss if gross_loss > 0 else 0

    return RiskMetrics(
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        tail_ratio=tail_ratio,
        omega_ratio=omega_ratio,
        gain_to_pain_ratio=gain_to_pain,
    )


def _calculate_breakdowns(trades: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """Calculate performance breakdowns by various dimensions."""

    if not trades:
        return {}

    df = pd.DataFrame(trades)

    breakdowns = {}

    # By symbol
    if "symbol" in df.columns:
        breakdowns["by_symbol"] = {}
        for symbol, group in df.groupby("symbol"):
            profits = group["profit"].values
            breakdowns["by_symbol"][symbol] = {
                "trades": len(group),
                "win_rate": (profits > 0).mean(),
                "total_profit": profits.sum(),
                "avg_profit": profits.mean(),
            }

    # By side (LONG/SHORT)
    if "side" in df.columns:
        breakdowns["by_side"] = {}
        for side, group in df.groupby("side"):
            profits = group["profit"].values
            breakdowns["by_side"][side] = {
                "trades": len(group),
                "win_rate": (profits > 0).mean(),
                "total_profit": profits.sum(),
            }

    # By session (if open_time available)
    if "open_time" in df.columns:
        df["hour"] = pd.to_datetime(df["open_time"]).dt.hour
        breakdowns["by_hour"] = {}
        for hour, group in df.groupby("hour"):
            profits = group["profit"].values
            breakdowns["by_hour"][int(hour)] = {
                "trades": len(group),
                "win_rate": (profits > 0).mean(),
                "total_profit": profits.sum(),
            }

    # By day of week
    if "open_time" in df.columns:
        df["dow"] = pd.to_datetime(df["open_time"]).dt.dayofweek
        breakdowns["by_dow"] = {}
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for dow, group in df.groupby("dow"):
            profits = group["profit"].values
            breakdowns["by_dow"][dow_names[int(dow)]] = {
                "trades": len(group),
                "win_rate": (profits > 0).mean(),
                "total_profit": profits.sum(),
            }

    # By month
    if "open_time" in df.columns:
        df["month"] = pd.to_datetime(df["open_time"]).dt.month
        breakdowns["by_month"] = {}
        for month, group in df.groupby("month"):
            profits = group["profit"].values
            breakdowns["by_month"][int(month)] = {
                "trades": len(group),
                "win_rate": (profits > 0).mean(),
                "total_profit": profits.sum(),
            }

    return breakdowns


def _resample_returns(times: List[datetime], values: np.ndarray, freq: str) -> np.ndarray:
    """Resample equity curve to specified frequency."""
    if len(values) < 2:
        return np.array([])

    df = pd.DataFrame({"value": values}, index=pd.DatetimeIndex(times))
    df = df.resample(freq).last().ffill()
    returns = df["value"].pct_change().dropna().values
    return returns


def bootstrap_confidence_interval(
    values: np.ndarray,
    statistic_fn: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for a statistic."""
    if len(values) < 10:
        return (0, 0)

    bootstrapped = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrapped.append(statistic_fn(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrapped, alpha * 100)
    upper = np.percentile(bootstrapped, (1 - alpha) * 100)
    return (lower, upper)


def monte_carlo_simulation(
    trades: List[Dict],
    n_simulations: int = 1000,
    initial_balance: float = 100000,
) -> Dict[str, Any]:
    """Run Monte Carlo simulation of trade sequence."""
    if not trades:
        return {}

    profits = np.array([t.get("profit", 0) for t in trades])

    final_equities = []
    max_drawdowns = []

    for _ in range(n_simulations):
        # Resample trades with replacement
        sampled = np.random.choice(profits, size=len(profits), replace=True)
        equity = 100000.0
        peak = equity
        max_dd = 0

        for p in sampled:
            equity += p
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)

        final_equities.append(equity)
        max_drawdowns.append(max_dd)

    return {
        "final_equity_mean": np.mean(final_equities),
        "final_equity_std": np.std(final_equities),
        "final_equity_5pct": np.percentile(final_equities, 5),
        "final_equity_95pct": np.percentile(final_equities, 95),
        "max_dd_mean": np.mean(max_drawdowns),
        "max_dd_std": np.std(max_drawdowns),
        "max_dd_95pct": np.percentile(max_drawdowns, 95),
        "prob_profit": np.mean(np.array(final_equities) > 100000),
        "prob_ruin": np.mean(np.array(final_equities) < 50000),
    }