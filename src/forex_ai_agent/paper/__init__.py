"""Paper Trading Package - Real-time paper trading with realistic execution."""

from __future__ import annotations

from .engine import (
    PaperTradingEngine,
    PaperAccount,
    PaperPosition,
    PaperTrade,
    PaperTradingMode,
    create_paper_engine,
)
from .manager import (
    AdvancedPositionManager,
    PositionLimits,
    ScaleRule,
    ScaleDirection,
    TrailingStopConfig,
)
from .tracking import (
    PerformanceTracker,
    TradeMetrics,
    SessionMetrics,
    StrategyPerformance,
)
from .reporting import (
    generate_daily_report,
    generate_weekly_report,
    generate_monthly_report,
    save_report,
)

__all__ = [
    # Engine
    "PaperTradingEngine",
    "PaperAccount",
    "PaperPosition",
    "PaperTrade",
    "PaperTradingMode",
    "create_paper_engine",
    # Manager
    "AdvancedPositionManager",
    "PositionLimits",
    "ScaleRule",
    "ScaleDirection",
    "TrailingStopConfig",
    # Tracking
    "PerformanceTracker",
    "TradeMetrics",
    "SessionMetrics",
    "StrategyPerformance",
    # Reporting
    "generate_daily_report",
    "generate_weekly_report",
    "generate_monthly_report",
    "save_report",
]