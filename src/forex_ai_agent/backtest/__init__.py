"""Backtest Package - Event-driven backtesting engine with comprehensive metrics."""

from __future__ import annotations

from .engine import (
    BacktestConfig,
    BacktestResult,
    DataFeed,
    BarData,
    SignalGenerator,
    BacktestEngine,
    run_backtest,
)
from .execution import (
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
    SessionSpreadModel,
    ATRSpreadModel,
    FixedSlippageModel,
    VolatilitySizeSlippageModel,
    PerLotCommissionModel,
    PercentageCommissionModel,
    FixedSwapModel,
    TrippleSwapWednesday,
    DEFAULT_SPREADS,
    DEFAULT_COMMISSION,
    DEFAULT_SWAP,
)
from .orders import (
    OrderManager,
    PositionManager,
    AccountState,
    ExecutionContext,
    FillPolicy,
    Fill,
)
from .metrics import (
    calculate_metrics,
    TradeMetrics,
    PortfolioMetrics,
    RiskMetrics,
    bootstrap_confidence_interval,
    monte_carlo_simulation,
)
from .walkforward import (
    WalkForwardConfig,
    WalkForwardWindow,
    WalkForwardResult,
    WalkForwardAnalyzer,
    run_walkforward,
)
from .reporting import (
    generate_html_report,
    generate_pdf_report,
    save_html_report,
)

__all__ = [
    # Engine
    "BacktestConfig",
    "BacktestResult",
    "DataFeed",
    "BarData",
    "SignalGenerator",
    "BacktestEngine",
    "run_backtest",
    # Execution
    "Order",
    "Position",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "SpreadModel",
    "SlippageModel",
    "CommissionModel",
    "SwapModel",
    "FixedSpreadModel",
    "SessionSpreadModel",
    "ATRSpreadModel",
    "FixedSlippageModel",
    "VolatilitySizeSlippageModel",
    "PerLotCommissionModel",
    "PercentageCommissionModel",
    "FixedSwapModel",
    "TrippleSwapWednesday",
    "DEFAULT_SPREADS",
    "DEFAULT_COMMISSION",
    "DEFAULT_SWAP",
    # Orders
    "OrderManager",
    "PositionManager",
    "AccountState",
    "ExecutionContext",
    "FillPolicy",
    "Fill",
    # Metrics
    "calculate_metrics",
    "TradeMetrics",
    "PortfolioMetrics",
    "RiskMetrics",
    "bootstrap_confidence_interval",
    "monte_carlo_simulation",
    # Walk-forward
    "WalkForwardConfig",
    "WalkForwardWindow",
    "WalkForwardResult",
    "WalkForwardAnalyzer",
    "run_walkforward",
    # Reporting
    "generate_html_report",
    "generate_pdf_report",
    "save_html_report",
]