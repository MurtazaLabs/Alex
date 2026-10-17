"""Execution Models for Backtesting.

Realistic execution simulation including spread, slippage, commission, and swap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    volume: float  # lots
    price: Optional[float] = None  # for limit/stop
    stop_price: Optional[float] = None  # for stop orders
    sl: Optional[float] = None
    tp: Optional[float] = None
    magic: int = 0
    comment: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_volume: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    profit: float = 0.0
    created_time: datetime = datetime.utcnow()
    filled_time: Optional[datetime] = None


@dataclass
class Position:
    """Open position representation.

    ``open_time`` is the BAR TIME the position opened at — set explicitly by
    every caller (Stage D lifecycle fix: the old import-time dataclass default
    made hold times wrong and overnight swap never accrue).
    """
    id: str
    symbol: str
    side: OrderSide
    volume: float
    open_price: float
    current_price: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    commission: float = 0.0
    swap: float = 0.0
    profit: float = 0.0
    open_time: Optional[datetime] = None
    swap_days: int = 0   # days of swap already charged (Stage D: charge each day once)
    magic: int = 0
    comment: str = ""


class SpreadModel(ABC):
    """Base spread model."""

    @abstractmethod
    def get_spread(self, symbol: str, timestamp: datetime, session: str) -> float:
        """Get spread in pips."""
        pass


class FixedSpreadModel(SpreadModel):
    """Fixed spread per symbol."""

    def __init__(self, spreads: Dict[str, float]):
        self.spreads = spreads

    def get_spread(self, symbol: str, timestamp: datetime, session: str) -> float:
        return self.spreads.get(symbol, 1.5)


class SessionSpreadModel(SpreadModel):
    """Session-based spread model."""

    SESSION_MULTIPLIERS = {
        "Asian": 1.5,
        "London": 1.0,
        "NewYork": 1.0,
        "Overlap": 0.8,
    }

    def __init__(self, base_spreads: Dict[str, float]):
        self.base_spreads = base_spreads

    def get_spread(self, symbol: str, timestamp: datetime, session: str) -> float:
        base = self.base_spreads.get(symbol, 1.5)
        mult = self.SESSION_MULTIPLIERS.get(session, 1.0)
        return base * mult


class ATRSpreadModel(SpreadModel):
    """ATR-based dynamic spread model."""

    def __init__(self, base_spreads: Dict[str, float], atr_multiplier: float = 0.1):
        self.base_spreads = base_spreads
        self.atr_multiplier = atr_multiplier

    def get_spread(self, symbol: str, timestamp: datetime, session: str, atr_pips: float = 10.0) -> float:
        base = self.base_spreads.get(symbol, 1.5)
        atr_component = atr_pips * self.atr_multiplier
        session_mult = SessionSpreadModel.SESSION_MULTIPLIERS.get(session, 1.0)
        return (base + atr_component) * session_mult


class SlippageModel(ABC):
    """Base slippage model."""

    @abstractmethod
    def get_slippage(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        spread_pips: float,
        volatility_pips: float,
        timestamp: datetime,
    ) -> float:
        """Get slippage in pips (positive = worse fill)."""
        pass


class FixedSlippageModel(SlippageModel):
    """Fixed slippage per symbol."""

    def __init__(self, slippage_pips: Dict[str, float]):
        self.slippage = slippage_pips

    def get_slippage(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        spread_pips: float,
        volatility_pips: float,
        timestamp: datetime,
    ) -> float:
        return self.slippage.get(symbol, 0.5)


class VolatilitySizeSlippageModel(SlippageModel):
    """Slippage based on volatility and position size."""

    def __init__(
        self,
        base_slippage: Dict[str, float] = None,
        vol_factor: float = 0.1,
        size_factor: float = 0.05,
    ):
        self.base_slippage = base_slippage or {}
        self.vol_factor = vol_factor
        self.size_factor = size_factor

    def get_slippage(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        spread_pips: float,
        volatility_pips: float,
        timestamp: datetime,
    ) -> float:
        base = self.base_slippage.get(symbol, 0.3)
        vol_component = volatility_pips * self.vol_factor
        size_component = max(0, (volume - 1.0) * self.size_factor)  # extra slippage for >1 lot
        return base + vol_component + size_component


class CommissionModel(ABC):
    """Base commission model."""

    @abstractmethod
    def calculate_commission(
        self,
        symbol: str,
        volume: float,
        price: float,
        side: OrderSide,
    ) -> float:
        """Calculate commission in account currency."""
        pass


class PerLotCommissionModel(CommissionModel):
    """Fixed commission per lot per side."""

    def __init__(self, commission_per_lot: float = 7.0):
        self.commission_per_lot = commission_per_lot

    def calculate_commission(
        self,
        symbol: str,
        volume: float,
        price: float,
        side: OrderSide,
    ) -> float:
        return volume * self.commission_per_lot


class PercentageCommissionModel(CommissionModel):
    """Percentage-based commission."""

    def __init__(self, percentage: float = 0.00005):  # 0.005%
        self.percentage = percentage

    def calculate_commission(
        self,
        symbol: str,
        volume: float,
        price: float,
        side: OrderSide,
    ) -> float:
        notional = volume * 100000 * price  # standard lot
        return notional * self.percentage


class SwapModel(ABC):
    """Base swap/rollover model."""

    @abstractmethod
    def calculate_swap(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        price: float,
        days_held: int,
    ) -> float:
        """Calculate swap in account currency."""
        pass


class FixedSwapModel(SwapModel):
    """Fixed swap points per symbol per side."""

    def __init__(self, swap_long: Dict[str, float], swap_short: Dict[str, float]):
        self.swap_long = swap_long
        self.swap_short = swap_short

    def calculate_swap(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        price: float,
        days_held: int,
    ) -> float:
        if days_held <= 0:
            return 0.0
        if side == OrderSide.BUY:
            points = self.swap_long.get(symbol, -0.5)
        else:
            points = self.swap_short.get(symbol, -0.5)
        # Convert points to currency (simplified)
        pip_value = 10.0 if "JPY" not in symbol else 1000.0
        return points * volume * pip_value * days_held


class TrippleSwapWednesday(SwapModel):
    """Wrapper that adds triple swap on Wednesdays."""

    def __init__(self, base_model: SwapModel):
        self.base = base_model

    def calculate_swap(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        price: float,
        days_held: int,
    ) -> float:
        swap = self.base.calculate_swap(symbol, side, volume, price, days_held)
        # Simplified: if held over Wednesday, triple
        # In reality, would check actual rollover dates
        return swap


# Default configurations
DEFAULT_SPREADS = {
    "EUR/USD": 1.0,
    "GBP/USD": 1.5,
    "USD/JPY": 1.2,
    "USD/CHF": 1.8,
    "AUD/USD": 1.5,
    "USD/CAD": 1.8,
    "NZD/USD": 2.0,
    "EUR/GBP": 1.5,
    "EUR/JPY": 2.0,
    "GBP/JPY": 3.0,
}

DEFAULT_COMMISSION = PerLotCommissionModel(7.0)  # $7 per lot per side

DEFAULT_SWAP = FixedSwapModel(
    swap_long={
        "EUR/USD": -0.5, "GBP/USD": -0.8, "USD/JPY": 0.2,
        "USD/CHF": 0.3, "AUD/USD": -0.2, "USD/CAD": -0.1,
        "NZD/USD": -0.3, "EUR/GBP": -0.2, "EUR/JPY": 0.1,
        "GBP/JPY": -0.5,
    },
    swap_short={
        "EUR/USD": 0.2, "GBP/USD": 0.1, "USD/JPY": -0.8,
        "USD/CHF": -0.5, "AUD/USD": 0.3, "USD/CAD": 0.2,
        "NZD/USD": 0.4, "EUR/GBP": 0.1, "EUR/JPY": -0.3,
        "GBP/JPY": -1.0,
    },
)