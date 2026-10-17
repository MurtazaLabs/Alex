"""Paper Trading Position Manager - Advanced position management features."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from forex_ai_agent.paper.engine import (
    PaperTradingEngine,
    PaperAccount,
    PaperPosition,
    PaperTrade,
    PaperTradingMode,
    OrderSide,
)
from forex_ai_agent.backtest.execution import OrderSide as BTOrderSide
from forex_ai_agent.core.event_bus import Event, EventBus


class ScaleDirection(Enum):
    IN = "IN"      # Adding to winning position
    OUT = "OUT"    # Reducing position


@dataclass
class ScaleRule:
    """Rule for scaling in/out of positions."""
    trigger_r_multiple: float  # R-multiple at which to scale
    scale_pct: float           # Percentage of position to add/reduce
    max_scales: int = 3        # Maximum number of scales
    direction: ScaleDirection = ScaleDirection.IN


@dataclass
class TrailingStopConfig:
    """Trailing stop configuration."""
    activation_r: float = 1.0      # Activate after this R-multiple profit
    trail_distance_pips: int = 20  # Trail distance in pips
    step_pips: int = 5             # Minimum step size in pips


@dataclass
class PositionLimits:
    """Position limits per symbol/correlation group."""
    max_positions_per_symbol: int = 1
    max_volume_per_symbol: float = 10.0  # lots
    max_correlated_exposure_pct: float = 5.0
    max_total_exposure_pct: float = 10.0
    max_positions_total: int = 10


class AdvancedPositionManager:
    """
    Advanced position management for paper trading.
    
    Features:
    - Trailing stops with configurable activation
    - Scale in/out rules based on R-multiples
    - Manual override (close, modify, partial close)
    - Correlation-aware position limits
    - Position grouping and basket management
    """

    def __init__(
        self,
        engine: "PaperTradingEngine",
        limits: PositionLimits = None,
        scale_rules: List["ScaleRule"] = None,
        trailing_config: "TrailingStopConfig" = None,
    ):
        self.engine = engine
        self.limits = limits or PositionLimits()
        self.scale_rules = scale_rules or [
            ScaleRule(trigger_r_multiple=1.0, scale_pct=0.5, max_scales=2, direction=ScaleDirection.IN),
            ScaleRule(trigger_r_multiple=2.0, scale_pct=0.33, max_scales=2, direction=ScaleDirection.IN),
            ScaleRule(trigger_r_multiple=1.5, scale_pct=0.5, max_scales=1, direction=ScaleDirection.OUT),
        ]
        self.trailing_config = trailing_config or TrailingStopConfig()

        # State
        self.scale_counts: Dict[str, int] = {}  # position_id -> scale count
        self.trailing_active: Dict[str, bool] = {}

        # Callbacks
        self.on_scale: Optional[Callable] = None
        self.on_trailing_activated: Optional[Callable] = None

    # Trailing Stops

    async def check_trailing_stops(self):
        """Check and update trailing stops for all positions."""
        for pos_id in list(self.engine.positions.keys()):
            position = self.engine.positions.get(pos_id)
            if position:
                await self._update_trailing_stop(position)

    async def _update_trailing_stop(self, position: "PaperPosition"):
        """Update trailing stop for a single position."""
        if position.trailing_sl is None:
            return

        # Check activation
        if not self.trailing_active.get(position.id, False):
            await self._check_trailing_activation(position)
            return

        # Update trailing stop
        current_price = position.current_price
        trail_distance = position.trailing_sl * 0.0001

        if position.side == BTOrderSide.BUY:
            new_sl = current_price - trail_distance
            if new_sl > (position.sl or 0):
                position.sl = new_sl
        else:
            new_sl = current_price + trail_distance
            if position.sl is None or new_sl < position.sl:
                position.sl = new_sl

    async def _check_trailing_activation(self, position: "PaperPosition"):
        """Check if trailing stop should activate."""
        if position.side == BTOrderSide.BUY:
            r_multiple = (position.current_price - position.entry_price) / abs(position.entry_price - position.sl) if position.sl else 0
        else:
            r_multiple = (position.entry_price - position.current_price) / abs(position.entry_price - position.sl) if position.sl else 0

        if r_multiple >= self.trailing_config.activation_r:
            self.trailing_active[position.id] = True
            if self.on_trailing_activated:
                await self.on_trailing_activated(position)

    async def set_trailing_stop(
        self,
        position_id: str,
        trail_distance_pips: int = None,
        activation_r: float = None,
    ) -> bool:
        """Set or update trailing stop for a position."""
        if position_id not in self.engine.positions:
            return False

        position = self.engine.positions[position_id]
        config = self.trailing_config

        if trail_distance_pips is not None:
            config.trail_distance_pips = trail_distance_pips
        if activation_r is not None:
            config.activation_r = activation_r

        position.trailing_sl = config.trail_distance_pips
        self.trailing_active[position_id] = False  # Will activate when R threshold met

        await self.engine._emit_event("trailing_stop.set", {
            "position_id": position_id,
            "trail_distance_pips": config.trail_distance_pips,
            "activation_r": config.activation_r,
        })
        return True

    async def remove_trailing_stop(self, position_id: str) -> bool:
        """Remove trailing stop from position."""
        if position_id not in self.engine.positions:
            return False

        position = self.engine.positions[position_id]
        position.trailing_sl = None
        self.trailing_active.pop(position_id, None)

        await self.engine._emit_event("trailing_stop.removed", {"position_id": position_id})
        return True

    # Scale In/Out

    async def check_scale_rules(self):
        """Check and execute scale in/out rules."""
        for pos_id in list(self.engine.positions.keys()):
            position = self.engine.positions.get(pos_id)
            if position:
                await self._check_scales(position)

    async def _check_scales(self, position: "PaperPosition"):
        """Check scale rules for a position."""
        if position.sl is None:
            return

        # Calculate current R-multiple
        if position.side == BTOrderSide.BUY:
            r_multiple = (position.current_price - position.entry_price) / abs(position.entry_price - position.sl)
        else:
            r_multiple = (position.entry_price - position.current_price) / abs(position.entry_price - position.sl)

        if r_multiple <= 0:
            return

        current_scales = self.scale_counts.get(position.id, 0)

        for rule in self.scale_rules:
            if r_multiple >= rule.trigger_r_multiple and current_scales < rule.max_scales:
                if rule.direction == ScaleDirection.IN:
                    await self._scale_in(position, rule)
                else:
                    await self._scale_out(position, rule)
                self.scale_counts[position.id] = current_scales + 1
                break

    async def _scale_in(self, position: "PaperPosition", rule: "ScaleRule"):
        """Scale into a winning position."""
        scale_volume = position.volume * rule.scale_pct

        # Check limits
        if not self._check_scale_limits(position, scale_volume):
            return

        # Create scale-in order (market)
        await self.engine.process_signal({
            "id": f"scale_{position.id}_{self.scale_counts.get(position.id, 0)}",
            "symbol": position.symbol,
            "direction": "LONG" if position.side == BTOrderSide.BUY else "SHORT",
            "entry_price": position.current_price,
            "sl": position.sl,
            "tp": position.tp,
            "volume": scale_volume,
            "is_scale": True,
            "parent_position": position.id,
        })

        await self.engine._emit_event("position.scaled_in", {
            "position_id": position.id,
            "scale_volume": scale_volume,
            "rule_trigger_r": rule.trigger_r_multiple,
        })

        if self.on_scale:
            await self.on_scale(position, ScaleDirection.IN, scale_volume)

    async def _scale_out(self, position: "PaperPosition", rule: "ScaleRule"):
        """Scale out of a position."""
        scale_volume = position.volume * rule.scale_pct

        # Partial close
        await self.engine.close_position(
            position.id,
            position.current_price,
            reason=f"SCALE_OUT_{rule.trigger_r_multiple}R",
        )

        # Re-open remaining volume
        remaining_volume = position.volume - scale_volume
        if remaining_volume > 0:
            await self.engine.process_signal({
                "id": f"rescale_{position.id}",
                "symbol": position.symbol,
                "direction": "LONG" if position.side == BTOrderSide.BUY else "SHORT",
                "entry_price": position.current_price,
                "sl": position.sl,
                "tp": position.tp,
                "volume": remaining_volume,
                "is_rescale": True,
            })

        await self.engine._emit_event("position.scaled_out", {
            "position_id": position.id,
            "scale_volume": scale_volume,
            "rule_trigger_r": rule.trigger_r_multiple,
        })

        if self.on_scale:
            await self.on_scale(position, ScaleDirection.OUT, scale_volume)

    def _check_scale_limits(self, position: "PaperPosition", scale_volume: float) -> bool:
        """Check if scaling in would exceed limits."""
        new_volume = position.volume + scale_volume
        return (
            new_volume <= self.limits.max_volume_per_symbol and
            len(self.engine.positions) < self.limits.max_positions_total
        )

    # Manual Override

    async def manual_close(self, position_id: str, pct: float = 1.0) -> bool:
        """Manually close position (full or partial)."""
        if position_id not in self.engine.positions:
            return False

        position = self.engine.positions[position_id]

        if pct >= 1.0:
            await self.engine.close_position(position_id, position.current_price, "MANUAL")
            return True

        # Partial close - close pct, keep rest
        close_volume = position.volume * pct
        await self.engine.close_position(position_id, position.current_price, "MANUAL_PARTIAL")

        # Re-open remaining
        remaining_volume = position.volume * (1 - pct)
        await self.engine.process_signal({
            "id": f"partial_{position.id}",
            "symbol": position.symbol,
            "direction": "LONG" if position.side == BTOrderSide.BUY else "SHORT",
            "entry_price": position.current_price,
            "sl": position.sl,
            "tp": position.tp,
            "volume": remaining_volume,
        })
        return True

    async def manual_modify(
        self,
        position_id: str,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        """Manually modify position SL/TP."""
        return await self.engine.modify_position(position_id, sl, tp)

    async def manual_reverse(self, position_id: str) -> bool:
        """Reverse position (close current, open opposite)."""
        if position_id not in self.engine.positions:
            return False

        position = self.engine.positions[position_id]
        await self.engine.close_position(position_id, position.current_price, "REVERSE")

        # Open opposite
        opposite_direction = "SHORT" if position.side == BTOrderSide.BUY else "LONG"
        await self.engine.process_signal({
            "id": f"reverse_{position_id}",
            "symbol": position.symbol,
            "direction": opposite_direction,
            "entry_price": position.current_price,
            "sl": position.sl,
            "tp": position.tp,
            "volume": position.volume,
            "is_reverse": True,
        })
        return True

    # Correlation & Limits

    def check_correlation_limit(self, signal: Dict) -> bool:
        """Check if new signal would exceed correlation limits."""
        symbol = signal["symbol"]
        base_currency = symbol.split("/")[0]

        correlated_volume = 0.0
        for pos in self.engine.positions.values():
            if pos.symbol.split("/")[0] == base_currency:
                correlated_volume += pos.volume

        new_volume = signal["volume"]
        total_correlated = correlated_volume + new_volume

        # Check against account equity
        total_exposure = sum(p.volume * p.current_price * 100000 for p in self.engine.positions.values())
        new_exposure = signal["volume"] * signal["entry_price"] * 100000
        total_equity = self.engine.account.equity

        if total_equity > 0:
            pct = ((total_exposure + new_exposure) / total_equity) * 100
            if pct > self.limits.max_total_exposure_pct:
                return False

        # Check correlated exposure
        if total_equity > 0:
            corr_pct = (total_exposure / total_equity) * 100  # simplified
            if corr_pct > self.limits.max_correlated_exposure_pct:
                return False

        return True

    def check_position_limits(self, signal: Dict) -> bool:
        """Check if new signal would exceed position limits."""
        # Max positions total
        if len(self.engine.positions) >= self.limits.max_positions_total:
            return False

        # Max per symbol
        symbol_count = sum(1 for p in self.engine.positions.values() if p.symbol == signal["symbol"])
        if symbol_count >= self.limits.max_positions_per_symbol:
            return False

        # Max volume per symbol
        symbol_volume = sum(p.volume for p in self.engine.positions.values() if p.symbol == signal["symbol"])
        if symbol_volume + signal["volume"] > self.limits.max_volume_per_symbol:
            return False

        return True

    # Basket Management

    async def close_all_by_symbol(self, symbol: str, reason: str = "SYMBOL_CLOSE") -> int:
        """Close all positions for a symbol."""
        closed = 0
        for pos_id, position in list(self.engine.positions.items()):
            if position.symbol == symbol:
                await self.engine.close_position(pos_id, position.current_price, reason)
                closed += 1
        return closed

    async def close_all_by_currency(self, currency: str, reason: str = "CURRENCY_CLOSE") -> int:
        """Close all positions involving a currency."""
        closed = 0
        for pos_id, position in list(self.engine.positions.items()):
            if currency in position.symbol:
                await self.engine.close_position(pos_id, position.current_price, reason)
                closed += 1
        return closed

    async def flatten_all(self, reason: str = "FLATTEN") -> int:
        """Close all positions."""
        closed = 0
        for pos_id in list(self.engine.positions.keys()):
            position = self.engine.positions[pos_id]
            await self.engine.close_position(pos_id, position.current_price, reason)
            closed += 1
        return closed

    # Reporting

    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions with management status."""
        return {
            "total_positions": len(self.engine.positions),
            "trailing_active": sum(1 for v in self.trailing_active.values() if v),
            "scales_pending": sum(self.scale_counts.values()),
            "by_symbol": {
                symbol: {
                    "count": sum(1 for p in self.engine.positions.values() if p.symbol == symbol),
                    "total_volume": sum(p.volume for p in self.engine.positions.values() if p.symbol == symbol),
                    "net_pnl": sum(p.unrealized_pnl for p in self.engine.positions.values() if p.symbol == symbol),
                }
                for symbol in set(p.symbol for p in self.engine.positions.values())
            },
            "limits": {
                "max_positions_total": self.limits.max_positions_total,
                "max_volume_per_symbol": self.limits.max_volume_per_symbol,
                "max_correlated_exposure_pct": self.limits.max_correlated_exposure_pct,
                "max_total_exposure_pct": self.limits.max_total_exposure_pct,
            },
        }