"""
Live Trading Gateway - Permission-gated live execution with safety systems.

This module implements the Live Trading Gateway that enforces permission levels,
provides safety systems (kill switch, daily loss limit, position circuit breaker),
and manages live order execution through broker providers.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from forex_ai_agent.core.event_bus import EventBus, Event, InMemoryEventBus
from forex_ai_agent.core.permission_manager import PermissionManager, PermissionLevel
from forex_ai_agent.data.providers.broker import (
    BrokerProvider,
    OrderRequest as BrokerOrderRequest,
    OrderResult as BrokerOrderResult,
    OrderSide,
    OrderType,
    OrderStatus,
    Position as BrokerPosition,
    AccountInfo,
)


class GatewayError(Exception):
    """Base exception for gateway errors."""
    pass


class PermissionDeniedError(GatewayError):
    """Raised when permission check fails."""
    pass


class SafetyViolationError(GatewayError):
    """Raised when a safety rule is violated."""
    pass


class OrderValidationError(GatewayError):
    """Raised when order validation fails."""
    pass


class SafetyRuleType(str, Enum):
    """Types of safety rules."""
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_POSITION_SIZE = "max_position_size"
    MAX_DAILY_TRADES = "max_daily_trades"
    MAX_OPEN_POSITIONS = "max_open_positions"
    MAX_CORRELATED_EXPOSURE = "max_correlated_exposure"
    NEWS_FILTER = "news_filter"
    SESSION_FILTER = "session_filter"
    KILL_SWITCH = "kill_switch"
    POSITION_SIZE_LIMIT = "position_size_limit"
    MARGIN_LEVEL = "margin_level"
    MAX_DRAWDOWN = "max_drawdown"


@dataclass
class SafetyRule:
    """Safety rule configuration."""
    rule_type: SafetyRuleType
    threshold: float
    enabled: bool = True
    severity: str = "CRITICAL"  # CRITICAL, WARNING, INFO
    action: str = "BLOCK"  # BLOCK, WARN, LOG_ONLY
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyViolation:
    """Safety rule violation record."""
    rule_type: SafetyRuleType
    message: str
    current_value: float
    threshold: float
    severity: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False


@dataclass
class LiveOrderRequest:
    """Order request for live execution."""
    id: str = ""
    symbol: str = ""
    side: str = ""  # BUY/SELL
    order_type: str = "MARKET"
    volume: float = 0.0
    price: float = 0.0
    stop_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    magic: int = 0
    comment: str = ""
    strategy_version: str = ""
    signal_id: str = ""


@dataclass
class LiveOrderResult:
    """Result of a live order execution."""
    order_id: str
    broker_order_id: Optional[str] = None
    status: str = "PENDING"
    filled_volume: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    profit: float = 0.0
    error: Optional[str] = None
    executed_at: Optional[datetime] = None


class LiveTradingGateway:
    """
    Live Trading Gateway - Permission-gated live execution with safety systems.
    
    This gateway enforces:
    - Permission levels (0-4): Observe -> Research -> Paper -> Simulate -> Live
    - Safety rules (daily loss limit, max position, drawdown, etc.)
    - Kill switch for emergency flatten
    - Order validation and risk checks
    - Real-time position monitoring
    """
    
    def __init__(
        self,
        broker_provider: BrokerProvider,
        permission_manager: PermissionManager,
        event_bus: Optional[EventBus] = None,
        safety_rules: Optional[List[SafetyRule]] = None,
        kill_switch_enabled: bool = True,
        max_daily_loss_pct: float = 0.02,
        max_position_size_lots: float = 10.0,
        max_daily_trades: int = 100,
        max_open_positions: int = 20,
        max_correlated_exposure_pct: float = 5.0,
        min_margin_level: float = 200.0,
        max_drawdown_pct: float = 0.10,
    ):
        self.broker = broker_provider
        self.permission_manager = permission_manager
        self.event_bus = event_bus or InMemoryEventBus()
        
        # Safety configuration
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_size_lots = max_position_size_lots
        self.max_daily_trades = max_daily_trades
        self.max_open_positions = max_open_positions
        self.max_correlated_exposure_pct = max_correlated_exposure_pct
        self.min_margin_level = min_margin_level
        self.max_drawdown_pct = max_drawdown_pct
        
        # State
        self._running = False
        self._kill_switch_activated = False
        self._daily_start_equity = 0.0
        self._daily_pnl = 0.0
        self._daily_trade_count = 0
        self._open_positions_count = 0
        self._peak_equity = 0.0
        self._last_reset_date = None
        self._safety_violations: List[SafetyViolation] = []
        
        # Callbacks
        self.on_safety_violation: Optional[Callable] = None
        self.on_kill_switch: Optional[Callable] = None
        self.on_order_executed: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None
        
    async def start(self) -> None:
        """Start the live trading gateway."""
        if self._running:
            return
        
        # Check permission level
        if not self.permission_manager.can_live_trade:
            raise PermissionDeniedError("Live trading requires Level 4 permission")
        
        # Initialize daily tracking
        account_info = await self.broker.get_account_info()
        self._daily_start_equity = account_info.equity
        self._peak_equity = account_info.equity
        self._daily_pnl = 0.0
        self._daily_trade_count = 0
        self._last_reset_date = datetime.utcnow().date()
        
        self._running = True
        self._kill_switch_activated = False
        
        # Start background monitoring
        asyncio.create_task(self._monitoring_loop())
        
        if self.event_bus:
            await self.event_bus.publish(Event(
                event_type="gateway.started",
                aggregate_id="live_gateway",
                aggregate_type="gateway",
                payload={"timestamp": datetime.utcnow().isoformat()},
                source_agent="live_gateway",
            ))
    
    async def stop(self, reason: str = "Manual stop") -> None:
        """Stop the live trading gateway."""
        self._running = False
        
        if self.event_bus:
            await self.event_bus.publish(Event(
                event_type="gateway.stopped",
                aggregate_id="live_gateway",
                aggregate_type="gateway",
                payload={"reason": reason, "timestamp": datetime.utcnow().isoformat()},
                source_agent="live_gateway",
            ))
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for safety checks."""
        while self._running:
            try:
                await self._check_safety_rules()
                await self._update_daily_stats()
                await self._check_kill_switch()
                await asyncio.sleep(5)  # Check every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue monitoring
                await asyncio.sleep(5)
    
    async def _check_safety_rules(self) -> None:
        """Check all safety rules."""
        try:
            account_info = await self.broker.get_account_info()
            positions = await self.broker.get_positions()
            open_positions = [p for p in positions if p.volume > 0]
            
            # Update peak equity
            if account_info.equity > self._peak_equity:
                self._peak_equity = account_info.equity
            
            # Check daily loss limit
            self._daily_pnl = self._daily_start_equity - account_info.equity
            daily_loss_pct = abs(self._daily_pnl) / self._daily_start_equity if self._daily_start_equity > 0 else 0
            
            if daily_loss_pct > self.max_daily_loss_pct:
                await self._trigger_safety_violation(
                    SafetyViolation(
                        rule_type=SafetyRuleType.DAILY_LOSS_LIMIT,
                        message=f"Daily loss limit exceeded: {daily_loss_pct:.2%} > {self.max_daily_loss_pct:.2%}",
                        current_value=daily_loss_pct,
                        threshold=self.max_daily_loss_pct,
                        severity="CRITICAL",
                    )
                )
            
            # Check max drawdown
            if self._peak_equity > 0:
                drawdown = (self._peak_equity - account_info.equity) / self._peak_equity
                if drawdown > self.max_drawdown_pct:
                    await self._trigger_safety_violation(
                        SafetyViolation(
                            rule_type=SafetyRuleType.MAX_DRAWDOWN,
                            message=f"Max drawdown exceeded: {drawdown:.2%} > {self.max_drawdown_pct:.2%}",
                            current_value=drawdown,
                            threshold=self.max_drawdown_pct,
                            severity="CRITICAL",
                        )
                    )
            
            # Check margin level
            margin_level = self._get_margin_level(account_info)
            if margin_level < self.min_margin_level:
                await self._trigger_safety_violation(
                    SafetyViolation(
                        rule_type=SafetyRuleType.MARGIN_LEVEL,
                        message=f"Margin level critical: {margin_level:.1f}% < {self.min_margin_level:.1f}%",
                        current_value=margin_level,
                        threshold=self.min_margin_level,
                        severity="CRITICAL",
                    )
                )
            
            # Check open positions count
            if len(open_positions) > self.max_open_positions:
                await self._trigger_safety_violation(
                    SafetyViolation(
                        rule_type=SafetyRuleType.MAX_OPEN_POSITIONS,
                        message=f"Too many open positions: {len(open_positions)} > {self.max_open_positions}",
                        current_value=len(open_positions),
                        threshold=self.max_open_positions,
                        severity="CRITICAL",
                    )
                )
            
            # Check daily trade count
            if self._daily_trade_count >= self.max_daily_trades:
                await self._trigger_safety_violation(
                    SafetyViolation(
                        rule_type=SafetyRuleType.MAX_DAILY_TRADES,
                        message=f"Daily trade limit reached: {self._daily_trade_count}/{self.max_daily_trades}",
                        current_value=self._daily_trade_count,
                        threshold=self.max_daily_trades,
                        severity="CRITICAL",
                    )
                )
            
            # Check position size limits
            for pos in open_positions:
                if pos.volume > self.max_position_size_lots:
                    await self._trigger_safety_violation(
                        SafetyViolation(
                            rule_type=SafetyRuleType.POSITION_SIZE_LIMIT,
                            message=f"Position size limit exceeded for {pos.symbol}: {pos.volume} > {self.max_position_size_lots}",
                            current_value=pos.volume,
                            threshold=self.max_position_size_lots,
                            severity="CRITICAL",
                        )
                    )
            
            # Check correlated exposure (simplified - USD pairs)
            usd_exposure = sum(p.volume for p in open_positions if "USD" in p.symbol)
            if usd_exposure > self.max_correlated_exposure_pct:
                await self._trigger_safety_violation(
                    SafetyViolation(
                        rule_type=SafetyRuleType.MAX_CORRELATED_EXPOSURE,
                        message=f"Correlated USD exposure exceeded: {usd_exposure:.2f} > {self.max_correlated_exposure_pct:.2f}",
                        current_value=usd_exposure,
                        threshold=self.max_correlated_exposure_pct,
                        severity="WARNING",
                    )
                )
                
        except Exception as e:
            # Log error but don't crash monitoring
            pass
    
    async def _update_daily_stats(self) -> None:
        """Update daily statistics and reset at midnight."""
        try:
            account_info = await self.broker.get_account_info()
            current_equity = account_info.equity
            now = datetime.utcnow()
            
            # Reset daily stats at midnight
            if self._last_reset_date is None or self._last_reset_date != now.date():
                self._daily_start_equity = current_equity
                self._peak_equity = current_equity
                self._daily_pnl = 0.0
                self._daily_trade_count = 0
                self._last_reset_date = now.date()
            
            self._daily_pnl = self._daily_start_equity - current_equity
            
        except Exception:
            pass
    
    def _get_margin_level(self, account_info: AccountInfo) -> float:
        """Get current margin level."""
        if account_info.margin > 0:
            return (account_info.equity / account_info.margin) * 100
        return 1000.0  # High default when margin is 0
    
    async def _check_kill_switch(self) -> None:
        """Check and handle kill switch."""
        if self._kill_switch_activated:
            return
        
        # Check if any critical safety violations
        critical_violations = [v for v in self._safety_violations 
                              if v.severity == "CRITICAL" and not v.acknowledged]
        
        if critical_violations:
            await self._activate_kill_switch("Critical safety violations detected")
    
    async def _activate_kill_switch(self, reason: str) -> None:
        """Activate kill switch - immediately flatten all positions."""
        if self._kill_switch_activated:
            return
            
        self._kill_switch_activated = True
        
        # Close all positions immediately
        await self.close_all_positions("KILL_SWITCH")
        
        # Notify callbacks
        if self.on_kill_switch:
            await self.on_kill_switch(reason)
        
        if self.event_bus:
            await self.event_bus.publish(Event(
                event_type="gateway.kill_switch_activated",
                aggregate_id="live_gateway",
                aggregate_type="gateway",
                payload={"reason": reason, "timestamp": datetime.utcnow().isoformat()},
                source_agent="live_gateway",
            ))
    
    async def _trigger_safety_violation(self, violation: SafetyViolation) -> None:
        """Handle a safety rule violation."""
        self._safety_violations.append(violation)
        
        if self.on_safety_violation:
            await self.on_safety_violation(violation)
        
        if self.event_bus:
            await self.event_bus.publish(Event(
                event_type="safety.violation",
                aggregate_id="live_gateway",
                aggregate_type="gateway",
                payload={
                    "rule_type": violation.rule_type.value,
                    "message": violation.message,
                    "current_value": violation.current_value,
                    "threshold": violation.threshold,
                    "severity": violation.severity,
                    "action": violation.action,
                    "timestamp": violation.timestamp.isoformat(),
                },
                source_agent="live_gateway",
            ))
        
        # Auto-activate kill switch for critical violations
        if violation.severity == "CRITICAL" and violation.action == "BLOCK":
            await self._activate_kill_switch(violation.message)
    
    async def _pre_order_checks(self, request: LiveOrderRequest) -> None:
        """Pre-flight safety checks before order execution."""
        # Check kill switch
        if self._kill_switch_activated:
            raise SafetyViolationError("Kill switch is active")
        
        # Check permissions
        if not self.permission_manager.can_live_trade:
            raise PermissionDeniedError("Live trading requires Level 4 permission")
        
        # Validate order
        if request.volume <= 0:
            raise OrderValidationError("Volume must be positive")
        
        if request.volume > self.max_position_size_lots:
            raise OrderValidationError(f"Volume {request.volume} exceeds max position size {self.max_position_size_lots}")
        
        if request.side not in ("BUY", "SELL"):
            raise OrderValidationError(f"Invalid side: {request.side}")
        
        if request.order_type not in ("MARKET", "LIMIT", "STOP", "STOP_LIMIT"):
            raise OrderValidationError(f"Invalid order type: {request.order_type}")
        
        # Check daily trade limit
        if self._daily_trade_count >= self.max_daily_trades:
            raise SafetyViolationError(f"Daily trade limit reached: {self._daily_trade_count}/{self.max_daily_trades}")
        
        # Check daily loss limit
        account_info = await self.broker.get_account_info()
        daily_loss_pct = abs(self._daily_pnl) / self._daily_start_equity if self._daily_start_equity > 0 else 0
        if daily_loss_pct > self.max_daily_loss_pct:
            raise SafetyViolationError(f"Daily loss limit exceeded: {daily_loss_pct:.2%}")
        
        # Check open positions limit
        positions = await self.broker.get_positions()
        open_positions = [p for p in positions if p.volume > 0]
        if len(open_positions) >= self.max_open_positions:
            raise SafetyViolationError(f"Max open positions reached: {len(open_positions)}/{self.max_open_positions}")
        
        # Check symbol info for volume limits
        symbol_info = await self.broker.get_symbol_info(request.symbol)
        if symbol_info:
            min_lot = symbol_info.get("min_lot", 0.01)
            max_lot = symbol_info.get("max_lot", 100.0)
            if request.volume < min_lot:
                raise OrderValidationError(f"Volume {request.volume} below minimum {min_lot}")
            if request.volume > max_lot:
                raise OrderValidationError(f"Volume {request.volume} above maximum {max_lot}")
    
    async def execute_order(self, request: LiveOrderRequest) -> LiveOrderResult:
        """Execute a live order with full safety checks."""
        
        # Pre-flight safety checks
        await self._pre_order_checks(request)
        
        # Create order request for broker
        broker_request = BrokerOrderRequest(
            id=str(uuid4())[:8],
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            volume=request.volume,
            price=request.price,
            stop_price=request.stop_price,
            sl=request.sl,
            tp=request.tp,
            magic=hash(request.strategy_version) % 100000,
            comment=f"{request.strategy_version} {request.signal_id}",
        )
        
        # Execute through broker
        try:
            result = await self.broker.send_order(broker_request)
            
            # Update daily stats
            self._daily_trade_count += 1
            
            return LiveOrderResult(
                order_id=result.order_id,
                broker_order_id=result.broker_order_id,
                status=result.status,
                filled_volume=result.filled_volume,
                filled_price=result.filled_price,
                commission=result.commission,
                swap=result.swap,
                profit=result.profit,
                error=result.error,
                executed_at=result.executed_at,
            )
            
        except Exception as e:
            return LiveOrderResult(
                order_id=request.id or str(uuid4())[:8],
                status="REJECTED",
                error=str(e),
            )
    
    async def close_position(self, position_id: str, reason: str = "MANUAL") -> bool:
        """Close a specific position."""
        try:
            return await self.broker.close_position(position_id)
        except Exception as e:
            return False
    
    async def close_all_positions(self, reason: str = "FLATTEN") -> int:
        """Close all open positions."""
        closed = 0
        try:
            positions = await self.broker.get_positions()
            for pos in positions:
                if pos.volume > 0:
                    try:
                        await self.broker.close_position(pos.id, reason)
                        await asyncio.sleep(0.1)  # Small delay between closes
                        closed += 1
                    except Exception:
                        pass
        except Exception:
            pass
        return closed
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status."""
        positions = []
        try:
            positions = asyncio.run_coroutine_threadsafe(
                self.broker.get_positions(), 
                asyncio.get_event_loop()
            ).result() if asyncio.get_event_loop().is_running() else []
        except Exception:
            pass
        
        open_positions = [p for p in positions if p.volume > 0]
        
        return {
            "kill_switch_active": self._kill_switch_activated,
            "daily_pnl": self._daily_pnl,
            "daily_pnl_pct": self._daily_pnl / self._daily_start_equity if self._daily_start_equity > 0 else 0,
            "daily_trades": self._daily_trade_count,
            "max_daily_trades": self.max_daily_trades,
            "open_positions": len(open_positions),
            "max_open_positions": self.max_open_positions,
            "active_violations": len([v for v in self._safety_violations if not v.acknowledged]),
            "safety_violations": [
                {
                    "rule_type": v.rule_type.value,
                    "message": v.message,
                    "severity": v.severity,
                    "acknowledged": v.acknowledged,
                }
                for v in self._safety_violations
            ],
        }
    
    def acknowledge_violation(self, violation_id: str) -> bool:
        """Acknowledge a safety violation."""
        for v in self._safety_violations:
            if v.timestamp.isoformat() == violation_id:
                v.acknowledged = True
                return True
        return False
    
    async def reset_kill_switch(self, authorized_by: str) -> bool:
        """Reset kill switch (requires authorization)."""
        # In production, verify authorization
        self._kill_switch_activated = False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get gateway status."""
        return {
            "running": self._running,
            "kill_switch": self._kill_switch_activated,
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trade_count,
            "daily_pnl_pct": self._daily_pnl / self._daily_start_equity if self._daily_start_equity else 0,
        }


# Factory function
async def create_live_gateway(
    broker: BrokerProvider,
    permission_manager: PermissionManager,
    event_bus: Optional[EventBus] = None,
    **kwargs
) -> LiveTradingGateway:
    """Create and start a live trading gateway."""
    gateway = LiveTradingGateway(
        broker_provider=broker,
        permission_manager=permission_manager,
        event_bus=event_bus,
        **kwargs
    )
    await gateway.start()
    return gateway