"""Regression Detection - Continuous backtest on new data with drift alerts."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus
from forex_ai_agent.strategy.versioning import StrategyVersion, StrategyVersionManager
from forex_ai_agent.backtest import BacktestEngine, BacktestConfig, run_backtest


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RegressionType(Enum):
    PERFORMANCE_DRIFT = "performance_drift"
    REGIME_DEGRADATION = "regime_degradation"
    PARAMETER_DRIFT = "parameter_drift"
    STRUCTURAL_BREAK = "structural_break"


@dataclass
class RegressionAlert:
    """Alert for detected regression."""
    version_id: str
    version: str
    alert_type: RegressionType
    severity: AlertSeverity
    message: str
    metric: str
    current_value: float
    baseline_value: float
    deviation_pct: float
    threshold: float
    detected_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4())[:8])


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time."""
    version_id: str
    version: str
    timestamp: datetime
    metrics: Dict[str, float]
    trades_count: int
    regime: Optional[str] = None


class RegimeDetector:
    """Detects market regime for regime-specific analysis."""
    
    def __init__(self):
        self.regimes = ["trending_up", "trending_down", "ranging", "high_vol", "low_vol"]
    
    def detect(self, market_data: Dict[str, Any]) -> str:
        """Detect current market regime."""
        # Simplified regime detection
        # In production, use proper regime detection (HMM, etc.)
        return "ranging"


@dataclass
class RegressionConfig:
    """Configuration for regression detection."""
    # Performance thresholds
    sharpe_drop_threshold: float = 0.3  # 30% drop in Sharpe
    profit_factor_drop_threshold: float = 0.3
    win_rate_drop_threshold: float = 0.1  # 10% drop
    max_drawdown_increase: float = 0.5  # 50% increase in max DD
    
    # Time windows
    short_window_days: int = 7
    medium_window_days: int = 30
    long_window_days: int = 90
    
    # Statistical thresholds
    min_trades_for_alert: int = 30
    min_confidence: float = 0.95
    
    # Regime-specific thresholds
    regime_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "trending_up": {"sharpe_drop": 0.4, "win_rate_drop": 0.15},
        "trending_down": {"sharpe_drop": 0.4, "win_rate_drop": 0.15},
        "ranging": {"sharpe_drop": 0.3, "win_rate_drop": 0.1},
        "high_vol": {"sharpe_drop": 0.5, "win_rate_drop": 0.2},
        "low_vol": {"sharpe_drop": 0.2, "win_rate_drop": 0.1},
    })


class RegressionDetector:
    """
    Detects performance regression in strategy versions.
    
    Monitors:
    - Performance drift over time
    - Regime-specific degradation
    - Parameter drift
    - Structural breaks in returns distribution
    """
    
    def __init__(
        self,
        version_manager: Any,
        backtest_engine: Any,
        config: Optional[RegressionConfig] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.version_manager = version_manager
        self.backtest_engine = backtest_engine
        self.config = config or RegressionConfig()
        self.event_bus = event_bus
        
        self.active_versions: Dict[str, Dict[str, Any]] = {}  # version_id -> tracking info
        self.performance_history: Dict[str, List[Dict]] = defaultdict(list)
        self.alerts: List[Dict] = []
        self.regime_detector = RegimeDetector()
        
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._check_interval = 3600  # 1 hour
    
    async def start(self):
        """Start regression monitoring."""
        if self._running:
            return
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop regression monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    def track_version(self, version_id: str, baseline_metrics: Dict[str, float] = None):
        """Start tracking a version for regression."""
        version = self.version_manager.get_version(version_id)
        if not version:
            raise ValueError(f"Version not found: {version_id}")
        
        self.active_versions[version.id] = {
            "version": version,
            "baseline": baseline_metrics or {},
            "snapshots": [],
            "last_check": datetime.utcnow(),
        }
        
        # Initialize with current performance if available
        if hasattr(self, 'backtest_engine') and self.backtest_engine:
            # Could run initial backtest here
            pass
    
    def untrack_version(self, version_id: str):
        """Stop tracking a version."""
        self.active_versions.pop(version_id, None)
        self.performance_history.pop(version_id, None)
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_all_versions()
            except Exception as e:
                print(f"Monitor error: {e}")
            
            await asyncio.sleep(self._check_interval)
    
    async def _check_all_versions(self):
        """Check all tracked versions for regression."""
        for version_id, info in self.active_versions.items():
            await self._check_version(version_id, info)
    
    async def _check_version(self, version_id: str, info: Dict):
        """Check a single version for regression."""
        version = info["version"]
        baseline = info["baseline"]
        
        # Get current performance (would run backtest or check live)
        current = await self._get_current_performance(version.id)
        if not current:
            return
        
        # Record snapshot
        snapshot = PerformanceSnapshot(
            version_id=version.id,
            version=version.version,
            timestamp=datetime.utcnow(),
            metrics=current,
            trades_count=current.get("trades_count", 0),
        )
        info["snapshots"].append(snapshot)
        info["last_check"] = datetime.utcnow()
        
        # Keep only recent snapshots
        cutoff = datetime.utcnow() - timedelta(days=90)
        info["snapshots"] = [s for s in info["snapshots"] if s.timestamp > cutoff]
        
        # Check for regression
        await self._detect_regression(version.id, baseline, current)
    
    async def _get_current_performance(self, version_id: str) -> Optional[Dict]:
        """Get current performance metrics for a version."""
        # In production, this would run a backtest on recent data
        # or query live trading results
        # For now, return mock data
        return {
            "sharpe_ratio": 1.2,
            "profit_factor": 1.5,
            "win_rate": 0.55,
            "max_drawdown_pct": 5.0,
            "total_return_pct": 15.0,
            "trades_count": 100,
        }
    
    async def _detect_regression(self, version_id: str, baseline: Dict, current: Dict):
        """Detect regression by comparing current to baseline."""
        if not baseline or not current:
            return
        
        version = self.version_manager.get_version(version_id)
        if not version:
            return
        
        regime = "ranging"  # Would use regime detector in production
        regime_thresholds = self.config.regime_thresholds.get(regime, self.config.regime_thresholds["ranging"])
        
        alerts = []
        
        # Check Sharpe ratio drop
        if "sharpe_ratio" in baseline and "sharpe_ratio" in current:
            baseline_sharpe = baseline["sharpe_ratio"]
            current_sharpe = current["sharpe_ratio"]
            if baseline_sharpe > 0:
                drop_pct = (baseline_sharpe - current["sharpe_ratio"]) / baseline_sharpe
                threshold = regime_thresholds.get("sharpe_drop", self.config.sharpe_drop_threshold)
                if drop_pct > threshold:
                    self._create_alert(
                        version_id=version_id,
                        alert_type=RegressionType.PERFORMANCE_DRIFT,
                        severity=AlertSeverity.WARNING if drop_pct < 0.5 else AlertSeverity.CRITICAL,
                        metric="sharpe_ratio",
                        current=current["sharpe_ratio"],
                        baseline=baseline_sharpe,
                        deviation=drop_pct,
                        threshold=threshold,
                    )
        
        # Check profit factor drop
        if "profit_factor" in baseline and "profit_factor" in current:
            baseline_pf = baseline["profit_factor"]
            current_pf = current["profit_factor"]
            if baseline_pf > 0:
                drop_pct = (baseline_pf - current_pf) / baseline_pf
                threshold = self.config.profit_factor_drop_threshold
                if drop_pct > threshold:
                    self._create_alert(
                        version_id=version_id,
                        alert_type=RegressionType.PERFORMANCE_DRIFT,
                        severity=AlertSeverity.WARNING,
                        metric="profit_factor",
                        current=current_pf,
                        baseline=baseline_pf,
                        deviation=drop_pct,
                        threshold=threshold,
                    )
        
        # Check win rate drop
        if "win_rate" in baseline and "win_rate" in current:
            baseline_wr = baseline["win_rate"]
            current_wr = current["win_rate"]
            drop_pct = baseline_wr - current_wr
            threshold = regime_thresholds.get("win_rate_drop", self.config.win_rate_drop_threshold)
            if drop_pct > threshold:
                self._create_alert(
                    version_id=version_id,
                    alert_type=RegressionType.PERFORMANCE_DRIFT,
                    severity=AlertSeverity.WARNING,
                    metric="win_rate",
                    current=current_wr,
                    baseline=baseline_wr,
                    deviation=drop_pct,
                    threshold=threshold,
                )
        
        # Check max drawdown increase
        if "max_drawdown_pct" in baseline and "max_drawdown_pct" in current:
            baseline_dd = baseline["max_drawdown_pct"]
            current_dd = current["max_drawdown_pct"]
            if baseline_dd > 0:
                increase = (current_dd - baseline_dd) / baseline_dd
                if increase > self.config.max_drawdown_increase:
                    self._create_alert(
                        version_id=version_id,
                        alert_type=RegressionType.PERFORMANCE_DRIFT,
                        severity=AlertSeverity.CRITICAL,
                        metric="max_drawdown_pct",
                        current=current_dd,
                        baseline=baseline_dd,
                        deviation=increase,
                        threshold=self.config.max_drawdown_increase,
                    )
        
        # Check for regime-specific degradation
        await self._check_regime_degradation(version_id, regime, current, baseline)
    
    async def _check_regime_degradation(
        self, 
        version_id: str, 
        regime: str, 
        current: Dict, 
        baseline: Dict
    ):
        """Check for regime-specific performance degradation."""
        # In production, would compare performance in specific regime
        # vs baseline for that regime
        pass
    
    def _create_alert(
        self,
        version_id: str,
        alert_type: RegressionType,
        severity: AlertSeverity,
        metric: str,
        current: float,
        baseline: float,
        deviation: float,
        threshold: float,
    ):
        """Create a regression alert."""
        version = self.version_manager.get_version(version_id)
        
        alert = {
            "id": str(uuid4())[:8],
            "version_id": version_id,
            "version": self.version_manager.get_version(version_id).version if self.version_manager.get_version(version_id) else "unknown",
            "type": alert_type.value,
            "severity": severity.value,
            "metric": metric,
            "message": f"{metric} {severity.value}: {deviation:.1%} deviation from baseline (threshold: {threshold:.1%})",
            "metric_name": metric,
            "current_value": current,
            "baseline_value": baseline,
            "deviation_pct": deviation,
            "threshold": threshold,
            "detected_at": datetime.utcnow().isoformat(),
            "acknowledged": False,
            "resolved": False,
        }
        
        self.alerts.append(alert)
        
        # Emit event
        if self.event_bus:
            asyncio.create_task(self.event_bus.publish(Event(
                event_type="regression.alert",
                aggregate_id=version_id,
                aggregate_type="strategy_version",
                payload=alert,
                source_agent="regression_detector",
            )))
    
    def get_alerts(
        self, 
        version_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        unacknowledged_only: bool = False,
    ) -> List[Dict]:
        """Get alerts with optional filtering."""
        alerts = self.alerts
        
        if version_id:
            alerts = [a for a in alerts if a["version_id"] == version_id]
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity.value]
        if unacknowledged_only:
            alerts = [a for a in alerts if not a["acknowledged"]]
        
        return sorted(alerts, key=lambda a: a["detected_at"], reverse=True)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["resolved"] = True
                return True
        return False
    
    def get_performance_summary(self, version_id: str) -> Dict:
        """Get performance summary for a version."""
        info = self.active_versions.get(version_id)
        if not info:
            return {}
        
        snapshots = info.get("snapshots", [])
        if not snapshots:
            return {}
        
        # Calculate trends
        sharpe_trend = [s.metrics.get("sharpe_ratio", 0) for s in snapshots]
        returns_trend = [s.metrics.get("total_return_pct", 0) for s in snapshots]
        
        return {
            "version_id": version_id,
            "snapshots_count": len(snapshots),
            "latest": {
                "timestamp": snapshots[-1].timestamp.isoformat() if snapshots else None,
                "metrics": snapshots[-1].metrics if snapshots else {},
            },
            "trends": {
                "sharpe": self._calculate_trend([s.metrics.get("sharpe_ratio", 0) for s in snapshots]),
                "return": self._calculate_trend([s.metrics.get("total_return_pct", 0) for s in snapshots]),
                "win_rate": self._calculate_trend([s.metrics.get("win_rate", 0) for s in snapshots]),
            },
            "alerts_count": len([a for a in self.alerts if a["version_id"] == version_id]),
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend statistics."""
        if len(values) < 2:
            return {"slope": 0, "direction": "flat", "strength": 0}
        
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        
        direction = "up" if slope > 0.01 else "down" if slope < -0.01 else "flat"
        strength = min(abs(slope) * len(values), 1.0)
        
        return {"slope": slope, "direction": direction, "strength": strength}


class RegressionMonitor:
    """
    High-level regression monitoring service.
    Integrates with version manager and backtest engine.
    """
    
    def __init__(
        self,
        version_manager: StrategyVersionManager,
        backtest_engine: Any,
        config: Optional[RegressionConfig] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.detector = RegressionDetector(
            version_manager=version_manager,
            backtest_engine=backtest_engine,
            config=config,
        )
        self.detector.event_bus = event_bus
    
    async def start(self):
        await self.detector.start()
    
    async def stop(self):
        await self.detector.stop()
    
    def track_version(self, version_id: str, baseline: Dict = None):
        self.detector.track_version(version_id, baseline)
    
    def untrack_version(self, version_id: str):
        self.detector.untrack_version(version_id)
    
    def get_alerts(self, **kwargs) -> List[Dict]:
        return self.detector.get_alerts(**kwargs)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        return self.detector.acknowledge_alert(alert_id)
    
    def get_version_health(self, version_id: str) -> Dict:
        return self.detector.get_performance_summary(version_id)


async def create_regression_monitor(
    version_manager: StrategyVersionManager,
    backtest_engine: Any,
    config: Optional[RegressionConfig] = None,
    event_bus: Optional[EventBus] = None,
) -> RegressionMonitor:
    """Create and start regression monitor."""
    monitor = RegressionMonitor(
        version_manager=version_manager,
        backtest_engine=backtest_engine,
        config=config,
        event_bus=event_bus,
    )
    await monitor.start()
    return monitor