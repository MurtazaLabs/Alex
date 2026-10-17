"""A/B Testing Framework - Parallel paper trading of strategy versions with statistical comparison."""

from __future__ import annotations

import asyncio
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus
from forex_ai_agent.paper import PaperTradingEngine, PaperTradingMode
from forex_ai_agent.strategy.versioning import StrategyVersion, StrategyVersionManager


class TrafficSplit(Enum):
    """How to split traffic between variants."""
    EQUAL = "equal"           # 50/50 split
    WEIGHTED = "weighted"     # Custom weights
    PROPORTIONAL = "proportional"  # Based on performance


@dataclass
class VariantConfig:
    """Configuration for a test variant."""
    version_id: str
    name: str
    traffic_percentage: float = 50.0
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0 <= self.traffic_percentage <= 100:
            raise ValueError("Traffic percentage must be between 0 and 100")


class TrafficSplit(Enum):
    """How to split traffic between variants."""
    EQUAL = "equal"           # 50/50 split
    WEIGHTED = "weighted"     # Custom weights
    PROPORTIONAL = "proportional"  # Based on performance


class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class VariantConfig:
    """Configuration for a test variant."""
    version_id: str
    name: str
    traffic_percentage: float = 50.0
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0 <= self.traffic_percentage <= 100:
            raise ValueError("Traffic percentage must be between 0 and 100")


class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ExperimentResult:
    """Results of an A/B test experiment."""
    experiment_id: str
    variant_a_id: str
    variant_b_id: str
    status: ExperimentStatus
    
    # Metrics
    variant_a_metrics: Dict[str, float] = field(default_factory=dict)
    variant_b_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Statistical results
    statistical_significance: float = 0.0
    p_value: float = 1.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    winner: Optional[str] = None
    confidence_level: float = 0.95
    
    # Sample sizes
    sample_size_a: int = 0
    sample_size_b: int = 0
    
    # Duration
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_hours: float = 0.0
    
    # Decision
    recommendation: str = ""  # "promote_a", "promote_b", "keep_both", "stop_both"
    confidence: float = 0.0


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    description: str = ""
    
    # Variants
    variants: List[VariantConfig] = field(default_factory=list)
    
    # Experiment settings
    min_sample_size: int = 100
    max_duration_hours: int = 168  # 1 week
    significance_level: float = 0.05
    minimum_effect_size: float = 0.05  # Minimum detectable effect
    
    # Traffic split
    traffic_split: TrafficSplit = TrafficSplit.EQUAL
    
    # Stopping rules
    early_stopping: bool = True
    min_running_hours: int = 24
    max_drawdown_stop: float = 0.10  # Stop if variant loses > 10%
    
    # Statistical settings
    confidence_level: float = 0.95
    power: float = 0.80


class ABTestExperiment:
    """A single A/B test experiment between two variants."""
    
    def __init__(
        self,
        config: ABTestConfig,
        version_manager: Any,
        paper_engine_factory: Callable,
        event_bus=None,
    ):
        self.config = config
        self.version_manager = version_manager
        self.paper_engine_factory = paper_engine_factory
        self.event_bus = event_bus
        
        self.status = ExperimentStatus.PENDING
        self.result: Optional[ExperimentResult] = None
        self.paper_engines: Dict[str, Any] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.variant_performance: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.variant_trades: Dict[str, List[Dict]] = defaultdict(list)
    
    async def start(self) -> bool:
        """Start the A/B test."""
        if len(self.config.variants) < 2:
            raise ValueError("Need at least 2 variants")
        
        self.status = ExperimentStatus.RUNNING
        self.result = ExperimentResult(
            experiment_id=self.config.id,
            variant_a_id=self.config.variants[0].version_id,
            variant_b_id=self.config.variants[1].version_id,
            status=ExperimentStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        
        # Create paper trading engines for each variant
        for variant in self.config.variants:
            engine = self.paper_engine_factory(variant.version_id, variant.config_overrides)
            self.paper_engines[variant.version_id] = engine
            await engine.start()
        
        self._running = True
        self._task = asyncio.create_task(self._run_experiment())
        
        return True
    
    async def stop(self, reason: str = "Manual stop"):
        """Stop the experiment."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        # Stop all engines
        for engine in self.paper_engines.values():
            await engine.stop()
        
        self.status = ExperimentStatus.STOPPED
        if self.result:
            self.result.status = ExperimentStatus.STOPPED
            self.result.completed_at = datetime.utcnow()
            if self.result.started_at:
                self.result.duration_hours = (datetime.utcnow() - self.result.started_at).total_seconds() / 3600
    
    async def _run_experiment(self):
        """Main experiment loop."""
        try:
            while self._running:
                await self._process_market_data()
                await self._check_stopping_rules()
                await self._update_statistics()
                
                if self._should_stop():
                    await self._complete_experiment()
                    break
                
                await asyncio.sleep(60)  # Check every minute
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.status = ExperimentStatus.FAILED
            if self.result:
                self.result.status = ExperimentStatus.FAILED
    
    async def _process_market_data(self):
        """Process market data through both variants."""
        # In real implementation, this would receive market data
        # and route to appropriate variant engines
        pass
    
    async def _check_stopping_rules(self):
        """Check if experiment should stop early."""
        if not self.result or not self.config.early_stopping:
            return
        
        # Check minimum running time
        if self.result.started_at:
            hours_run = (datetime.utcnow() - self.result.started_at).total_seconds() / 3600
            if hours_run < self.config.min_running_hours:
                return
        
        # Check max drawdown
        for variant in self.config.variants:
            metrics = self._get_variant_metrics(variant.version_id)
            if metrics.get("max_drawdown_pct", 0) > self.config.max_drawdown_stop * 100:
                await self.stop(f"Max drawdown exceeded for {variant.name}")
                return
        
        # Check minimum sample size
        if (self.result.sample_size_a >= self.config.min_sample_size and 
            self.result.sample_size_b >= self.config.min_sample_size):
            # Run statistical test
            await self._run_statistical_test()
    
    async def _run_statistical_test(self):
        """Run statistical significance test."""
        # Simplified t-test for demonstration
        # In production, use scipy.stats or proper statistical library
        pass
    
    def _should_stop(self) -> bool:
        """Check if experiment should stop."""
        if not self._running:
            return True
        
        if self.result and self.result.status in [ExperimentStatus.COMPLETED, ExperimentStatus.STOPPED, ExperimentStatus.FAILED]:
            return True
        
        # Check max duration
        if self.result and self.result.started_at:
            hours_run = (datetime.utcnow() - self.result.started_at).total_seconds() / 3600
            if hours_run >= self.config.max_duration_hours:
                return True
        
        return False
    
    async def _complete_experiment(self):
        """Complete the experiment and determine winner."""
        self.status = ExperimentStatus.COMPLETED
        if self.result:
            self.result.status = ExperimentStatus.COMPLETED
            self.result.completed_at = datetime.utcnow()
            if self.result.started_at:
                self.result.duration_hours = (self.result.completed_at - self.result.started_at).total_seconds() / 3600
            
            # Determine winner based on metrics
            await self._determine_winner()
    
    async def _determine_winner(self):
        """Determine the winning variant."""
        if not self.result:
            return
        
        # Get final metrics
        metrics_a = self._get_variant_metrics(self.result.variant_a_id)
        metrics_b = self._get_variant_metrics(self.result.variant_b_id)
        
        self.result.variant_a_metrics = metrics_a
        self.result.variant_b_metrics = metrics_b
        
        # Simple comparison - in production use proper statistical tests
        score_a = self._calculate_score(metrics_a)
        score_b = self._calculate_score(metrics_b)
        
        if score_a > score_b:
            self.result.winner = self.result.variant_a_id
            self.result.recommendation = "promote_a"
        elif score_b > score_a:
            self.result.winner = self.result.variant_b_id
            self.result.recommendation = "promote_b"
        else:
            self.result.recommendation = "keep_both"
        
        self.result.confidence = min(abs(score_a - score_b) * 2, 1.0)
    
    def _calculate_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite score for a variant."""
        # Weighted score: Sharpe * 0.3 + Profit Factor * 0.2 + Win Rate * 0.2 + Return * 0.3
        sharpe = metrics.get("sharpe_ratio", 0)
        profit_factor = metrics.get("profit_factor", 1)
        win_rate = metrics.get("win_rate", 0.5)
        total_return = metrics.get("total_return_pct", 0)
        
        score = (sharpe * 0.3 + 
                min(profit_factor, 5) * 0.2 + 
                win_rate * 0.2 + 
                max(total_return, -50) / 50 * 0.3)
        return score
    
    def _get_variant_metrics(self, version_id: str) -> Dict[str, float]:
        """Get current metrics for a variant."""
        # In production, this would query the paper trading engine
        return {}
    
    async def _update_statistics(self):
        """Update experiment statistics."""
        pass
    
    async def _process_market_data(self):
        """Process market data through both variants."""
        # In real implementation, this would receive market data
        # and route to appropriate variant engines
        pass


class ABTestManager:
    """
    Manages multiple A/B test experiments with automatic promotion/demotion.
    """
    
    def __init__(
        self,
        version_manager: Any,
        paper_engine_factory: Callable,
        event_bus: Optional[EventBus] = None,
        max_concurrent_experiments: int = 5,
    ):
        self.version_manager = version_manager
        self.paper_engine_factory = paper_engine_factory
        self.event_bus = event_bus or InMemoryEventBus()
        self.max_concurrent = max_concurrent_experiments
        
        self.experiments: Dict[str, ABTestExperiment] = {}
        self.running_experiments: Dict[str, ABTestExperiment] = {}
        self.completed_experiments: Dict[str, ABTestExperiment] = {}
        
        # Auto-promotion rules
        self.auto_promote: bool = True
        self.promotion_threshold: float = 0.95  # Confidence threshold
        self.demotion_threshold: float = 0.05   # Demote if confidence < 5%
        
        # Callbacks
        self.on_experiment_complete: Optional[Callable] = None
        self.on_promotion: Optional[Callable] = None
        self.on_demotion: Optional[Callable] = None
    
    async def create_experiment(
        self,
        name: str,
        description: str,
        variant_a: VariantConfig,
        variant_b: VariantConfig,
        config: Optional[ABTestConfig] = None,
    ) -> ABTestExperiment:
        """Create a new A/B test experiment."""
        if len(self.running_experiments) >= self.max_concurrent_experiments:
            raise ValueError("Maximum concurrent experiments reached")
        
        config = config or ABTestConfig(
            name=name,
            description=description,
            variants=[variant_a, variant_b],
        )
        
        experiment = ABTestExperiment(
            config=config,
            version_manager=self.version_manager,
            paper_engine_factory=self.paper_engine_factory,
            event_bus=self.event_bus,
        )
        
        self.experiments[experiment.config.id] = experiment
        return experiment
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return False
        
        if len(self.running_experiments) >= self.max_concurrent_experiments:
            return False
        
        success = await experiment.start()
        if success:
            self.running_experiments[experiment_id] = experiment
            return True
        return False
    
    async def stop_experiment(self, experiment_id: str, reason: str = "Manual") -> bool:
        """Stop an experiment."""
        experiment = self.running_experiments.get(experiment_id) or self.experiments.get(experiment_id)
        if not experiment:
            return False
        
        await experiment.stop(reason)
        self.running_experiments.pop(experiment_id, None)
        self.completed_experiments[experiment_id] = experiment
        return True
    
    async def _create_paper_engine(self, version_id: str, overrides: Dict) -> Any:
        """Create a paper trading engine for a variant."""
        # This would be implemented with the actual paper trading engine
        # For now, return a mock
        class MockEngine:
            async def start(self): pass
            async def stop(self): pass
        return MockEngine()
    
    async def check_promotions(self):
        """Check running experiments for promotion/demotion opportunities."""
        for exp_id, experiment in list(self.running_experiments.items()):
            if experiment.result and experiment.result.confidence >= self.promotion_threshold:
                if experiment.result.recommendation == "promote_a":
                    await self._promote_variant(experiment.result.variant_a_id, experiment.config.id)
                elif experiment.result.recommendation == "promote_b":
                    await self._promote_variant(experiment.result.variant_b_id, experiment.config.id)
            
            # Check for demotion
            if experiment.result and experiment.result.confidence <= self.demotion_threshold:
                if experiment.result.recommendation == "promote_a":
                    await self._demote_variant(experiment.result.variant_b_id, experiment.config.id)
                elif experiment.result.recommendation == "promote_b":
                    await self._demote_variant(experiment.result.variant_a_id, experiment.config.id)
    
    async def _promote_variant(self, version_id: str, experiment_id: str):
        """Promote a variant to active."""
        self.version_manager.set_active(version_id)
        if self.on_promotion:
            await self.on_promotion(version_id, experiment_id)
    
    async def _demote_variant(self, version_id: str, experiment_id: str):
        """Demote a variant (archive)."""
        version = self.version_manager.get_version(version_id)
        if version:
            version.status = "archived"
        if self.on_demotion:
            await self.on_demotion(version_id, experiment_id)
    
    def get_experiment_status(self, experiment_id: str) -> Optional[Dict]:
        """Get status of an experiment."""
        exp = self.experiments.get(experiment_id) or self.running_experiments.get(experiment_id) or self.completed_experiments.get(experiment_id)
        if not exp:
            return None
        
        return {
            "id": exp.config.id,
            "name": exp.config.name,
            "status": exp.status.value,
            "result": exp.result.__dict__ if exp.result else None,
        }
    
    def list_experiments(self) -> List[Dict]:
        """List all experiments."""
        all_experiments = {**self.experiments, **self.running_experiments, **self.completed_experiments}
        return [self.get_experiment_status(eid) for eid in all_experiments]


# Factory
async def create_ab_test_manager(
    version_manager: Any,
    paper_engine_factory: Callable,
    event_bus: Optional[EventBus] = None,
) -> ABTestManager:
    """Create and initialize A/B test manager."""
    manager = ABTestManager(
        version_manager=version_manager,
        paper_engine_factory=paper_engine_factory,
        event_bus=event_bus,
    )
    return manager