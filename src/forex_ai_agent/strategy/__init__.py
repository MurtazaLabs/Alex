"""Strategy Package - Versioning, A/B testing, regression detection, marketplace."""

from __future__ import annotations

from .versioning import (
    VersionType,
    StrategyVersion,
    StrategyDiff,
    StrategyVersionManager,
    create_version_manager,
)
from .ab_testing import (
    TrafficSplit,
    ExperimentStatus,
    VariantConfig,
    ABTestConfig,
    ABTestExperiment,
    ABTestManager,
    create_ab_test_manager,
)
from .regression import (
    RegressionType,
    AlertSeverity,
    RegressionAlert,
    PerformanceSnapshot,
    RegressionConfig,
    RegressionDetector,
    RegressionMonitor,
    create_regression_monitor,
)
from .marketplace import (
    StrategyListing,
    ListingStatus,
    ActivationStatus,
    ActivationRequest,
    StrategyReview,
    MarketplaceConfig,
    StrategyMarketplace,
    create_report_generator,
)
from .versioning import (
    StrategyVersionManager,
    create_version_manager,
)
from forex_ai_agent.research.multi_agent import (
    ParallelResearchManager,
    ResourceAllocator,
    ResourceType,
    TaskDependencyManager,
    HypothesisTournament,
    TournamentConfig,
    TournamentMode,
    TournamentStage,
    PatternMiner,
    PatternType,
    DiscoveredPattern,
    MarketConditionClusterer,
    ConceptProposer,
    ConceptDiscovery,
    ResearchReportGenerator,
    ReportConfig,
    LatexGenerator,
    ExecutiveSummaryGenerator,
    create_parallel_manager,
    create_tournament,
    create_concept_discovery,
    create_report_generator,
)

__all__ = [
    # Versioning
    "VersionType",
    "StrategyVersion",
    "StrategyDiff",
    "StrategyVersionManager",
    "create_version_manager",
    
    # A/B Testing
    "TrafficSplit",
    "ExperimentStatus",
    "VariantConfig",
    "ABTestConfig",
    "ABTestExperiment",
    "ABTestManager",
    "create_ab_test_manager",
    
    # Regression Detection
    "RegressionType",
    "AlertSeverity",
    "RegressionAlert",
    "PerformanceSnapshot",
    "RegressionConfig",
    "RegressionDetector",
    "RegressionMonitor",
    "create_regression_monitor",
    
    # Marketplace
    "StrategyListing",
    "ListingStatus",
    "ActivationStatus",
    "ActivationRequest",
    "StrategyReview",
    "MarketplaceConfig",
    "StrategyMarketplace",
    "create_report_generator",
    
    # Versioning
    "StrategyVersionManager",
    "create_version_manager",
    
    # Multi-Agent Research
    "ParallelResearchManager",
    "ResourceAllocator",
    "ResourceType",
    "TaskDependencyManager",
    "HypothesisTournament",
    "TournamentConfig",
    "TournamentMode",
    "TournamentStage",
    "PatternMiner",
    "PatternType",
    "DiscoveredPattern",
    "MarketConditionClusterer",
    "ConceptProposer",
    "ConceptDiscovery",
    "ResearchReportGenerator",
    "ReportConfig",
    "LatexGenerator",
    "ExecutiveSummaryGenerator",
    "create_parallel_manager",
    "create_tournament",
    "create_concept_discovery",
    "create_report_generator",
]