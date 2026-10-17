"""Multi-Agent Research Package - Phase 8: Multi-Agent Research."""

from __future__ import annotations

from .parallel_manager import (
    ParallelResearchManager,
    ResourceAllocator,
    ResourceLimit,
    ResourceAllocation,
    ResourceType,
    TaskDependencyManager,
    TaskDependency,
    TaskPriority,
    ScheduledTask,
    create_parallel_manager,
)
from .tournament import (
    HypothesisTournament,
    TournamentConfig,
    TournamentMode,
    TournamentStage,
    Match,
    MatchResult,
    TournamentParticipant,
    TournamentFactory,
    create_tournament,
)
from .concept_discovery import (
    PatternMiner,
    PatternType,
    DiscoveredPattern,
    MarketConditionClusterer,
    ConceptProposer,
    ConceptDiscovery,
    create_concept_discovery,
)
from .report_generator import (
    ResearchReportGenerator,
    ReportConfig,
    LatexGenerator,
    ExecutiveSummaryGenerator,
    create_report_generator,
)

__all__ = [
    "ParallelResearchManager",
    "ResourceAllocator",
    "ResourceLimit",
    "ResourceAllocation",
    "ResourceType",
    "TaskDependencyManager",
    "TaskDependency",
    "TaskPriority",
    "ScheduledTask",
    "create_parallel_manager",
    "HypothesisTournament",
    "TournamentConfig",
    "TournamentMode",
    "TournamentStage",
    "Match",
    "MatchResult",
    "TournamentParticipant",
    "TournamentFactory",
    "create_tournament",
    "PatternMiner",
    "PatternType",
    "DiscoveredPattern",
    "MarketConditionClusterer",
    "ConceptProposer",
    "ConceptDiscovery",
    "create_concept_discovery",
    "ResearchReportGenerator",
    "ReportConfig",
    "LatexGenerator",
    "ExecutiveSummaryGenerator",
    "create_report_generator",
]