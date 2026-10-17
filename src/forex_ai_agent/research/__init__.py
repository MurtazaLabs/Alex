"""Research System Package - Multi-agent research pipeline for Forex AI."""

from __future__ import annotations

from .models import (
    AgentType,
    ResearchPhase,
    ResearchStatus,
    Source,
    SourceType,
    Claim,
    ClaimType,
    Hypothesis,
    ResearchTask,
    ResearchSession,
    Concept,
    KnowledgeEdge,
    CritiqueResult,
    StatisticalValidation,
)

from .orchestrator import (
    ResearchOrchestrator,
    OrchestratorConfig,
    create_orchestrator,
)
from .pipeline import (
    ResearchPipeline,
    PipelineConfig,
    create_pipeline,
)

from .agents import (
    BaseResearchAgent,
    ResearcherAgent,
    create_researcher_agent,
    QuantAgent,
    QuantConfig,
    create_quant_agent,
    BacktestAgent,
    BacktestAgentConfig,
    create_backtest_agent,
    CriticAgent,
    create_critic_agent,
    StatisticalAgent,
    create_statistical_agent,
    KnowledgeAgent,
    create_knowledge_agent,
)

from .neo4j_knowledge import (
    Neo4jConfig,
    Neo4jKnowledgeGraph,
    Neo4jKnowledgeAgent,
    create_neo4j_knowledge_graph,
    create_neo4j_knowledge_agent,
)

from .multi_agent import (
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
    ResearchReportGenerator,
    ReportConfig,
    LatexGenerator,
    ExecutiveSummaryGenerator,
    create_parallel_manager,
    create_tournament,
    create_concept_discovery,
    create_report_generator,
    ConceptDiscovery,
    create_concept_discovery,
)

__all__ = [
    # Models
    "AgentType",
    "ResearchPhase",
    "ResearchStatus",
    "Source",
    "SourceType",
    "Claim",
    "ClaimType",
    "Hypothesis",
    "ResearchTask",
    "ResearchSession",
    "Concept",
    "KnowledgeEdge",
    "CritiqueResult",
    "StatisticalValidation",
    
    # Orchestrator
    "ResearchOrchestrator",
    "OrchestratorConfig",
    "create_orchestrator",
    
    # Agents
    "BaseResearchAgent",
    "ResearcherAgent",
    "create_researcher_agent",
    "QuantAgent",
    "QuantConfig",
    "create_quant_agent",
    "BacktestAgent",
    "BacktestAgentConfig",
    "create_backtest_agent",
    "CriticAgent",
    "create_critic_agent",
    "StatisticalAgent",
    "create_statistical_agent",
    "KnowledgeAgent",
    "create_knowledge_agent",
    
    # Neo4j Knowledge Graph
    "Neo4jConfig",
    "Neo4jKnowledgeGraph",
    "Neo4jKnowledgeAgent",
    "create_neo4j_knowledge_graph",
    "create_neo4j_knowledge_agent",
    
    # Pipeline
    "ResearchPipeline",
    "PipelineConfig",
    "create_pipeline",
    "quick_research",
    
    # Multi-Agent
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
    "ResearchReportGenerator",
    "ReportConfig",
    "LatexGenerator",
    "ExecutiveSummaryGenerator",
    "create_parallel_manager",
    "create_tournament",
    "create_concept_discovery",
    "create_report_generator",
    "ConceptDiscovery",
    "create_concept_discovery",
]