"""Research System Core Models and Types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4


class ResearchStatus(Enum):
    """Status of a research task."""
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    AWAITING_REVIEW = "AWAITING_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ResearchPhase(Enum):
    """Phase of research pipeline."""
    INITIALIZATION = "INITIALIZATION"
    LITERATURE_REVIEW = "LITERATURE_REVIEW"
    HYPOTHESIS_FORMATION = "HYPOTHESIS_FORMATION"
    QUANT_FORMALIZATION = "QUANT_FORMALIZATION"
    BACKTESTING = "BACKTESTING"
    CRITIQUE = "CRITIQUE"
    STATISTICAL_VALIDATION = "STATISTICAL_VALIDATION"
    KNOWLEDGE_INTEGRATION = "KNOWLEDGE_INTEGRATION"
    REPORT_GENERATION = "REPORT_GENERATION"
    COMPLETED = "COMPLETED"


class AgentType(Enum):
    """Types of research agents."""
    ORCHESTRATOR = "ORCHESTRATOR"
    RESEARCHER = "RESEARCHER"
    QUANT = "QUANT"
    BACKTEST = "BACKTEST"
    CRITIC = "CRITIC"
    STATISTICAL = "STATISTICAL"
    KNOWLEDGE = "KNOWLEDGE"
    MEMORY = "MEMORY"


class SourceType(Enum):
    """Types of information sources."""
    ACADEMIC_PAPER = "ACADEMIC_PAPER"
    BOOK = "BOOK"
    BLOG_POST = "BLOG_POST"
    DOCUMENTATION = "DOCUMENTATION"
    NEWS_ARTICLE = "NEWS_ARTICLE"
    FORUM_POST = "FORUM_POST"
    VIDEO = "VIDEO"
    INTERNAL_NOTE = "INTERNAL_NOTE"
    BACKTEST_RESULT = "BACKTEST_RESULT"


class ClaimType(Enum):
    """Types of claims extracted from sources."""
    EMPIRICAL_FINDING = "EMPIRICAL_FINDING"
    THEORETICAL_CLAIM = "THEORETICAL_CLAIM"
    METHODOLOGY = "METHODOLOGY"
    PARAMETER_VALUE = "PARAMETER_VALUE"
    MARKET_OBSERVATION = "MARKET_OBSERVATION"
    STRATEGY_RULE = "STRATEGY_RULE"
    RISK_FACTOR = "RISK_FACTOR"
    PERFORMANCE_METRIC = "PERFORMANCE_METRIC"


@dataclass
class Source:
    """Information source with metadata."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    type: SourceType = SourceType.ACADEMIC_PAPER
    title: str = ""
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[datetime] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    abstract: str = ""
    credibility_score: float = 0.5  # 0-1
    tags: List[str] = field(default_factory=list)
    content_hash: str = ""
    retrieved_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Claim:
    """Extracted claim with citation."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    text: str = ""
    claim_type: ClaimType = ClaimType.EMPIRICAL_FINDING
    source_id: str = ""
    confidence: float = 0.5  # 0-1
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    extracted_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """Research hypothesis with formalization."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    title: str = ""
    description: str = ""
    research_question: str = ""
    
    # Formalization
    formal_statement: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)
    expected_relationship: str = ""
    
    # Testable rules
    entry_rules: List[str] = field(default_factory=list)
    exit_rules: List[str] = field(default_factory=list)
    risk_rules: List[str] = field(default_factory=list)
    parameter_space: Dict[str, Any] = field(default_factory=dict)
    
    # Evidence
    supporting_claims: List[str] = field(default_factory=list)  # claim IDs
    contradicting_claims: List[str] = field(default_factory=list)
    
    # Status
    status: ResearchStatus = ResearchStatus.PENDING
    current_phase: ResearchPhase = ResearchPhase.INITIALIZATION
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Results
    backtest_results: Optional[Dict[str, Any]] = None
    critique_results: Optional[Dict[str, Any]] = None
    statistical_results: Optional[Dict[str, Any]] = None
    final_verdict: Optional[str] = None


@dataclass
class ResearchTask:
    """A research task in the pipeline."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    hypothesis_id: str = ""
    assigned_agent: AgentType = AgentType.ORCHESTRATOR
    phase: ResearchPhase = ResearchPhase.INITIALIZATION
    status: ResearchStatus = ResearchStatus.PENDING
    
    # Input/Output
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    
    # Progress
    progress: float = 0.0  # 0-1
    progress_details: str = ""
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Error handling
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # task IDs


@dataclass
class ResearchSession:
    """A complete research session for a hypothesis."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    hypothesis: Hypothesis = field(default_factory=Hypothesis)
    tasks: List[ResearchTask] = field(default_factory=list)
    
    # Session state
    current_phase: ResearchPhase = ResearchPhase.INITIALIZATION
    overall_status: ResearchStatus = ResearchStatus.PENDING
    overall_progress: float = 0.0
    
    # Timeline
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Events
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Concept:
    """A research concept in the knowledge graph."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    description: str = ""
    category: str = ""
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    
    # Relationships
    relationships: List[Dict[str, str]] = field(default_factory=list)  # {target_id, relationship_type}
    
    # Evidence
    supporting_sources: List[str] = field(default_factory=list)  # source IDs
    confidence: float = 0.5
    
    # Versioning
    version: int = 1
    parent_concept_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Tags
    tags: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    config: Any = None
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    daily_returns: List[Tuple[datetime, float]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    position_history: List[Dict] = field(default_factory=list)
    order_history: List[Dict] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    """Relationship between concepts in knowledge graph."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    source_concept_id: str = ""
    target_concept_id: str = ""
    relationship_type: str = ""  # implies, contradicts, supports, is_a, part_of, parameter_of
    weight: float = 1.0
    evidence: List[str] = field(default_factory=list)  # source/claim IDs
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CritiqueResult:
    """Result from critic agent analysis."""
    hypothesis_id: str = ""
    overall_score: float = 0.0  # 0-1
    
    # Specific checks
    parameter_sensitivity: Dict[str, float] = field(default_factory=dict)
    permutation_test_p_value: Optional[float] = None
    regime_dependence_score: float = 0.0
    confounder_risk: float = 0.0
    look_ahead_bias_risk: float = 0.0
    overfitting_score: float = 0.0
    cherry_picking_score: float = 0.0
    
    # Issues found
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    verdict: str = ""  # PASS, CONDITIONAL_PASS, FAIL
    reviewed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StatisticalValidation:
    """Statistical validation results."""
    hypothesis_id: str = ""
    
    # Out-of-sample
    oos_sharpe: Optional[float] = None
    oos_profit_factor: Optional[float] = None
    oos_win_rate: Optional[float] = None
    
    # Walk-forward
    wf_sharpe_mean: Optional[float] = None
    wf_sharpe_std: Optional[float] = None
    wf_consistency: Optional[float] = None  # % windows positive
    
    # Monte Carlo
    mc_prob_profit: Optional[float] = None
    mc_prob_ruin: Optional[float] = None
    mc_median_dd: Optional[float] = None
    
    # Significance
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    statistically_significant: bool = False
    
    # Regime analysis
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    validated_at: datetime = field(default_factory=datetime.utcnow)


# Type aliases
ResearchEvent = Dict[str, Any]
AgentMessage = Dict[str, Any]