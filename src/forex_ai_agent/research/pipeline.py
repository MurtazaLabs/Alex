"""Research Pipeline - End-to-end research pipeline orchestration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus
from forex_ai_agent.research.models import (
    Hypothesis,
    ResearchSession,
    ResearchPhase,
    ResearchStatus,
    AgentType,
    Source,
    Claim,
    CritiqueResult,
    StatisticalValidation,
)
from forex_ai_agent.research.orchestrator import ResearchOrchestrator, OrchestratorConfig
from forex_ai_agent.research.agents import (
    ResearcherAgent,
    QuantAgent,
    BacktestAgent,
    CriticAgent,
    StatisticalAgent,
    KnowledgeAgent,
    create_researcher_agent,
    create_quant_agent,
    create_backtest_agent,
    create_critic_agent,
    create_statistical_agent,
    create_knowledge_agent,
)
from forex_ai_agent.research.orchestrator import create_orchestrator


@dataclass
class PipelineConfig:
    """Configuration for the research pipeline."""
    orchestrator_config: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    researcher_config: Any = None
    quant_config: Any = None
    backtest_config: Any = None
    critic_config: Any = None
    statistical_config: Any = None
    knowledge_config: Any = None
    auto_start: bool = True
    require_approval: bool = True


class ResearchPipeline:
    """
    End-to-end Research Pipeline.
    
    Manages the complete research lifecycle:
    1. Literature Review (Researcher Agent)
    2. Hypothesis Formation (Researcher Agent)
    3. Quant Formalization (Quant Agent)
    4. Backtesting (Backtest Agent)
    5. Critique (Critic Agent)
    6. Statistical Validation (Statistical Agent)
    7. Knowledge Integration (Knowledge Agent)
    """
    
    def __init__(
        self,
        config: PipelineConfig = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.config = config or PipelineConfig()
        self.event_bus = event_bus or InMemoryEventBus()
        
        # Create orchestrator
        self.orchestrator = ResearchOrchestrator(
            config=self.config.orchestrator_config,
            event_bus=self.event_bus,
        )
        
        # Create agents
        self.agents = {
            AgentType.RESEARCHER: create_researcher_agent(self.config.researcher_config, self.event_bus),
            AgentType.QUANT: create_quant_agent(self.config.quant_config, self.event_bus),
            AgentType.BACKTEST: create_backtest_agent(self.config.backtest_config, self.event_bus),
            AgentType.CRITIC: create_critic_agent(self.config.critic_config, self.event_bus),
            AgentType.STATISTICAL: create_statistical_agent(self.config.statistical_config, self.event_bus),
            AgentType.KNOWLEDGE: create_knowledge_agent(self.config.knowledge_config, self.event_bus),
        }
        
        # Override orchestrator's agents with ours
        self.orchestrator.agents = self.agents
        
        # Pipeline state
        self.sessions: Dict[str, ResearchSession] = {}
        self._running = False
        
        # Callbacks
        self.on_pipeline_complete: Optional[Callable] = None
        self.on_phase_complete: Optional[Callable] = None
    
    async def start(self):
        """Start the pipeline."""
        if self._running:
            return
        self._running = True
        await self.orchestrator.start()
        
        # Set up orchestrator callbacks
        self.orchestrator.on_session_complete = self._on_session_complete
        self.orchestrator.on_phase_transition = self._on_phase_transition
        self.orchestrator.on_human_approval_required = self._on_approval_required
    
    async def stop(self):
        """Stop the pipeline."""
        self._running = False
        await self.orchestrator.stop()
    
    async def research(
        self,
        title: str,
        description: str,
        research_question: str,
        variables: Dict[str, Any] = None,
        require_approval: bool = None,
    ) -> ResearchSession:
        """
        Start a new research project.
        
        Args:
            title: Short title for the hypothesis
            description: Detailed description
            research_question: The core research question
            variables: Dictionary of variables involved
            require_approval: Override default approval requirement
        
        Returns:
            ResearchSession object
        """
        # Create hypothesis
        hypothesis = Hypothesis(
            id=str(uuid4())[:8],
            title=title,
            description=description,
            research_question=research_question,
            variables=variables or {},
            status=ResearchStatus.PENDING,
            current_phase=ResearchPhase.INITIALIZATION,
        )
        
        # Create session via orchestrator
        session = await self.orchestrator.create_session(
            hypothesis=hypothesis,
            require_approval=require_approval if require_approval is not None else self.config.require_approval,
        )
        
        self.sessions[session.id] = session
        
        return session
    
    async def _on_session_complete(self, session: ResearchSession):
        """Handle session completion."""
        self.sessions[session.id] = session
        
        if self.on_pipeline_complete:
            await self.on_pipeline_complete(session)
        
        # Auto-start next if queued
        await self.orchestrator._start_queued_sessions()
    
    async def _on_phase_transition(self, session_id: str, phase: ResearchPhase):
        """Handle phase transition."""
        session = self.sessions.get(session_id)
        if session:
            session.current_phase = phase
            session.hypothesis.current_phase = phase
        
        if self.on_phase_complete:
            await self.on_phase_complete(session_id, phase)
    
    async def _on_approval_required(self, session: ResearchSession, task: Any):
        """Handle approval requirement."""
        # In a real implementation, this would notify the user
        print(f"APPROVAL REQUIRED for session {session.id}, task {task.id}, phase {task.phase.value}")
    
    async def approve_task(self, session_id: str, task_id: str, approved: bool, feedback: str = "") -> bool:
        """Approve or reject a task requiring human approval."""
        return await self.orchestrator.approve_task(session_id, task_id, approved, feedback)
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed session status."""
        return self.orchestrator.get_session_status(session_id)
    
    def get_session(self, session_id: str) -> Optional[ResearchSession]:
        return self.sessions.get(session_id)
    
    def get_all_sessions(self) -> List[ResearchSession]:
        return list(self.sessions.values())
    
    async def query_knowledge(
        self,
        query_type: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Query the knowledge graph."""
        knowledge_agent = self.agents[AgentType.KNOWLEDGE]
        
        if query_type == "search":
            return await knowledge_agent.query_concepts({
                "query_type": "search",
                "query": kwargs.get("query", ""),
                "category": kwargs.get("category"),
                "tags": kwargs.get("tags", []),
                "limit": kwargs.get("limit", 20),
            })
        elif query_type == "neighbors":
            return await knowledge_agent.query_concepts({
                "query_type": "neighbors",
                "concept_id": kwargs.get("concept_id"),
                "depth": kwargs.get("depth", 1),
            })
        elif query_type == "path":
            return await knowledge_agent.query_concepts({
                "query_type": "path",
                "source_id": kwargs.get("source_id"),
                "target_id": kwargs.get("target_id"),
                "max_depth": kwargs.get("max_depth", 5),
            })
        elif query_type == "subgraph":
            return await knowledge_agent.query_concepts({
                "query_type": "subgraph",
                "concept_id": kwargs.get("concept_id"),
                "radius": kwargs.get("radius", 2),
            })
        else:
            return {"error": f"Unknown query type: {query_type}"}
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status."""
        active = [s for s in self.sessions.values() if s.id in self.orchestrator.active_sessions]
        completed = [s for s in self.sessions.values() if s.overall_status.value == "COMPLETED"]
        pending = [s for s in self.sessions.values() if s.overall_status.value == "PENDING"]
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": len(active),
            "completed_sessions": len(completed),
            "pending_sessions": len(pending),
            "agents": {k.value: v.get_capabilities() for k, v in self.agents.items()},
        }


# Factory
async def create_pipeline(
    config: PipelineConfig = None,
    event_bus: Optional[EventBus] = None,
) -> ResearchPipeline:
    """Create and start a research pipeline."""
    pipeline = ResearchPipeline(config=config, event_bus=event_bus)
    await pipeline.start()
    return pipeline


# Convenience function for quick research
async def quick_research(
    title: str,
    description: str,
    research_question: str,
    variables: Dict[str, Any] = None,
    config: PipelineConfig = None,
) -> ResearchSession:
    """
    Quick research function - runs complete pipeline automatically.
    
    Example:
        session = await quick_research(
            title="Liquidity Sweep Reversal",
            description="Investigate if liquidity sweep reversals predict short-term reversals",
            research_question="Do liquidity sweeps at key levels predict short-term price reversals in EUR/USD?",
            variables={"symbol": "EUR/USD", "timeframe": "H1", "session": "London"}
        )
    """
    pipeline = await create_pipeline(config=config)
    session = await pipeline.research(
        title=title,
        description=description,
        research_question=research_question,
        variables=variables,
        require_approval=False,
    )
    
    # Wait for completion (with timeout)
    max_wait = 300  # 5 minutes
    wait_interval = 10
    waited = 0
    
    while waited < max_wait:
        status = await pipeline.get_session_status(session.id)
        if status and status["overall_status"] in ["COMPLETED", "FAILED", "REJECTED", "CANCELLED"]:
            break
        await asyncio.sleep(wait_interval)
        waited += wait_interval
    
    return session