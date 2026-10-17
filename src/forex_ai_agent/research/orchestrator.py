"""Research Orchestrator - Manages the multi-agent research pipeline."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from forex_ai_agent.core.event_bus import Event, EventBus, InMemoryEventBus
from forex_ai_agent.research.models import (
    AgentType,
    AgentMessage,
    Hypothesis,
    KnowledgeEdge,
    ResearchEvent,
    ResearchPhase,
    ResearchSession,
    ResearchStatus,
    ResearchTask,
)

from forex_ai_agent.research.agents import (
    create_researcher_agent,
    create_quant_agent,
    create_backtest_agent,
    create_critic_agent,
    create_statistical_agent,
    create_knowledge_agent,
    create_report_agent,
)


@dataclass
class OrchestratorConfig:
    """Configuration for the research orchestrator."""
    max_concurrent_tasks: int = 3
    max_concurrent_sessions: int = 2
    task_timeout_minutes: int = 30
    enable_auto_retry: bool = True
    max_retries: int = 3
    progress_update_interval_seconds: int = 10
    require_human_approval: bool = True
    approval_required_phases: List[str] = field(default_factory=lambda: [
        "BACKTESTING", "CRITIQUE", "STATISTICAL_VALIDATION"
    ])


class ResearchOrchestrator:
    """
    Orchestrates the multi-agent research pipeline.
    
    Manages:
    - Session lifecycle (creation, phase transitions, completion)
    - Task scheduling and agent delegation
    - Progress tracking and event emission
    - Human approval gates
    - Error handling and retries
    """
    
    def __init__(
        self,
        config: OrchestratorConfig = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.config = config or OrchestratorConfig()
        self.event_bus = event_bus or InMemoryEventBus()
        
        # Session management
        self.sessions: Dict[str, ResearchSession] = {}
        self.active_sessions: Set[str] = set()
        
        # Agent registry
        self.agents: Dict[AgentType, Any] = {}
        self._initialize_agents()
        
        # Task queue
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Background tasks
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.on_session_complete: Optional[Callable] = None
        self.on_phase_transition: Optional[Callable] = None
        self.on_human_approval_required: Optional[Callable] = None
    
    def _initialize_agents(self):
        """Create and register all research agents."""
        self.agents[AgentType.RESEARCHER] = create_researcher_agent(event_bus=self.event_bus)
        self.agents[AgentType.QUANT] = create_quant_agent(event_bus=self.event_bus)
        self.agents[AgentType.BACKTEST] = create_backtest_agent(event_bus=self.event_bus)
        self.agents[AgentType.CRITIC] = create_critic_agent(event_bus=self.event_bus)
        self.agents[AgentType.STATISTICAL] = create_statistical_agent(event_bus=self.event_bus)
        self.agents[AgentType.KNOWLEDGE] = create_knowledge_agent(event_bus=self.event_bus)
        self.agents[AgentType.ORCHESTRATOR] = create_report_agent(event_bus=self.event_bus)
    
    async def start(self):
        """Start the orchestrator background tasks."""
        if self._running:
            return
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        await self._emit_event("orchestrator.started", {"config": self.config.__dict__})
    
    async def stop(self):
        """Stop the orchestrator."""
        self._running = False
        
        # Cancel running tasks
        for task in self.running_tasks.values():
            task.cancel()
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        await self._emit_event("orchestrator.stopped", {})
    
    # Session Management
    
    async def create_session(
        self,
        hypothesis: Hypothesis,
        require_approval: bool = None,
    ) -> ResearchSession:
        """Create a new research session for a hypothesis."""
        session = ResearchSession(hypothesis=hypothesis)
        
        # Create initial tasks for each phase
        tasks = self._create_pipeline_tasks(hypothesis.id)
        session.tasks = tasks
        
        self.sessions[session.id] = session
        
        await self._emit_event("session.created", {
            "session_id": session.id,
            "hypothesis_id": hypothesis.id,
            "hypothesis_title": hypothesis.title,
        })
        
        # Auto-start if not requiring approval for first phase
        if not (require_approval or self.config.require_human_approval):
            await self.start_session(session.id)
        
        return session
    
    def _create_pipeline_tasks(self, hypothesis_id: str) -> List[ResearchTask]:
        """Create the standard pipeline tasks for a hypothesis."""
        phases = [
            (ResearchPhase.LITERATURE_REVIEW, AgentType.RESEARCHER),
            (ResearchPhase.HYPOTHESIS_FORMATION, AgentType.RESEARCHER),
            (ResearchPhase.QUANT_FORMALIZATION, AgentType.QUANT),
            (ResearchPhase.BACKTESTING, AgentType.BACKTEST),
            (ResearchPhase.CRITIQUE, AgentType.CRITIC),
            (ResearchPhase.STATISTICAL_VALIDATION, AgentType.STATISTICAL),
            (ResearchPhase.KNOWLEDGE_INTEGRATION, AgentType.KNOWLEDGE),
            (ResearchPhase.REPORT_GENERATION, AgentType.ORCHESTRATOR),
        ]
        
        tasks = []
        for i, (phase, agent_type) in enumerate(phases):
            task = ResearchTask(
                hypothesis_id=hypothesis_id,
                assigned_agent=agent_type,
                phase=phase,
                status=ResearchStatus.PENDING,
                depends_on=[tasks[i-1].id] if i > 0 else [],
            )
            tasks.append(task)
        
        return tasks
    
    async def start_session(self, session_id: str) -> bool:
        """Start a research session."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        if session_id in self.active_sessions:
            return False
        
        if len(self.active_sessions) >= self.config.max_concurrent_sessions:
            await self._emit_event("session.queued", {"session_id": session_id})
            return False
        
        session.overall_status = ResearchStatus.IN_PROGRESS
        session.started_at = datetime.utcnow()
        session.hypothesis.status = ResearchStatus.IN_PROGRESS
        self.active_sessions.add(session_id)
        
        # Start first task
        await self._start_next_task(session)
        
        await self._emit_event("session.started", {
            "session_id": session_id,
            "hypothesis_id": session.hypothesis.id,
        })
        return True
    
    async def _start_next_task(self, session: ResearchSession):
        """Start the next pending task in the session."""
        # Find next task that can run
        for task in session.tasks:
            if task.status == ResearchStatus.PENDING:
                # Check dependencies
                deps_met = all(
                    self._is_task_completed(session, dep_id) 
                    for dep_id in task.depends_on
                )
                if deps_met:
                    await self._execute_task(session, task)
                    return
    
    def _is_task_completed(self, session: ResearchSession, task_id: str) -> bool:
        task = next((t for t in session.tasks if t.id == task_id), None)
        return task and task.status == ResearchStatus.COMPLETED
    
    async def _execute_task(self, session: ResearchSession, task: ResearchTask):
        """Execute a research task with the assigned agent."""
        if len(self.running_tasks) >= self.config.max_concurrent_tasks:
            return  # Queue full
        
        task.status = ResearchStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        
        # Get agent
        agent = self.agents.get(task.assigned_agent)
        if not agent:
            task.status = ResearchStatus.FAILED
            task.error = f"Agent {task.assigned_agent} not found"
            await self._on_task_complete(session, task)
            return
        
        # Check for human approval requirement
        phase_name = task.phase.value
        if (self.config.require_human_approval and 
            phase_name in self.config.approval_required_phases):
            task.status = ResearchStatus.AWAITING_REVIEW
            await self._request_human_approval(session, task)
            return
        
        # Execute task
        coro = self._run_task_with_timeout(session, task, agent)
        running_task = asyncio.create_task(coro)
        self.running_tasks[task.id] = running_task
        
        # Add done callback
        running_task.add_done_callback(
            lambda t: self._on_task_done(session, task, t)
        )
    
    async def _run_task_with_timeout(
        self,
        session: ResearchSession,
        task: ResearchTask,
        agent: Any,
    ):
        """Run task with timeout."""
        try:
            # Prepare input
            input_data = self._prepare_task_input(session, task)
            
            # Call agent
            method_name = f"execute_{task.phase.value.lower()}"
            if hasattr(agent, method_name):
                method = getattr(agent, method_name)
                result = await asyncio.wait_for(
                    method(session.hypothesis, input_data),
                    timeout=self.config.task_timeout_minutes * 60,
                )
            else:
                # Generic execute
                result = await asyncio.wait_for(
                    agent.execute(session.hypothesis, input_data),
                    timeout=self.config.task_timeout_minutes * 60,
                )
            
            task.output_data = result
            task.status = ResearchStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.progress = 1.0
            
        except asyncio.TimeoutError:
            task.status = ResearchStatus.FAILED
            task.error = f"Task timed out after {self.config.task_timeout_minutes} minutes"
        except Exception as e:
            task.status = ResearchStatus.FAILED
            task.error = str(e)
        
        return task

    async def _retry_task(self, session: ResearchSession, task: ResearchTask):
        """Retry a failed task with exponential backoff."""
        await asyncio.sleep(2 ** task.retry_count)
        await self._execute_task(session, task)

    def _prepare_task_input(self, session: ResearchSession, task: ResearchTask) -> Dict[str, Any]:
        """Prepare input data for a task based on previous task outputs."""
        input_data = {
            "hypothesis": session.hypothesis.__dict__,
            "session_id": session.id,
            "task_id": task.id,
        }
        
        # Add outputs from dependency tasks
        for dep_id in task.depends_on:
            dep_task = next((t for t in session.tasks if t.id == dep_id), None)
            if dep_task and dep_task.output_data:
                input_data[f"{dep_task.phase.value.lower()}_output"] = dep_task.output_data
        
        return input_data
    
    def _on_task_done(self, session: ResearchSession, task: ResearchTask, future: asyncio.Future):
        """Callback when task completes."""
        self.running_tasks.pop(task.id, None)
        
        try:
            future.result()  # Raise any exception
        except Exception as e:
            task.status = ResearchStatus.FAILED
            task.error = str(e)
        
        # Handle retries
        if task.status == ResearchStatus.FAILED and self.config.enable_auto_retry:
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = ResearchStatus.PENDING
                task.error = None
                # Schedule retry with exponential backoff
                asyncio.create_task(self._retry_task(session, task))
                return
        
        # Schedule task completion handler
        asyncio.create_task(self._on_task_complete(session, task))
    
    async def _on_task_complete(self, session: ResearchSession, task: ResearchTask):
        """Handle task completion."""
        await self._emit_event("task.completed", {
            "session_id": session.id,
            "task_id": task.id,
            "phase": task.phase.value,
            "status": task.status.value,
            "output": task.output_data,
        })
        
        # Update session progress
        completed = sum(1 for t in session.tasks if t.status == ResearchStatus.COMPLETED)
        session.overall_progress = completed / len(session.tasks)
        
        # Check for human approval requirement for next task
        next_task = self._get_next_pending_task(session)
        if next_task:
            phase_name = next_task.phase.value
            if (self.config.require_human_approval and 
                phase_name in self.config.approval_required_phases):
                await self._request_human_approval(session, next_task)
                return
        
        # Start next task
        await self._start_next_task(session)
        
        # Check if session complete
        if all(t.status == ResearchStatus.COMPLETED for t in session.tasks):
            await self._complete_session(session)
    
    def _get_next_pending_task(self, session: ResearchSession) -> Optional[ResearchTask]:
        for task in session.tasks:
            if task.status == ResearchStatus.PENDING:
                deps_met = all(
                    self._is_task_completed(session, dep_id) 
                    for dep_id in task.depends_on
                )
                if deps_met:
                    return task
        return None
    
    async def _request_human_approval(self, session: ResearchSession, task: ResearchTask):
        """Request human approval for a task."""
        await self._emit_event("approval.required", {
            "session_id": session.id,
            "task_id": task.id,
            "phase": task.phase.value,
            "hypothesis": session.hypothesis.__dict__,
            "task_input": task.input_data,
        })
        
        if self.on_human_approval_required:
            await self.on_human_approval_required(session, task)
    
    async def approve_task(self, session_id: str, task_id: str, approved: bool, feedback: str = ""):
        """Handle human approval decision."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        task = next((t for t in session.tasks if t.id == task_id), None)
        if not task:
            return False
        
        if approved:
            task.status = ResearchStatus.PENDING
            await self._execute_task(session, task)
        else:
            task.status = ResearchStatus.REJECTED
            task.error = f"Rejected: {feedback}"
            session.overall_status = ResearchStatus.REJECTED
            await self._complete_session(session)
        
        return True
    
    async def _complete_session(self, session: ResearchSession):
        """Complete a research session."""
        session.overall_status = ResearchStatus.COMPLETED
        session.completed_at = datetime.utcnow()
        session.overall_progress = 1.0
        session.hypothesis.status = ResearchStatus.COMPLETED
        session.hypothesis.current_phase = ResearchPhase.COMPLETED
        
        self.active_sessions.discard(session.id)
        
        await self._emit_event("session.completed", {
            "session_id": session.id,
            "hypothesis_id": session.hypothesis.id,
            "verdict": session.hypothesis.final_verdict,
        })
        
        if self.on_session_complete:
            await self.on_session_complete(session)
        
        # Start next queued session
        await self._start_queued_sessions()
    
    async def _start_queued_sessions(self):
        """Start any queued sessions if capacity available."""
        for session in self.sessions.values():
            if (session.overall_status == ResearchStatus.PENDING and 
                session.id not in self.active_sessions):
                await self.start_session(session.id)
    
    # Monitoring
    
    async def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.progress_update_interval_seconds)
                await self._update_progress()
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._emit_event("orchestrator.error", {"error": str(e)})
    
    async def _update_progress(self):
        """Update progress for all active sessions."""
        for session in self.sessions.values():
            if session.id in self.active_sessions:
                # Update task progress
                for task in session.tasks:
                    if task.status == ResearchStatus.IN_PROGRESS:
                        # Could query agent for progress
                        pass
                
                # Update overall progress
                completed = sum(1 for t in session.tasks if t.status == ResearchStatus.COMPLETED)
                session.overall_progress = completed / len(session.tasks) if session.tasks else 0
                
                await self._emit_event("progress.update", {
                    "session_id": session.id,
                    "progress": session.overall_progress,
                    "current_phase": session.current_phase.value,
                })
    
    # Event emission
    
    async def _emit_event(self, event_type: str, payload: Dict[str, Any]):
        if self.event_bus:
            event = Event(
                event_type=event_type,
                aggregate_id="orchestrator",
                aggregate_type="research_orchestrator",
                payload=payload,
                source_agent="research_orchestrator",
            )
            await self.event_bus.publish(event)
    
    # Query methods
    
    def get_session(self, session_id: str) -> Optional[ResearchSession]:
        return self.sessions.get(session_id)
    
    def get_active_sessions(self) -> List[ResearchSession]:
        return [s for s in self.sessions.values() if s.id in self.active_sessions]
    
    def get_all_sessions(self) -> List[ResearchSession]:
        return list(self.sessions.values())
    
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel a research session."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        # Cancel running tasks
        for task in session.tasks:
            if task.id in self.running_tasks:
                self.running_tasks[task.id].cancel()
                self.running_tasks.pop(task.id, None)
            if task.status == ResearchStatus.IN_PROGRESS:
                task.status = ResearchStatus.CANCELLED
        
        session.overall_status = ResearchStatus.CANCELLED
        self.active_sessions.discard(session_id)
        
        await self._emit_event("session.cancelled", {"session_id": session_id})
        return True
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed session status."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.id,
            "hypothesis_id": session.hypothesis.id,
            "hypothesis_title": session.hypothesis.title,
            "overall_status": session.overall_status.value,
            "overall_progress": session.overall_progress,
            "current_phase": session.current_phase.value,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "tasks": [
                {
                    "id": t.id,
                    "phase": t.phase.value,
                    "agent": t.assigned_agent.value,
                    "status": t.status.value,
                    "progress": t.progress,
                    "started_at": t.started_at.isoformat() if t.started_at else None,
                    "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                    "error": t.error,
                }
                for t in session.tasks
            ],
        }


# Factory
async def create_orchestrator(
    config: OrchestratorConfig = None,
    event_bus: Optional[EventBus] = None,
) -> ResearchOrchestrator:
    """Create and start a research orchestrator."""
    orchestrator = ResearchOrchestrator(config=config, event_bus=event_bus)
    await orchestrator.start()
    return orchestrator