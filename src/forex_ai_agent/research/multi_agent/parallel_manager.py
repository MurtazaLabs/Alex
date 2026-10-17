"""Parallel Research Manager - Manages concurrent research tasks with resource allocation."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus
from forex_ai_agent.research.models import ResearchTask, ResearchStatus, ResearchPhase


class ResourceType(Enum):
    """Types of resources that can be allocated."""
    API_QUOTA = "api_quota"
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ResourceLimit:
    """Resource limit configuration."""
    resource_type: ResourceType
    max_units: float
    current_usage: float = 0.0
    reserved: float = 0.0

    @property
    def available(self) -> float:
        return self.max_units - self.current_usage - self.reserved

    def can_allocate(self, units: float) -> bool:
        return self.available >= units

    def allocate(self, units: float) -> bool:
        if self.can_allocate(units):
            self.current_usage += units
            return True
        return False

    def release(self, units: float):
        self.current_usage = max(0.0, self.current_usage - units)


@dataclass
class ResourceAllocation:
    """Resource allocation for a task."""
    task_id: str
    allocations: Dict[ResourceType, float] = field(default_factory=dict)
    allocated_at: datetime = field(default_factory=datetime.utcnow)
    released_at: Optional[datetime] = None


class ResourceAllocator:
    """Manages resource allocation for concurrent research tasks."""

    def __init__(self, limits: Dict[ResourceType, ResourceLimit] = None):
        self.limits = limits or self._default_limits()
        self.allocations: Dict[str, ResourceAllocation] = {}
        self._lock = asyncio.Lock()

    def _default_limits(self) -> Dict[ResourceType, ResourceLimit]:
        return {
            ResourceType.API_QUOTA: ResourceLimit(ResourceType.API_QUOTA, max_units=1000.0),
            ResourceType.COMPUTE: ResourceLimit(ResourceType.COMPUTE, max_units=100.0),
            ResourceType.MEMORY: ResourceLimit(ResourceType.MEMORY, max_units=16384.0),  # MB
            ResourceType.STORAGE: ResourceLimit(ResourceType.STORAGE, max_units=10240.0),  # MB
            ResourceType.NETWORK: ResourceLimit(ResourceType.NETWORK, max_units=100.0),  # Mbps
        }

    async def allocate(self, task_id: str, requirements: Dict[ResourceType, float]) -> bool:
        """Allocate resources for a task."""
        async with self._lock:
            # Check if all resources can be allocated
            for resource_type, units in requirements.items():
                limit = self.limits.get(resource_type)
                if not limit or not limit.can_allocate(units):
                    return False

            # Allocate all resources
            allocation = ResourceAllocation(task_id=task_id)
            for resource_type, units in requirements.items():
                self.limits[resource_type].allocate(units)
                allocation.allocations[resource_type] = units

            self.allocations[task_id] = allocation
            return True

    async def release(self, task_id: str):
        """Release resources allocated to a task."""
        async with self._lock:
            allocation = self.allocations.pop(task_id, None)
            if allocation:
                for resource_type, units in allocation.allocations.items():
                    self.limits[resource_type].release(units)
                allocation.released_at = datetime.utcnow()

    def get_usage(self) -> Dict[ResourceType, Dict[str, float]]:
        """Get current resource usage."""
        return {
            rt: {
                "max": limit.max_units,
                "current": limit.current_usage,
                "reserved": limit.reserved,
                "available": limit.available,
            }
            for rt, limit in self.limits.items()
        }


@dataclass
class TaskDependency:
    """Task dependency specification."""
    task_id: str
    depends_on: List[str] = field(default_factory=list)
    blocking: bool = True  # If True, task waits for dependencies to complete


class TaskDependencyManager:
    """Manages task dependencies and execution order."""

    def __init__(self):
        self.dependencies: Dict[str, TaskDependency] = {}
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()

    def add_dependency(self, task_id: str, depends_on: List[str], blocking: bool = True):
        """Add a task dependency."""
        self.dependencies[task_id] = TaskDependency(
            task_id=task_id,
            depends_on=depends_on,
            blocking=blocking
        )

    def mark_completed(self, task_id: str):
        """Mark a task as completed."""
        self.completed_tasks.add(task_id)
        self.failed_tasks.discard(task_id)

    def mark_failed(self, task_id: str):
        """Mark a task as failed."""
        self.failed_tasks.add(task_id)

    def can_execute(self, task_id: str) -> bool:
        """Check if a task can be executed (all dependencies met)."""
        dep = self.dependencies.get(task_id)
        if not dep:
            return True

        if dep.blocking:
            return all(d in self.completed_tasks for d in dep.depends_on)
        else:
            return all(d in self.completed_tasks or d in self.failed_tasks for d in dep.depends_on)

    def get_ready_tasks(self, pending_tasks: List[str]) -> List[str]:
        """Get list of tasks that are ready to execute."""
        return [t for t in pending_tasks if self.can_execute(t)]


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ScheduledTask:
    """A scheduled research task."""
    task: ResearchTask
    priority: TaskPriority = TaskPriority.NORMAL
    scheduled_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0
    max_retries: int = 3


class ParallelResearchManager:
    """
    Manages concurrent research tasks with resource allocation and dependency management.
    
    Features:
    - Concurrent task execution with configurable concurrency limits
    - Resource allocation (API quotas, compute, memory, storage, network)
    - Task dependency management with topological ordering
    - Priority-based scheduling
    - Automatic retry with exponential backoff
    - Progress tracking and monitoring
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 5,
        resource_allocator: Optional[ResourceAllocator] = None,
        dependency_manager: Optional[TaskDependencyManager] = None,
        event_bus: Optional[EventBus] = None,
        default_task_timeout_minutes: int = 30,
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.resource_allocator = resource_allocator or ResourceAllocator()
        self.dependency_manager = dependency_manager or TaskDependencyManager()
        self.event_bus = event_bus or InMemoryEventBus()
        
        self.default_task_timeout = timedelta(minutes=default_task_timeout_minutes)
        
        # Task queues
        self.pending_tasks: Dict[str, ScheduledTask] = {}
        self.running_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: Dict[str, ResearchTask] = {}
        self.failed_tasks: Dict[str, ResearchTask] = {}
        
        # State
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Callbacks
        self.on_task_start: Optional[Callable] = None
        self.on_task_complete: Optional[Callable] = None
        self.on_task_failed: Optional[Callable] = None
        
        # Default resource requirements per task type
        self.default_requirements: Dict[ResearchPhase, Dict[ResourceType, float]] = {
            ResearchPhase.LITERATURE_REVIEW: {
                ResourceType.API_QUOTA: 10.0,
                ResourceType.COMPUTE: 5.0,
                ResourceType.MEMORY: 512.0,
            },
            ResearchPhase.HYPOTHESIS_FORMATION: {
                ResourceType.API_QUOTA: 5.0,
                ResourceType.COMPUTE: 10.0,
                ResourceType.MEMORY: 256.0,
            },
            ResearchPhase.QUANT_FORMALIZATION: {
                ResourceType.COMPUTE: 20.0,
                ResourceType.MEMORY: 1024.0,
            },
            ResearchPhase.BACKTESTING: {
                ResourceType.COMPUTE: 50.0,
                ResourceType.MEMORY: 2048.0,
                ResourceType.STORAGE: 100.0,
            },
            ResearchPhase.CRITIQUE: {
                ResourceType.COMPUTE: 30.0,
                ResourceType.MEMORY: 512.0,
            },
            ResearchPhase.STATISTICAL_VALIDATION: {
                ResourceType.COMPUTE: 40.0,
                ResourceType.MEMORY: 1024.0,
            },
            ResearchPhase.KNOWLEDGE_INTEGRATION: {
                ResourceType.API_QUOTA: 10.0,
                ResourceType.COMPUTE: 10.0,
                ResourceType.MEMORY: 256.0,
            },
            ResearchPhase.REPORT_GENERATION: {
                ResourceType.COMPUTE: 10.0,
                ResourceType.MEMORY: 256.0,
            },
        }

    async def start(self):
        """Start the parallel research manager."""
        if self._running:
            return
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop the parallel research manager."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def submit_task(self, task: ResearchTask, priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Submit a research task for execution."""
        scheduled = ScheduledTask(task=task, priority=priority)
        self.pending_tasks[task.id] = scheduled
        
        # Add dependency info
        if hasattr(task, 'depends_on') and task.depends_on:
            self.dependency_manager.add_dependency(task.id, task.depends_on)
        
        return task.id

    async def submit_tasks(self, tasks: List[ResearchTask], priorities: List[TaskPriority] = None) -> List[str]:
        """Submit multiple tasks at once."""
        if priorities is None:
            priorities = [TaskPriority.NORMAL] * len(tasks)
        
        task_ids = []
        for task, priority in zip(tasks, priorities):
            task_id = await self.submit_task(task, priority)
            task_ids.append(task_id)
        return task_ids

    async def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                await self._schedule_tasks()
                await self._check_running_tasks()
                await asyncio.sleep(5)  # Check every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue monitoring
                pass

    async def _schedule_tasks(self):
        """Schedule pending tasks for execution."""
        # Get ready tasks (dependencies met)
        pending_ids = list(self.pending_tasks.keys())
        ready_ids = self.dependency_manager.get_ready_tasks(pending_ids)
        
        # Sort by priority
        ready_tasks = [
            self.pending_tasks[tid] 
            for tid in ready_ids 
            if tid in self.pending_tasks
        ]
        ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        
        # Schedule up to concurrency limit
        available_slots = self.max_concurrent_tasks - len(self.running_tasks)
        for task in ready_tasks[:available_slots]:
            await self._start_task(task)

    async def _start_task(self, scheduled_task: ScheduledTask):
        """Start executing a task."""
        task = scheduled_task.task
        
        # Check resource requirements
        requirements = self.default_requirements.get(task.phase, {})
        if not await self.resource_allocator.allocate(task.id, requirements):
            # Not enough resources, will retry later
            return

        # Move from pending to running
        self.pending_tasks.pop(task.id, None)
        scheduled_task.started_at = datetime.utcnow()
        self.running_tasks[task.id] = scheduled_task
        
        # Execute task with semaphore
        asyncio.create_task(self._execute_task(scheduled_task))

    async def _execute_task(self, scheduled_task: ScheduledTask):
        """Execute a single task with timeout and retry logic."""
        task = scheduled_task.task
        
        try:
            async with self._semaphore:
                # Check resource allocation
                requirements = self.default_requirements.get(task.phase, {})
                if not await self.resource_allocator.allocate(task.id, {}):
                    raise RuntimeError("Resource allocation failed")
                
                # Call the actual task handler (would be implemented by orchestrator)
                if self.on_task_start:
                    await self.on_task_start(scheduled_task)
                
                scheduled_task.completed_at = datetime.utcnow()
                self.completed_tasks[task.id] = scheduled_task.task
                self.dependency_manager.mark_completed(task.id)
                
                # Release resources
                await self.resource_allocator.release(task.id)
                
                if self.on_task_complete:
                    await self.on_task_complete(scheduled_task)
                    
        except Exception as e:
            await self._handle_task_failure(scheduled_task, e)

    async def _handle_task_failure(self, scheduled_task: ScheduledTask, error: Exception):
        """Handle task failure with retry logic."""
        task = scheduled_task.task
        scheduled_task.retries += 1
        
        # Release resources
        await self.resource_allocator.release(task.id)
        
        if scheduled_task.retries < scheduled_task.max_retries:
            # Retry with exponential backoff
            delay = 2 ** scheduled_task.retries
            await asyncio.sleep(delay)
            self.pending_tasks[task.id] = scheduled_task
        else:
            # Max retries exceeded
            self.failed_tasks[task.id] = scheduled_task.task
            self.dependency_manager.mark_failed(task.id)
            
            if self.on_task_failed:
                await self.on_task_failed(scheduled_task, error)
        
        # Clean up running tasks
        self.running_tasks.pop(task.id, None)

    async def _check_running_tasks(self):
        """Check for timed out tasks."""
        now = datetime.utcnow()
        timed_out = []
        
        for task_id, scheduled in self.running_tasks.items():
            if scheduled.started_at and now - scheduled.started_at > self.default_task_timeout:
                timed_out.append(task_id)
        
        for task_id in timed_out:
            scheduled = self.running_tasks.pop(task_id, None)
            if scheduled:
                await self._handle_task_failure(scheduled, TimeoutError("Task timed out"))

    def get_status(self) -> Dict[str, Any]:
        """Get current manager status."""
        return {
            "pending": len(self.pending_tasks),
            "running": len(self.running_tasks),
            "completed": len(self.completed_tasks),
            "failed": len(self.failed_tasks),
            "resource_usage": self.resource_allocator.get_usage(),
        }

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        for task_dict in [self.pending_tasks, self.running_tasks, self.completed_tasks, self.failed_tasks]:
            if task_id in task_dict:
                scheduled = task_dict[task_id]
                return {
                    "task_id": task_id,
                    "phase": scheduled.task.phase.value,
                    "status": "pending" if task_id in self.pending_tasks else
                              "running" if task_id in self.running_tasks else
                              "completed" if task_id in self.completed_tasks else
                              "failed",
                    "priority": scheduled.priority.name,
                    "scheduled_at": scheduled.scheduled_at.isoformat(),
                    "started_at": scheduled.started_at.isoformat() if scheduled.started_at else None,
                    "completed_at": scheduled.completed_at.isoformat() if scheduled.completed_at else None,
                    "retries": scheduled.retries,
                }
        return None


# Factory function
async def create_parallel_manager(
    max_concurrent_tasks: int = 5,
    event_bus: Optional[EventBus] = None,
) -> ParallelResearchManager:
    """Create and start a parallel research manager."""
    manager = ParallelResearchManager(
        max_concurrent_tasks=max_concurrent_tasks,
        event_bus=event_bus,
    )
    await manager.start()
    return manager