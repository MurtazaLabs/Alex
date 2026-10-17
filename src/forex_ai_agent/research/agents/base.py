"""Base Research Agent Class."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from forex_ai_agent.core.event_bus import Event, EventBus, InMemoryEventBus
from forex_ai_agent.research.models import (
    AgentType,
    AgentMessage,
    Hypothesis,
    ResearchPhase,
    Source,
)


@dataclass
class AgentCapability:
    """Describes an agent capability."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    estimated_duration_seconds: int = 60


class BaseResearchAgent:
    """
    Base class for all research agents.
    
    Provides common infrastructure for:
    - Event bus communication
    - Progress reporting
    - Error handling
    - Configuration
    """
    
    def __init__(
        self,
        agent_type: str,
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.agent_type = agent_type
        self.event_bus = event_bus or InMemoryEventBus()
        self.config = config or {}
        self.capabilities: List[str] = []
        
        # Register capabilities
        self._register_capabilities()
    
    @abstractmethod
    def _register_capabilities(self):
        """Register agent capabilities."""
        pass
    
    @abstractmethod
    async def execute(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the agent's main task."""
        pass
    
    def get_capabilities(self) -> List[str]:
        return self.capabilities
    
    async def emit_progress(
        self,
        event_type: str,
        payload: Dict[str, Any],
    ):
        """Emit progress event."""
        # In a real implementation, this would use the event bus
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, None)
    
    async def _emit_event(self, event_type: str, payload: Dict[str, Any]):
        """Emit event via event bus."""
        # In a real implementation, this would use the event bus
        pass
    
    def get_timestamp(self) -> str:
        return datetime.utcnow().isoformat()
    
    def create_output(
        self,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create standardized output."""
        return {
            "data": data,
            "metadata": metadata or {},
            "timestamp": self.get_timestamp(),
        }
    
    def validate_input(self, input_data: Dict[str, Any], required_keys: List[str]) -> bool:
        """Validate input has required keys."""
        return all(key in input_data for key in self.config.get("required_keys", []))
    
    async def emit_progress_update(
        self,
        task_id: str,
        progress: float,
        message: str = "",
    ):
        """Emit progress update."""
        await self._emit_event("progress.update", {
            "agent": self.agent_type,
            "task_id": task_id,
            "progress": progress,
            "message": message,
        })