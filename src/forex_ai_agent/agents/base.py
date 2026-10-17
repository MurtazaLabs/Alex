"""Agent lifecycle hooks.

Agents may opt in to events by decorating methods with ``@on_event`` and be
auto-registered with the event bus via ``register_agent``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..core.event_bus import Event, EventBus


class BaseAgent:
    """Common infrastructure for all specialists."""

    name: str = "base"
    domains: list = []

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus
        if event_bus is not None:
            event_bus.register_agent(self)

    async def publish(
        self,
        event_type: str,
        aggregate_type: str,
        aggregate_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """Publish an event through the configured event bus."""
        if self.event_bus is None:
            return
        await self.event_bus.publish(
            Event(
                event_type=event_type,
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id,
                payload=payload,
                source_agent=self.name,
            )
        )

    async def run(
        self, task: Dict[str, Any], context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute the assigned task.

        Subclasses override this. ``task`` contains the routed request, and
        ``context`` may carry upstream specialist results for sequential pipelines.
        """
        raise NotImplementedError(f"{self.name} does not implement run()")
