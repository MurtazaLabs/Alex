"""Strategy specialist (Phase 1 stub). Full strategy/versioning is Phase 9."""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseAgent


class StrategyAgent(BaseAgent):
    """Routing name: ``strategy_agent`` (domain: strategy)."""

    name = "strategy_agent"
    domains = ["strategy"]

    async def run(
        self, task: Dict[str, Any], context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        context = context or {}
        return {
            "status": "SUCCESS",
            "specialist": self.name,
            "summary": (
                "Strategy placeholder. Versioned strategy management and "
                "research-to-production gates arrive in Phase 9."
            ),
            "direction": None,
            "data_complete": False,
        }
