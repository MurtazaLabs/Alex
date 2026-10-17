"""Research specialist (Phase 1 stub).

Full multi-source research pipeline lands in Phase 7. This stub keeps the
research routing chain (orchestrator -> research) operational.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseAgent


class ResearchAgent(BaseAgent):
    """Routing name: ``research_agent`` (domain: research)."""

    name = "research_agent"
    domains = ["research"]

    async def run(
        self, task: Dict[str, Any], context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        context = context or {}
        return {
            "status": "SUCCESS",
            "specialist": self.name,
            "summary": (
                "Research placeholder. The multi-agent research pipeline "
                "(sources -> hypotheses -> backtests) is implemented in Phase 7."
            ),
            "direction": None,
            "data_complete": False,
        }
