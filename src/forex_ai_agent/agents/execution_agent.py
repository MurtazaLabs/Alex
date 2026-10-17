"""Execution specialist (Phase 1 stub).

Never executes anything until it is permission-gated at level 3 (paper) /
level 4 (live) in later phases. The orchestration layer rejects this domain
when the current permission level is insufficient.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseAgent


class ExecutionAgent(BaseAgent):
    """Routing name: ``execution_agent`` (domain: execution)."""

    name = "execution_agent"
    domains = ["execution"]

    async def run(
        self, task: Dict[str, Any], context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        context = context or {}
        return {
            "status": "BLOCKED",
            "specialist": self.name,
            "summary": (
                "Execution is not available yet. Paper trading is Phase 6 and "
                "live broker execution is Phase 10, both behind permission gates."
            ),
            "direction": None,
            "data_complete": False,
        }
