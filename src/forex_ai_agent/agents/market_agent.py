"""Market/market-structure specialist (Phase 1 stub).

Phase 2 wires this agent to the market structure detection and indicators.
For Phase 1 it returns a deterministic, evidence-shaped placeholder so the
orchestration pipeline can be exercised end-to-end.
"""

from __future__ import annotations

from typing import Any, Dict

from ..core.event_bus import on_event
from .base import BaseAgent


class MarketAgent(BaseAgent):
    """Routing name: ``market_agent`` (domain: market)."""

    name = "market_agent"
    domains = ["market", "analysis"]

    @on_event("analysis.indicators.calculated")
    async def _on_indicators(self, event) -> None:
        """Placeholder handler demonstrating event-driven integration."""

    async def run(
        self, task: Dict[str, Any], context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        context = context or {}
        query = task.get("domain", "market")

        # Phase 1 stub: no live data path yet. Replace in Phase 2 with real
        # structure + indicator analysis driven by market data.
        return {
            "status": "SUCCESS",
            "specialist": self.name,
            "summary": (
                f"Market analysis placeholder for domain {query!r}. "
                "Live market-structure analysis arrives in Phase 2."
            ),
            "direction": None,
            "data_complete": False,
        }
