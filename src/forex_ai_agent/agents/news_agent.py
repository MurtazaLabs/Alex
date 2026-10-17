"""News specialist (Phase 1 stub). Full news integration is Phase 4."""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseAgent


class NewsAgent(BaseAgent):
    """Routing name: ``news_agent`` (domain: news)."""

    name = "news_agent"
    domains = ["news"]

    async def run(
        self, task: Dict[str, Any], context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        context = context or {}
        return {
            "status": "SUCCESS",
            "specialist": self.name,
            "summary": (
                "News placeholder. Economic-calendar and news-risk integration "
                "arrives in Phase 4."
            ),
            "direction": None,
            "data_complete": False,
        }
