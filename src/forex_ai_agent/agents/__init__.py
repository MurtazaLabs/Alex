"""Specialist agents for the Forex AI Suite."""

from .base import BaseAgent
from .execution_agent import ExecutionAgent
from .market_agent import MarketAgent
from .news_agent import NewsAgent
from .research_agent import ResearchAgent
from .risk_agent import RiskAgent
from .strategy_agent import StrategyAgent
from .technical_analysis_agent import TechnicalAnalysisAgent

# Convenience list for wiring an orchestrator with all Phase 1 stubs.
DEFAULT_SPECIALISTS = [
    MarketAgent,
    ResearchAgent,
    RiskAgent,
    StrategyAgent,
    ExecutionAgent,
    NewsAgent,
    TechnicalAnalysisAgent,
]


def build_default_specialists(event_bus=None) -> list:
    """Instantiate one of each default specialist, sharing the event bus."""
    return [cls(event_bus=event_bus) for cls in DEFAULT_SPECIALISTS]


__all__ = [
    "BaseAgent",
    "MarketAgent",
    "ResearchAgent",
    "RiskAgent",
    "StrategyAgent",
    "ExecutionAgent",
    "NewsAgent",
    "TechnicalAnalysisAgent",
    "DEFAULT_SPECIALISTS",
    "build_default_specialists",
]
