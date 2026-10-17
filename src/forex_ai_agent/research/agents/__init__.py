"""Research Agents Package."""

from __future__ import annotations

from .base import BaseResearchAgent
from .researcher import ResearcherAgent, create_researcher_agent
from .quant import QuantAgent, QuantConfig, create_quant_agent
from .backtest import BacktestAgent, BacktestAgentConfig, create_backtest_agent
from .critic import CriticAgent, create_critic_agent
from .statistical import StatisticalAgent, create_statistical_agent
from .knowledge import KnowledgeAgent, create_knowledge_agent
from .report import ReportAgent, create_report_agent

__all__ = [
    "BaseResearchAgent",
    "ResearcherAgent",
    "create_researcher_agent",
    "QuantAgent",
    "QuantConfig",
    "create_quant_agent",
    "BacktestAgent",
    "BacktestAgentConfig",
    "create_backtest_agent",
    "CriticAgent",
    "create_critic_agent",
    "StatisticalAgent",
    "create_statistical_agent",
    "KnowledgeAgent",
    "create_knowledge_agent",
    "ReportAgent",
    "create_report_agent",
]