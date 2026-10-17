"""News Integration Package.

Provides economic calendar data, news risk classification, and integration
with signal/risk agents for news-aware trading.
"""

from __future__ import annotations

from .agent import NewsAgent, CurrencyRiskState, create_news_agent
from .providers import (
    EconomicEvent,
    NewsImpact,
    NewsCategory,
    NewsProvider,
    MockNewsProvider,
    get_news_provider,
    set_news_provider,
)
from .forex_factory_provider import (
    ForexFactoryNewsProvider,
    InvestingComNewsProvider,
)

__all__ = [
    "NewsAgent",
    "CurrencyRiskState",
    "create_news_agent",
    "EconomicEvent",
    "NewsImpact",
    "NewsCategory",
    "NewsProvider",
    "MockNewsProvider",
    "ForexFactoryNewsProvider",
    "InvestingComNewsProvider",
    "get_news_provider",
    "set_news_provider",
]