"""News Provider Interface and Implementations.

Provides economic calendar data and news events for risk management.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4


class NewsImpact(Enum):
    """News impact level."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    NONE = "NONE"


class NewsCategory(Enum):
    """Economic event categories."""
    INTEREST_RATE = "INTEREST_RATE"
    INFLATION = "INFLATION"
    EMPLOYMENT = "EMPLOYMENT"
    GDP = "GDP"
    TRADE_BALANCE = "TRADE_BALANCE"
    CONSUMER_CONFIDENCE = "CONSUMER_CONFIDENCE"
    MANUFACTURING = "MANUFACTURING"
    SERVICES = "SERVICES"
    HOUSING = "HOUSING"
    CENTRAL_BANK_SPEECH = "CENTRAL_BANK_SPEECH"
    GEOPOLITICAL = "GEOPOLITICAL"
    OTHER = "OTHER"


@dataclass
class EconomicEvent:
    """Normalized economic calendar event."""
    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    currency: str = ""
    country: str = ""
    category: NewsCategory = NewsCategory.OTHER
    impact: NewsImpact = NewsImpact.LOW
    scheduled_time: datetime = field(default_factory=datetime.utcnow)
    actual_time: Optional[datetime] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    unit: str = ""
    source: str = ""
    url: Optional[str] = None

    # Derived fields
    affected_pairs: List[str] = field(default_factory=list)
    risk_state: NewsImpact = NewsImpact.NONE
    minutes_until_event: Optional[int] = None

    def __post_init__(self):
        """Derive affected pairs from currency."""
        if self.currency and not self.affected_pairs:
            # Major pairs involving this currency
            major_pairs = {
                "USD": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"],
                "EUR": ["EUR/USD", "EUR/GBP", "EUR/JPY", "EUR/CHF", "EUR/AUD", "EUR/CAD"],
                "GBP": ["GBP/USD", "EUR/GBP", "GBP/JPY", "GBP/CHF", "GBP/AUD", "GBP/CAD"],
                "JPY": ["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY", "CAD/JPY", "CHF/JPY"],
                "CHF": ["USD/CHF", "EUR/CHF", "GBP/CHF", "CHF/JPY"],
                "AUD": ["AUD/USD", "EUR/AUD", "GBP/AUD", "AUD/JPY", "AUD/CAD", "AUD/CHF", "AUD/NZD"],
                "CAD": ["USD/CAD", "EUR/CAD", "GBP/CAD", "CAD/JPY", "AUD/CAD", "CAD/CHF"],
                "NZD": ["NZD/USD", "EUR/NZD", "GBP/NZD", "AUD/NZD", "NZD/JPY", "NZD/CHF", "NZD/CAD"],
            }
            self.affected_pairs = major_pairs.get(self.currency.upper(), [])


class NewsProvider(ABC):
    """Abstract interface for news/economic calendar providers."""

    @abstractmethod
    async def fetch_events(
        self,
        start: datetime,
        end: datetime,
        currencies: Optional[List[str]] = None,
        impact_filter: Optional[Set[NewsImpact]] = None,
    ) -> List[EconomicEvent]:
        """Fetch economic events for date range."""
        pass

    @abstractmethod
    async def fetch_upcoming_events(
        self,
        hours_ahead: int = 24,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        """Fetch upcoming events within hours_ahead."""
        pass

    @abstractmethod
    async def get_current_risk_state(self, currency: str) -> NewsImpact:
        """Get current news risk state for a currency."""
        pass

    @abstractmethod
    def get_supported_currencies(self) -> List[str]:
        """Return list of supported currency codes."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass


class MockNewsProvider(NewsProvider):
    """Mock news provider for testing and development."""

    def __init__(self):
        self._events: List[EconomicEvent] = []
        self._generate_mock_events()

    def _generate_mock_events(self):
        """Generate realistic mock economic events."""
        base_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]

        mock_events = [
            # High impact events
            ("FOMC Rate Decision", "USD", NewsCategory.INTEREST_RATE, NewsImpact.HIGH, 14, 18),
            ("NFP Employment Change", "USD", NewsCategory.EMPLOYMENT, NewsImpact.HIGH, 12, 13),
            ("CPI Inflation Rate", "USD", NewsCategory.INFLATION, NewsImpact.HIGH, 10, 11),
            ("ECB Rate Decision", "EUR", NewsCategory.INTEREST_RATE, NewsImpact.HIGH, 14, 18),
            ("BoE Rate Decision", "GBP", NewsCategory.INTEREST_RATE, NewsImpact.HIGH, 14, 18),
            ("BoJ Rate Decision", "JPY", NewsCategory.INTEREST_RATE, NewsImpact.HIGH, 14, 18),
            # Medium impact
            ("ISM Manufacturing PMI", "USD", NewsCategory.MANUFACTURING, NewsImpact.MEDIUM, 10, 10),
            ("Retail Sales", "USD", NewsCategory.OTHER, NewsImpact.MEDIUM, 10, 10),
            ("GDP Quarterly", "EUR", NewsCategory.GDP, NewsImpact.MEDIUM, 10, 10),
            ("CPI Flash", "EUR", NewsCategory.INFLATION, NewsImpact.MEDIUM, 10, 10),
            ("Claimant Count", "GBP", NewsCategory.EMPLOYMENT, NewsImpact.MEDIUM, 10, 10),
            # Low impact
            ("Building Permits", "USD", NewsCategory.HOUSING, NewsImpact.LOW, 10, 10),
            ("Industrial Production", "EUR", NewsCategory.MANUFACTURING, NewsImpact.LOW, 10, 10),
        ]

        for i, (title, currency, category, impact, day_offset, hour) in enumerate(mock_events):
            event_time = base_time + timedelta(days=day_offset, hours=hour)
            event = EconomicEvent(
                title=title,
                currency=currency,
                country=currency,
                category=category,
                impact=impact,
                scheduled_time=event_time,
                forecast="--",
                previous="--",
                unit="%",
                source="mock",
            )
            self._events.append(event)

        # Add some near-future events for testing
        for hours_ahead in [1, 3, 6, 12, 24]:
            for currency in ["USD", "EUR", "GBP"]:
                event = EconomicEvent(
                    title=f"Mock {currency} Event",
                    currency=currency,
                    country=currency,
                    category=NewsCategory.OTHER,
                    impact=NewsImpact.MEDIUM,
                    scheduled_time=base_time + timedelta(hours=hours_ahead),
                    forecast="--",
                    previous="--",
                    source="mock",
                )
                self._events.append(event)

    async def fetch_events(
        self,
        start: datetime,
        end: datetime,
        currencies: Optional[List[str]] = None,
        impact_filter: Optional[Set[NewsImpact]] = None,
    ) -> List[EconomicEvent]:
        filtered = [
            e for e in self._events
            if start <= e.scheduled_time <= end
        ]
        if currencies:
            filtered = [e for e in filtered if e.currency.upper() in [c.upper() for c in currencies]]
        if impact_filter:
            filtered = [e for e in filtered if e.impact in impact_filter]
        return filtered

    async def fetch_upcoming_events(
        self,
        hours_ahead: int = 24,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        now = datetime.utcnow()
        end = now + timedelta(hours=hours_ahead)
        return await self.fetch_events(now, end, currencies)

    async def get_current_risk_state(self, currency: str) -> NewsImpact:
        """Check if there's a high/medium impact event near now for currency."""
        now = datetime.utcnow()
        window = timedelta(hours=2)  # ±2 hours
        for event in self._events:
            if event.currency.upper() == currency.upper():
                if abs((event.scheduled_time - now).total_seconds()) <= window.total_seconds():
                    return event.impact
        return NewsImpact.NONE

    def get_supported_currencies(self) -> List[str]:
        return ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]

    @property
    def name(self) -> str:
        return "MockNewsProvider"


# Global provider instance
_news_provider: Optional[NewsProvider] = None


def get_news_provider() -> NewsProvider:
    """Get global news provider instance."""
    global _news_provider
    if _news_provider is None:
        _news_provider = MockNewsProvider()
    return _news_provider


def set_news_provider(provider: NewsProvider) -> None:
    """Set global news provider."""
    global _news_provider
    _news_provider = provider