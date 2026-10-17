"""Forex Factory Economic Calendar Provider.

Fetches economic events from Forex Factory's calendar.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import httpx
from bs4 import BeautifulSoup

from .providers import (
    EconomicEvent,
    NewsCategory,
    NewsImpact,
    NewsProvider,
)

logger = logging.getLogger(__name__)


class ForexFactoryNewsProvider(NewsProvider):
    """Forex Factory economic calendar provider.

    Scrapes the Forex Factory calendar page for economic events.
    """

    BASE_URL = "https://www.forexfactory.com/calendar"
    CURRENCY_MAP = {
        "USD": "usd",
        "EUR": "eur",
        "GBP": "gbp",
        "JPY": "jpy",
        "CHF": "chf",
        "AUD": "aud",
        "CAD": "cad",
        "NZD": "nzd",
        "CNY": "cny",
    }

    IMPACT_MAP = {
        "high": NewsImpact.HIGH,
        "medium": NewsImpact.MEDIUM,
        "low": NewsImpact.LOW,
        "non-economic": NewsImpact.NONE,
    }

    CATEGORY_MAP = {
        "interest rate": NewsCategory.INTEREST_RATE,
        "inflation": NewsCategory.INFLATION,
        "employment": NewsCategory.EMPLOYMENT,
        "gdp": NewsCategory.GDP,
        "trade": NewsCategory.TRADE_BALANCE,
        "consumer": NewsCategory.CONSUMER_CONFIDENCE,
        "manufacturing": NewsCategory.MANUFACTURING,
        "services": NewsCategory.SERVICES,
        "housing": NewsCategory.HOUSING,
        "speech": NewsCategory.CENTRAL_BANK_SPEECH,
    }

    def __init__(
        self,
        timeout: int = 30,
        user_agent: str = None,
        cache_ttl: int = 3600,
    ):
        self.timeout = timeout
        self.user_agent = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, List[EconomicEvent]] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            )
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _get_cache_key(self, start: datetime, end: datetime, currencies: Optional[List[str]]) -> str:
        """Generate cache key."""
        cur_str = ",".join(sorted(currencies)) if currencies else "all"
        return f"{start.date()}_{end.date()}_{cur_str}"

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache is still valid."""
        if key not in self._cache_time:
            return False
        age = (datetime.utcnow() - self._cache_time[key]).total_seconds()
        return age < self.cache_ttl

    async def _fetch_calendar_page(
        self,
        start: datetime,
        end: datetime,
        currencies: Optional[List[str]] = None,
    ) -> str:
        """Fetch calendar page HTML."""
        client = await self._get_client()

        params = {
            "day": start.strftime("%b%d.%Y"),
            "week": "this",
        }

        if currencies:
            currency_params = ",".join([self.CURRENCY_MAP.get(c.upper(), c.lower()) for c in currencies])
            params["currency"] = currency_params

        url = f"{self.BASE_URL}.php"

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch Forex Factory calendar: {e}")
            raise

    def _parse_calendar_html(self, html: str, start: datetime, end: datetime) -> List[EconomicEvent]:
        """Parse Forex Factory calendar HTML into EconomicEvent objects."""
        soup = BeautifulSoup(html, "html.parser")
        events = []

        calendar_table = soup.find("table", class_="calendar__table")
        if not calendar_table:
            logger.warning("Could not find calendar table in Forex Factory HTML")
            return events

        current_date = None

        for row in calendar_table.find_all("tr", class_="calendar__row"):
            # Check for date row
            date_cell = row.find("td", class_="calendar__date")
            if date_cell and date_cell.get_text(strip=True):
                date_text = date_cell.get_text(strip=True)
                try:
                    current_date = datetime.strptime(f"{date_text} {start.year}", "%a %b %d %Y")
                except ValueError:
                    pass
                continue

            # Skip non-event rows
            if "calendar__row--day-break" in row.get("class", []):
                continue

            # Parse event row
            event = self._parse_event_row(row, current_date)
            if event and start <= event.scheduled_time <= end:
                events.append(event)

        return events

    def _parse_event_row(self, row: BeautifulSoup, current_date: Optional[datetime]) -> Optional[EconomicEvent]:
        """Parse a single event row."""
        try:
            # Time
            time_cell = row.find("td", class_="calendar__time")
            time_text = time_cell.get_text(strip=True) if time_cell else ""
            event_time = self._parse_time(time_text, current_date)

            # Currency
            currency_cell = row.find("td", class_="calendar__currency")
            currency = currency_cell.get_text(strip=True).upper() if currency_cell else "USD"

            # Impact
            impact_cell = row.find("td", class_="calendar__impact")
            impact = NewsImpact.LOW
            if impact_cell:
                impact_span = impact_cell.find("span")
                if impact_span:
                    impact_class = impact_span.get("class", [])
                    for cls in impact_class:
                        if cls in self.IMPACT_MAP:
                            impact = self.IMPACT_MAP[cls]
                            break

            # Event name
            event_cell = row.find("td", class_="calendar__event")
            title = event_cell.get_text(strip=True) if event_cell else ""

            # Category inference from title
            category = self._infer_category(title)

            # Forecast/Previous/Actual
            forecast_cell = row.find("td", class_="calendar__forecast")
            forecast = forecast_cell.get_text(strip=True) if forecast_cell else None

            previous_cell = row.find("td", class_="calendar__previous")
            previous = previous_cell.get_text(strip=True) if previous_cell else None

            actual_cell = row.find("td", class_="calendar__actual")
            actual = actual_cell.get_text(strip=True) if actual_cell else None

            # Create event
            event = EconomicEvent(
                title=title,
                currency=currency,
                country=currency,
                category=category,
                impact=impact,
                scheduled_time=event_time,
                forecast=forecast,
                previous=previous,
                actual=actual,
                source="forex_factory",
                url=f"https://www.forexfactory.com/calendar.php?day={event_time.strftime('%b%d.%Y')}",
            )

            return event

        except Exception as e:
            logger.debug(f"Failed to parse event row: {e}")
            return None

    def _parse_time(self, time_text: str, current_date: Optional[datetime]) -> datetime:
        """Parse time string to datetime."""
        if not current_date:
            current_date = datetime.utcnow()

        # Handle "All Day" or empty
        if not time_text or time_text.lower() in ["all day", "tentative"]:
            return current_date.replace(hour=0, minute=0, second=0, microsecond=0)

        # Parse time like "8:30am" or "10:00am"
        try:
            time_obj = datetime.strptime(time_text.strip(), "%I:%M%p")
            return current_date.replace(
                hour=time_obj.hour,
                minute=time_obj.minute,
                second=0,
                microsecond=0,
            )
        except ValueError:
            try:
                time_obj = datetime.strptime(time_text.strip(), "%H:%M")
                return current_date.replace(
                    hour=time_obj.hour,
                    minute=time_obj.minute,
                    second=0,
                    microsecond=0,
                )
            except ValueError:
                return current_date

    def _infer_category(self, title: str) -> NewsCategory:
        """Infer category from event title."""
        title_lower = title.lower()
        for keyword, category in self.CATEGORY_MAP.items():
            if keyword in title_lower:
                return category
        return NewsCategory.OTHER

    async def fetch_events(
        self,
        start: datetime,
        end: datetime,
        currencies: Optional[List[str]] = None,
        impact_filter: Optional[Set[NewsImpact]] = None,
    ) -> List[EconomicEvent]:
        """Fetch economic events from Forex Factory."""
        cache_key = self._get_cache_key(start, end, currencies)

        if self._is_cache_valid(cache_key):
            cached = self._cache[cache_key]
            if impact_filter:
                cached = [e for e in cached if e.impact in impact_filter]
            return cached

        try:
            # Fetch pages for each week in range
            all_events = []
            current = start
            while current <= end:
                html = await self._fetch_calendar_page(current, end, currencies)
                events = self._parse_calendar_html(html, current, end)
                all_events.extend(events)
                current += timedelta(weeks=1)

            # Deduplicate by title + time + currency
            seen = set()
            unique_events = []
            for e in all_events:
                key = (e.title, e.scheduled_time, e.currency)
                if key not in seen:
                    seen.add(key)
                    unique_events.append(e)

            self._cache[cache_key] = unique_events
            self._cache_time[cache_key] = datetime.utcnow()

            if impact_filter:
                unique_events = [e for e in unique_events if e.impact in impact_filter]

            return unique_events

        except Exception as e:
            logger.error(f"Forex Factory fetch failed: {e}")
            # Return cached if available
            if cache_key in self._cache:
                return self._cache[cache_key]
            return []

    async def fetch_upcoming_events(
        self,
        hours_ahead: int = 24,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        """Fetch upcoming events within hours_ahead."""
        now = datetime.utcnow()
        end = now + timedelta(hours=hours_ahead)
        return await self.fetch_events(now, end, currencies)

    async def get_current_risk_state(self, currency: str) -> NewsImpact:
        """Get current news risk state for a currency."""
        now = datetime.utcnow()
        events = await self.fetch_upcoming_events(hours_ahead=2, currencies=[currency])

        max_impact = NewsImpact.NONE
        for event in events:
            if event.currency.upper() == currency.upper():
                time_diff = abs((event.scheduled_time - now).total_seconds())
                if time_diff <= 7200:  # 2 hours
                    if event.impact.value > max_impact.value:
                        max_impact = event.impact

        return max_impact

    def get_supported_currencies(self) -> List[str]:
        return list(self.CURRENCY_MAP.keys())

    @property
    def name(self) -> str:
        return "ForexFactoryNewsProvider"


class InvestingComNewsProvider(NewsProvider):
    """Investing.com Economic Calendar Provider.

    Uses Investing.com's economic calendar API/endpoint.
    """

    BASE_URL = "https://api.investing.com/api/finance/calendar"
    CURRENCY_MAP = {
        "USD": 5,
        "EUR": 4,
        "GBP": 3,
        "JPY": 6,
        "CHF": 7,
        "AUD": 8,
        "CAD": 9,
        "NZD": 10,
    }

    IMPACT_MAP = {
        3: NewsImpact.HIGH,
        2: NewsImpact.MEDIUM,
        1: NewsImpact.LOW,
        0: NewsImpact.NONE,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        cache_ttl: int = 3600,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, List[EconomicEvent]] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
                "X-Requested-With": "XMLHttpRequest",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=headers,
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _get_cache_key(self, start: datetime, end: datetime, currencies: Optional[List[str]]) -> str:
        cur_str = ",".join(sorted(currencies)) if currencies else "all"
        return f"{start.date()}_{end.date()}_{cur_str}"

    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache_time:
            return False
        age = (datetime.utcnow() - self._cache_time[key]).total_seconds()
        return age < self.cache_ttl

    async def _fetch_calendar(
        self,
        start: datetime,
        end: datetime,
        currencies: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch calendar data from Investing.com API."""
        client = await self._get_client()

        params = {
            "dateFrom": start.strftime("%Y-%m-%d"),
            "dateTo": end.strftime("%Y-%m-%d"),
        }

        if currencies:
            currency_ids = [str(self.CURRENCY_MAP.get(c.upper(), 5)) for c in currencies]
            params["country"] = ",".join(currency_ids)

        try:
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            # Handle different response formats
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            elif isinstance(data, list):
                return data
            return []

        except httpx.HTTPError as e:
            logger.error(f"Investing.com fetch failed: {e}")
            return []

    def _parse_investing_event(self, item: Dict[str, Any]) -> Optional[EconomicEvent]:
        """Parse Investing.com event format."""
        try:
            # Parse timestamp
            timestamp = item.get("timestamp") or item.get("date")
            if isinstance(timestamp, (int, float)):
                event_time = datetime.fromtimestamp(timestamp / 1000 if timestamp > 1e10 else timestamp)
            elif isinstance(timestamp, str):
                event_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                event_time = datetime.utcnow()

            # Currency
            currency = item.get("currency", "USD")
            country = item.get("country", currency)

            # Impact
            importance = item.get("importance", 1)
            impact = self.IMPACT_MAP.get(importance, NewsImpact.LOW)

            # Category
            category = self._infer_category(item.get("event", ""))

            return EconomicEvent(
                id=str(item.get("id", "")),
                title=item.get("event", ""),
                currency=currency,
                country=country,
                category=category,
                impact=impact,
                scheduled_time=event_time,
                forecast=str(item.get("forecast", "")) if item.get("forecast") else None,
                previous=str(item.get("previous", "")) if item.get("previous") else None,
                actual=str(item.get("actual", "")) if item.get("actual") else None,
                unit=item.get("unit", ""),
                source="investing_com",
                url=item.get("url"),
            )
        except Exception as e:
            logger.debug(f"Failed to parse Investing.com event: {e}")
            return None

    def _infer_category(self, title: str) -> NewsCategory:
        title_lower = title.lower()
        if any(k in title_lower for k in ["rate", "interest", "fed", "ecb", "boe", "boj"]):
            return NewsCategory.INTEREST_RATE
        elif any(k in title_lower for k in ["cpi", "inflation", "pce", "ppi"]):
            return NewsCategory.INFLATION
        elif any(k in title_lower for k in ["employment", "jobs", "unemployment", "nfp", "claims"]):
            return NewsCategory.EMPLOYMENT
        elif "gdp" in title_lower:
            return NewsCategory.GDP
        elif any(k in title_lower for k in ["trade", "balance", "exports", "imports"]):
            return NewsCategory.TRADE_BALANCE
        elif any(k in title_lower for k in ["consumer", "confidence", "sentiment"]):
            return NewsCategory.CONSUMER_CONFIDENCE
        elif any(k in title_lower for k in ["manufacturing", "pmi", "industrial"]):
            return NewsCategory.MANUFACTURING
        elif any(k in title_lower for k in ["services", "ism", "pmi"]):
            return NewsCategory.SERVICES
        elif any(k in title_lower for k in ["housing", "building", "permits", "starts"]):
            return NewsCategory.HOUSING
        elif any(k in title_lower for k in ["speech", "testimony", "minutes"]):
            return NewsCategory.CENTRAL_BANK_SPEECH
        return NewsCategory.OTHER

    async def fetch_events(
        self,
        start: datetime,
        end: datetime,
        currencies: Optional[List[str]] = None,
        impact_filter: Optional[Set[NewsImpact]] = None,
    ) -> List[EconomicEvent]:
        cache_key = self._get_cache_key(start, end, currencies)

        if self._is_cache_valid(cache_key):
            cached = self._cache[cache_key]
            if impact_filter:
                cached = [e for e in cached if e.impact in impact_filter]
            return cached

        raw_events = await self._fetch_calendar(start, end, currencies)
        events = []
        for item in raw_events:
            event = self._parse_investing_event(item)
            if event and start <= event.scheduled_time <= end:
                events.append(event)

        self._cache[cache_key] = events
        self._cache_time[cache_key] = datetime.utcnow()

        if impact_filter:
            events = [e for e in events if e.impact in impact_filter]

        return events

    async def fetch_upcoming_events(
        self,
        hours_ahead: int = 24,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        now = datetime.utcnow()
        end = now + timedelta(hours=hours_ahead)
        return await self.fetch_events(now, end, currencies)

    async def get_current_risk_state(self, currency: str) -> NewsImpact:
        now = datetime.utcnow()
        events = await self.fetch_upcoming_events(hours_ahead=2, currencies=[currency])

        max_impact = NewsImpact.NONE
        for event in events:
            if event.currency.upper() == currency.upper():
                time_diff = abs((event.scheduled_time - now).total_seconds())
                if time_diff <= 7200:
                    if event.impact.value > max_impact.value:
                        max_impact = event.impact

        return max_impact

    def get_supported_currencies(self) -> List[str]:
        return list(self.CURRENCY_MAP.keys())

    @property
    def name(self) -> str:
        return "InvestingComNewsProvider"