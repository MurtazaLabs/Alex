"""News Agent - Event detection, classification, and risk state management."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from forex_ai_agent.core.event_bus import Event, EventBus
from forex_ai_agent.news.providers import (
    EconomicEvent,
    NewsImpact,
    NewsProvider,
    get_news_provider,
)


@dataclass
class CurrencyRiskState:
    """Current news risk state for a currency."""
    currency: str
    risk_level: NewsImpact = NewsImpact.NONE
    active_events: List[str] = field(default_factory=list)  # event IDs
    next_event: Optional[datetime] = None
    next_event_title: Optional[str] = None
    minutes_to_next_event: Optional[int] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)


class NewsAgent:
    """
    News Agent monitors economic calendar and maintains real-time risk states.
    
    Responsibilities:
    - Fetch and cache economic events
    - Classify events by impact
    - Maintain per-currency risk states
    - Emit risk state change events
    - Provide upcoming event alerts
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        provider: Optional[NewsProvider] = None,
        refresh_interval_minutes: int = 60,
        alert_window_minutes: int = 30,
    ):
        self.event_bus = event_bus
        self.provider = provider or get_news_provider()
        self.refresh_interval = timedelta(minutes=refresh_interval_minutes)
        self.alert_window = timedelta(minutes=alert_window_minutes)

        # State
        self._events: Dict[str, EconomicEvent] = {}
        self._risk_states: Dict[str, CurrencyRiskState] = {}
        self._last_fetch: Optional[datetime] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Initialize risk states for all supported currencies
        for currency in self.provider.get_supported_currencies():
            self._risk_states[currency.upper()] = CurrencyRiskState(currency=currency.upper())

    async def start(self) -> None:
        """Start the news agent background tasks."""
        if self._running:
            return
        self._running = True
        await self._refresh_events()
        self._task = asyncio.create_task(self._background_loop())

    async def stop(self) -> None:
        """Stop the news agent."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _background_loop(self) -> None:
        """Background refresh loop."""
        while self._running:
            try:
                await asyncio.sleep(self.refresh_interval.total_seconds())
                await self._refresh_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                print(f"News agent refresh error: {e}")

    async def _refresh_events(self) -> None:
        """Fetch and update economic events."""
        now = datetime.utcnow()
        end = now + timedelta(days=7)

        try:
            events = await self.provider.fetch_events(now, end)
            for event in events:
                self._events[event.id] = event

            # Update risk states
            await self._update_risk_states()

            self._last_fetch = now

            # Emit event for new data
            if self.event_bus:
                await self.event_bus.publish(Event(
                    event_type="news.events.updated",
                    aggregate_id="news_agent",
                    aggregate_type="news_events",
                    payload={
                        "event_count": len(events),
                        "currencies": list(set(e.currency for e in events)),
                        "high_impact_count": sum(1 for e in events if e.impact == NewsImpact.HIGH),
                    },
                    source_agent="news_agent",
                ))
        except Exception as e:
            print(f"Failed to refresh news events: {e}")

    async def _update_risk_states(self) -> None:
        """Update per-currency risk states based on current events."""
        now = datetime.utcnow()
        alert_window = self.alert_window

        # Group events by currency
        events_by_currency: Dict[str, List[EconomicEvent]] = defaultdict(list)
        for event in self._events.values():
            events_by_currency[event.currency.upper()].append(event)

        for currency, state in self._risk_states.items():
            events = events_by_currency.get(currency, [])
            if not events:
                # No events - clear risk state
                if state.risk_level != NewsImpact.NONE:
                    old_level = state.risk_level
                    state.risk_level = NewsImpact.NONE
                    state.active_events = []
                    state.next_event = None
                    state.next_event_title = None
                    state.minutes_to_next_event = None
                    state.last_updated = now
                    await self._emit_risk_change(currency, old_level, NewsImpact.NONE)
                continue

            # Find active events (within alert window)
            active = []
            upcoming = []
            for event in events:
                time_diff = event.scheduled_time - now
                if abs(time_diff) <= self.alert_window:
                    active.append(event)
                elif time_diff > timedelta(0):
                    upcoming.append(event)

            # Determine current risk level
            old_level = state.risk_level
            if active:
                # Highest impact among active events
                state.risk_level = max((e.impact for e in active), default=NewsImpact.NONE)
                state.active_events = [e.id for e in active]
            else:
                state.risk_level = NewsImpact.NONE
                state.active_events = []

            # Next upcoming event
            if upcoming:
                next_event = min(upcoming, key=lambda e: e.scheduled_time)
                state.next_event = next_event.scheduled_time
                state.next_event_title = next_event.title
                state.minutes_to_next_event = int((next_event.scheduled_time - now).total_seconds() / 60)
            else:
                state.next_event = None
                state.next_event_title = None
                state.minutes_to_next_event = None

            state.last_updated = now

            # Emit change if risk level changed
            if old_level != state.risk_level:
                await self._emit_risk_change(currency, old_level, state.risk_level)

    async def _emit_risk_change(self, currency: str, old_level: NewsImpact, new_level: NewsImpact) -> None:
        """Emit risk state change event."""
        if not self.event_bus:
            return

        await self.event_bus.publish(Event(
            event_type="news.risk_state.changed",
            aggregate_id="news_agent",
            aggregate_type="currency_risk",
            payload={
                "currency": currency,
                "old_risk_level": old_level.value,
                "new_risk_level": new_level.value,
                "timestamp": datetime.utcnow().isoformat(),
            },
            source_agent="news_agent",
        ))

    # Public query methods

    def get_risk_state(self, currency: str) -> NewsImpact:
        """Get current risk state for a currency."""
        state = self._risk_states.get(currency.upper())
        return state.risk_level if state else NewsImpact.NONE

    def get_all_risk_states(self) -> Dict[str, CurrencyRiskState]:
        """Get all currency risk states."""
        return self._risk_states.copy()

    def get_upcoming_events(
        self,
        hours_ahead: int = 24,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        """Get upcoming events (synchronous, from cache)."""
        now = datetime.utcnow()
        end = now + timedelta(hours=hours_ahead)

        events = [
            e for e in self._events.values()
            if now <= e.scheduled_time <= end
        ]
        if currencies:
            currencies_upper = {c.upper() for c in currencies}
            events = [e for e in events if e.currency.upper() in currencies_upper]
        return sorted(events, key=lambda e: e.scheduled_time)

    def get_events_for_currency(self, currency: str) -> List[EconomicEvent]:
        """Get all cached events for a currency."""
        currency_upper = currency.upper()
        return [e for e in self._events.values() if e.currency.upper() == currency_upper]

    async def get_current_risk_state_async(self, currency: str) -> NewsImpact:
        """Get current risk state, refreshing if needed."""
        if self._last_fetch is None or datetime.utcnow() - self._last_fetch > self.refresh_interval:
            await self._refresh_events()
        return self.get_risk_state(currency)

    # Alert methods

    def get_imminent_events(self, minutes_ahead: int = 60) -> List[EconomicEvent]:
        """Get events happening within minutes_ahead."""
        now = datetime.utcnow()
        end = now + timedelta(minutes=minutes_ahead)
        return [
            e for e in self._events.values()
            if now <= e.scheduled_time <= end
        ]

    def get_high_impact_upcoming(self, hours_ahead: int = 24) -> List[EconomicEvent]:
        """Get high impact events within hours_ahead."""
        return [
            e for e in self.get_upcoming_events(hours_ahead)
            if e.impact == NewsImpact.HIGH
        ]


# Factory function
async def create_news_agent(
    event_bus: Optional[EventBus] = None,
    provider: Optional[NewsProvider] = None,
) -> NewsAgent:
    """Create and start a NewsAgent."""
    agent = NewsAgent(event_bus=event_bus, provider=provider)
    await agent.start()
    return agent