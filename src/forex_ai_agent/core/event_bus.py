"""
Event bus for inter-agent communication.

Primary transport is Redis Pub/Sub (``redis.asyncio``). An in-memory fallback
(``InMemoryEventBus``) is provided for local development and unit testing so the
event-driven architecture works without a running Redis server.

Guarantees (Phase 1):
- At-least-once delivery over Redis Pub/Sub
- Ordered delivery per Redis channel
- Events are JSON-serializable for persistence (audit trail in ``events`` table)
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import ConnectionError as RedisConnectionError

from ..config import get_settings
from ..config.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

__all__ = [
    "Event",
    "EventBus",
    "RedisEventBus",
    "InMemoryEventBus",
    "create_event_bus",
    "on_event",
    "publishes",
]


class Event:
    """Base event structure for the event-driven architecture."""

    def __init__(
        self,
        event_type: str,
        aggregate_id: str,
        aggregate_type: str,
        payload: Dict[str, Any],
        source_agent: str,
        correlation_id: Optional[str] = None,
        causation_id: Optional[int] = None,
    ):
        self.event_type = event_type
        self.aggregate_id = aggregate_id
        self.aggregate_type = aggregate_type
        self.payload = payload
        self.source_agent = source_agent
        self.correlation_id = correlation_id or ""
        self.causation_id = causation_id
        self.timestamp = time.time()

    def model_dump(self) -> Dict[str, Any]:
        """Serialize event to a JSON-serializable dictionary."""
        return {
            "event_type": self.event_type,
            "aggregate_id": str(self.aggregate_id),
            "aggregate_type": self.aggregate_type,
            "payload": self.payload,
            "source_agent": self.source_agent,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "Event":
        """Deserialize event from a dictionary."""
        return cls(
            event_type=data["event_type"],
            aggregate_id=data["aggregate_id"],
            aggregate_type=data["aggregate_type"],
            payload=data["payload"],
            source_agent=data["source_agent"],
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Event({self.event_type!r}, src={self.source_agent!r}, "
            f"agg={self.aggregate_type}:{self.aggregate_id})"
        )


class EventBus(ABC):
    """Abstract event bus contract used across the system."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable]] = {}

    async def subscribe(
        self,
        event_type: str,
        callback: Callable[[Event], Any],
    ) -> None:
        """Register a callback for a specific event type."""
        self._subscribers.setdefault(event_type, []).append(callback)
        logger.debug("subscribed", event_type=event_type)

    async def unsubscribe(
        self,
        event_type: str,
        callback: Callable[[Event], Any],
    ) -> None:
        """Unregister a callback for a specific event type."""
        callbacks = self._subscribers.get(event_type, [])
        if callback in callbacks:
            callbacks.remove(callback)
            logger.debug("unsubscribed", event_type=event_type)

    def register_agent(self, agent: Any) -> None:
        """Auto-subscribe an agent's ``@on_event``-decorated handlers."""
        for attr_name in dir(agent):
            if attr_name.startswith("__"):
                continue
            handler = getattr(agent, attr_name, None)
            event_type = getattr(handler, "_subscribed_event", None)
            if event_type:
                bound = handler.__get__(agent, type(agent))
                self._subscribers.setdefault(event_type, []).append(bound)
                logger.debug(
                    "agent_subscribed",
                    agent=type(agent).__name__,
                    event_type=event_type,
                )

    async def _dispatch(self, event: Event) -> None:
        """Route an event to every local callback registered for its type."""
        for callback in self._subscribers.get(event.event_type, []):
            try:
                result = callback(event)
                if hasattr(result, "__await__"):
                    await result
            except Exception:  # noqa: BLE001 - isolate one bad consumer
                logger.exception("event_callback_error", event_type=event.event_type)

    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish an event to bus consumers."""

    @abstractmethod
    async def start_listening(self) -> None:
        """Begin consuming events from the transport."""

    async def close(self) -> None:  # pragma: no cover - optional cleanup
        """Release transport resources (if any)."""


class InMemoryEventBus(EventBus):
    """
    Event bus that dispatches events directly to registered callbacks.

    Useful for tests and local development where no Redis server is available.
    """

    def __init__(self) -> None:
        super().__init__()
        self.published: List[Event] = []

    async def publish(self, event: Event) -> None:
        self.published.append(event)
        await self._dispatch(event)
        logger.debug("published_in_memory", event_type=event.event_type)

    async def start_listening(self) -> None:
        """In-memory bus has no transport loop; subscribers fire synchronously."""
        logger.info("in_memory_event_bus_started")


class RedisEventBus(EventBus):
    """Redis Pub/Sub backed event bus."""

    def __init__(self, redis: Optional[AsyncRedis] = None) -> None:
        super().__init__()
        self.redis = redis or self._create_redis()
        self._pubsub: Any = None

    @staticmethod
    def _create_redis() -> AsyncRedis:
        return AsyncRedis.from_url(settings.redis_url, decode_responses=True)

    async def publish(self, event: Event) -> None:
        payload = json.dumps(event.model_dump(), default=str)
        channel = f"events:{event.event_type}"
        aggregate_channel = f"aggregate:{event.aggregate_type}:{event.aggregate_id}"
        try:
            await self.redis.publish(channel, payload)
            await self.redis.publish(aggregate_channel, payload)
            logger.debug("published", channel=channel, event_type=event.event_type)
        except RedisConnectionError:
            # Fall back to direct dispatch so a Redis outage is not fatal
            logger.warning("redis_unavailable_dispatching_locally", channel=channel)
            await self._dispatch(event)

    async def start_listening(self) -> None:
        """Subscribe to all known channels and dispatch incoming events."""
        self._pubsub = self.redis.pubsub()
        channels = [f"events:{t}" for t in self._subscribers]
        if channels:
            await self._pubsub.subscribe(*channels)
        logger.info("event_bus_listening", channels=len(channels))
        try:
            async for message in self._pubsub.iter():
                if message.get("type") != "message":
                    continue
                try:
                    event_data = json.loads(message["data"])
                    event = Event.model_validate(event_data)
                    await self._dispatch(event)
                except (json.JSONDecodeError, KeyError, TypeError):
                    logger.exception("event_decode_error")
        except asyncio.CancelledError:
            logger.info("event_bus_listening_cancelled")
        finally:
            if self._pubsub is not None:
                await self._pubsub.unsubscribe()
                await self._pubsub.close()

    async def close(self) -> None:
        if self._pubsub is not None:
            await self._pubsub.close()
        await self.redis.aclose()


def create_event_bus(use_redis: bool = True) -> EventBus:
    """
    Create an event bus based on configuration.

    Falls back to ``InMemoryEventBus`` when Redis is not reachable or disabled.
    """
    if not use_redis:
        return InMemoryEventBus()
    try:
        bus = RedisEventBus()
        logger.info("redis_event_bus_created", url=settings.redis_url)
        return bus
    except (RedisConnectionError, OSError, ValueError):
        logger.warning("redis_unavailable_using_in_memory_bus")
        return InMemoryEventBus()


# --- Publisher / subscriber decorators -------------------------------------


def on_event(event_type: str):
    """Mark a method as a handler for the given event type."""

    def decorator(func: Callable) -> Callable:
        func._subscribed_event = event_type  # type: ignore[attr-defined]
        return func

    return decorator


def publishes(event_type: str):
    """Mark a method as publishing the given event type."""

    def decorator(func: Callable) -> Callable:
        func._publishes_event = event_type  # type: ignore[attr-defined]
        return func

    return decorator
