"""Unit tests for the event bus (event model + in-memory dispatch)."""

from __future__ import annotations

import asyncio

import pytest

from forex_ai_agent.core.event_bus import (
    Event,
    InMemoryEventBus,
    on_event,
    publishes,
)


def test_event_roundtrip():
    event = Event(
        event_type="market.tick.received",
        aggregate_id="7468e6ea-4f2c-4baa-bd0b-23349e1b1e5e",
        aggregate_type="pair",
        payload={"bid": 1.08, "ask": 1.0802},
        source_agent="market_agent",
        correlation_id="corr-123",
    )
    dumped = event.model_dump()
    restored = Event.model_validate(dumped)
    assert restored.event_type == event.event_type
    assert restored.payload == event.payload
    assert restored.aggregate_id == event.aggregate_id
    assert restored.causation_id is None


@pytest.mark.asyncio
async def test_in_memory_publish_dispatches_to_subscribers():
    bus = InMemoryEventBus()
    received = []

    async def handler(event: Event):
        received.append(event)

    await bus.subscribe("signal.created", handler)
    event = Event(
        event_type="signal.created",
        aggregate_id="agg-1",
        aggregate_type="signal",
        payload={"direction": "LONG"},
        source_agent="signal_agent",
    )
    await bus.publish(event)
    assert len(received) == 1
    assert received[0].payload["direction"] == "LONG"


@pytest.mark.asyncio
async def test_subscriber_isolation_on_error():
    bus = InMemoryEventBus()
    calls = []

    async def bad_handler(event: Event):
        raise RuntimeError("boom")

    async def good_handler(event: Event):
        calls.append(event)

    await bus.subscribe("x.y", bad_handler)
    await bus.subscribe("x.y", good_handler)
    await bus.publish(Event("x.y", "agg", "pair", {}, "test_agent"))
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_unsubscribe():
    bus = InMemoryEventBus()
    seen = []

    async def handler(event: Event):
        seen.append(event)

    await bus.subscribe("a.b", handler)
    await bus.unsubscribe("a.b", handler)
    await bus.publish(Event("a.b", "agg", "pair", {}, "t"))
    assert seen == []


def test_on_event_decorator_and_register_agent():
    bus = InMemoryEventBus()

    class Spy:
        name = "spy"

        def __init__(self):
            self.calls = []

        @on_event("research.requested")
        async def handle_research(self, event):
            self.calls.append(event.event_type)

        def ignore_me(self, event):
            pass

    spy = Spy()
    bus.register_agent(spy)
    assert "research.requested" in bus._subscribers

    async def drive():
        await bus.publish(Event("research.requested", "a", "task", {}, "orch"))

    asyncio.run(drive())
    assert spy.calls == ["research.requested"]


def test_publishes_decorator_marks_method():
    class Producer:
        @publishes("market.data.updated")
        def emit(self):
            return "hi"

    assert getattr(Producer().emit, "_publishes_event") == "market.data.updated"


@pytest.mark.asyncio
async def test_published_history():
    bus = InMemoryEventBus()
    await bus.publish(Event("one", "a", "pair", {}, "x"))
    await bus.publish(Event("two", "b", "pair", {}, "x"))
    assert [e.event_type for e in bus.published] == ["one", "two"]
