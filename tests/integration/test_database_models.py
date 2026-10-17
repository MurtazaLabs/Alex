"""Integration tests for the database layer (models + repositories)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from sqlalchemy import select

from forex_ai_agent.core.event_bus import Event
from forex_ai_agent.data.models import (
    Pair,
    Signal,
    Strategy,
    User,
    Watchlist,
)
from forex_ai_agent.data.repositories import (
    EventRepository,
    PairRepository,
    WatchlistRepository,
)


@pytest.mark.asyncio
async def test_create_user_and_watchlist(session):
    user = User(username="trader1", email="trader1@example.com", password_hash="x")
    session.add(user)
    await session.commit()
    await session.refresh(user)

    wl = Watchlist(user_id=user.id, name="Majors", pairs=["EUR/USD", "GBP/USD"])
    session.add(wl)
    await session.commit()

    assert user.permission_level == 0
    assert wl.id is not None
    assert wl.pairs == ["EUR/USD", "GBP/USD"]


@pytest.mark.asyncio
async def test_pair_repository(session):
    repo = PairRepository(session)
    pair = Pair(
        symbol="EUR/USD",
        base_currency="EUR",
        quote_currency="USD",
        pip_value=Decimal("0.0001"),
    )
    created = await repo.add(pair)

    fetched = await repo.get_by_symbol("eur/usd")
    assert fetched is not None and fetched.id == created.id

    dup_check = await repo.get_by_symbol("EUR/USD")
    assert dup_check is not None


@pytest.mark.asyncio
async def test_watchlist_repository(session):
    user = User(username="u2", email="u2@example.com", password_hash="x")
    session.add(user)
    await session.commit()
    await session.refresh(user)

    repo = WatchlistRepository(session)
    created = await repo.create(user.id, "Default", ["EUR/USD"])
    assert created.pairs == ["EUR/USD"]

    updated = await repo.add_pairs(created, ["EUR/USD", "USD/JPY"])
    assert set(updated.pairs) == {"EUR/USD", "USD/JPY"}

    items = await repo.list_for_user(user.id)
    assert len(items) == 1

    by_name = await repo.get_by_name(user.id, "Default")
    assert by_name is not None


@pytest.mark.asyncio
async def test_event_repository_roundtrip(session):
    repo = EventRepository(session)
    bus_event = Event(
        event_type="signal.created",
        aggregate_id="e57f7e9e-0000-4000-8000-000000000000",
        aggregate_type="signal",
        payload={"direction": "LONG"},
        source_agent="signal_agent",
        correlation_id="corr-1",
    )
    record = await repo.save_event(bus_event)
    assert record.event_type == "signal.created"
    assert record.payload_json == {"direction": "LONG"}

    records = await repo.list_for_aggregate("signal", bus_event.aggregate_id)
    assert len(records) == 1


@pytest.mark.asyncio
async def test_signal_and_strategy_relations(session):
    strategy = Strategy(
        name="Break-Retest v1",
        version="1.0.0",
        rules_json={"rules": []},
        parameters_json={"params": {}},
    )
    session.add(strategy)
    await session.commit()
    await session.refresh(strategy)

    pair = Pair(
        symbol="GBP/USD",
        base_currency="GBP",
        quote_currency="USD",
        pip_value=Decimal("0.0001"),
    )
    session.add(pair)
    await session.commit()
    await session.refresh(pair)

    signal = Signal(
        strategy_id=strategy.id,
        pair_id=pair.id,
        direction="LONG",
        timeframe="H1",
        setup_type="structure_break_retest",
        entry_price=Decimal("1.2500"),
        stop_loss=Decimal("1.2480"),
        take_profit_1=Decimal("1.2540"),
        risk_reward=Decimal("2.0"),
        market_state_json={"trend": "bullish"},
        supporting_evidence_json=[],
        invalidation_condition="close below SL",
        detected_at=datetime.utcnow(),
        strategy_version="1.0.0",
    )
    session.add(signal)
    await session.commit()

    stmt = select(Signal).where(Signal.pair_id == pair.id)
    result = await session.execute(stmt)
    assert result.scalar_one().direction == "LONG"
