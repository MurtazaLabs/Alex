"""Tests for watchlist management: repo remove_pairs, symbol normalization,
and the /alex/watchlist/pairs API routes backing the Markets page UI."""

from __future__ import annotations

import pytest

from forex_ai_agent.data.models import User
from forex_ai_agent.data.models.user import Watchlist
from forex_ai_agent.data.repositories import WatchlistRepository
from forex_ai_agent.interfaces.api.routes.alex import _normalize_symbol


# ---------------------------------------------------------------------------
# Repository: remove_pairs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_pairs_removes_only_requested(session):
    user = User(username="wl-u1", email="wl-u1@example.com", password_hash="x")
    session.add(user)
    await session.commit()
    await session.refresh(user)

    repo = WatchlistRepository(session)
    wl = await repo.create(user.id, "default", ["EUR/USD", "GBP/USD", "USD/JPY"])

    updated = await repo.remove_pairs(wl, ["GBP/USD"])
    assert updated.pairs == ["EUR/USD", "USD/JPY"]


@pytest.mark.asyncio
async def test_remove_pairs_ignores_absent_symbols(session):
    user = User(username="wl-u2", email="wl-u2@example.com", password_hash="x")
    session.add(user)
    await session.commit()
    await session.refresh(user)

    repo = WatchlistRepository(session)
    wl = await repo.create(user.id, "default", ["EUR/USD"])

    updated = await repo.remove_pairs(wl, ["AUD/USD", "NZD/USD"])
    assert updated.pairs == ["EUR/USD"]


@pytest.mark.asyncio
async def test_remove_then_add_roundtrip(session):
    user = User(username="wl-u3", email="wl-u3@example.com", password_hash="x")
    session.add(user)
    await session.commit()
    await session.refresh(user)

    repo = WatchlistRepository(session)
    wl = await repo.create(user.id, "default", ["EUR/USD"])

    await repo.remove_pairs(wl, ["EUR/USD"])
    assert wl.pairs == []

    await repo.add_pairs(wl, ["EUR/USD", "GBP/JPY"])
    assert sorted(wl.pairs) == ["EUR/USD", "GBP/JPY"]

    back = await repo.get_by_name(user.id, "default")
    assert sorted(back.pairs) == ["EUR/USD", "GBP/JPY"]


# ---------------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("EUR/USD", "EUR/USD"),
        ("eurusd", "EUR/USD"),
        ("eur-usd", "EUR/USD"),
        ("  usdjpy ", "USD/JPY"),
        ("gbp/jpy", "GBP/JPY"),
    ],
)
def test_normalize_symbol_accepts_common_forms(raw, expected):
    assert _normalize_symbol(raw) == expected


@pytest.mark.parametrize("raw", ["EUR", "EUR/USDD", "123456", "euro!", ""])
def test_normalize_symbol_rejects_garbage(raw):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _normalize_symbol(raw)
    assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# API routes (in-memory app, same pattern as test_alex_routes.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def wl_client(monkeypatch):
    """App + auth headers wired to an in-memory DB with the schema created."""
    import asyncio
    import os

    from fastapi.testclient import TestClient
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.pool import StaticPool

    from forex_ai_agent.core.event_bus import InMemoryEventBus
    from forex_ai_agent.data.database import get_session
    from forex_ai_agent.data.models import Base
    from forex_ai_agent.interfaces.api.app import create_app

    os.environ["ENVIRONMENT"] = "test"
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_session():
        async with maker() as s:
            yield s

    from forex_ai_agent.config.settings import Settings

    settings = Settings(
        environment="test", database_url="sqlite+aiosqlite:///:memory:"
    )
    app = create_app(settings=settings, event_bus=InMemoryEventBus())
    app.dependency_overrides[get_session] = override_get_session

    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(setup())

    with TestClient(app) as c:
        # user + login
        res = c.post(
            "/api/users",
            json={
                "username": "wluser",
                "email": "wluser@example.com",
                "password": "secret-pass-123",
            },
        )
        assert res.status_code == 201, res.text
        login = c.post(
            "/api/auth/login", json={"username": "wluser", "password": "secret-pass-123"}
        )
        assert login.status_code == 200, login.text
        token = login.json()["access_token"]
        yield c, {"Authorization": f"Bearer {token}"}

    asyncio.run(engine.dispose())


def test_watchlist_mutations_roundtrip(wl_client):
    client, headers = wl_client

    # add
    res = client.post(
        "/api/alex/watchlist/pairs", json={"pair": "usdcad"}, headers=headers
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["pairs"] == ["USD/CAD"]
    assert body["source"] == "watchlist"

    # add is idempotent
    res = client.post("/api/alex/watchlist/pairs", json={"pair": "USD/CAD"}, headers=headers)
    assert res.status_code == 200
    assert res.json()["pairs"] == ["USD/CAD"]

    # remove
    res = client.delete("/api/alex/watchlist/pairs?symbol=USDCAD", headers=headers)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["pairs"] == []
    # empty watchlist -> UI falls back to defaults
    assert body["source"] == "defaults"

    # removing an absent symbol is a no-op, not an error
    res = client.delete("/api/alex/watchlist/pairs?symbol=AUD/USD", headers=headers)
    assert res.status_code == 200
    assert res.json()["pairs"] == []


def test_watchlist_requires_auth(wl_client):
    client, _ = wl_client
    res = client.post("/api/alex/watchlist/pairs", json={"pair": "EUR/USD"})
    assert res.status_code in (401, 403)
    res = client.delete("/api/alex/watchlist/pairs?symbol=EUR/USD")
    assert res.status_code in (401, 403)


def test_watchlist_rejects_bad_symbol(wl_client):
    client, headers = wl_client
    res = client.post("/api/alex/watchlist/pairs", json={"pair": "nope"}, headers=headers)
    assert res.status_code == 422
