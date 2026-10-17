"""Integration tests for the FastAPI application (health, pairs, users, watchlists)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from forex_ai_agent.config.settings import Settings
from forex_ai_agent.core.event_bus import InMemoryEventBus
from forex_ai_agent.data.database import get_session
from forex_ai_agent.data.models import Base
from forex_ai_agent.interfaces.api.app import create_app


@pytest.fixture
def api_fixtures():
    """Build an app wired to an in-memory SQLite database."""
    import os

    os.environ["ENVIRONMENT"] = "test"
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    settings = Settings(
        environment="test", database_url="sqlite+aiosqlite:///:memory:"
    )

    async def override_get_session():
        async with maker() as s:
            yield s

    app = create_app(settings=settings, event_bus=InMemoryEventBus())
    app.dependency_overrides[get_session] = override_get_session
    return app, engine


@pytest.fixture
def client(api_fixtures):
    app, engine = api_fixtures
    with TestClient(app) as c:
        # create schema inside the same loop the app will use
        import asyncio

        async def setup():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

        asyncio.run(setup())
        yield c
    import asyncio

    asyncio.run(engine.dispose())


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"


def test_root_endpoint(client):
    res = client.get("/api")
    assert res.status_code == 200
    assert res.json()["name"] == "forex-ai-agent"


def test_pair_crud(client):
    created = client.post(
        "/api/pairs",
        json={
            "symbol": "EUR/USD",
            "base_currency": "EUR",
            "quote_currency": "USD",
            "pip_value": "0.0001",
        },
    )
    assert created.status_code == 201, created.text
    assert created.json()["symbol"] == "EUR/USD"

    duplicate = client.post(
        "/api/pairs",
        json={
            "symbol": "EUR/USD",
            "base_currency": "EUR",
            "quote_currency": "USD",
            "pip_value": "0.0001",
        },
    )
    assert duplicate.status_code == 409

    listed = client.get("/api/pairs")
    assert listed.status_code == 200
    assert len(listed.json()) == 1


def test_user_and_watchlist_flow(client):
    created = client.post(
        "/api/users",
        json={
            "username": "alice",
            "email": "alice@example.com",
            "password": "hunter2-secret",
        },
    )
    assert created.status_code == 201, created.text
    user_id = created.json()["id"]

    wl = client.post(
        f"/api/users/{user_id}/watchlists",
        json={"name": "Majors", "pairs": ["EUR/USD", "GBP/USD"]},
    )
    assert wl.status_code == 201, wl.text
    wl_id = wl.json()["id"]

    listed = client.get(f"/api/users/{user_id}/watchlists")
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    added = client.post(
        f"/api/watchlists/{wl_id}/pairs", json={"pairs": ["EUR/USD", "USD/JPY"]}
    )
    assert added.status_code == 200
    assert set(added.json()["pairs"]) == {"EUR/USD", "GBP/USD", "USD/JPY"}

    fetched = client.get(f"/api/watchlists/{wl_id}")
    assert fetched.status_code == 200

    deleted = client.delete(f"/api/watchlists/{wl_id}")
    assert deleted.status_code == 204

    missing = client.get(f"/api/watchlists/{wl_id}")
    assert missing.status_code == 404


def test_watchlist_duplicate_name_conflict(client):
    created = client.post(
        "/api/users",
        json={
            "username": "bob",
            "email": "bob@example.com",
            "password": "hunter2-secret",
        },
    )
    user_id = created.json()["id"]
    client.post(f"/api/users/{user_id}/watchlists", json={"name": "Main", "pairs": []})
    dup = client.post(
        f"/api/users/{user_id}/watchlists", json={"name": "Main", "pairs": []}
    )
    assert dup.status_code == 409
