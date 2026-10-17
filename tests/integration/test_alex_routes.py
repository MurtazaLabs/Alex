"""Integration tests for the /api/alex routes backing the alex-ui SPA."""

from __future__ import annotations

import os

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
    """App wired to an in-memory SQLite database (same pattern as test_api)."""
    os.environ["ENVIRONMENT"] = "test"
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    settings = Settings(environment="test", database_url="sqlite+aiosqlite:///:memory:")

    async def override_get_session():
        async with maker() as s:
            yield s

    app = create_app(settings=settings, event_bus=InMemoryEventBus())
    app.dependency_overrides[get_session] = override_get_session

    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    import asyncio

    asyncio.run(setup())
    yield app, engine

    asyncio.run(engine.dispose())


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


def _register_and_login(client) -> dict:
    """Create a user via the API and return auth headers."""
    res = client.post(
        "/api/users",
        json={
            "username": "uiuser",
            "email": "uiuser@example.com",
            "password": "secret-pass-123",
        },
    )
    assert res.status_code == 201, res.text
    login = client.post(
        "/api/auth/login", json={"username": "uiuser", "password": "secret-pass-123"}
    )
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_alex_routes_require_auth(client):
    for path in ("/api/alex/overview", "/api/alex/signals", "/api/alex/strategies/probation"):
        res = client.get(path)
        assert res.status_code in (401, 403), f"{path} must be authenticated"


def test_alex_overview_shape(client, monkeypatch):
    headers = _register_and_login(client)

    async def fake_snapshot():
        return {"available": False, "note": "no mt5 in tests"}

    monkeypatch.setattr(
        "forex_ai_agent.interfaces.api.routes.alex._account_snapshot_safe",
        fake_snapshot,
    )
    res = client.get("/api/alex/overview", headers=headers)
    assert res.status_code == 200, res.text
    body = res.json()
    assert "data_source" in body
    assert "watchlist" in body
    assert "probation_active" in body
    assert "recent_signals" in body


def test_alex_chat_endpoint(client):
    headers = _register_and_login(client)
    res = client.post(
        "/api/alex/chat", json={"message": "What's happening with EUR/USD?"}, headers=headers
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert isinstance(body.get("answer"), str) and body["answer"]
    assert isinstance(body.get("tools_used"), list)


def test_probation_approval_requires_pass(client):
    """A non-eligible strategy must NOT be approvable — the human gate holds."""
    headers = _register_and_login(client)
    res = client.post("/api/alex/strategies/probation/doesnotexist/approve", headers=headers)
    assert res.status_code == 404


def test_spa_served(client):
    """The static SPA mount serves the alex-ui shell at /."""
    res = client.get("/")
    if res.status_code == 404:
        pytest.skip("alex-ui directory not mounted in this environment")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
