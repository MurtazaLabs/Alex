"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Ensure the src layout is importable for tests run from anywhere
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("ENVIRONMENT", "test")

from forex_ai_agent.core.event_bus import InMemoryEventBus  # noqa: E402
from forex_ai_agent.data.models import Base  # noqa: E402


@pytest.fixture
def event_bus() -> InMemoryEventBus:
    """An in-memory event bus for unit tests."""
    return InMemoryEventBus()


@pytest.fixture
async def db_engine():
    """In-memory SQLite engine (shared connection) with the full schema."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session_factory(db_engine) -> async_sessionmaker:
    """Session factory bound to the in-memory database."""
    return async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def session(session_factory) -> AsyncSession:
    """An async session for repository / model tests."""
    async with session_factory() as s:
        yield s
