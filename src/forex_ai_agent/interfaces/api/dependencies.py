"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from ...data.database import get_session


async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Alias dependency so routers can inject a database session."""
    async for session in get_session():
        yield session


__all__ = ["db_session", "get_session"]
