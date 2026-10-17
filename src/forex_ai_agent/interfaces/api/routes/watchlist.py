"""Watchlist CRUD endpoints (Phase 1.7)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ....data.database import get_session
from ....data.repositories import WatchlistRepository
from ..schemas import WatchlistAddPairs, WatchlistCreate, WatchlistRead

router = APIRouter(tags=["watchlists"])


@router.get("/users/{user_id}/watchlists", response_model=list[WatchlistRead])
async def list_watchlists(
    user_id: UUID, session: AsyncSession = Depends(get_session)
) -> list[WatchlistRead]:
    """List all watchlists for a user."""
    repo = WatchlistRepository(session)
    items = await repo.list_for_user(user_id)
    return [WatchlistRead.model_validate(w) for w in items]


@router.post(
    "/users/{user_id}/watchlists", response_model=WatchlistRead, status_code=201
)
async def create_watchlist(
    user_id: UUID,
    payload: WatchlistCreate,
    session: AsyncSession = Depends(get_session),
) -> WatchlistRead:
    """Create a named watchlist for a user."""
    repo = WatchlistRepository(session)
    if await repo.get_by_name(user_id, payload.name) is not None:
        raise HTTPException(
            status_code=409, detail=f"Watchlist {payload.name!r} already exists"
        )
    watchlist = await repo.create(user_id, payload.name, payload.pairs)
    return WatchlistRead.model_validate(watchlist)


@router.get("/watchlists/{watchlist_id}", response_model=WatchlistRead)
async def get_watchlist(
    watchlist_id: UUID, session: AsyncSession = Depends(get_session)
) -> WatchlistRead:
    """Fetch a single watchlist."""
    repo = WatchlistRepository(session)
    watchlist = await repo.get(watchlist_id)
    if watchlist is None:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    return WatchlistRead.model_validate(watchlist)


@router.delete("/watchlists/{watchlist_id}", status_code=204)
async def delete_watchlist(
    watchlist_id: UUID, session: AsyncSession = Depends(get_session)
) -> None:
    """Delete a watchlist."""
    repo = WatchlistRepository(session)
    watchlist = await repo.get(watchlist_id)
    if watchlist is None:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    await repo.delete(watchlist)


@router.post("/watchlists/{watchlist_id}/pairs", response_model=WatchlistRead)
async def add_pairs(
    watchlist_id: UUID,
    payload: WatchlistAddPairs,
    session: AsyncSession = Depends(get_session),
) -> WatchlistRead:
    """Append pairs to a watchlist (de-duplicated)."""
    repo = WatchlistRepository(session)
    watchlist = await repo.get(watchlist_id)
    if watchlist is None:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    updated = await repo.add_pairs(watchlist, payload.pairs)
    return WatchlistRead.model_validate(updated)
