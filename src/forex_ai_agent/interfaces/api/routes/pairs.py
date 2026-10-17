"""Pair registry endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ....data.database import get_session
from ....data.repositories import PairRepository
from ..schemas import PairCreate, PairRead

router = APIRouter(prefix="/pairs", tags=["pairs"])


@router.get("", response_model=list[PairRead])
async def list_pairs(session: AsyncSession = Depends(get_session)) -> list[PairRead]:
    """List all registered pairs."""
    repo = PairRepository(session)
    pairs = await repo.list_active()
    return [PairRead.model_validate(p) for p in pairs]


@router.post("", response_model=PairRead, status_code=201)
async def create_pair(
    payload: PairCreate, session: AsyncSession = Depends(get_session)
) -> PairRead:
    """Register a new pair."""
    from ....data.models import Pair

    repo = PairRepository(session)
    existing = await repo.get_by_symbol(payload.symbol)
    if existing is not None:
        raise HTTPException(
            status_code=409, detail=f"Pair {payload.symbol} already exists"
        )

    pair = await repo.add(
        Pair(symbol=payload.symbol.upper(), **payload.model_dump(exclude={"symbol"}))
    )
    return PairRead.model_validate(pair)


@router.get("/{symbol}", response_model=PairRead)
async def get_pair(
    symbol: str, session: AsyncSession = Depends(get_session)
) -> PairRead:
    """Fetch a pair by symbol."""
    repo = PairRepository(session)
    pair = await repo.get_by_symbol(symbol)
    if pair is None:
        raise HTTPException(status_code=404, detail=f"Pair {symbol} not found")
    return PairRead.model_validate(pair)
