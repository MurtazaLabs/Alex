"""FastAPI request/response schemas (Pydantic v2)."""

from __future__ import annotations

from decimal import Decimal
from typing import List
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PairCreate(BaseModel):
    symbol: str = Field(..., pattern=r"^[A-Z]{3}/[A-Z]{3}$")
    base_currency: str = Field(..., min_length=3, max_length=3)
    quote_currency: str = Field(..., min_length=3, max_length=3)
    pip_value: Decimal = Field(..., gt=0)
    contract_size: Decimal = Field(default=100_000, gt=0)
    min_lot_size: Decimal = Field(default=0.01, ge=0)
    max_lot_size: Decimal = Field(default=100.00, gt=0)
    spread_avg_pips: Decimal = Field(default=1.0, ge=0)
    trading_sessions: List[str] = Field(default_factory=lambda: ["London", "NewYork"])


class PairRead(PairCreate):
    model_config = ConfigDict(from_attributes=True)

    id: UUID


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(..., min_length=8, max_length=255)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    username: str
    email: str
    permission_level: int


class WatchlistCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    pairs: List[str] = Field(default_factory=list)


class WatchlistAddPairs(BaseModel):
    pairs: List[str] = Field(min_length=1)


class WatchlistRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    name: str
    pairs: List[str]
    is_default: bool
