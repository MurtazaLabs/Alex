"""User endpoints with authentication."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field, SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from forex_ai_agent.data.database import get_session
from forex_ai_agent.data.models import User
from forex_ai_agent.data.repositories import UserRepository
from forex_ai_agent.auth import (
    verify_password,
    get_password_hash,
    create_tokens_for_user,
    refresh_access_token,
    get_current_user,
    get_current_user_optional,
    require_permission,
    Token,
    auth_rate_limiter,
)
from forex_ai_agent.core.permission_manager import PermissionLevel

router = APIRouter(prefix="/users", tags=["users"])


class UserCreate(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: SecretStr = Field(..., min_length=8)


class UserRead(BaseModel):
    """User response model."""
    id: UUID
    username: str
    email: str
    permission_level: int
    is_active: bool
    created_at: str


class UserLogin(BaseModel):
    """User login request."""
    username: str
    password: SecretStr


class TokenRefresh(BaseModel):
    """Token refresh request."""
    refresh_token: str


@router.post("/register", response_model=UserRead, status_code=201)
async def register_user(
    payload: UserCreate,
    session: AsyncSession = Depends(get_session),
) -> UserRead:
    """Register a new user (permission level 0 - researcher)."""
    repo = UserRepository(session)
    
    existing = await repo.get_by_username(payload.username)
    if existing:
        raise HTTPException(
            status_code=409, detail="Username already registered"
        )
    
    existing = await repo.get_by_email(payload.email)
    if existing:
        raise HTTPException(
            status_code=409, detail="Email already registered"
        )
    
    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=get_password_hash(payload.password.get_secret_value()),
        permission_level=0,  # Researcher only by default
        is_active=True,
    )
    created = await repo.add(user)
    
    return UserRead(
        id=created.id,
        username=created.username,
        email=created.email,
        permission_level=created.permission_level,
        is_active=created.is_active,
        created_at=created.created_at.isoformat(),
    )


@router.post("", response_model=UserRead, status_code=201)
async def create_user(
    payload: UserCreate,
    session: AsyncSession = Depends(get_session),
) -> UserRead:
    """RESTful alias for ``POST /users/register``."""
    return await register_user(payload, session)


@router.post("/login", response_model=Token)
async def login_user(
    payload: UserLogin,
    session: AsyncSession = Depends(get_session),
) -> Token:
    """Authenticate user and return access/refresh tokens."""
    repo = UserRepository(session)
    user = await repo.get_by_username(payload.username)
    
    if not user or not verify_password(
        payload.password.get_secret_value(),
        user.password_hash
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )
    
    return await create_tokens_for_user(user)


@router.post("/refresh", response_model=Token)
async def refresh_token(
    payload: TokenRefresh,
    session: AsyncSession = Depends(get_session),
) -> Token:
    """Refresh access token using refresh token."""
    return await refresh_access_token(payload.refresh_token, session)


@router.get("/me", response_model=UserRead)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
) -> UserRead:
    """Get current authenticated user's info."""
    return UserRead(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        permission_level=current_user.permission_level,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat(),
    )


@router.get("/{user_id}", response_model=UserRead)
async def get_user(
    user_id: UUID,
    current_user: User = Depends(require_permission(PermissionLevel.MARKET_MONITORING)),
    session: AsyncSession = Depends(get_session),
) -> UserRead:
    """Fetch a user by id (requires researcher permission)."""
    repo = UserRepository(session)
    user = await repo.get(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserRead(
        id=user.id,
        username=user.username,
        email=user.email,
        permission_level=user.permission_level,
        is_active=user.is_active,
        created_at=user.created_at.isoformat(),
    )


@router.patch("/me/permission", response_model=UserRead)
async def update_permission(
    permission_level: int,
    current_user: User = Depends(require_permission(PermissionLevel.LIVE_TRADING)),
    session: AsyncSession = Depends(get_session),
) -> UserRead:
    """Update user permission level (requires live trader permission)."""
    if not 0 <= permission_level <= 4:
        raise HTTPException(status_code=400, detail="Invalid permission level")
    
    repo = UserRepository(session)
    user = await repo.get(current_user.id)
    user.permission_level = permission_level
    await session.commit()
    await session.refresh(user)
    
    return UserRead(
        id=user.id,
        username=user.username,
        email=user.email,
        permission_level=user.permission_level,
        is_active=user.is_active,
        created_at=user.created_at.isoformat(),
    )