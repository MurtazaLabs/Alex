"""Authentication routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, SecretStr
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
    Token,
)


class UserLogin(BaseModel):
    """User login request."""
    username: str
    password: SecretStr


class TokenRefresh(BaseModel):
    """Token refresh request."""
    refresh_token: str


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=Token)
async def login(
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
async def refresh(
    payload: TokenRefresh,
    session: AsyncSession = Depends(get_session),
) -> Token:
    """Refresh access token using refresh token."""
    return await refresh_access_token(payload.refresh_token, session)


@router.get("/me")
async def get_me(
    current_user: User = Depends(get_current_user),
) -> dict:
    """Get current user info."""
    return {
        "id": str(current_user.id),
        "username": current_user.username,
        "email": current_user.email,
        "permission_level": current_user.permission_level,
        "is_active": current_user.is_active,
    }


@router.post("/logout")
async def logout() -> dict:
    """Logout endpoint (client-side token removal)."""
    return {"message": "Logged out successfully"}