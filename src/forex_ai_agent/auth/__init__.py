"""Authentication and authorization module for the Forex AI Agent API.

Provides:
- JWT token generation and validation
- Password hashing (bcrypt)
- Access and refresh tokens
- Rate limiting
- Permission-based access control
"""

from __future__ import annotations

import hashlib
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from forex_ai_agent.config import get_settings
from forex_ai_agent.data.database import get_session
from forex_ai_agent.data.models import User
from forex_ai_agent.data.repositories import UserRepository
from forex_ai_agent.core.permission_manager import PermissionLevel as CorePermissionLevel

settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

# JWT settings
SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Security scheme
security = HTTPBearer(auto_error=False)


class TokenData(BaseModel):
    """Decoded JWT token data."""
    sub: str  # user_id
    username: str
    permission_level: int
    exp: int
    iat: int
    token_type: str  # "access" or "refresh"


class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseModel):
    """Token payload for encoding."""
    sub: str
    username: str
    permission_level: int
    token_type: str
    exp: int
    iat: int


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_token(
    user_id: UUID,
    username: str,
    permission_level: int,
    token_type: str = "access",
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT token."""
    now = datetime.now(timezone.utc)
    
    if expires_delta:
        expire = now + expires_delta
    else:
        if token_type == "access":
            expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        else:
            expire = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    payload = TokenPayload(
        sub=str(user_id),
        username=username,
        permission_level=permission_level,
        token_type=token_type,
        exp=int(expire.timestamp()),
        iat=int(now.timestamp()),
    )
    
    encoded_jwt = jwt.encode(payload.model_dump(), SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> TokenData:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TokenData(**payload)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    session: AsyncSession = Depends(get_session),
) -> User:
    """Get current authenticated user from JWT token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_data = decode_token(credentials.credentials)
    
    if token_data.token_type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    repo = UserRepository(session)
    user = await repo.get(UUID(token_data.sub))
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    session: AsyncSession = Depends(get_session),
) -> Optional[User]:
    """Get current user if authenticated, otherwise None."""
    if not credentials:
        return None
    
    try:
        token_data = decode_token(credentials.credentials)
        
        if token_data.token_type != "access":
            return None
        
        repo = UserRepository(session)
        user = await repo.get(UUID(token_data.sub))
        
        if user and user.is_active:
            return user
    except HTTPException:
        pass
    
    return None


def require_permission(min_level: int):
    """Dependency factory for permission level requirements."""
    async def permission_checker(
        current_user: User = Depends(get_current_user),
    ) -> User:
        if current_user.permission_level < min_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission level {min_level} required",
            )
        return current_user
    
    return permission_checker


# Convenience dependencies for each permission level
require_researcher = require_permission(CorePermissionLevel.MARKET_MONITORING)      # Level 1
require_paper_trader = require_permission(CorePermissionLevel.PAPER_TRADING)        # Level 3
require_simulator = require_permission(CorePermissionLevel.SIGNAL_GENERATION)       # Level 2
require_live_trader = require_permission(CorePermissionLevel.LIVE_TRADING)          # Level 4


class RateLimitConfig(BaseModel):
    """Rate limit configuration."""
    requests: int = 100
    window_seconds: int = 60
    key_prefix: str = "ratelimit"


# In-memory rate limiter (use Redis in production)
_rate_limit_store: Dict[str, List[float]] = {}


def check_rate_limit(
    key: str,
    config: RateLimitConfig = None,
) -> bool:
    """Check if request is within rate limit."""
    if config is None:
        config = RateLimitConfig()
    
    now = time.time()
    window_start = now - config.window_seconds
    
    if key not in _rate_limit_store:
        _rate_limit_store[key] = []
    
    # Clean old entries
    _rate_limit_store[key] = [
        t for t in _rate_limit_store[key] if t > window_start
    ]
    
    if len(_rate_limit_store[key]) >= config.requests:
        return False
    
    _rate_limit_store[key].append(now)
    return True


async def rate_limit_dependency(
    request: Request,
    config: RateLimitConfig = None,
) -> None:
    """FastAPI dependency for rate limiting."""
    if config is None:
        config = RateLimitConfig()
    
    # Use client IP as key
    client_ip = request.client.host if request.client else "unknown"
    key = f"{config.key_prefix}:{client_ip}"
    
    if not check_rate_limit(key, config):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(config.window_seconds)},
        )


def create_rate_limiter(
    requests: int = 100,
    window_seconds: int = 60,
    key_prefix: str = "ratelimit",
):
    """Create a rate limiter dependency with custom config."""
    config = RateLimitConfig(
        requests=requests,
        window_seconds=window_seconds,
        key_prefix=key_prefix,
    )
    
    async def limiter(request: Request) -> None:
        await rate_limit_dependency(request, config)
    
    return limiter


# Pre-configured rate limiters
auth_rate_limiter = create_rate_limiter(
    requests=10,
    window_seconds=60,
    key_prefix="auth",
)

api_rate_limiter = create_rate_limiter(
    requests=100,
    window_seconds=60,
    key_prefix="api",
)

strict_rate_limiter = create_rate_limiter(
    requests=20,
    window_seconds=60,
    key_prefix="strict",
)


async def create_tokens_for_user(user: User) -> Token:
    """Create access and refresh tokens for a user."""
    access_token = create_token(
        user_id=user.id,
        username=user.username,
        permission_level=user.permission_level,
        token_type="access",
    )
    
    refresh_token = create_token(
        user_id=user.id,
        username=user.username,
        permission_level=user.permission_level,
        token_type="refresh",
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


async def refresh_access_token(
    refresh_token: str,
    session: AsyncSession,
) -> Token:
    """Refresh an access token using a refresh token."""
    try:
        token_data = decode_token(refresh_token)
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    
    if token_data.token_type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )
    
    repo = UserRepository(session)
    user = await repo.get(UUID(token_data.sub))
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled",
        )
    
    return await create_tokens_for_user(user)


class APIKeyManager:
    """Manage API keys for service-to-service authentication."""
    
    def __init__(self):
        self._keys: Dict[str, Dict[str, Any]] = {}
    
    def generate_key(self, name: str, permissions: List[str] = None) -> str:
        """Generate a new API key."""
        key = f"fxai_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        self._keys[key_hash] = {
            "name": name,
            "permissions": permissions or [],
            "created_at": datetime.utcnow(),
            "last_used": None,
        }
        
        return key
    
    def validate_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        key_data = self._keys.get(key_hash)
        
        if key_data:
            key_data["last_used"] = datetime.utcnow()
            return key_data
        
        return None
    
    def revoke_key(self, key: str) -> bool:
        """Revoke an API key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        if key_hash in self._keys:
            del self._keys[key_hash]
            return True
        return False


# Global API key manager
api_key_manager = APIKeyManager()


async def get_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """Validate API key from Authorization header."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    key_data = api_key_manager.validate_key(credentials.credentials)
    
    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return key_data