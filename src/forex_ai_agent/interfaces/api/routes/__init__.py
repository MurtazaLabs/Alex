"""API route registration."""

from .health import router as health_router
from .pairs import router as pairs_router
from .users import router as users_router
from .watchlist import router as watchlist_router
from .auth import router as auth_router
from .alex import router as alex_router
from .execution import router as execution_router

__all__ = [
    "health_router", "pairs_router", "users_router",
    "watchlist_router", "auth_router", "alex_router", "execution_router",
]
