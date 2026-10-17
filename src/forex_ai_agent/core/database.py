"""
Database configuration and session management.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from forex_ai_agent.config import get_settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
    
    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        if self._engine is not None:
            return
        
        # Create async engine
        self._engine = create_async_engine(
            self.settings.database_url,
            echo=self.settings.environment == "development",
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            poolclass=NullPool if self.settings.environment == "test" else None,
        )
        
        # Create session factory
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        
        # Test connection
        async with self._engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
    
    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
    
    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._engine
    
    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._session_factory
    
    async def get_session(self) -> AsyncSession:
        """Get a new database session."""
        async with self.session_factory() as session:
            yield session
    
    async def create_tables(self) -> None:
        """Create all tables (development only)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self) -> None:
        """Drop all tables (development only)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


# Global database manager instance
_database_manager: DatabaseManager | None = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager


async def init_db(database_url: str | None = None) -> None:
    """Initialize database connection."""
    manager = get_database_manager()
    if database_url:
        # Override database URL if provided
        pass
    await manager.initialize()


async def close_db() -> None:
    """Close database connections."""
    global _database_manager
    if _database_manager:
        await _database_manager.close()
        _database_manager = None


async def get_db_session():
    """Get database session for dependency injection."""
    manager = get_database_manager()
    async with manager.session_factory() as session:
        yield session