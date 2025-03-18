"""Database session management."""
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, configure_mappers

SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./data/proxy.db"

# Configure SQLAlchemy mappers
configure_mappers()

# Create engine with echo for debugging
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=True,  # Enable SQL logging
    pool_pre_ping=True,  # Enable connection health checks
    pool_size=20,  # Increase from default 5
    max_overflow=30,  # Increase from default 10
    pool_timeout=60,  # Increase from default 30
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_use_lifo=True  # Use LIFO to reduce number of connections
)

# Create session factory with explicit configuration
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Prevent expired object errors
    future=True  # Use future API
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def get_async_session() -> AsyncSession:
    """Get an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close() 