"""Database session management."""
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./data/proxy.db"

engine = create_async_engine(SQLALCHEMY_DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session."""
    async with AsyncSessionLocal() as session:
        yield session

async def get_async_session() -> AsyncSession:
    """Get an async database session."""
    return AsyncSessionLocal() 