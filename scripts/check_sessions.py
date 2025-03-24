"""Script to check active proxy sessions."""
import asyncio
from sqlalchemy import text
from database import AsyncSessionLocal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_sessions():
    """Check for active proxy sessions."""
    async with AsyncSessionLocal() as db:
        # Check proxy sessions
        result = await db.execute(
            text("SELECT * FROM proxy_sessions WHERE is_active = true")
        )
        sessions = result.fetchall()
        logger.info(f"Found {len(sessions) if sessions else 0} active sessions")
        for session in sessions:
            logger.info(f"Session: {session}")

        # Check if table exists
        result = await db.execute(
            text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='proxy_sessions';
            """)
        )
        table = result.scalar()
        logger.info(f"proxy_sessions table exists: {bool(table)}")

if __name__ == "__main__":
    asyncio.run(check_sessions()) 