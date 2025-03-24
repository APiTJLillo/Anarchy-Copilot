"""Script to check database tables."""
import asyncio
from sqlalchemy import text
from database import AsyncSessionLocal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_tables():
    """Check all tables in the database."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("""
                SELECT name FROM sqlite_master 
                WHERE type='table'
                ORDER BY name;
            """)
        )
        tables = result.fetchall()
        
        logger.info("Database tables:")
        for table in tables:
            logger.info(f"- {table[0]}")
            
            # Get table schema
            schema = await db.execute(
                text(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=:name"),
                {"name": table[0]}
            )
            schema_sql = schema.scalar()
            logger.info(f"Schema:\n{schema_sql}\n")

if __name__ == "__main__":
    asyncio.run(check_tables()) 