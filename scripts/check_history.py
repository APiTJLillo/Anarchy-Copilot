"""Script to check proxy history entries."""
import asyncio
from sqlalchemy import text
from database import AsyncSessionLocal
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_history():
    """Check proxy history entries."""
    async with AsyncSessionLocal() as db:
        # Get latest entries
        result = await db.execute(
            text("""
                SELECT id, session_id, timestamp, method, url, 
                       request_headers, request_body,
                       status_code, response_headers, response_body,
                       duration, tags, is_intercepted,
                       tls_version, cipher_suite
                FROM proxy_history 
                ORDER BY timestamp DESC LIMIT 5
            """)
        )
        entries = result.fetchall()
        
        if not entries:
            logger.info("No proxy history entries found")
            return
            
        logger.info(f"Found {len(entries)} recent proxy history entries:")
        for entry in entries:
            logger.info("-" * 50)
            logger.info(f"ID: {entry.id}")
            logger.info(f"Session: {entry.session_id}")
            logger.info(f"Time: {entry.timestamp}")
            logger.info(f"Method: {entry.method}")
            logger.info(f"URL: {entry.url}")
            logger.info(f"Request headers: {entry.request_headers}")
            logger.info(f"Request body: {entry.request_body[:200] if entry.request_body else None}")
            logger.info(f"Response status: {entry.status_code}")
            logger.info(f"Response headers: {entry.response_headers}")
            logger.info(f"Response body: {entry.response_body[:200] if entry.response_body else None}")
            logger.info(f"Duration: {entry.duration}")
            logger.info(f"Tags: {entry.tags}")
            logger.info(f"Intercepted: {entry.is_intercepted}")
            logger.info(f"TLS Version: {entry.tls_version}")
            logger.info(f"Cipher Suite: {entry.cipher_suite}")

if __name__ == "__main__":
    asyncio.run(check_history()) 