"""History management for proxy requests."""
import asyncio
import logging
import aiohttp
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from .database_models import ProxyHistoryEntry, ProxySession
from .connection import connection_manager

logger = logging.getLogger("proxy.core")

# Global variables for WebSocket connection
_session: Optional[aiohttp.ClientSession] = None
_ws: Optional[aiohttp.ClientWebSocketResponse] = None
_connect_lock = asyncio.Lock()
_reconnect_task: Optional[asyncio.Task] = None
_MAX_RETRIES = 5
_RETRY_DELAY = 3

async def _reconnect_websocket():
    """Background task to handle WebSocket reconnection."""
    global _ws, _session
    retries = 0
    
    while retries < _MAX_RETRIES:
        try:
            if _ws is None or _ws.closed:
                logger.info("Attempting to reconnect to dev container WebSocket...")
                if _session is None or _session.closed:
                    _session = aiohttp.ClientSession()
                    logger.debug("Created new aiohttp session")
                
                logger.debug("Attempting WebSocket connection to ws://localhost:8000/api/proxy/ws/internal")
                logger.debug("Current WebSocket state: _ws=%s, _session=%s", 
                           "closed" if _ws and _ws.closed else "None" if _ws is None else "open",
                           "closed" if _session and _session.closed else "None" if _session is None else "open")
                
                _ws = await _session.ws_connect('ws://localhost:8000/api/proxy/ws/internal')
                logger.info("Successfully reconnected to dev container WebSocket")
                
                # Send a test message to verify connection
                logger.debug("Sending test message to verify connection")
                await _ws.send_json({"type": "test_connection"})
                logger.debug("Waiting for test message response...")
                
                # Wait for response with timeout
                try:
                    response = await asyncio.wait_for(_ws.receive_json(), timeout=5.0)
                    if response.get("type") == "test_connection_response" and response.get("status") == "ok":
                        logger.info("Test message successful, connection verified")
                        return True
                    else:
                        logger.error(f"Unexpected test message response: {response}")
                        raise Exception("Invalid test message response")
                except asyncio.TimeoutError:
                    logger.error("Timeout waiting for test message response")
                    raise Exception("Test message timeout")
                
        except Exception as e:
            retries += 1
            logger.error(f"Failed to reconnect (attempt {retries}/{_MAX_RETRIES}): {e}")
            if _session and not _session.closed:
                await _session.close()
            _session = None
            _ws = None
            await asyncio.sleep(_RETRY_DELAY)
    
    logger.error("Max reconnection attempts reached")
    return False

async def ensure_dev_connection():
    """Ensure a connection to the dev container exists."""
    global _session, _ws, _reconnect_task
    
    async with _connect_lock:
        try:
            # Check if we need to reconnect
            if _ws is None or _ws.closed:
                logger.info("WebSocket connection is not active, attempting to reconnect")
                # Cancel any existing reconnect task
                if _reconnect_task and not _reconnect_task.done():
                    _reconnect_task.cancel()
                    try:
                        await _reconnect_task
                    except asyncio.CancelledError:
                        pass
                
                # Start new reconnect task
                _reconnect_task = asyncio.create_task(_reconnect_websocket())
                success = await _reconnect_task
                
                if not success:
                    raise Exception("Failed to establish WebSocket connection after retries")
            
            return True
                
        except Exception as e:
            logger.error(f"Failed to connect to dev container: {e}")
            # Clean up if connection failed
            if _session and not _session.closed:
                await _session.close()
            _session = None
            _ws = None
            raise

async def broadcast_to_dev(data: dict):
    """Send data to dev container via WebSocket."""
    global _ws
    
    try:
        await ensure_dev_connection()
        if _ws and not _ws.closed:
            logger.debug(f"Broadcasting to dev container: {data}")
            await _ws.send_json(data)
            logger.debug(f"Successfully sent data to dev container: {data['type']}")
        else:
            logger.error("WebSocket connection is not available")
            # Try to reconnect
            await ensure_dev_connection()
            if _ws and not _ws.closed:
                await _ws.send_json(data)
            else:
                logger.error("Failed to send data after reconnection attempt")
    except Exception as e:
        logger.error(f"Failed to broadcast to dev container: {e}")
        # Reset connection on failure
        _ws = None

async def create_history_entry(
    db: AsyncSession,
    session_id: int,
    method: str,
    url: str,
    host: Optional[str] = None,
    path: Optional[str] = None,
    request_headers: Optional[Dict[str, Any]] = None,
    request_body: Optional[str] = None,
    response_headers: Optional[Dict[str, Any]] = None,
    response_body: Optional[str] = None,
    status_code: Optional[int] = None,
    duration: Optional[float] = None,
    is_intercepted: bool = True,
    is_encrypted: bool = False,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None
) -> ProxyHistoryEntry:
    """Create a new history entry and broadcast it to WebSocket clients."""
    try:
        logger.debug(f"Creating history entry for {method} {url}")
        # Create the history entry
        entry = ProxyHistoryEntry(
            session_id=session_id,
            method=method,
            url=url,
            host=host,
            path=path,
            request_headers=request_headers or {},
            request_body=request_body,
            response_headers=response_headers or {},
            response_body=response_body,
            status_code=status_code,
            duration=duration,
            is_intercepted=is_intercepted,
            is_encrypted=is_encrypted,
            tags=tags or [],
            notes=notes,
            timestamp=datetime.utcnow()
        )
        
        # Add to database
        db.add(entry)
        await db.commit()
        await db.refresh(entry)
        logger.debug(f"Created history entry with ID {entry.id}")
        
        # Convert entry to dict for broadcasting
        entry_dict = {
            "id": entry.id,
            "timestamp": entry.timestamp.isoformat(),
            "method": entry.method,
            "url": entry.url,
            "host": entry.host,
            "path": entry.path,
            "status_code": entry.status_code,
            "response_status": entry.status_code,
            "duration": entry.duration,
            "is_intercepted": entry.is_intercepted,
            "is_encrypted": entry.is_encrypted,
            "tags": entry.tags,
            "notes": entry.notes,
            "request_headers": entry.request_headers,
            "request_body": entry.request_body,
            "response_headers": entry.response_headers,
            "response_body": entry.response_body,
            "raw_request": entry.request_body,
            "raw_response": entry.response_body,
            "decrypted_request": entry.decrypted_request,
            "decrypted_response": entry.decrypted_response,
            "applied_rules": None,
            "session_id": entry.session_id
        }

        # Broadcast to dev container
        logger.debug("Broadcasting history entry to dev container")
        await broadcast_to_dev({
            "type": "proxy_history",
            "data": entry_dict
        })
        logger.debug("Successfully broadcast history entry")
        
        return entry
        
    except Exception as e:
        logger.error(f"Failed to create history entry: {e}")
        raise

async def get_history_entries(
    db: AsyncSession,
    session_id: Optional[int] = None,
    limit: int = 100
) -> List[ProxyHistoryEntry]:
    """Get history entries, optionally filtered by session."""
    try:
        logger.debug(f"Fetching history entries (session_id={session_id}, limit={limit})")
        # Build query
        stmt = (
            select(ProxyHistoryEntry)
            .order_by(ProxyHistoryEntry.timestamp.desc())
            .limit(limit)
        )
        if session_id:
            stmt = stmt.where(ProxyHistoryEntry.session_id == session_id)
            
        # Execute query
        result = await db.execute(stmt)
        entries = list(result.scalars().all())
        logger.debug(f"Found {len(entries)} history entries")
        
        # Process entries
        for entry in entries:
            # Ensure headers are dictionaries
            if entry.request_headers is None:
                entry.request_headers = {}
            if entry.response_headers is None:
                entry.response_headers = {}
            
            # Ensure tags is a list
            if entry.tags is None:
                entry.tags = []
                
        return entries
        
    except Exception as e:
        logger.error(f"Failed to get history entries: {e}")
        raise

async def clear_history(db: AsyncSession, session_id: Optional[int] = None) -> None:
    """Clear history entries, optionally filtered by session."""
    try:
        logger.debug(f"Clearing history entries (session_id={session_id})")
        if session_id:
            await db.execute(
                """DELETE FROM proxy_history WHERE session_id = :session_id""",
                {"session_id": session_id}
            )
        else:
            await db.execute("""DELETE FROM proxy_history""")
            
        await db.commit()
        logger.debug("Successfully cleared history entries")
        
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        raise 