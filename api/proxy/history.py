"""History management for proxy requests."""
import asyncio
import logging
import aiohttp
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from aiohttp import WSMsgType, ClientWebSocketResponse
from .database_models import ProxyHistoryEntry, ProxySession
from .connection import connection_manager, ConnectionType
from .state import proxy_state
import os
import base64

logger = logging.getLogger("proxy.core")

class ClientWebSocket:
    """WebSocket client wrapper."""
    def __init__(self, websocket: ClientWebSocketResponse):
        self.websocket = websocket
        self.id = str(id(self))
        self.connected_at = datetime.utcnow()
        self.last_activity = self.connected_at
        self.message_count = 0
        self.error_count = 0
        self.state = "connected" if not websocket.closed else "closed"
        self.last_error: Optional[str] = None
        self.last_message_type: Optional[str] = None
        self.connection_history: List[Dict[str, Any]] = []

    @property
    def closed(self) -> bool:
        """Check if the WebSocket is closed."""
        return self.websocket.closed

    async def close(self, code: int = 1000):
        """Close the WebSocket connection."""
        try:
            await self.websocket.close(code=code)
            self.state = "closed"
            self.connection_history.append({
                "timestamp": datetime.utcnow(),
                "event": "closed",
                "details": f"WebSocket connection closed with code {code}"
            })
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")
            self.error_count += 1
            self.last_error = str(e)

    async def send_json(self, data: Dict[str, Any]):
        """Send JSON data over the WebSocket."""
        try:
            await self.websocket.send_json(data)
            self.message_count += 1
            self.last_activity = datetime.utcnow()
            self.last_message_type = data.get("type")
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            self.error_count += 1
            self.last_error = str(e)
            raise

    async def receive_json(self) -> Dict[str, Any]:
        """Receive JSON data from the WebSocket."""
        try:
            data = await self.websocket.receive_json()
            self.message_count += 1
            self.last_activity = datetime.utcnow()
            self.last_message_type = data.get("type")
            return data
        except Exception as e:
            logger.error(f"Error receiving WebSocket message: {e}")
            self.error_count += 1
            self.last_error = str(e)
            raise

# Global variables for WebSocket connection
_session: Optional[aiohttp.ClientSession] = None
_ws: Optional[ClientWebSocket] = None
_connect_lock = asyncio.Lock()
_is_shutting_down = False
_connection_task: Optional[asyncio.Task] = None
_RETRY_DELAY = 5  # Base delay in seconds between retries
_MAX_RETRY_DELAY = 60  # Maximum retry delay in seconds

async def _maintain_connection() -> None:
    """Maintain WebSocket connection to dev container."""
    global _ws, _session
    dev_host = os.getenv('ANARCHY_DEV_HOST', 'dev' if os.getenv('DOCKER_ENV') else 'localhost')
    url = f"ws://{dev_host}:8000/api/proxy/internal"
    retry_delay = _RETRY_DELAY
    
    while not _is_shutting_down:
        try:
            # Clean up existing session if needed
            if _session and not _session.closed:
                await _session.close()
            _session = aiohttp.ClientSession()
            
            # First check if the server is ready
            health_url = f"http://{dev_host}:8000/api/proxy/health"
            async with _session.get(health_url) as resp:
                if resp.status != 200:
                    logger.warning(f"[WebSocket] Server not ready (status {resp.status})")
                    await asyncio.sleep(retry_delay)
                    continue
                
                health_data = await resp.json()
                if not health_data.get("status") == "healthy":
                    logger.warning("[WebSocket] Server reports unhealthy status")
                    await asyncio.sleep(retry_delay)
                    continue
            
            logger.debug(f"[WebSocket] Connecting to {url}")
            raw_ws = await _session.ws_connect(
                url,
                timeout=15,
                heartbeat=15,
                autoclose=True,
                headers={
                    'Origin': f'http://{dev_host}:8000',
                    'User-Agent': 'Anarchy-Copilot-Proxy/0.1.0',
                    'x-connection-type': 'internal',
                    'x-proxy-version': '0.1.0'
                },
                protocols=['proxy-internal']
            )
            
            # Wrap the raw WebSocket
            _ws = ClientWebSocket(raw_ws)
            logger.info("[WebSocket] Connection established")
            
            # Register with connection manager
            connection_id = await connection_manager.connect(_ws, ConnectionType.INTERNAL.value)
            logger.info(f"[WebSocket] Registered with connection manager, ID: {connection_id}")
            
            # Send test message
            test_msg = {
                "type": "test_connection",
                "timestamp": datetime.utcnow().isoformat(),
                "connection_id": connection_id
            }
            await _ws.send_json(test_msg)
            
            async for msg in raw_ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        logger.debug(f"[WebSocket] Received message: {data}")
                        
                        if data.get("type") == "test_connection_response":
                            logger.info("[WebSocket] Test connection successful")
                            retry_delay = _RETRY_DELAY  # Reset retry delay on successful connection
                        elif data.get("type") == "heartbeat":
                            proxy_state.last_heartbeat = datetime.utcnow().timestamp()
                            proxy_state.is_running = True
                            await _ws.send_json({
                                "type": "heartbeat_response",
                                "timestamp": datetime.utcnow().isoformat(),
                                "connection_id": connection_id
                            })
                    except json.JSONDecodeError as e:
                        logger.error(f"[WebSocket] Error decoding JSON: {e}")
                elif msg.type == WSMsgType.CLOSED:
                    logger.warning("[WebSocket] Connection closed by server")
                    break
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"[WebSocket] Connection error: {msg.data}")
                    break
                elif msg.type == WSMsgType.CLOSE:
                    logger.warning("[WebSocket] Server requested close")
                    break
                elif msg.type == WSMsgType.PING:
                    await raw_ws.pong()
                    logger.debug("[WebSocket] Received PING, sent PONG")
                elif msg.type == WSMsgType.PONG:
                    logger.debug("[WebSocket] Received PONG")
                
        except aiohttp.ClientError as e:
            logger.error(f"[WebSocket] Connection error: {e}")
            if _ws:
                try:
                    await connection_manager.disconnect(_ws)
                except:
                    pass
                _ws = None
            
            retry_delay = min(retry_delay * 2, _MAX_RETRY_DELAY)
            logger.info(f"[WebSocket] Backing off for {retry_delay}s before retry")
            await asyncio.sleep(retry_delay)
            continue
        except Exception as e:
            logger.error(f"[WebSocket] Unexpected error: {e}")
            if _ws:
                try:
                    await connection_manager.disconnect(_ws)
                except:
                    pass
                _ws = None
            
            retry_delay = min(retry_delay * 2, _MAX_RETRY_DELAY)
            logger.info(f"[WebSocket] Backing off for {retry_delay}s before retry")
            await asyncio.sleep(retry_delay)
            continue
        
        logger.info("[WebSocket] Connection closed, retrying...")
        await asyncio.sleep(retry_delay)

async def _check_dev_container_health() -> bool:
    """Check if the dev container is healthy."""
    try:
        dev_host = os.getenv('ANARCHY_DEV_HOST', 'dev' if os.getenv('DOCKER_ENV') else 'localhost')
        health_url = f'http://{dev_host}:8000/api/proxy/health'
        logger.info(f"[TRACE] Checking dev container health at {health_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(health_url, timeout=15) as resp:
                if resp.status == 200:
                    response_text = await resp.text()
                    logger.info(f"[TRACE] Dev container health check successful: {response_text}")
                    return True
                logger.warning(f"[TRACE] Dev container health check failed with status {resp.status}")
                response_text = await resp.text()
                logger.warning(f"[TRACE] Health check response: {response_text}")
                return False
    except Exception as e:
        logger.error(f"[TRACE] Dev container health check failed with error: {e}")
        return False

async def ensure_dev_connection():
    """Ensure a connection to the dev container exists."""
    global _connection_task
    
    async with _connect_lock:
        try:
            # Start the connection maintenance task if not running
            if _connection_task is None or _connection_task.done():
                if _connection_task and _connection_task.done() and _connection_task.exception():
                    logger.error(f"[TRACE] Previous connection task failed: {_connection_task.exception()}")
                _connection_task = asyncio.create_task(_maintain_connection())
                logger.info("[TRACE] Started WebSocket connection maintenance task")
            return True
        except Exception as e:
            logger.error(f"[TRACE] Failed to ensure dev connection: {e}")
            return False

async def broadcast_to_dev(data: dict):
    """Send data to dev container via WebSocket."""
    global _ws
    
    try:
        logger.info(f"[TRACE] Starting broadcast_to_dev with data type: {data.get('type')}")
        await ensure_dev_connection()
        if _ws and not _ws.closed:
            logger.info(f"[TRACE] WebSocket connection is ready, sending data to dev container")
            await _ws.send_json(data)
            logger.info(f"[TRACE] Successfully sent data to dev container: {data['type']}")
        else:
            logger.error("[TRACE] WebSocket connection is not available")
    except Exception as e:
        logger.error(f"[TRACE] Failed to broadcast to dev container: {e}")
        logger.exception("[TRACE] Full exception details:")
        # Reset connection on failure
        _ws = None

async def cleanup():
    """Clean up WebSocket connection and tasks."""
    global _is_shutting_down, _ws, _session, _connection_task
    
    _is_shutting_down = True
    
    # Cancel connection task
    if _connection_task and not _connection_task.done():
        _connection_task.cancel()
        try:
            await _connection_task
        except asyncio.CancelledError:
            pass
    
    # Close WebSocket connection
    if _ws:
        try:
            await connection_manager.disconnect(_ws)
        except:
            pass
        _ws = None
    
    # Close session
    if _session and not _session.closed:
        await _session.close()
    _session = None
    
    _is_shutting_down = False

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
        print("[DEBUG] Starting create_history_entry")  # Added print for visibility
        logger.error("[DEBUG] Starting create_history_entry")  # Added error level for visibility
        
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
        print(f"[DEBUG] Created history entry with ID {entry.id}")  # Added print for visibility
        logger.error(f"[DEBUG] Created history entry with ID {entry.id}")  # Added error level for visibility
        
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
        print("[DEBUG] About to broadcast history entry to dev container")  # Added print for visibility
        logger.error("[DEBUG] About to broadcast history entry to dev container")  # Added error level for visibility
        await broadcast_to_dev({
            "type": "proxy_history",
            "data": entry_dict
        })
        print("[DEBUG] Successfully broadcast history entry")  # Added print for visibility
        logger.error("[DEBUG] Successfully broadcast history entry")  # Added error level for visibility
        
        return entry
        
    except Exception as e:
        print(f"[DEBUG] Failed to create history entry: {e}")  # Added print for visibility
        logger.error(f"[DEBUG] Failed to create history entry: {e}")  # Added error level for visibility
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
