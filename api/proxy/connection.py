"""Connection management for proxy server."""
import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Set, List, Any, Callable, Union
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionType(Enum):
    UI = "ui"
    INTERNAL = "internal"
    PROXY = "proxy"

class ConnectionState(Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"

class Connection:
    """Represents a WebSocket connection."""
    def __init__(self, websocket: WebSocket, type: str):
        self.id = str(uuid.uuid4())
        self.websocket = websocket
        self.type = type
        self.state = ConnectionState.CONNECTING.value
        self.connected_at = datetime.utcnow()
        self.last_activity = self.connected_at
        self.message_count = 0
        self.error_count = 0
        self.metadata: Dict[str, Any] = {}
        self.disconnect_time: Optional[datetime] = None
        self.last_error: Optional[str] = None

    def record_activity(self):
        """Record connection activity."""
        self.last_activity = datetime.utcnow()
        
    def record_message(self):
        """Record a message being sent/received."""
        self.message_count += 1
        self.record_activity()

    def record_error(self, error: str):
        """Record an error occurring."""
        self.error_count += 1
        self.last_error = error
        self.record_activity()

    def record_disconnect(self):
        """Record connection disconnect."""
        self.state = ConnectionState.DISCONNECTED.value
        self.disconnect_time = datetime.utcnow()

class ConnectionManager:
    """Manages WebSocket connections."""
    def __init__(self):
        """Initialize connection manager."""
        self._connections: Dict[str, Connection] = {}
        self._active_connections: Dict[str, Dict[str, Connection]] = {
            ConnectionType.UI.value: {},
            ConnectionType.INTERNAL.value: {},
            ConnectionType.PROXY.value: {}
        }
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 60  # seconds
        self._max_inactivity = 300  # seconds
        self._event_handlers: Dict[str, List[Callable]] = {
            "connect": [],
            "disconnect": [],
            "state_change": [],
            "error": []
        }

    @property
    def active_connections(self) -> Dict[str, Dict[str, Connection]]:
        """Get active connections dictionary."""
        return self._active_connections

    def add_event_handler(self, event: str, handler: Callable) -> None:
        """Add an event handler."""
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)

    async def _trigger_event(self, event: str, *args, **kwargs) -> None:
        """Trigger event handlers."""
        for handler in self._event_handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {event} handler: {e}")

    async def start_cleanup_task(self):
        """Start the periodic cleanup task."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Started connection cleanup task")

    async def stop_cleanup_task(self):
        """Stop the periodic cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Stopped connection cleanup task")

    async def _periodic_cleanup(self):
        """Periodically clean up stale connections."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_stale_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")

    async def _cleanup_stale_connections(self):
        """Clean up stale connections."""
        now = datetime.utcnow()
        async with self._lock:
            for conn_type in self._active_connections:
                stale_connections = []
                for conn_id, conn in self._active_connections[conn_type].items():
                    inactivity_time = (now - conn.last_activity).total_seconds()
                    if inactivity_time > self._max_inactivity:
                        stale_connections.append(conn_id)
                        logger.warning(f"Connection {conn_id} inactive for {inactivity_time}s, cleaning up")
                
                for conn_id in stale_connections:
                    conn = self._active_connections[conn_type].pop(conn_id)
                    try:
                        await conn.websocket.close(code=1000, reason="Inactivity timeout")
                    except:
                        pass
                    await self._trigger_event("disconnect", conn_id, conn_type)

    async def connect(self, websocket: WebSocket, type: str = ConnectionType.UI.value) -> str:
        """Connect a new WebSocket client."""
        async with self._lock:
            connection = Connection(websocket, type)
            self._connections[connection.id] = connection
            self._active_connections[type][connection.id] = connection
            connection.state = ConnectionState.CONNECTED.value
            await self._trigger_event("connect", connection.id, type)
            return connection.id

    async def disconnect(self, websocket: WebSocket, connection_type: Optional[str] = None) -> None:
        """Disconnect a WebSocket client."""
        async with self._lock:
            # Find the connection ID by websocket object
            connection_id = None
            for conn_id, conn in self._connections.items():
                if conn.websocket == websocket:
                    connection_id = conn_id
                    break

            if connection_id:
                connection = self._connections.pop(connection_id)
                if connection_type:
                    self._active_connections[connection_type].pop(connection_id, None)
                connection.record_disconnect()
                await self._trigger_event("disconnect", connection_id, connection_type or connection.type)

    async def broadcast_json(
        self,
        message: dict,
        exclude: Optional[Set[WebSocket]] = None,
        connection_type: Optional[str] = None
    ) -> None:
        """Broadcast a JSON message to connected clients."""
        exclude = exclude or set()
        tasks = []

        async with self._lock:
            if connection_type:
                connections = self._active_connections[connection_type].values()
            else:
                connections = self._connections.values()

            for connection in connections:
                if connection.websocket not in exclude and connection.state == ConnectionState.CONNECTED.value:
                    try:
                        tasks.append(connection.websocket.send_json(message))
                        connection.record_message()
                    except Exception as e:
                        connection.record_error(str(e))
                        logger.error(f"Error queuing broadcast to {connection.id}: {e}")

        if tasks:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
            for task in done:
                if task.exception():
                    logger.error(f"Error in broadcast: {task.exception()}")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        stats = {
            ConnectionType.UI.value: {
                "connected": 0,
                "connection_count": 0,
                "message_count": 0,
                "error_count": 0,
                "last_message": None
            },
            ConnectionType.INTERNAL.value: {
                "connected": 0,
                "connection_count": 0,
                "message_count": 0,
                "error_count": 0,
                "last_message": None
            },
            ConnectionType.PROXY.value: {
                "connected": 0,
                "connection_count": 0,
                "message_count": 0,
                "error_count": 0,
                "last_message": None
            }
        }

        for conn in self._connections.values():
            type_stats = stats[conn.type]
            if conn.state == ConnectionState.CONNECTED.value:
                type_stats["connected"] += 1
            type_stats["connection_count"] += 1
            type_stats["message_count"] += conn.message_count
            type_stats["error_count"] += conn.error_count
            if conn.last_activity:
                if not type_stats["last_message"] or conn.last_activity > type_stats["last_message"]:
                    type_stats["last_message"] = conn.last_activity

        return stats

    def get_active_connections(self) -> List[Connection]:
        """Get all active connections."""
        active = []
        for conn_type in self._active_connections:
            active.extend([
                conn for conn in self._active_connections[conn_type].values()
                if conn.state == ConnectionState.CONNECTED.value
            ])
        return active

    def get_connection(self, connection_id: str) -> Optional[Connection]:
        """Get a specific connection by ID."""
        return self._connections.get(connection_id)

    def get_connection_by_type(self, connection_type: str) -> List[Connection]:
        """Get all active connections of a specific type."""
        return [
            conn for conn in self._active_connections[connection_type].values()
            if conn.state == ConnectionState.CONNECTED.value
        ]

    async def update_connection_state(
        self,
        connection_id: str,
        new_state: Union[ConnectionState, str],
        error: Optional[str] = None
    ) -> None:
        """Update a connection's state."""
        async with self._lock:
            if connection_id in self._connections:
                connection = self._connections[connection_id]
                old_state = connection.state
                if isinstance(new_state, ConnectionState):
                    new_state = new_state.value
                connection.state = new_state
                if error:
                    connection.record_error(error)
                await self._trigger_event(
                    "state_change",
                    connection_id,
                    connection.type,
                    old_state,
                    new_state,
                    error
                )

# Create singleton instance
connection_manager = ConnectionManager()