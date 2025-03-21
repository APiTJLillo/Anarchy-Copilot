"""WebSocket connection management."""
import asyncio
import logging
from typing import List, Dict, Any, Set, Optional
from fastapi import WebSocket
import uuid
from datetime import datetime

logger = logging.getLogger("proxy.core")

class Connection:
    def __init__(self, websocket: WebSocket, connection_type: str):
        self.id = str(uuid.uuid4())
        self.websocket = websocket
        self.type = connection_type
        self.connected_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.message_count = 0
        self.error_count = 0
        self.connection_attempts = 1
        self.last_error = None
        self.last_message_type = None
        self.state = "connected"  # connected, disconnected, error
        self.connection_history = [{
            "timestamp": self.connected_at,
            "event": "connected",
            "details": f"Initial connection established for {connection_type}"
        }]

    def update_activity(self, message_type: str = None):
        self.last_activity = datetime.utcnow()
        self.message_count += 1
        self.last_message_type = message_type
        self.connection_history.append({
            "timestamp": self.last_activity,
            "event": "message",
            "details": f"Received message type: {message_type}"
        })

    def record_error(self, error: str):
        self.error_count += 1
        self.last_error = error
        self.state = "error"
        self.connection_history.append({
            "timestamp": datetime.utcnow(),
            "event": "error",
            "details": str(error)
        })

    def record_disconnect(self):
        self.state = "disconnected"
        self.connection_history.append({
            "timestamp": datetime.utcnow(),
            "event": "disconnected",
            "details": "Connection closed"
        })

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Connection] = {}
        self.connection_types: Dict[str, str] = {}
        self._last_message = {
            "ui": datetime.utcnow(),
            "internal": datetime.utcnow()
        }
        self._message_count = {
            "ui": 0,
            "internal": 0
        }
        self._error_count = {
            "ui": 0,
            "internal": 0
        }
        self._connection_history = {
            "ui": [],
            "internal": []
        }
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, connection_type: str = "ui"):
        """Connect a new WebSocket client.
        
        Args:
            websocket: The WebSocket connection
            connection_type: Type of connection ("ui", "proxy", or "internal")
        """
        await websocket.accept()
        connection = Connection(websocket, connection_type)
        self.active_connections[connection.id] = connection
        self.connection_types[connection.id] = connection_type
        
        # Record connection in history
        self._connection_history[connection_type].append({
            "timestamp": connection.connected_at,
            "event": "connected",
            "connection_id": connection.id,
            "details": f"New {connection_type} connection established"
        })
        
        logger.info(f"New {connection_type} connection established with ID: {connection.id}")

    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client."""
        # Find connection by websocket
        for conn_id, conn in list(self.active_connections.items()):
            if conn.websocket == websocket:
                conn.record_disconnect()
                # Record disconnection in history
                self._connection_history[conn.type].append({
                    "timestamp": datetime.utcnow(),
                    "event": "disconnected",
                    "connection_id": conn.id,
                    "details": f"Connection {conn_id} disconnected"
                })
                del self.active_connections[conn_id]
                del self.connection_types[conn_id]
                logger.info(f"Connection {conn_id} disconnected")
                break

    async def broadcast_json(self, message: dict, exclude: Optional[Set[WebSocket]] = None, connection_type: Optional[str] = None):
        """Broadcast a JSON message to all connected clients."""
        exclude = exclude or set()
        
        for conn in self.active_connections.values():
            if conn.websocket in exclude:
                continue
            if connection_type and conn.type != connection_type:
                continue
                
            try:
                await conn.websocket.send_json(message)
                conn.update_activity(message.get("type"))
                self._last_message[conn.type] = datetime.utcnow()
                self._message_count[conn.type] = self._message_count.get(conn.type, 0) + 1
            except Exception as e:
                logger.error(f"Error broadcasting to connection {conn.id}: {e}")
                conn.record_error(str(e))
                self._error_count[conn.type] = self._error_count.get(conn.type, 0) + 1
                self.disconnect(conn.websocket)

    async def broadcast_message(self, message: str):
        """Broadcast a text message to all UI clients."""
        await self.broadcast_json(
            {"type": "message", "data": message},
            connection_type="ui"
        )

    async def broadcast_history_update(self, history_data: dict):
        """Broadcast a history update to all UI clients."""
        await self.broadcast_json(
            {"type": "proxy_history", "data": history_data},
            connection_type="ui"
        )

    async def broadcast_state_update(self, state_data: dict):
        """Broadcast a state update to all UI clients."""
        await self.broadcast_json(
            {"type": "proxy_state", "data": state_data},
            connection_type="ui"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed connection manager statistics."""
        ui_connections = [conn for conn in self.active_connections.values() if conn.type == "ui"]
        internal_connections = [conn for conn in self.active_connections.values() if conn.type == "internal"]
        
        now = datetime.utcnow()
        
        return {
            "ui": {
                "connected": len(ui_connections) > 0,
                "connection_count": len(ui_connections),
                "last_message": self._last_message.get("ui", now),
                "message_count": self._message_count.get("ui", 0),
                "error_count": self._error_count.get("ui", 0),
                "active_connections": [{
                    "id": conn.id,
                    "connected_since": conn.connected_at,
                    "last_activity": conn.last_activity,
                    "message_count": conn.message_count,
                    "error_count": conn.error_count,
                    "state": conn.state,
                    "last_error": conn.last_error,
                    "last_message_type": conn.last_message_type,
                    "connection_history": conn.connection_history
                } for conn in ui_connections],
                "connection_history": self._connection_history["ui"][-50:]  # Keep last 50 events
            },
            "internal": {
                "connected": len(internal_connections) > 0,
                "connection_count": len(internal_connections),
                "last_message": self._last_message.get("internal", now),
                "message_count": self._message_count.get("internal", 0),
                "error_count": self._error_count.get("internal", 0),
                "active_connections": [{
                    "id": conn.id,
                    "connected_since": conn.connected_at,
                    "last_activity": conn.last_activity,
                    "message_count": conn.message_count,
                    "error_count": conn.error_count,
                    "state": conn.state,
                    "last_error": conn.last_error,
                    "last_message_type": conn.last_message_type,
                    "connection_history": conn.connection_history
                } for conn in internal_connections],
                "connection_history": self._connection_history["internal"][-50:]  # Keep last 50 events
            }
        }

    def get_active_connections(self) -> List[Connection]:
        """Get list of active connections with detailed status."""
        return list(self.active_connections.values())

# Global instance
connection_manager = ConnectionManager() 