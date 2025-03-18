"""WebSocket connection management."""
import asyncio
import logging
from typing import List, Dict, Any
from fastapi import WebSocket

logger = logging.getLogger("proxy.core")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
                # Remove failed connection
                self.disconnect(connection)

    async def broadcast_history_update(self, entry: Dict[str, Any]):
        """Broadcast a history update to all connected clients."""
        message = {
            "type": "proxy_history",
            "data": entry
        }
        await self.broadcast_message(message)

    async def broadcast_state_update(self, state: Dict[str, Any]):
        """Broadcast a state update to all connected clients."""
        message = {
            "type": "state_update",
            "data": state
        }
        await self.broadcast_message(message)

    async def broadcast_connection_update(self, connection: Dict[str, Any]):
        """Broadcast connection updates to all connected clients."""
        message = {
            "type": "connectionUpdate",
            "data": {
                "id": connection.get("id"),
                "interceptRequests": connection.get("intercept_requests", True),
                "interceptResponses": connection.get("intercept_responses", True),
                "allowedHosts": connection.get("allowed_hosts", []),
                "excludedHosts": connection.get("excluded_hosts", []),
                **{k: v for k, v in connection.items() if k not in ["id", "intercept_requests", "intercept_responses", "allowed_hosts", "excluded_hosts"]}
            }
        }
        await self.broadcast_message(message)

connection_manager = ConnectionManager() 