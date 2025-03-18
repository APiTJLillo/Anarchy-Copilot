"""WebSocket client connection management."""
import asyncio
import logging
from typing import Dict, Set, Any, Optional
import json
from fastapi import WebSocket
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketClientManager:
    """Manages WebSocket client connections."""

    def __init__(self):
        """Initialize the client manager."""
        self.active_clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Add a WebSocket client connection."""
        async with self._lock:
            self.active_clients.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket client connection."""
        async with self._lock:
            self.active_clients.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        async with self._lock:
            dead_clients = set()
            for websocket in self.active_clients:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")
                    dead_clients.add(websocket)
            
            # Remove dead clients
            self.active_clients.difference_update(dead_clients)

    def has_active_clients(self) -> bool:
        """Check if there are any active clients."""
        return bool(self.active_clients)

    async def send_to_client(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Send a message to a specific client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            await self.disconnect(websocket)
