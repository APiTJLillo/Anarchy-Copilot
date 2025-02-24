"""WebSocket endpoints for proxy connection monitoring."""
from typing import List
from fastapi import WebSocket, WebSocketDisconnect
from .models import ConnectionInfo

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast_connection_update(self, connection: ConnectionInfo):
        """Broadcast connection updates to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json({
                    "type": "connection_update",
                    "data": connection.dict()
                })
            except Exception:
                # Remove dead connections
                await connection.close()
                self.active_connections.remove(connection)

    async def broadcast_connection_closed(self, connection_id: str):
        """Broadcast when a connection is closed."""
        for connection in self.active_connections:
            try:
                await connection.send_json({
                    "type": "connection_closed",
                    "data": {"id": connection_id}
                })
            except Exception:
                await connection.close()
                self.active_connections.remove(connection)

connection_manager = ConnectionManager()

async def handle_proxy_connection_updates(websocket: WebSocket):
    """Handle WebSocket connections for proxy connection updates."""
    await connection_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
