"""WebSocket endpoints for proxy connection monitoring."""
from typing import List
from fastapi import WebSocket, WebSocketDisconnect
from .models import ConnectionInfo
import asyncio
import logging
from proxy.server.state import proxy_state, ConnectionEventBroadcaster

logger = logging.getLogger("proxy.core")

class ConnectionManager(ConnectionEventBroadcaster):
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        # Register as the event broadcaster
        proxy_state.set_event_broadcaster(self)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Send initial state
        try:
            connections = await proxy_state.get_all_connections()
            for conn_id, conn_data in connections.items():
                await websocket.send_json({
                    "type": "connection_update",
                    "data": {
                        "id": conn_id,
                        **conn_data
                    }
                })
        except Exception as e:
            logger.error(f"Failed to send initial state: {e}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast_connection_update(self, connection: ConnectionInfo):
        """Broadcast connection updates to all connected clients."""
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json({
                    "type": "connection_update",
                    "data": connection
                })
            except Exception:
                dead_connections.append(connection)
                
        # Cleanup dead connections
        for dead in dead_connections:
            try:
                await dead.close()
            except Exception:
                pass
            if dead in self.active_connections:
                self.active_connections.remove(dead)

    async def broadcast_connection_closed(self, connection_id: str):
        """Broadcast when a connection is closed."""
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json({
                    "type": "connection_closed",
                    "data": {"id": connection_id}
                })
            except Exception:
                dead_connections.append(connection)
                
        # Cleanup dead connections
        for dead in dead_connections:
            try:
                await dead.close()
            except Exception:
                pass
            if dead in self.active_connections:
                self.active_connections.remove(dead)

connection_manager = ConnectionManager()

async def handle_proxy_connection_updates(websocket: WebSocket):
    """Handle WebSocket connections for proxy connection updates."""
    await connection_manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive and watch for state changes
            await websocket.receive_text()
            
            # Get latest state
            connections = await proxy_state.get_all_connections()
            for conn_id, conn_data in connections.items():
                await websocket.send_json({
                    "type": "connection_update",
                    "data": {
                        "id": conn_id,
                        **conn_data
                    }
                })
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        if websocket in connection_manager.active_connections:
            connection_manager.disconnect(websocket)
