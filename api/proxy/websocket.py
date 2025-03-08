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
        self._lock = asyncio.Lock()
        # Register as the event broadcaster
        proxy_state.set_event_broadcaster(self)

    async def connect(self, websocket: WebSocket):
        try:
            # Accept the connection first
            await websocket.accept()
            logger.debug("WebSocket connection accepted")

            # Add to active connections under lock
            async with self._lock:
                self.active_connections.append(websocket)
            
            # Send initial state
            try:
                connections = await proxy_state.get_all_connections()
                # Send an initial message first to verify connection
                await websocket.send_json({"type": "connected", "status": "ok"})
                
                # Then send the actual state
                if connections:
                    for conn_id, conn_data in connections.items():
                        if websocket.client_state.value == 1:  # Check if still connected
                            await websocket.send_json({
                                "type": "connection_update",
                                "data": {
                                    "id": conn_id,
                                    **conn_data
                                }
                            })
                else:
                    if websocket.client_state.value == 1:  # Check if still connected
                        await websocket.send_json({
                            "type": "initial_state",
                            "data": {"connections": {}}
                        })
                logger.debug("Sent initial state to WebSocket client")
            except Exception as e:
                logger.error(f"Failed to send initial state: {e}")
                # Don't raise here, just log the error
                
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            try:
                await websocket.close()
            except:
                pass
            raise

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                logger.debug("WebSocket connection removed from active connections")

    async def broadcast_connection_update(self, connection: ConnectionInfo):
        """Broadcast connection updates to all connected clients."""
        message = {
            "type": "connection_update",
            "data": connection
        }
        await self._broadcast(message)

    async def broadcast_connection_closed(self, connection_id: str):
        """Broadcast when a connection is closed."""
        message = {
            "type": "connection_closed",
            "data": {"id": connection_id}
        }
        await self._broadcast(message)

    async def _broadcast(self, message: dict):
        """Helper method to broadcast messages with error handling."""
        async with self._lock:
            dead_connections = []
            for connection in self.active_connections:
                try:
                    if connection.client_state.value == 1:  # Only send if connection is open
                        await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send message to WebSocket client: {e}")
                    dead_connections.append(connection)
            
            # Cleanup dead connections
            for dead in dead_connections:
                if dead in self.active_connections:
                    self.active_connections.remove(dead)
                    try:
                        await dead.close()
                    except:
                        pass

connection_manager = ConnectionManager()

async def handle_proxy_connection_updates(websocket: WebSocket):
    """Handle WebSocket connections for proxy connection updates."""
    try:
        await connection_manager.connect(websocket)
        
        while websocket.client_state.value == 1:  # Only continue if connection is open
            try:
                # Keep connection alive and watch for state changes
                data = await websocket.receive_text()
                
                # Verify connection is still open before sending
                if websocket.client_state.value == 1:
                    # Send current state
                    connections = await proxy_state.get_all_connections()
                    await websocket.send_json({
                        "type": "state_update",
                        "data": {"connections": connections}
                    })
                
            except WebSocketDisconnect:
                logger.debug("WebSocket client disconnected normally")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                break
                
    except Exception as e:
        logger.error(f"Error in WebSocket connection handler: {e}")
        
    finally:
        # Always ensure we clean up
        await connection_manager.disconnect(websocket)
        try:
            if websocket.client_state.value == 1:
                await websocket.close()
        except:
            pass
