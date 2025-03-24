"""WebSocket client manager."""
import asyncio
import logging
import aiohttp
import time
import os
from typing import Dict, List, Set, Optional, Callable, Any, cast, Literal, Union, TypeVar, Mapping
from fastapi import WebSocket 
from datetime import datetime

from .types import (
    ConnectionState, ConnectionStatus, ConnectionType,
    MessageType, MessageDirection, WSMessage,
    create_status_message, create_state_message,
    create_heartbeat_message, ConnectionEventType
)

logger = logging.getLogger(__name__)

# Type variables and aliases
T = TypeVar('T')
StatusDict = Dict[str, ConnectionStatus]
ConnStatusDict = Dict[str, Dict[str, ConnectionStatus]]
ConnStatusType = Dict[str, Dict[str, ConnectionStatus]]

class WebSocketClientManager:
    """Manages WebSocket client connections."""
    
    def __init__(self):
        """Initialize client manager."""
        self._clients: Dict[str, Dict[str, WebSocket]] = {
            ConnectionType.UI.value: {},
            ConnectionType.INTERNAL.value: {},
            ConnectionType.PROXY.value: {}
        }
        self._lock = asyncio.Lock()
        self._connection_status: ConnStatusDict = {
            ConnectionType.UI.value: {},
            ConnectionType.INTERNAL.value: {},
            ConnectionType.PROXY.value: {}
        }
        self._event_handlers: Dict[str, List[Callable]] = {
            "connect": [],
            "disconnect": [],
            "error": [],
            "state_change": []
        }
        self._heartbeat_tasks: Dict[str, Set[asyncio.Task]] = {}
        self._reconnect_delays = {
            ConnectionType.UI.value: 1,      # 1 second
            ConnectionType.INTERNAL.value: 2, # 2 seconds
            ConnectionType.PROXY.value: 5     # 5 seconds
        }
        self._max_reconnect_attempts = {
            ConnectionType.UI.value: 3,
            ConnectionType.INTERNAL.value: 5,  # More attempts for internal connections
            ConnectionType.PROXY.value: 3
        }

    def on(self, event_type: ConnectionEventType, handler: Callable) -> None:
        """Register an event handler."""
        if event_type in self._event_handlers:
            self._event_handlers[event_type].append(handler)
            
    async def _trigger_event(self, event_type: ConnectionEventType, *args: Any, **kwargs: Any) -> None:
        """Trigger handlers for an event."""
        for handler in self._event_handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {event_type} handler: {e}")

    async def _update_connection_state(
        self, 
        connection_id: str,
        connection_type: ConnectionType,
        new_state: ConnectionState,
        error: Optional[str] = None
    ) -> None:
        """Update connection state and trigger events."""
        async with self._lock:
            if connection_id not in self._connection_status[connection_type.value]:
                return
                
            status = self._connection_status[connection_type.value][connection_id]
            old_state = status.state
            
            if old_state == new_state:
                return
                
            status.update_state(new_state, error)
            
            # Create state change message
            state_msg = create_state_message(connection_id, old_state, new_state)
            
            # Trigger state change event
            await self._trigger_event(
                "state_change",
                connection_id,
                connection_type,
                old_state,
                new_state,
                error
            )
            
            # Broadcast state change to UI clients
            await self._broadcast_to_type(
                ConnectionType.UI.value,
                state_msg.data
            )
        
    async def connect(self, websocket: WebSocket, connection_type: str = ConnectionType.UI.value) -> None:
        """Connect a new client.
        
        Args:
            websocket: WebSocket connection to add
            connection_type: Type of connection ("ui", "internal", or "proxy")
        """
        try:
            conn_type = ConnectionType(connection_type)
        except ValueError:
            raise ValueError("connection_type must be 'ui', 'internal', or 'proxy'")
            
        async with self._lock:
            connection_id = str(id(websocket))
            if connection_id in self._clients[conn_type.value]:
                logger.warning(f"Connection {connection_id} already exists, updating")
                try:
                    await self.disconnect(websocket, conn_type.value)
                except Exception as e:
                    logger.error(f"Error cleaning up existing connection: {e}")
                    
            self._clients[conn_type.value][connection_id] = websocket
            
            # Initialize connection status
            self._connection_status[conn_type.value][connection_id] = ConnectionStatus.create(
                conn_type,
                connected=True,
                metadata={
                    "client_id": connection_id,
                    "type": conn_type.value
                }
            )
            
        # Update state to connected
        await self._update_connection_state(
            connection_id,
            conn_type,
            ConnectionState.CONNECTED
        )
            
        # Trigger connect event
        await self._trigger_event("connect", connection_id, conn_type.value)
        
        # Broadcast connection status
        await self._broadcast_connection_status(connection_id, conn_type.value, "connect")

        # Start heartbeat monitoring if internal or proxy connection
        if conn_type in (ConnectionType.INTERNAL, ConnectionType.PROXY):
            self._heartbeat_tasks[connection_id] = set()
            asyncio.create_task(
                self._monitor_connection(websocket, connection_id, conn_type)
            )
            
    async def _monitor_connection(
        self,
        websocket: WebSocket,
        connection_id: str,
        connection_type: ConnectionType
    ) -> None:
        """Monitor connection with heartbeat."""
        heartbeat_interval = 15  # seconds
        heartbeat_timeout = 20   # seconds
        last_heartbeat_response = asyncio.Event()
        monitor_state = {"active": True}

        async def send_heartbeats():
            while monitor_state["active"]:
                try:
                    if not self._is_connected(connection_id, connection_type.value):
                        logger.warning(f"Connection {connection_id} no longer active")
                        break

                    await websocket.send_json(
                        create_heartbeat_message(
                            direction=MessageDirection.OUTGOING
                        ).data
                    )

                    # Update last activity
                    async with self._lock:
                        if connection_id in self._connection_status[connection_type.value]:
                            status = self._connection_status[connection_type.value][connection_id]
                            status.last_activity = datetime.utcnow()
                            status.duration = (status.last_activity - status.connected_at).total_seconds()

                    await asyncio.sleep(heartbeat_interval)

                except Exception as e:
                    logger.error(f"Error sending heartbeat: {e}")
                    await self._handle_connection_error(connection_id, connection_type, str(e))
                    if not monitor_state["active"]:
                        break
                    await asyncio.sleep(self._reconnect_delays[connection_type.value])

        async def receive_messages():
            while monitor_state["active"]:
                try:
                    if not self._is_connected(connection_id, connection_type.value):
                        break

                    msg = await websocket.receive_json()
                    
                    if msg.get("type") == MessageType.HEARTBEAT.value:
                        last_heartbeat_response.set()
                        await websocket.send_json(
                            create_heartbeat_message(
                                direction=MessageDirection.OUTGOING,
                                response=True
                            ).data
                        )
                    
                    elif msg.get("type") == MessageType.HEARTBEAT_RESPONSE.value:
                        last_heartbeat_response.set()
                        await self._update_connection_state(
                            connection_id,
                            connection_type,
                            ConnectionState.CONNECTED
                        )

                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    await self._handle_connection_error(connection_id, connection_type, str(e))
                    if not monitor_state["active"]:
                        break
                    await asyncio.sleep(1)

        async def monitor_heartbeat():
            while monitor_state["active"]:
                try:
                    try:
                        await asyncio.wait_for(last_heartbeat_response.wait(), heartbeat_timeout)
                        last_heartbeat_response.clear()
                        await self._update_connection_state(
                            connection_id,
                            connection_type,
                            ConnectionState.CONNECTED
                        )
                    except asyncio.TimeoutError:
                        await self._handle_connection_error(
                            connection_id,
                            connection_type,
                            "Heartbeat timeout"
                        )
                        if not monitor_state["active"]:
                            break
                    
                    await asyncio.sleep(heartbeat_interval)
                    
                except Exception as e:
                    logger.error(f"Error monitoring heartbeat: {e}")
                    await self._handle_connection_error(connection_id, connection_type, str(e))
                    if not monitor_state["active"]:
                        break

        tasks = set()
        try:
            tasks.add(asyncio.create_task(send_heartbeats(), name=f"heartbeat_sender_{connection_id}"))
            tasks.add(asyncio.create_task(receive_messages(), name=f"heartbeat_receiver_{connection_id}"))
            tasks.add(asyncio.create_task(monitor_heartbeat(), name=f"heartbeat_monitor_{connection_id}"))

            self._heartbeat_tasks[connection_id] = tasks

            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            monitor_state["active"] = False
            
            for task in pending:
                task.cancel()

        except asyncio.CancelledError:
            logger.info(f"Heartbeat monitoring cancelled for {connection_id}")
        except Exception as e:
            logger.error(f"Unexpected error in heartbeat monitoring for {connection_id}: {e}")
        finally:
            monitor_state["active"] = False
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _handle_connection_error(
        self,
        connection_id: str,
        connection_type: ConnectionType,
        error: str
    ) -> None:
        """Handle connection errors with reconnection logic."""
        async with self._lock:
            if connection_id not in self._connection_status[connection_type.value]:
                return

            status = self._connection_status[connection_type.value][connection_id]
            status.reconnect_count += 1
            status.last_reconnect = datetime.utcnow()

            max_attempts = self._max_reconnect_attempts[connection_type.value]
            
            if status.reconnect_count > max_attempts:
                logger.error(f"Max reconnection attempts ({max_attempts}) reached for {connection_id}")
                await self._update_connection_state(
                    connection_id,
                    connection_type,
                    ConnectionState.ERROR,
                    error
                )
                return True  # Signal to stop monitoring
            
            logger.info(f"Attempting reconnect {status.reconnect_count}/{max_attempts} for {connection_id}")
            await self._update_connection_state(
                connection_id,
                connection_type,
                ConnectionState.CONNECTING,
                f"Reconnecting: {error}"
            )
            
            return False  # Continue monitoring

    async def _broadcast_connection_status(
        self,
        connection_id: str,
        connection_type: str,
        event: str
    ) -> None:
        """Broadcast connection status update."""
        async with self._lock:
            if connection_id not in self._connection_status[connection_type]:
                return
                
            status = self._connection_status[connection_type][connection_id]
            msg = create_status_message(
                connection_id,
                status,
                cast(ConnectionEventType, event)  # Safe cast since we control event values
            )
            
        await self._broadcast_to_type(ConnectionType.UI.value, msg.data)

    async def disconnect(self, websocket: WebSocket, connection_type: Optional[str] = None) -> None:
        """Disconnect a client.
        
        Args:
            websocket: WebSocket connection to remove
            connection_type: Type of connection ("ui", "internal", or "proxy"), if known
        """
        connection_id = str(id(websocket))
        
        # Cancel any heartbeat tasks
        if connection_id in self._heartbeat_tasks:
            for task in self._heartbeat_tasks[connection_id]:
                if not task.done():
                    task.cancel()
            del self._heartbeat_tasks[connection_id]
            
        async with self._lock:
            # Find the connection type if not provided
            if connection_type is None:
                for ctype in ConnectionType:
                    if connection_id in self._clients[ctype.value]:
                        connection_type = ctype.value
                        break
            
            if connection_type:
                conn_type = ConnectionType(connection_type)
                
                if connection_id in self._clients[conn_type.value]:
                    # Update state before removing
                    await self._update_connection_state(
                        connection_id,
                        conn_type,
                        ConnectionState.CLOSED
                    )
                    
                    # Trigger disconnect event
                    await self._trigger_event("disconnect", connection_id, conn_type.value)
                    
                    # Broadcast disconnect status
                    try:
                        await self._broadcast_connection_status(
                            connection_id,
                            conn_type.value,
                            "disconnect"
                        )
                    except Exception as e:
                        logger.error(f"Error broadcasting disconnect: {e}")
                
                    # Clean up
                    del self._clients[conn_type.value][connection_id]
                    if connection_id in self._connection_status[conn_type.value]:
                        del self._connection_status[conn_type.value][connection_id]
                
                    logger.info(f"[TRACE] Removed {conn_type.value} connection: {connection_id}")
            
    async def _broadcast_to_type(self, connection_type: str, message: dict) -> None:
        """Broadcast a message to all clients of a specific type."""
        disconnected = []
        
        for conn_id, websocket in self._clients[connection_type].items():
            try:
                if websocket.client_state.value == 1:  # WebSocketState.CONNECTED
                    await websocket.send_json(message)
                    
                    # Update last activity and message count
                    async with self._lock:
                        if conn_id in self._connection_status[connection_type]:
                            status = self._connection_status[connection_type][conn_id]
                            status.last_activity = datetime.utcnow()
                            status.message_count += 1
                            status.duration = (status.last_activity - status.connected_at).total_seconds()
                else:
                    logger.warning(f"[TRACE] Client {conn_id} not connected, marking for cleanup")
                    disconnected.append((websocket, conn_id))
            except Exception as e:
                logger.error(f"[TRACE] Failed to send message to {connection_type} client {conn_id}: {e}")
                disconnected.append((websocket, conn_id))
        
        # Clean up disconnected clients
        for websocket, conn_id in disconnected:
            await self.disconnect(websocket, connection_type)
            
    async def broadcast(self, message: dict) -> None:
        """Broadcast a message to all UI clients."""
        await self._broadcast_to_type(ConnectionType.UI.value, message)
                        
    async def broadcast_connection_update(self, data: dict) -> None:
        """Broadcast a connection update to all UI clients."""
        await self.broadcast({
            "type": MessageType.CONNECTION_STATUS.value,
            "data": data
        })
        
    def _is_connected(self, connection_id: str, connection_type: str) -> bool:
        """Check if a client is still connected."""
        try:
            conn_type = ConnectionType(connection_type)
        except ValueError:
            return False
            
        if connection_id not in self._clients[conn_type.value]:
            return False
        if connection_id not in self._connection_status[conn_type.value]:
            return False
            
        status = self._connection_status[conn_type.value][connection_id]
        return status.connected and status.state == ConnectionState.CONNECTED
    def get_clients(self, connection_type: Optional[str] = None) -> Dict[str, WebSocket]:
        """Get all connected clients of a specific type or all clients if type not specified."""
        if connection_type:
            return self._clients.get(ConnectionType(connection_type).value, {})
            
        return {
            ctype.value: self._clients[ctype.value]
            for ctype in ConnectionType
        }
        
    def get_connection_status(self, connection_type: Optional[str] = None) -> ConnStatusType:
        """Get status of all connections of a specific type or all connections if type not specified."""
        if connection_type:
            result: Dict[str, Dict[str, ConnectionStatus]] = {
                connection_type: self._connection_status.get(ConnectionType(connection_type).value, {})
            }
            return result
            
        return {
            ctype.value: self._connection_status[ctype.value]
            for ctype in ConnectionType
        }

class DevConnectionManager:
    """Manages WebSocket connection to dev container."""
    
    def __init__(self):
        """Initialize the connection manager."""
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._connect_lock = asyncio.Lock()
        self._is_shutting_down = False
        self._connection_task: Optional[asyncio.Task] = None
        self._retry_delay = 5  # Base delay in seconds
        self._max_retry_delay = 60  # Maximum retry delay in seconds
        self.last_heartbeat = 0
        self.is_connected = False

    async def _maintain_connection(self) -> None:
        """Maintain WebSocket connection to dev container."""
        dev_host = 'dev' if 'DOCKER_ENV' in os.environ else 'localhost'
        url = f"ws://{dev_host}:8000/api/proxy/internal"
        retry_delay = self._retry_delay
        
        while not self._is_shutting_down:
            try:
                # Clean up existing session if needed
                if self._session and not self._session.closed:
                    await self._session.close()
                self._session = aiohttp.ClientSession()
                
                # First check if the server is ready
                health_url = f"http://{dev_host}:8000/api/proxy/health"
                async with self._session.get(health_url) as resp:
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
                self._ws = await self._session.ws_connect(
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
                
                logger.info("[WebSocket] Connection established")
                self.is_connected = True
                self.last_heartbeat = time.time()
                
                # Send test message
                await self._ws.send_json({
                    "type": "test_connection",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                async for msg in self._ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = msg.json()
                            logger.debug(f"[WebSocket] Received message: {data}")
                            
                            if data.get("type") == "test_connection_response":
                                logger.info("[WebSocket] Test connection successful")
                                retry_delay = self._retry_delay  # Reset retry delay on successful connection
                            elif data.get("type") == "heartbeat":
                                self.last_heartbeat = time.time()
                                await self._ws.send_json({
                                    "type": "heartbeat_response",
                                    "timestamp": datetime.utcnow().isoformat()
                                })
                        except Exception as e:
                            logger.error(f"[WebSocket] Error processing message: {e}")
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("[WebSocket] Connection closed by server")
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"[WebSocket] Connection error: {msg.data}")
                        break
                
            except aiohttp.ClientError as e:
                logger.error(f"[WebSocket] Connection error: {e}")
                self.is_connected = False
                retry_delay = min(retry_delay * 2, self._max_retry_delay)
                logger.info(f"[WebSocket] Backing off for {retry_delay}s before retry")
                await asyncio.sleep(retry_delay)
                continue
            except Exception as e:
                logger.error(f"[WebSocket] Unexpected error: {e}")
                self.is_connected = False
                retry_delay = min(retry_delay * 2, self._max_retry_delay)
                logger.info(f"[WebSocket] Backing off for {retry_delay}s before retry")
                await asyncio.sleep(retry_delay)
                continue
            
            logger.info("[WebSocket] Connection closed, retrying...")
            self.is_connected = False
            await asyncio.sleep(retry_delay)

    async def ensure_connection(self) -> None:
        """Ensure a connection to the dev container exists."""
        async with self._connect_lock:
            try:
                # Start the connection maintenance task if not running
                if self._connection_task is None or self._connection_task.done():
                    if self._connection_task and self._connection_task.done() and self._connection_task.exception():
                        logger.error(f"[WebSocket] Previous connection task failed: {self._connection_task.exception()}")
                    self._connection_task = asyncio.create_task(self._maintain_connection())
                    logger.info("[WebSocket] Started WebSocket connection maintenance task")
            except Exception as e:
                logger.error(f"[WebSocket] Failed to ensure connection: {e}")

    async def cleanup(self) -> None:
        """Clean up WebSocket connection and tasks."""
        self._is_shutting_down = True
        
        # Cancel connection task
        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket connection
        if self._ws and not self._ws.closed:
            await self._ws.close()
        
        # Close session
        if self._session and not self._session.closed:
            await self._session.close()
        
        self.is_connected = False
        self._is_shutting_down = False

# Create singleton instance
dev_connection_manager = DevConnectionManager()
