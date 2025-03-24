"""
WebSocket session and proxy management.
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from fastapi import WebSocket

import aiohttp
from aiohttp import ClientWebSocketResponse, WSMessage, WSMsgType, web

from .conversation import WSConversation
from .fuzzing import WSFuzzer
from .types import (
    WSMessage as WSProxyMessage, 
    MessageType,
    MessageDirection,
    ConnectionType
)
from .interceptor import WebSocketInterceptor
from .client_manager import WebSocketClientManager

logger = logging.getLogger(__name__)

# Constants
CONNECTION_TIMEOUT = 30  # seconds
CLEANUP_DELAY = 5.0  # seconds

class WebSocketManager:
    """Manages WebSocket connections and sessions."""

    def __init__(self):
        """Initialize WebSocket manager."""
        self.conversations: Dict[str, WSConversation] = {}
        self.active_sessions: Set[str] = set()
        self._fuzzer = WSFuzzer()
        self._interceptors: List[WebSocketInterceptor] = []
        self.client_manager = WebSocketClientManager()
        self._lock = asyncio.Lock()
        self._cleanup_tasks: Set[asyncio.Task] = set()
        self._connection_timeouts: Dict[str, asyncio.Task] = {}
        self._last_activity: Dict[str, float] = {}

        # Register event handlers for client manager
        self.client_manager.on("connect", self._on_client_connect)
        self.client_manager.on("disconnect", self._on_client_disconnect)
        self.client_manager.on("error", self._on_client_error)

    async def _on_client_connect(self, connection_id: str, connection_type: str) -> None:
        """Handle client connection event."""
        logger.info(f"[TRACE] Client {connection_id} ({connection_type}) connected")
        await self._refresh_connection_timeout(connection_id)
        await self.broadcast_state_update()

    async def _on_client_disconnect(self, connection_id: str, connection_type: str) -> None:
        """Handle client disconnection event."""
        logger.info(f"[TRACE] Client {connection_id} ({connection_type}) disconnected")
        if connection_id in self._connection_timeouts:
            self._connection_timeouts[connection_id].cancel()
            del self._connection_timeouts[connection_id]
        await self.broadcast_state_update()

    async def _on_client_error(self, connection_id: str, connection_type: str, error: str) -> None:
        """Handle client error event."""
        logger.error(f"[TRACE] Client {connection_id} ({connection_type}) error: {error}")
    
    def add_interceptor(self, interceptor: WebSocketInterceptor) -> None:
        """Add an interceptor to the processing pipeline."""
        self._interceptors.append(interceptor)
    
    def create_conversation(self, conv_id: str, url: str) -> WSConversation:
        """Create a new WebSocket conversation."""
        conversation = WSConversation(conv_id, url)
        self.conversations[conv_id] = conversation
        self.active_sessions.add(conv_id)
        self._last_activity[conv_id] = asyncio.get_event_loop().time()
        return conversation
    
    def get_conversation(self, conv_id: str) -> Optional[WSConversation]:
        """Get a WebSocket conversation by ID."""
        return self.conversations.get(conv_id)
    
    async def close_conversation(self, conv_id: str) -> None:
        """Close a WebSocket conversation."""
        if conv_id in self.active_sessions:
            # Cancel any existing timeout task
            if conv_id in self._connection_timeouts:
                self._connection_timeouts[conv_id].cancel()
                del self._connection_timeouts[conv_id]
            
            # Remove last activity timestamp
            self._last_activity.pop(conv_id, None)
            
            # Schedule delayed cleanup to allow for reconnection
            cleanup_task = asyncio.create_task(
                self._delayed_cleanup(conv_id),
                name=f"cleanup_{conv_id}"
            )
            self._cleanup_tasks.add(cleanup_task)
            cleanup_task.add_done_callback(self._cleanup_tasks.discard)

    async def _delayed_cleanup(self, conv_id: str, delay: float = CLEANUP_DELAY) -> None:
        """Delayed cleanup of connection resources."""
        try:
            await asyncio.sleep(delay)
            
            async with self._lock:
                if conv_id in self.active_sessions:
                    self.active_sessions.remove(conv_id)
                    conversation = self.conversations[conv_id]
                    conversation.closed_at = datetime.now()
                    
                    # Notify interceptors
                    for interceptor in self._interceptors:
                        try:
                            await interceptor.on_close(conversation)
                        except Exception as e:
                            logger.error(f"Error in interceptor cleanup: {e}")
                    
                    # Broadcast state update
                    await self.broadcast_state_update()
                    
                    logger.info(f"[TRACE] Cleaned up connection {conv_id}")
                
        except Exception as e:
            logger.error(f"Error in delayed cleanup for {conv_id}: {e}")

    async def broadcast_state_update(self) -> None:
        """Broadcast state update to all connected clients."""
        connections = []
        statuses = self.client_manager.get_connection_status()
        
        for conv_id in self.active_sessions:
            conv = self.conversations[conv_id]
            if conv and conv_id in statuses.get(ConnectionType.PROXY.value, {}):
                status = statuses[ConnectionType.PROXY.value][conv_id]
                connections.append({
                    "id": conv.id,
                    "url": conv.url,
                    "status": status.state.value,
                    "timestamp": conv.created_at.isoformat(),
                    "interceptorEnabled": any(i.is_enabled for i in self._interceptors),
                    "fuzzingEnabled": self._fuzzer.is_enabled,
                    "messages": len(conv.messages),
                    "healthy": status.healthy,
                    "last_error": status.last_error
                })

        message = {
            "type": "state_update",
            "data": {
                "connections": connections
            }
        }
        await self.client_manager.broadcast(message)

    async def broadcast_connection_update(self, data: dict) -> None:
        """Broadcast a connection update to all clients."""
        await self.client_manager.broadcast_connection_update(data)

    def get_active_connections(self) -> List[Dict[str, Any]]:
        """Get list of active connections."""
        connections = []
        for conv_id in self.active_sessions:
            conv = self.conversations[conv_id]
            if conv:
                connections.append({
                    "id": conv.id,
                    "url": conv.url,
                    "status": "ACTIVE",
                    "timestamp": conv.created_at.isoformat(),
                    "interceptorEnabled": any(i.is_enabled for i in self._interceptors),
                    "fuzzingEnabled": self._fuzzer.is_enabled,
                    "messages": len(conv.messages)
                })
        return connections
    
    async def handle_client_message(
        self, 
        message: WSMessage, 
        client_ws: web.WebSocketResponse,
        server_ws: aiohttp.ClientWebSocketResponse,
        conversation: WSConversation
    ) -> None:
        """Handle a WebSocket message from client."""
        # Update connection timeout
        await self._refresh_connection_timeout(conversation.id)
        
        if message.type == WSMsgType.TEXT:
            msg_type = MessageType.TEXT
        elif message.type == WSMsgType.BINARY:
            msg_type = MessageType.BINARY
        elif message.type == WSMsgType.PING:
            msg_type = MessageType.PING
        elif message.type == WSMsgType.PONG:
            msg_type = MessageType.PONG
        elif message.type == WSMsgType.CLOSE:
            msg_type = MessageType.CLOSE
        else:
            logger.warning(f"Unknown message type: {message.type}")
            return

        proxy_msg = WSProxyMessage(
            id=uuid.uuid4(),
            type=msg_type,
            data=message.data,
            direction=MessageDirection.OUTGOING,
            timestamp=datetime.now(),
            metadata={"source": "client"}
        )

        for interceptor in self._interceptors:
            proxy_msg = await interceptor.on_message(proxy_msg, conversation)
            if proxy_msg is None:
                return

        conversation.add_message(proxy_msg)
        await self._forward_message(proxy_msg, server_ws)
    
    async def handle_server_message(
        self,
        message: WSMessage,
        client_ws: web.WebSocketResponse,
        server_ws: aiohttp.ClientWebSocketResponse,
        conversation: WSConversation
    ) -> None:
        """Handle a WebSocket message from server."""
        # Update connection timeout
        await self._refresh_connection_timeout(conversation.id)
        
        if message.type == WSMsgType.TEXT:
            msg_type = MessageType.TEXT
        elif message.type == WSMsgType.BINARY:
            msg_type = MessageType.BINARY
        elif message.type == WSMsgType.PING:
            msg_type = MessageType.PING
        elif message.type == WSMsgType.PONG:
            msg_type = MessageType.PONG
        elif message.type == WSMsgType.CLOSE:
            msg_type = MessageType.CLOSE
        else:
            logger.warning(f"Unknown message type: {message.type}")
            return

        proxy_msg = WSProxyMessage(
            id=uuid.uuid4(),
            type=msg_type,
            data=message.data,
            direction=MessageDirection.INCOMING,
            timestamp=datetime.now(),
            metadata={"source": "server"}
        )

        for interceptor in self._interceptors:
            proxy_msg = await interceptor.on_message(proxy_msg, conversation)
            if proxy_msg is None:
                return

        conversation.add_message(proxy_msg)
        await self._forward_message(proxy_msg, client_ws)

    async def _refresh_connection_timeout(self, conv_id: str) -> None:
        """Refresh the connection timeout for a conversation."""
        current_time = asyncio.get_event_loop().time()
        
        async with self._lock:
            # Update last activity time
            self._last_activity[conv_id] = current_time
            
            # Cancel existing timeout task if it exists
            if conv_id in self._connection_timeouts:
                self._connection_timeouts[conv_id].cancel()
                del self._connection_timeouts[conv_id]
            
            # Create new timeout task with longer timeout
            async def timeout_handler() -> None:
                await asyncio.sleep(CONNECTION_TIMEOUT)
                # Check if there has been activity since we started waiting
                if conv_id in self._last_activity:
                    last_activity_time = self._last_activity[conv_id]
                    if current_time - last_activity_time >= CONNECTION_TIMEOUT:
                        logger.warning(f"Connection {conv_id} timed out after {CONNECTION_TIMEOUT}s of inactivity")
                        await self.close_conversation(conv_id)
                
            self._connection_timeouts[conv_id] = asyncio.create_task(timeout_handler())

    async def _forward_message(
        self,
        message: WSProxyMessage,
        ws: Union[web.WebSocketResponse, aiohttp.ClientWebSocketResponse]
    ) -> None:
        """Forward a message to a WebSocket connection."""
        try:
            if message.type == MessageType.TEXT:
                await ws.send_str(message.data)
            elif message.type == MessageType.BINARY:
                await ws.send_bytes(message.data)
            elif message.type == MessageType.PING:
                await ws.ping(message.data)
            elif message.type == MessageType.PONG:
                await ws.pong(message.data)
            elif message.type == MessageType.CLOSE:
                await ws.close()
        except Exception as e:
            logger.error(f"Error forwarding message: {e}")

    async def replay_conversation(
        self,
        conv_id: str,
        client_ws: web.WebSocketResponse,
        server_ws: aiohttp.ClientWebSocketResponse
    ) -> None:
        """Replay a recorded WebSocket conversation."""
        conversation = self.get_conversation(conv_id)
        if conversation:
            await conversation.replay_messages(client_ws, server_ws)
    
    async def fuzz_conversation(self, conv_id: str) -> Optional[List[WSProxyMessage]]:
        """Fuzz a recorded WebSocket conversation."""
        conversation = self.get_conversation(conv_id)
        if conversation:
            return await self._fuzzer.fuzz_conversation(conversation)
        return None

    async def handle_websocket(self, info: Tuple[web.Request, str, Dict[str, str]]) -> web.WebSocketResponse:
        """Handle a WebSocket connection proxy request."""
        request, target_url, target_headers = info
        
        # Create WebSocket connection to client
        client_ws = web.WebSocketResponse()
        await client_ws.prepare(request)
        
        # Create conversation record and generate unique ID
        conv_id = str(uuid.uuid4())
        conversation = self.create_conversation(conv_id, target_url)
        
        try:
            # Let interceptors validate connection
            for interceptor in self._interceptors:
                if not await interceptor.on_connect(request):
                    await client_ws.close()
                    return client_ws

            # Register client with client manager as proxy connection
            await self.client_manager.connect(client_ws, ConnectionType.PROXY.value)

            # Connect to target server
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    target_url,
                    headers=target_headers or {},
                    protocols=request.headers.get('Sec-WebSocket-Protocol', '').split(','),
                    heartbeat=5.0,  # Send ping every 5 seconds to match health check interval
                    timeout=10.0  # Connection timeout
                ) as server_ws:
                    # Set up initial connection timeout
                    await self._refresh_connection_timeout(conv_id)
                    
                    try:
                        # Send initial connection update
                        await self.broadcast_connection_update({
                            "id": conv_id,
                            "status": "connected",
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Handle bidirectional communication
                        await asyncio.gather(
                            self._forward_client_messages(client_ws, server_ws, conversation),
                            self._forward_server_messages(client_ws, server_ws, conversation)
                        )
                    except Exception as e:
                        logger.error(f"Error in websocket communication: {e}")
                        raise

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await client_ws.close()
        finally:
            await self.client_manager.disconnect(client_ws, ConnectionType.PROXY.value)
            await self.close_conversation(conv_id)
                
        return client_ws

    async def connect(self, websocket: WebSocket) -> None:
        """Connect a new WebSocket client."""
        await self.client_manager.connect(websocket, ConnectionType.UI.value)
        await self.broadcast_state_update()

    async def disconnect(self, websocket: WebSocket) -> None:
        """Disconnect a WebSocket client."""
        await self.client_manager.disconnect(websocket, ConnectionType.UI.value)
        await self.broadcast_state_update()

    async def broadcast(self, message: Union[str, dict]) -> None:
        """Broadcast a message to all connected clients."""
        if isinstance(message, dict):
            message = json.dumps(message)
        await self.client_manager.broadcast({"type": "message", "data": message})

    async def _forward_client_messages(
        self,
        client_ws: web.WebSocketResponse,
        server_ws: aiohttp.ClientWebSocketResponse,
        conversation: WSConversation
    ) -> None:
        """Forward messages from client to server."""
        async for msg in client_ws:
            if msg.type in (WSMsgType.TEXT, WSMsgType.BINARY):
                await self.handle_client_message(msg, client_ws, server_ws, conversation)
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"Client WebSocket error: {msg.data}")
                break
            elif msg.type == WSMsgType.CLOSING:
                break

    async def _forward_server_messages(
        self,
        client_ws: web.WebSocketResponse,
        server_ws: aiohttp.ClientWebSocketResponse,
        conversation: WSConversation
    ) -> None:
        """Forward messages from server to client."""
        async for msg in server_ws:
            if msg.type in (WSMsgType.TEXT, WSMsgType.BINARY):
                await self.handle_server_message(msg, client_ws, server_ws, conversation)
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"Server WebSocket error: {msg.data}")
                break
            elif msg.type == WSMsgType.CLOSING:
                break
