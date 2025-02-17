"""
WebSocket session and proxy management.
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union

import aiohttp
from aiohttp import ClientWebSocketResponse, WSMessage, WSMsgType, web

from .conversation import WSConversation
from .fuzzing import WSFuzzer
from .types import WSMessage as WSProxyMessage, WSMessageType
from .interceptor import WebSocketInterceptor

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections and sessions."""

    def __init__(self):
        """Initialize WebSocket manager."""
        self.conversations: Dict[str, WSConversation] = {}
        self.active_sessions: Set[str] = set()
        self._fuzzer = WSFuzzer()
        self._interceptors: List[WebSocketInterceptor] = []
    
    def add_interceptor(self, interceptor: WebSocketInterceptor) -> None:
        """Add an interceptor to the processing pipeline.
        
        Args:
            interceptor: WebSocket interceptor instance
        """
        self._interceptors.append(interceptor)
    
    def create_conversation(self, conv_id: str, url: str) -> WSConversation:
        """Create a new WebSocket conversation.
        
        Args:
            conv_id: Unique conversation ID
            url: WebSocket URL
            
        Returns:
            New conversation instance
        """
        conversation = WSConversation(conv_id, url)
        self.conversations[conv_id] = conversation
        self.active_sessions.add(conv_id)
        return conversation
    
    def get_conversation(self, conv_id: str) -> Optional[WSConversation]:
        """Get a WebSocket conversation by ID.
        
        Args:
            conv_id: Conversation ID to retrieve
            
        Returns:
            WSConversation if found, None otherwise
        """
        return self.conversations.get(conv_id)
    
    def close_conversation(self, conv_id: str) -> None:
        """Close a WebSocket conversation.
        
        Args:
            conv_id: ID of conversation to close
        """
        if conv_id in self.active_sessions:
            self.active_sessions.remove(conv_id)
            conversation = self.conversations[conv_id]
            conversation.closed_at = datetime.now()
    
    async def handle_client_message(
        self, 
        message: WSMessage, 
        client_ws: web.WebSocketResponse,
        server_ws: aiohttp.ClientWebSocketResponse,
        conversation: WSConversation
    ) -> None:
        """Handle a WebSocket message from client.
        
        Args:
            message: Client WebSocket message
            client_ws: Client WebSocket connection
            server_ws: Server WebSocket connection  
            conversation: Current WebSocket conversation
        """
        # Convert to internal message format
        proxy_msg = WSProxyMessage(
            type=WSMessageType(message.type),
            data=message.data,
            metadata={}
        )

        # Run interceptors
        for interceptor in self._interceptors:
            proxy_msg = await interceptor.on_message(proxy_msg, conversation)
            if proxy_msg is None:
                # Message was blocked
                return

        # Add to conversation history
        conversation.add_message(proxy_msg)

        # Forward to server
        await self._forward_message(proxy_msg, server_ws)
    
    async def handle_server_message(
        self,
        message: WSMessage,
        client_ws: web.WebSocketResponse,
        server_ws: aiohttp.ClientWebSocketResponse,
        conversation: WSConversation
    ) -> None:
        """Handle a WebSocket message from server.
        
        Args:
            message: Server WebSocket message
            client_ws: Client WebSocket connection
            server_ws: Server WebSocket connection
            conversation: Current WebSocket conversation
        """
        # Convert to internal message format
        proxy_msg = WSProxyMessage(
            type=WSMessageType(message.type),
            data=message.data,
            metadata={}
        )

        # Run interceptors
        for interceptor in self._interceptors:
            proxy_msg = await interceptor.on_message(proxy_msg, conversation)
            if proxy_msg is None:
                # Message was blocked
                return

        # Add to conversation history  
        conversation.add_message(proxy_msg)

        # Forward to client
        await self._forward_message(proxy_msg, client_ws)

    async def _forward_message(
        self,
        message: WSProxyMessage,
        ws: Union[web.WebSocketResponse, aiohttp.ClientWebSocketResponse]
    ) -> None:
        """Forward a message to a WebSocket connection.
        
        Args:
            message: Message to forward
            ws: WebSocket connection to send to
        """
        try:
            if message.type == WSMessageType.TEXT:
                await ws.send_str(message.data)
            elif message.type == WSMessageType.BINARY:
                await ws.send_bytes(message.data)
            elif message.type == WSMessageType.PING:
                await ws.ping(message.data)
            elif message.type == WSMessageType.PONG:
                await ws.pong(message.data)
            elif message.type == WSMessageType.CLOSE:
                await ws.close()
        except Exception as e:
            logger.error(f"Error forwarding message: {e}")

    async def replay_conversation(
        self,
        conv_id: str,
        client_ws: web.WebSocketResponse,
        server_ws: aiohttp.ClientWebSocketResponse
    ) -> None:
        """Replay a recorded WebSocket conversation.
        
        Args:
            conv_id: Conversation ID to replay
            client_ws: Client WebSocket connection
            server_ws: Server WebSocket connection
        """
        conversation = self.get_conversation(conv_id)
        if conversation:
            await conversation.replay_messages(client_ws, server_ws)
    
    async def fuzz_conversation(self, conv_id: str) -> Optional[List[WSProxyMessage]]:
        """Fuzz a recorded WebSocket conversation.
        
        Args:
            conv_id: Conversation ID to fuzz
            
        Returns:
            List of fuzzed messages if conversation exists
        """
        conversation = self.get_conversation(conv_id)
        if conversation:
            return await self._fuzzer.fuzz_conversation(conversation)
        return None

    async def handle_websocket(self, info: Tuple[web.Request, str, Dict[str, str]]) -> web.WebSocketResponse:
        """Handle a WebSocket connection proxy request.
        
        Args:
            request: The HTTP request that initiated the WebSocket
            target_url: The target WebSocket URL to connect to
            target_headers: Optional headers to send to target
            
        Returns:
            WebSocket response object
        """
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

            # Connect to target server
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    target_url,
                    headers=target_headers or {},
                    protocols=request.headers.get('Sec-WebSocket-Protocol', '').split(','),
                ) as server_ws:
                    # Handle bidirectional communication
                    await asyncio.gather(
                        self._forward_client_messages(client_ws, server_ws, conversation),
                        self._forward_server_messages(client_ws, server_ws, conversation)
                    )
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await client_ws.close()
        finally:
            self.close_conversation(conv_id)
            # Notify interceptors
            for interceptor in self._interceptors:
                await interceptor.on_close(conversation)
                
        return client_ws

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

    async def cleanup(self) -> None:
        """Cleanup all WebSocket resources."""
        try:
            # Close all active conversations
            for conv_id in list(self.active_sessions):
                try:
                    conversation = self.conversations[conv_id]
                    # Mark as closed
                    conversation.closed_at = datetime.now()
                    # Trigger interceptor close events
                    for interceptor in self._interceptors:
                        try:
                            await interceptor.on_close(conversation)
                        except Exception as e:
                            logger.error(f"Error in interceptor cleanup: {e}")
                except Exception as e:
                    logger.error(f"Error closing conversation {conv_id}: {e}")

            # Clear all state
            self.active_sessions.clear()
            self.conversations.clear()
            self._interceptors.clear()
            
        except Exception as e:
            logger.error(f"Error during WebSocket manager cleanup: {e}", exc_info=True)

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
