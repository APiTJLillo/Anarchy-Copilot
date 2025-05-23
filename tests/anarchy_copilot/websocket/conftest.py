"""
WebSocket testing fixtures for Anarchy Copilot.
"""

import pytest
from datetime import datetime
from typing import AsyncGenerator, Dict, Generator, List
from uuid import uuid4
from aiohttp import web
from aiohttp.test_utils import TestClient
from aiohttp.web import Application

from proxy.websocket.types import (
    WSMessage, ConnectionType, MessageType, MessageDirection,
    create_text_message, create_binary_message
)
from proxy.websocket.manager import WebSocketManager
from proxy.websocket.conversation import WSConversation

@pytest.fixture
async def ws_manager() -> AsyncGenerator[WebSocketManager, None]:
    """Fixture for a WebSocket manager instance."""
    manager = WebSocketManager()
    yield manager

@pytest.fixture
async def ws_conversation() -> AsyncGenerator[WSConversation, None]:
    """Fixture for a WebSocket conversation."""
    conversation = WSConversation(
        id="test_conv",
        url="ws://test.local/ws"
    )
    yield conversation

@pytest.fixture
def sample_ws_messages() -> List[WSMessage]:
    """Fixture providing sample WebSocket messages for testing."""
    now = datetime.utcnow()
    return [
        create_text_message(
            data="Hello world",
            direction=MessageDirection.OUTGOING,
            id=uuid4(),
            timestamp=now,
            metadata={"test": True}
        ),
        create_binary_message(
            data=b"Binary data",
            direction=MessageDirection.INCOMING,
            id=uuid4(),
            timestamp=now,
            metadata={"test": True}
        ),
        create_text_message(
            data='{"password": "secret123"}',
            direction=MessageDirection.OUTGOING,
            id=uuid4(),
            timestamp=now,
            metadata={"sensitive": True}
        )
    ]

@pytest.fixture
async def ws_test_app() -> Application:
    """Fixture providing a test aiohttp application with WebSocket endpoint."""
    app = web.Application()

    async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                await ws.send_str(f"Echo: {msg.data}")
            elif msg.type == web.WSMsgType.BINARY:
                await ws.send_bytes(msg.data)
            elif msg.type == web.WSMsgType.CLOSE:
                break

        return ws

    app.router.add_get("/ws", websocket_handler)
    return app

@pytest.fixture
async def ws_test_client(ws_test_app: Application, aiohttp_client) -> AsyncGenerator[TestClient, None]:
    """Fixture providing a test client with WebSocket support."""
    client = await aiohttp_client(ws_test_app)
    yield client
    await client.close()
