"""WebSocket protocol compliance tests."""
import pytest
import asyncio
import aiohttp
import websockets
import json
import base64
import hashlib
from typing import Tuple, Optional
import struct
from dataclasses import dataclass
import secrets
import json

from .mock_server import MockHttpsServer, ServerConfig, ResponseTemplate

@dataclass
class WebSocketFrame:
    """WebSocket frame structure."""
    fin: bool = True
    rsv1: bool = False
    rsv2: bool = False
    rsv3: bool = False
    opcode: int = 0x1  # 0x1 for text, 0x2 for binary
    masked: bool = True
    payload_length: int = 0
    mask_key: Optional[bytes] = None
    payload: bytes = b""

    def serialize(self) -> bytes:
        """Convert frame to bytes."""
        header = (
            (0x80 if self.fin else 0) |
            (0x40 if self.rsv1 else 0) |
            (0x20 if self.rsv2 else 0) |
            (0x10 if self.rsv3 else 0) |
            (self.opcode & 0x0f)
        )
        
        mask_bit = 0x80 if self.masked else 0
        
        if self.payload_length < 126:
            length = self.payload_length
            length_bytes = struct.pack("!B", mask_bit | length)
        elif self.payload_length < 65536:
            length_bytes = struct.pack("!BH", mask_bit | 126, self.payload_length)
        else:
            length_bytes = struct.pack("!BQ", mask_bit | 127, self.payload_length)
        
        frame = bytes([header]) + length_bytes
        
        if self.masked:
            if not self.mask_key:
                self.mask_key = secrets.token_bytes(4)
            frame += self.mask_key
            masked_payload = bytes(b ^ k for b, k in zip(
                self.payload,
                self.mask_key * (len(self.payload) // 4 + 1)
            ))
        else:
            masked_payload = self.payload
            
        return frame + masked_payload

@pytest.fixture
async def ws_server():
    """Create a server with WebSocket support."""
    config = ServerConfig(port=0)
    server = MockHttpsServer(config)
    
    # Add WebSocket upgrade handler
    @server.route("/ws")
    async def ws_handler(request):
        if request["headers"].get("upgrade", "").lower() != "websocket":
            return ResponseTemplate(status=400)
            
        key = request["headers"].get("sec-websocket-key")
        if not key:
            return ResponseTemplate(status=400)
            
        # Calculate accept key
        accept = base64.b64encode(
            hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
            ).digest()
        ).decode()
        
        return ResponseTemplate(
            status=101,
            headers={
                "Upgrade": "websocket",
                "Connection": "Upgrade",
                "Sec-WebSocket-Accept": accept
            }
        )
    
    await server.start()
    port = server.server.sockets[0].getsockname()[1]
    yield server, port
    await server.stop()

async def create_ws_connection(port: int) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Create a raw WebSocket connection."""
    reader, writer = await asyncio.open_connection("localhost", port)
    
    # Send WebSocket upgrade request
    key = base64.b64encode(secrets.token_bytes(16)).decode()
    request = (
        f"GET /ws HTTP/1.1\r\n"
        f"Host: localhost:{port}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    ).encode()
    
    writer.write(request)
    await writer.drain()
    
    # Read upgrade response
    response = await reader.readuntil(b"\r\n\r\n")
    if not response.startswith(b"HTTP/1.1 101"):
        raise ValueError("WebSocket upgrade failed")
        
    return reader, writer

@pytest.mark.asyncio
async def test_websocket_handshake(ws_server):
    """Test WebSocket handshake compliance."""
    server, port = ws_server
    
    async with aiohttp.ClientSession() as session:
        # Test without upgrade header
        async with session.get(f"http://localhost:{port}/ws") as response:
            assert response.status == 400
        
        # Test with invalid version
        async with session.get(
            f"http://localhost:{port}/ws",
            headers={
                "Upgrade": "websocket",
                "Connection": "Upgrade",
                "Sec-WebSocket-Version": "12"
            }
        ) as response:
            assert response.status == 400
        
        # Test successful upgrade
        async with session.ws_connect(f"http://localhost:{port}/ws") as ws:
            assert ws.closed is False

@pytest.mark.asyncio
async def test_frame_masking(ws_server):
    """Test WebSocket frame masking requirements."""
    server, port = ws_server
    reader, writer = await create_ws_connection(port)
    
    try:
        # Test unmasked frame (should be rejected)
        frame = WebSocketFrame(
            opcode=0x1,
            masked=False,
            payload=b"test",
            payload_length=4
        )
        writer.write(frame.serialize())
        await writer.drain()
        
        # Server should close connection
        response = await reader.read(100)
        assert response[0] & 0x0f == 0x8  # Close frame
        
        # Test properly masked frame
        reader, writer = await create_ws_connection(port)
        frame = WebSocketFrame(
            opcode=0x1,
            masked=True,
            payload=b"test",
            payload_length=4
        )
        writer.write(frame.serialize())
        await writer.drain()
        
        # Server should accept the frame
        response = await reader.read(100)
        assert response[0] & 0x0f != 0x8  # Not a close frame
        
    finally:
        writer.close()
        await writer.wait_closed()

@pytest.mark.asyncio
async def test_fragmented_messages(ws_server):
    """Test handling of fragmented WebSocket messages."""
    server, port = ws_server
    
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"http://localhost:{port}/ws") as ws:
            # Send fragmented message
            await ws.send_str("Hello", compress=False)
            await ws.send_str(" ", compress=False)
            await ws.send_str("World", compress=False)
            
            # Receive complete message
            msg = await ws.receive_str()
            assert msg == "Hello World"

@pytest.mark.asyncio
async def test_control_frames(ws_server):
    """Test WebSocket control frame handling."""
    server, port = ws_server
    reader, writer = await create_ws_connection(port)
    
    try:
        # Send ping frame
        frame = WebSocketFrame(
            opcode=0x9,  # Ping
            masked=True,
            payload=b"ping",
            payload_length=4
        )
        writer.write(frame.serialize())
        await writer.drain()
        
        # Expect pong response
        response = await reader.read(100)
        assert response[0] & 0x0f == 0xA  # Pong frame
        
        # Send close frame
        frame = WebSocketFrame(
            opcode=0x8,  # Close
            masked=True,
            payload=struct.pack("!H", 1000),  # Normal closure
            payload_length=2
        )
        writer.write(frame.serialize())
        await writer.drain()
        
        # Expect close response
        response = await reader.read(100)
        assert response[0] & 0x0f == 0x8  # Close frame
        
    finally:
        writer.close()
        await writer.wait_closed()

@pytest.mark.asyncio
async def test_max_frame_size(ws_server):
    """Test handling of large WebSocket frames."""
    server, port = ws_server
    
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"http://localhost:{port}/ws") as ws:
            # Send large frame (16MB)
            large_data = "X" * (16 * 1024 * 1024)
            await ws.send_str(large_data)
            
            # Should receive same data back
            msg = await ws.receive_str()
            assert len(msg) == len(large_data)

@pytest.mark.asyncio
async def test_concurrent_connections(ws_server):
    """Test handling multiple WebSocket connections."""
    server, port = ws_server
    
    async def client():
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"http://localhost:{port}/ws") as ws:
                await ws.send_str("test")
                msg = await ws.receive_str()
                assert msg == "test"
    
    # Create multiple concurrent connections
    clients = [client() for _ in range(10)]
    await asyncio.gather(*clients)

@pytest.mark.asyncio
async def test_protocol_errors(ws_server):
    """Test handling of WebSocket protocol errors."""
    server, port = ws_server
    reader, writer = await create_ws_connection(port)
    
    try:
        # Test invalid opcode
        frame = WebSocketFrame(
            opcode=0xF,  # Invalid opcode
            masked=True,
            payload=b"test",
            payload_length=4
        )
        writer.write(frame.serialize())
        await writer.drain()
        
        # Should receive close frame with protocol error
        response = await reader.read(100)
        assert response[0] & 0x0f == 0x8  # Close frame
        code = struct.unpack("!H", response[2:4])[0]
        assert code == 1002  # Protocol error
        
    finally:
        writer.close()
        await writer.wait_closed()

if __name__ == "__main__":
    pytest.main([__file__])
