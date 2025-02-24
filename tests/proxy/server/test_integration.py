"""Integration tests for HTTPS interception components."""
import pytest
import asyncio
import ssl
import aiohttp
from typing import Tuple
import json
from pathlib import Path

from proxy.server.https_intercept_protocol import HttpsInterceptProtocol
from proxy.server.certificates import CertificateAuthority
from proxy.server.tls.connection_manager import connection_mgr

@pytest.fixture
async def proxy_server(unused_tcp_port: int, tmp_path: Path, ca_handler):
    """Create and start a test proxy server."""
    # Create protocol factory
    factory = HttpsInterceptProtocol.create_protocol_factory(ca=ca_handler)
    
    # Start server
    server = await asyncio.get_event_loop().create_server(
        factory,
        '127.0.0.1',
        unused_tcp_port
    )
    
    async with server:
        yield ('127.0.0.1', unused_tcp_port)

@pytest.fixture
async def https_echo_server(unused_tcp_port: int, ca_handler) -> Tuple[str, int]:
    """Create an HTTPS echo server for testing."""
    # Create self-signed certificate
    cert_path, key_path = ca_handler.get_certificate("localhost")
    
    # SSL context for server
    ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_ctx.load_cert_chain(cert_path, key_path)
    
    async def handle_request(reader: asyncio.StreamReader, 
                           writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                data = await reader.read(8192)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except Exception as e:
            print(f"Echo server error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    server = await asyncio.start_server(
        handle_request,
        '127.0.0.1',
        unused_tcp_port,
        ssl=ssl_ctx
    )
    
    async with server:
        yield ('127.0.0.1', unused_tcp_port)

@pytest.fixture
def proxy_session(proxy_server):
    """Create aiohttp client session using the proxy."""
    host, port = proxy_server
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=False),
        proxy=f"http://{host}:{port}"
    )

@pytest.mark.asyncio
async def test_https_request_response(proxy_server, https_echo_server, proxy_session):
    """Test complete HTTPS request/response cycle through proxy."""
    # Prepare test data
    test_data = {
        "method": "POST",
        "path": "/test",
        "headers": {"Content-Type": "application/json"},
        "body": {"test": "data"}
    }
    
    echo_host, echo_port = https_echo_server
    url = f"https://localhost:{echo_port}/test"
    
    # Send request through proxy
    async with proxy_session.post(
        url,
        json=test_data["body"],
        headers=test_data["headers"]
    ) as response:
        # Verify response
        assert response.status == 200
        data = await response.json()
        assert data == test_data["body"]

@pytest.mark.asyncio
async def test_connection_tracking(proxy_server, https_echo_server, proxy_session):
    """Test connection tracking and metrics through proxy."""
    echo_host, echo_port = https_echo_server
    url = f"https://localhost:{echo_port}/test"
    
    # Get initial connection count
    initial_count = connection_mgr.get_active_connection_count()
    
    # Make request
    async with proxy_session.get(url) as response:
        # Verify connection was tracked
        assert connection_mgr.get_active_connection_count() > initial_count
        assert response.status == 200
    
    # Wait for cleanup
    await asyncio.sleep(0.1)
    
    # Verify connection was cleaned up
    assert connection_mgr.get_active_connection_count() == initial_count

@pytest.mark.asyncio
async def test_large_data_transfer(proxy_server, https_echo_server, proxy_session):
    """Test large data transfer through proxy."""
    # Create large test data
    large_data = b"X" * (1024 * 1024)  # 1MB
    echo_host, echo_port = https_echo_server
    url = f"https://localhost:{echo_port}/test"
    
    # Send large data
    async with proxy_session.post(url, data=large_data) as response:
        assert response.status == 200
        response_data = await response.read()
        assert len(response_data) == len(large_data)
        assert response_data == large_data

@pytest.mark.asyncio
async def test_concurrent_requests(proxy_server, https_echo_server, proxy_session):
    """Test handling multiple concurrent requests."""
    echo_host, echo_port = https_echo_server
    url = f"https://localhost:{echo_port}/test"
    
    # Create multiple requests
    async def make_request(i: int):
        data = {"request": i}
        async with proxy_session.post(url, json=data) as response:
            assert response.status == 200
            result = await response.json()
            assert result == data
    
    # Send concurrent requests
    tasks = [make_request(i) for i in range(10)]
    await asyncio.gather(*tasks)

@pytest.mark.asyncio
async def test_error_handling(proxy_server, proxy_session):
    """Test error handling in proxy chain."""
    # Try invalid host
    with pytest.raises(aiohttp.ClientError):
        async with proxy_session.get("https://invalid.local/test") as response:
            assert response.status in (502, 504)  # Bad Gateway or Gateway Timeout

@pytest.mark.asyncio
async def test_protocol_state_transitions(proxy_server, https_echo_server, proxy_session):
    """Test protocol state transitions through complete cycle."""
    echo_host, echo_port = https_echo_server
    url = f"https://localhost:{echo_port}/test"
    
    # Monitor active connections before request
    pre_request_conns = connection_mgr.get_active_connection_count()
    
    async with proxy_session.get(url) as response:
        # Verify connection established
        assert connection_mgr.get_active_connection_count() > pre_request_conns
        assert response.status == 200
    
    # Wait for cleanup
    await asyncio.sleep(0.1)
    
    # Verify cleanup completed
    assert connection_mgr.get_active_connection_count() == pre_request_conns

@pytest.mark.asyncio
async def test_tls_version_negotiation(proxy_server, https_echo_server, proxy_session):
    """Test TLS version negotiation through proxy."""
    echo_host, echo_port = https_echo_server
    url = f"https://localhost:{echo_port}/test"
    
    async with proxy_session.get(url) as response:
        assert response.status == 200
        
        # Get active connection
        connections = [
            conn for conn in connection_mgr._active_connections.values()
            if conn.get("host") == "localhost"
        ]
        assert len(connections) > 0
        
        # Verify TLS version
        conn = connections[0]
        assert conn["tls_version"] is not None
        assert "TLSv1.2" in conn["tls_version"] or "TLSv1.3" in conn["tls_version"]

@pytest.mark.asyncio
async def test_connection_lifecycle_events(proxy_server, https_echo_server, proxy_session):
    """Test connection lifecycle events are properly tracked."""
    echo_host, echo_port = https_echo_server
    url = f"https://localhost:{echo_port}/test"
    
    events_received = []
    
    def event_callback(event):
        events_received.append(event)
    
    # Monitor connection events
    connection_mgr.on_connection_event = event_callback
    
    async with proxy_session.get(url) as response:
        assert response.status == 200
    
    # Wait for events
    await asyncio.sleep(0.1)
    
    # Verify event sequence
    event_types = [e["type"] for e in events_received]
    assert "connect" in event_types
    assert "data" in event_types
    assert "close" in event_types

@pytest.mark.asyncio
async def test_proxy_chain_metrics(proxy_server, https_echo_server, proxy_session):
    """Test metrics collection through the proxy chain."""
    echo_host, echo_port = https_echo_server
    url = f"https://localhost:{echo_port}/test"
    
    # Send request with known size
    test_data = b"X" * 1000
    
    async with proxy_session.post(url, data=test_data) as response:
        assert response.status == 200
        response_data = await response.read()
        
        # Get connection metrics
        connections = list(connection_mgr._active_connections.values())
        assert len(connections) > 0
        
        conn = connections[0]
        assert conn["bytes_received"] >= len(test_data)
        assert conn["bytes_sent"] >= len(response_data)
        assert conn["requests_processed"] > 0
