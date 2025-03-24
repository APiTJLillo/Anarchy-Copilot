"""Tests for CONNECT method and TLS tunnel setup."""
import pytest
import asyncio
import ssl
from unittest.mock import Mock, AsyncMock, patch

from proxy.server.handlers.connect import ConnectHandler
from proxy.server.tls.transport import BufferedTransport

@pytest.fixture
def mock_ca_handler():
    """Create a mock CA handler."""
    ca = Mock()
    ca.get_certificate.return_value = ("test.crt", "test.key")
    return ca

@pytest.fixture
def connect_handler(connection_id, mock_transport):
    """Create a CONNECT handler instance."""
    return ConnectHandler(connection_id, mock_transport)

@pytest.mark.asyncio
async def test_successful_connect(connect_handler, mock_ca_handler, echo_server):
    """Test successful CONNECT handling."""
    host, port = echo_server
    
    # Handle CONNECT request
    success = await connect_handler.handle_connect(host, port, mock_ca_handler)
    
    # Verify success
    assert success
    assert connect_handler.client_transport is not None
    assert connect_handler.server_transport is not None

@pytest.mark.asyncio
async def test_connection_failure(connect_handler, mock_ca_handler):
    """Test CONNECT failure handling."""
    # Try to connect to non-existent server
    success = await connect_handler.handle_connect("nonexistent.local", 12345, mock_ca_handler)
    
    # Verify failure
    assert not success
    assert connect_handler.client_transport is None
    assert connect_handler.server_transport is None

@pytest.mark.asyncio
async def test_certificate_handling(connect_handler, mock_ca_handler):
    """Test certificate acquisition and setup."""
    # Mock certificate paths
    mock_ca_handler.get_certificate.return_value = ("cert.pem", "key.pem")
    
    with patch("ssl.create_default_context") as mock_ctx:
        mock_ctx.return_value = Mock()
        
        await connect_handler._setup_tls("example.com", 443, mock_ca_handler)
        
        # Verify certificate was requested
        mock_ca_handler.get_certificate.assert_called_once_with("example.com")
        
        # Verify SSL contexts were created
        mock_ctx.assert_called()

@pytest.mark.asyncio
async def test_connection_test_retries(connect_handler):
    """Test connection testing with retries."""
    # Mock failing connection that succeeds on retry
    fail_count = [0]
    
    async def mock_open_connection(*args, **kwargs):
        fail_count[0] += 1
        if fail_count[0] < 2:
            raise ConnectionRefusedError()
        return Mock(), Mock()
    
    with patch("asyncio.open_connection", mock_open_connection):
        result = await connect_handler._test_connection("example.com", 443)
        
        # Should succeed after retry
        assert result
        assert fail_count[0] == 2

@pytest.mark.asyncio
async def test_tls_upgrade(connect_handler, mock_ca_handler):
    """Test TLS connection upgrade."""
    # Mock SSL context and transport
    ssl_context = Mock()
    transport = Mock()
    
    with patch("asyncio.get_event_loop") as mock_loop:
        mock_loop.return_value = Mock()
        mock_loop.return_value.start_tls.return_value = transport
        
        success = await connect_handler._upgrade_client_tls(ssl_context)
        
        # Verify TLS upgrade
        assert success
        assert connect_handler._tls_server_transport == transport

@pytest.mark.asyncio
async def test_server_connection_establishment(connect_handler, mock_ca_handler):
    """Test server connection establishment."""
    ssl_context = Mock()
    transport = Mock()
    protocol = Mock()
    
    with patch("asyncio.get_event_loop") as mock_loop:
        mock_loop.return_value = Mock()
        mock_loop.return_value.create_connection.return_value = (transport, protocol)
        
        success = await connect_handler._establish_server_tls(
            "example.com", 443, ssl_context
        )
        
        # Verify connection
        assert success
        assert connect_handler._remote_transport == transport

def test_connection_cleanup(connect_handler):
    """Test resource cleanup."""
    # Create mock transports
    client_transport = Mock()
    server_transport = Mock()
    connect_handler._tls_server_transport = client_transport
    connect_handler._remote_transport = server_transport
    
    # Perform cleanup
    connect_handler.close()
    
    # Verify transports were closed
    client_transport.close.assert_called_once()
    server_transport.close.assert_called_once()
    assert connect_handler._tls_server_transport is None
    assert connect_handler._remote_transport is None

@pytest.mark.asyncio
async def test_error_responses(connect_handler):
    """Test error response sending."""
    # Test various error scenarios
    error_cases = [
        (502, "Connection failed"),
        (504, "Gateway timeout"),
        (400, "Bad request")
    ]
    
    for status, message in error_cases:
        await connect_handler._send_error(status, message)
        
        # Verify error was sent
        data = connect_handler.transport.write.call_args[0][0]
        assert str(status).encode() in data
        assert message.encode() in data

@pytest.mark.asyncio
async def test_tls_info_update(connect_handler, mock_ssl_object):
    """Test TLS information updating."""
    # Setup mock TLS transport
    transport = Mock()
    transport.get_extra_info.return_value = mock_ssl_object
    connect_handler._tls_server_transport = transport
    
    # Update TLS info
    connect_handler._update_tls_info()
    
    # Verify SSL info was extracted
    transport.get_extra_info.assert_called_with('ssl_object')

@pytest.mark.asyncio
async def test_concurrent_connections(event_loop):
    """Test handling multiple concurrent connections."""
    # Create multiple handlers
    handlers = [
        ConnectHandler(f"conn_{i}", Mock())
        for i in range(5)
    ]
    
    # Setup echo server
    async def echo_server(reader, writer):
        while True:
            data = await reader.read(1024)
            if not data:
                break
            writer.write(data)
            await writer.drain()
        writer.close()
        await writer.wait_closed()
    
    server = await asyncio.start_server(echo_server, '127.0.0.1', 0)
    host, port = server.sockets[0].getsockname()
    
    async with server:
        # Test concurrent connections
        tasks = [
            handler.handle_connect(host, port, Mock())
            for handler in handlers
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all connections succeeded
        assert all(results)
        
        # Cleanup
        for handler in handlers:
            handler.close()

@pytest.mark.asyncio
async def test_connection_timeouts(connect_handler):
    """Test connection timeout handling."""
    with patch("asyncio.open_connection") as mock_connect:
        # Simulate timeout
        mock_connect.side_effect = asyncio.TimeoutError()
        
        success = await connect_handler.handle_connect(
            "example.com", 443, Mock()
        )
        
        # Verify timeout was handled
        assert not success
        
        # Verify error response
        connect_handler.transport.write.assert_called()
        data = connect_handler.transport.write.call_args[0][0]
        assert b"504" in data  # Gateway Timeout
