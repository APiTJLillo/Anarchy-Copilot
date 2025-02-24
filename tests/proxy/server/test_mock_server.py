"""Tests for mock HTTPS server."""
import pytest
import asyncio
import aiohttp
import ssl
import json
from pathlib import Path
from typing import Tuple
import tempfile
from datetime import datetime

from .mock_server import MockHttpsServer, ServerConfig, ResponseTemplate

@pytest.fixture
async def mock_server():
    """Create and start a mock server instance."""
    config = ServerConfig(port=0)  # Use random port
    server = MockHttpsServer(config)
    await server.start()
    
    # Get the actual port
    port = server.server.sockets[0].getsockname()[1]
    
    yield server, port
    
    await server.stop()

@pytest.fixture
async def ssl_server():
    """Create and start a mock server with SSL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cert_path = Path(tmpdir) / "cert.pem"
        key_path = Path(tmpdir) / "key.pem"
        
        # Create self-signed certificate
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
        import datetime
        
        # Generate key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost")
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.now()
        ).not_valid_after(
            datetime.datetime.now() + datetime.timedelta(days=1)
        ).sign(private_key, hashes.SHA256())
        
        # Save certificate and key
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
            
        with open(key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        config = ServerConfig(
            port=0,
            ssl_cert=str(cert_path),
            ssl_key=str(key_path)
        )
        server = MockHttpsServer(config)
        await server.start()
        
        port = server.server.sockets[0].getsockname()[1]
        
        yield server, port, cert_path
        
        await server.stop()

@pytest.mark.asyncio
async def test_basic_request(mock_server):
    """Test basic request handling."""
    server, port = mock_server
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:{port}/test") as response:
            assert response.status == 200
            data = await response.json()
            assert data["method"] == "GET"
            assert data["path"] == "/test"

@pytest.mark.asyncio
async def test_custom_route(mock_server):
    """Test custom route handler."""
    server, port = mock_server
    
    @server.route("/custom")
    async def custom_handler(request):
        return ResponseTemplate(
            status=201,
            headers={"X-Custom": "test"},
            body=b"Custom response"
        )
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:{port}/custom") as response:
            assert response.status == 201
            assert response.headers["X-Custom"] == "test"
            assert await response.text() == "Custom response"

@pytest.mark.asyncio
async def test_ssl_connection(ssl_server):
    """Test SSL/TLS connection."""
    server, port, cert_path = ssl_server
    
    # Create SSL context that trusts our self-signed cert
    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(cert_path)
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://localhost:{port}/test",
            ssl=ssl_context
        ) as response:
            assert response.status == 200

@pytest.mark.asyncio
async def test_error_simulation(mock_server):
    """Test error rate simulation."""
    server, port = mock_server
    server.config.error_rate = 1.0  # Always generate errors
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:{port}/test") as response:
            assert response.status == 500
            data = await response.json()
            assert "error" in data

@pytest.mark.asyncio
async def test_latency_simulation(mock_server):
    """Test latency simulation."""
    server, port = mock_server
    server.config.latency_range = (0.1, 0.1)  # Fixed latency
    
    start_time = datetime.now()
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:{port}/test"):
            duration = (datetime.now() - start_time).total_seconds()
            assert duration >= 0.1

@pytest.mark.asyncio
async def test_chunked_response(mock_server):
    """Test chunked response handling."""
    server, port = mock_server
    large_data = b"X" * 100000  # 100KB
    
    @server.route("/large")
    async def large_handler(request):
        return ResponseTemplate(
            body=large_data,
            chunk_size=8192
        )
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:{port}/large") as response:
            data = await response.read()
            assert len(data) == len(large_data)
            assert data == large_data

@pytest.mark.asyncio
async def test_keep_alive(mock_server):
    """Test keep-alive connection handling."""
    server, port = mock_server
    
    async with aiohttp.ClientSession() as session:
        # Send multiple requests on same connection
        for _ in range(5):
            async with session.get(
                f"http://localhost:{port}/test",
                headers={"Connection": "keep-alive"}
            ) as response:
                assert response.status == 200
        
        stats = server.get_stats()
        assert stats["requests"]["total"] == 5
        assert stats["requests"]["active_connections"] <= 1

@pytest.mark.asyncio
async def test_concurrent_requests(mock_server):
    """Test handling of concurrent requests."""
    server, port = mock_server
    
    async def make_request():
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://localhost:{port}/test") as response:
                return response.status
    
    # Send 10 concurrent requests
    tasks = [make_request() for _ in range(10)]
    results = await asyncio.gather(*tasks)
    
    assert all(status == 200 for status in results)
    stats = server.get_stats()
    assert stats["requests"]["total"] == 10

@pytest.mark.asyncio
async def test_server_stats(mock_server):
    """Test server statistics collection."""
    server, port = mock_server
    
    async with aiohttp.ClientSession() as session:
        # Send requests with different sizes
        await session.post(f"http://localhost:{port}/test", data="small")
        await session.post(f"http://localhost:{port}/test", data="X" * 1000)
        
    stats = server.get_stats()
    assert stats["requests"]["total"] == 2
    assert stats["bytes"]["received"] > 0
    assert stats["bytes"]["sent"] > 0
    assert stats["uptime"] > 0

@pytest.mark.asyncio
async def test_error_handling(mock_server):
    """Test server error handling."""
    server, port = mock_server
    
    @server.route("/error")
    async def error_handler(request):
        raise Exception("Test error")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:{port}/error") as response:
            assert response.status == 500
            stats = server.get_stats()
            assert stats["requests"]["errors"] > 0
