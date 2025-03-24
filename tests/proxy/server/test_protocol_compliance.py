"""Protocol compliance tests for mock server."""
import pytest
import asyncio
import aiohttp
import re
import ssl
import json
from datetime import datetime
from typing import Dict, Any
import email.utils
from http import HTTPStatus
import socket

from tests.proxy.server.mock_server import MockHttpsServer, ServerConfig, ResponseTemplate
from aiohttp import hdrs

@pytest.fixture
async def compliance_server(event_loop):
    """Create a server instance for protocol compliance testing."""
    config = ServerConfig(
        port=0,
        latency_range=(0, 0),  # No artificial latency for compliance tests
        error_rate=0
    )
    server = MockHttpsServer(config)
    await server.start()
    await asyncio.sleep(0.1)  # Give server time to fully start
    
    for sock in server.server.sockets:
        if sock.family == socket.AF_INET:
            port = sock.getsockname()[1]
            break
    else:
        raise RuntimeError("No IPv4 socket found")
    
    return server, port  # Use return instead of yield

@pytest.fixture
async def cleanup_compliance_server(compliance_server):
    """Cleanup fixture for compliance server."""
    server, _ = await compliance_server
    yield
    await server.stop()
    await asyncio.sleep(0.1)  # Allow time for cleanup

@pytest.fixture
async def compliance_server_factory():
    """Factory fixture for compliance server creation."""
    servers = []

    async def create_server():
        config = ServerConfig(
            port=0,
            latency_range=(0, 0),
            error_rate=0
        )
        server = MockHttpsServer(config)
        await server.start()
        servers.append(server)
        port = server.server.sockets[0].getsockname()[1]
        await asyncio.sleep(0)
        return server, port

    yield create_server

    # Cleanup all servers
    for server in servers:
        await server.stop()

def validate_http_date(date_str: str) -> bool:
    """Validate HTTP date format (RFC 7231)."""
    try:
        parsed = email.utils.parsedate_to_datetime(date_str)
        return isinstance(parsed, datetime)
    except (TypeError, ValueError):
        return False

async def get_response_headers(url: str) -> Dict[str, str]:
    """Get headers from a response."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return dict(response.headers)

@pytest.mark.asyncio
async def test_required_headers(compliance_server):
    """Test presence and format of required HTTP headers."""
    server, port = await compliance_server
    
    # Register test endpoint
    @server.route("/test")
    async def handler(request):
        return ResponseTemplate(
            body=b"test",
            headers={
                "Date": email.utils.formatdate(usegmt=True),
                "Server": "TestServer/1.0",
                "Content-Type": "text/plain"
            }
        )
    
    # Make request and verify headers
    headers = await get_response_headers(f"http://localhost:{port}/test")
    
    # Required headers (RFC 7231)
    assert "Date" in headers
    assert validate_http_date(headers["Date"])
    assert "Content-Length" in headers or "Transfer-Encoding" in headers
    assert "Server" in headers

@pytest.mark.asyncio
async def test_response_line_format(compliance_server):
    """Test HTTP response line format."""
    server, port = await compliance_server
    
    async def validate_status_line(status: int, description: str):
        @server.route(f"/status/{status}")
        async def handler(request):
            return ResponseTemplate(status=status)
            
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://localhost:{port}/status/{status}") as response:
                assert response.status == status
                assert response.reason == HTTPStatus(status).phrase
    
    # Test various status codes
    test_cases = [
        (200, "OK"),
        (201, "Created"),
        (404, "Not Found"),
        (500, "Internal Server Error")
    ]
    
    for status, description in test_cases:
        await validate_status_line(status, description)

@pytest.mark.asyncio
async def test_header_format(compliance_server):
    """Test HTTP header format compliance."""
    server, port = await compliance_server
    
    @server.route("/headers")
    async def handler(request):
        return ResponseTemplate(
            headers={
                "X-Test": "value",
                "X-Multiple": "value1, value2",
                "X-Special": "!#$%&'*+-.^_`|~"  # Valid header chars
            }
        )
    
    headers = await get_response_headers(f"http://localhost:{port}/headers")
    
    # Validate header format (RFC 7230)
    header_pattern = re.compile(r"^[\x21-\x7E]+$")  # Printable ASCII
    for name, value in headers.items():
        # Check header name format
        assert ":" not in name
        assert " " not in name
        assert header_pattern.match(name)
        
        # Check header value format
        assert "\n" not in value
        assert "\r" not in value

@pytest.mark.asyncio
async def test_content_length_accuracy(compliance_server):
    """Test Content-Length header accuracy."""
    server, port = await compliance_server
    
    test_bodies = [
        b"",
        b"Hello",
        b"X" * 1000,
        "Hello 世界".encode()  # Unicode test
    ]
    
    for i, body in enumerate(test_bodies):
        @server.route(f"/body/{i}")
        async def handler(request):
            return ResponseTemplate(body=body)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://localhost:{port}/body/{i}") as response:
                content = await response.read()
                assert len(content) == int(response.headers["Content-Length"])
                assert content == body

@pytest.mark.asyncio
async def test_transfer_encoding(compliance_server):
    """Test Transfer-Encoding handling."""
    server, port = await compliance_server
    large_data = b"X" * 100000
    
    @server.route("/chunked")
    async def handler(request):
        return ResponseTemplate(
            body=large_data,
            headers={"Transfer-Encoding": "chunked", "Content-Length": None},
            chunk_size=8192
        )
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:{port}/chunked") as response:
            assert "Transfer-Encoding" in response.headers
            content = await response.read()
            assert len(content) == len(large_data)

@pytest.mark.asyncio
async def test_method_handling(compliance_server):
    """Test HTTP method handling compliance."""
    server, port = await compliance_server
    
    # Register test endpoint
    @server.route("/test")
    async def handler(request):
        return ResponseTemplate(
            status=200 if request["method"] in ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"] else 405
        )
    methods = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]
    
    async with aiohttp.ClientSession() as session:
        for method in methods:
            async with session.request(method, f"http://localhost:{port}/test") as response:
                assert response.status in (200, 405)
                if method == "HEAD":
                    assert len(await response.read()) == 0

@pytest.mark.asyncio
async def test_connection_header(compliance_server):
    """Test Connection header handling."""
    server, port = await compliance_server
    
    # Register test endpoint with connection header handling
    @server.route("/test")
    async def handler(request):
        return ResponseTemplate(
            headers={"Connection": request["headers"].get("connection", "keep-alive")}
        )
    
    # Test keep-alive
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"http://localhost:{port}/test",
            headers={"Connection": "keep-alive"}
        ) as response:
            assert response.headers.get("Connection", "").lower() == "keep-alive"
    
    # Test close
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"http://localhost:{port}/test",
            headers={"Connection": "close"}
        ) as response:
            assert response.headers.get("Connection", "").lower() == "close"

@pytest.mark.asyncio
async def test_host_header(compliance_server):
    """Test Host header handling."""
    server, port = await compliance_server
    
    @server.route("/host")
    async def handler(request):
        return ResponseTemplate(
            body=json.dumps({
                "host": request["headers"].get("host")
            }).encode()
        )
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:{port}/host") as response:
            data = await response.json()
            assert "host" in data
            assert data["host"] == f"localhost:{port}"

@pytest.mark.asyncio
async def test_malformed_requests(compliance_server):
    """Test handling of malformed requests."""
    server, port = await compliance_server
    
    async def send_raw(data: bytes) -> int:
        reader, writer = await asyncio.open_connection("localhost", port)
        writer.write(data)
        await writer.drain()
        
        try:
            response = await reader.readuntil(b"\r\n\r\n")
            status_line = response.split(b"\r\n")[0].decode()
            status_code = int(status_line.split(" ")[1])
            return status_code
        finally:
            writer.close()
            await writer.wait_closed()
    
    # Test cases for malformed requests
    test_cases = [
        (b"INVALID / HTTP/1.1\r\n\r\n", 400),  # Invalid method
        (b"GET /test HTTP/9.9\r\n\r\n", 400),  # Invalid HTTP version
        (b"GET /test\r\n\r\n", 400),           # Missing HTTP version
        (b"GET /test HTTP/1.1\n", 400),        # Missing CRLF
        (b"GET /test HTTP/1.1\r\nBad Header\r\n\r\n", 400),  # Invalid header
        (b"GET /test HTTP/1.1\r\n" * 100 + b"\r\n", 431),  # Too many headers
        (b"\x00GET /test HTTP/1.1\r\n\r\n", 400),  # Invalid characters
        (b"GET /../../../etc/passwd HTTP/1.1\r\n\r\n", 400)  # Path traversal
    ]
    
    for request, expected_status in test_cases:
        status = await send_raw(request)
        assert status == expected_status

@pytest.mark.asyncio
async def test_max_header_size(compliance_server):
    """Test handling of oversized headers."""
    server, port = await compliance_server
    
    @server.route("/test")
    async def handler(request):
        return ResponseTemplate()  # Simple response for header size testing
    
    # Create request with very large header
    large_header = "X-Large: " + "X" * 8192
    
    async with aiohttp.ClientSession() as session:
        with pytest.raises(aiohttp.ClientError):
            await session.get(
                f"http://localhost:{port}/test",
                headers={"X-Large": "X" * 8192}
            )

@pytest.mark.asyncio
async def test_invalid_certificate_handling(compliance_server, tmp_path):
    """Test handling of invalid SSL certificates."""
    server, port = await compliance_server

    # Register test endpoint for certificate validation
    @server.route("/test")
    async def handler(request):
        return ResponseTemplate(
            body=b"secure content",
            headers={
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "Content-Security-Policy": "upgrade-insecure-requests"
            }
        )
    
    # Test cases for invalid certificates
    test_cases = [
        ("expired_cert.pem", ssl.SSLError),     # Expired certificate
        ("wrong_host.pem", ssl.SSLCertVerificationError),  # Wrong hostname
        ("self_signed.pem", ssl.SSLCertVerificationError), # Self-signed cert
        ("corrupted.pem", ssl.SSLError)         # Corrupted certificate
    ]
    
    for cert_file, expected_error in test_cases:
        # Create invalid cert
        cert_path = tmp_path / cert_file
        with open(cert_path, 'w') as f:
            f.write("INVALID CERTIFICATE")
        
        # Attempt connection with invalid cert
        ssl_context = ssl.create_default_context()
        ssl_context.load_verify_locations(cafile=str(cert_path))
        
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            with pytest.raises(expected_error):
                await session.get(f"https://localhost:{port}/test")

@pytest.mark.asyncio
async def test_protocol_violations(compliance_server):
    """Test handling of HTTP protocol violations."""
    server, port = await compliance_server
    
    # Register test endpoint for handling violations
    @server.route("/test")
    async def handler(request):
        return ResponseTemplate(
            headers={"Connection": "close"}  # Force connection close after violation
        )
    
    async def send_violation(data: bytes) -> int:
        reader, writer = await asyncio.open_connection("localhost", port)
        
        try:
            # Send initial valid request
            writer.write(b"GET /test HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            
            # Wait for response
            await reader.readuntil(b"\r\n\r\n")
            
            # Send violation data
            writer.write(data)
            await writer.drain()
            
            try:
                response = await reader.readuntil(b"\r\n\r\n")
                return int(response.split(b"\r\n")[0].split()[1])
            except asyncio.IncompleteReadError:
                return 0  # Connection closed
                
        finally:
            writer.close()
            await writer.wait_closed()
    
    violations = [
        (b"GET / HTTP/1.0\r\n" * 10, 400),  # Pipelined requests in HTTP/1.0
        (b"\x00\x01\x02\x03", 400),         # Binary garbage
        (b"GET / HTTP/1.1\n" * 100, 431),   # Request flooding
        (b"POST / HTTP/1.1\r\n" * 50, 400)  # Multiple simultaneous POSTs
    ]
    
    for violation, expected_status in violations:
        status = await send_violation(violation)
        assert status in (expected_status, 0)  # 0 means connection was closed

@pytest.mark.asyncio
async def test_rate_limiting(compliance_server):
    """Test rate limiting and anti-DoS protections."""
    server, port = await compliance_server

    # Configure server with stricter rate limits for testing
    server.config.rate_limit = 10  # requests per second
    server.config.burst_limit = 20  # max burst size
    
    async def make_request(session):
        try:
            async with session.get(f"http://localhost:{port}/test") as response:
                return response.status
        except aiohttp.ClientError as e:
            return 429  # Assume rate limit error
    
    # Send requests in bursts
    async with aiohttp.ClientSession() as session:
        # First burst - should mostly succeed
        initial_results = await asyncio.gather(
            *[make_request(session) for _ in range(10)],
            return_exceptions=True
        )
        
        # Immediate second burst - should hit rate limits
        await asyncio.sleep(0.1)  # Small delay to ensure rate tracking
        burst_results = await asyncio.gather(
            *[make_request(session) for _ in range(20)],
            return_exceptions=True
        )
    
    # Verify rate limiting behavior
    success_count = sum(1 for r in initial_results if r == 200)
    limited_count = sum(1 for r in burst_results if r == 429)
    
    assert success_count > 0, "Some initial requests should succeed"
    assert limited_count > 0, "Some burst requests should be rate limited"
    
    # Verify rate limit headers if present
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:{port}/test") as response:
            headers = response.headers
            assert any(h in headers for h in (
                'X-RateLimit-Limit',
                'X-RateLimit-Remaining',
                'X-RateLimit-Reset',
                'Retry-After'
            )), "Rate limit headers should be present"

@pytest.mark.asyncio
async def test_protocol_downgrade_prevention(compliance_server):
    """Test prevention of protocol downgrade attacks."""
    server, port = await compliance_server
    
    # Register endpoint that blocks downgrade attacks
    @server.route("/test")
    async def handler(request):
        # Reject HTTP/1.0 requests trying to use newer features
        if request["http_version"] == "1.0" and any(
            h in request["headers"] for h in ("upgrade", "http2-settings")
        ):
            return ResponseTemplate(
                status=400,
                headers={"Connection": "close"}
            )
        return ResponseTemplate(status=200)
    
    # Test HTTP/1.0 downgrade attempt with HTTP/1.1 features
    reader, writer = await asyncio.open_connection("localhost", port)
    try:
        request = (
            b"GET /test HTTP/1.0\r\n"
            b"Host: localhost\r\n"
            b"Connection: keep-alive\r\n"
            b"Upgrade: h2c\r\n"
            b"HTTP2-Settings: AAMAAABkAAQAAP__\r\n\r\n"
        )
        writer.write(request)
        await writer.drain()
        
        response = await reader.readuntil(b"\r\n\r\n")
        status_line = response.split(b"\r\n")[0].decode()
        status_code = int(status_line.split(" ")[1])
        headers = {}
        for header_line in response.split(b"\r\n")[1:-2]:  # Skip status line and empty line
            if header_line:
                name, value = header_line.decode().split(": ", 1)
                headers[name.lower()] = value
        
        assert status_code == 400
        assert headers.get("connection") == "close"
    finally:
        writer.close()
        await writer.wait_closed()
    
    # Verify normal HTTP/1.1 request works
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:{port}/test") as response:
            assert response.status == 200

if __name__ == "__main__":
    pytest.main([__file__])
