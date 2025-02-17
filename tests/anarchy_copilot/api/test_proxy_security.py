"""Test proxy security handling."""
import pytest
from typing import Dict, Any, AsyncGenerator
from httpx import Timeout
import json
import os

pytestmark = pytest.mark.asyncio

async def test_null_byte_injection(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of null byte injection attempts."""
    async for client_data in proxy_client:
        client = client_data["client"]
        
        # Try to send request with null bytes in headers and body
        null_data = b"testdata"  # Add null byte in middle of data
        null_header = "testinjection"  # Add null byte in middle of header

        response = await client.post(
            "/post",
            content=null_data,
            headers={
                "Content-Type": "text/plain",
                "X-Test": null_header
            }
        )
        
        # Should detect and reject null bytes
        assert response.status_code in (400, 422)  # Accept either status code
        assert any(word in response.text.lower() for word in ["invalid", "null", "illegal"])

async def test_large_header_injection(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of oversized headers."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Create very large header
        large_header = "X" * 32768  # 32KB header
        response = await client.get(
            "/get",
            headers={"X-Large": large_header}
        )
        
        # Should reject oversized headers
        assert response.status_code in (400, 413)  # Accept either status code
        assert any(word in response.text.lower() for word in ["large", "too long", "header"])

async def test_protocol_downgrade(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test prevention of protocol downgrade attacks."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Try to downgrade HTTPS to HTTP
        response = await client.get(
            "/get",  # Test with simpler path
            headers={
                "Upgrade-Insecure-Requests": "1",
                "X-Forwarded-Proto": "http"
            }
        )
        
        # Should reject downgrade attempt
        assert response.status_code in (400, 403)
        assert any(word in response.text.lower() for word in ["invalid", "insecure", "downgrade"])

async def test_host_header_injection(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test prevention of Host header injection."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Try host header injection
        response = await client.get(
            "/get",
            headers={"Host": "malicious.com"}
        )
        
        # Should reject request to unauthorized host
        assert response.status_code == 403
        assert "not in scope" in response.text.lower()

async def test_directory_traversal(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test prevention of directory traversal attempts."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Try different path traversal attempts
        traversal_paths = [
            "/../../../etc/passwd",
            "/..%2f..%2f..%2fetc/passwd",
            "/%2e%2e/%2e%2e/etc/passwd",
            f"/{os.path.pardir}/{os.path.pardir}/etc/passwd"
        ]

        for path in traversal_paths:
            response = await client.get(path)
            
            # Should reject path traversal
            assert response.status_code in (400, 403), f"Failed for path: {path}"
            assert any(word in response.text.lower() for word in ["invalid", "path", "traversal"])

async def test_request_smuggling(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test prevention of HTTP request smuggling."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Test various smuggling attempts
        smuggling_headers = [
            {
                "Content-Length": "4",
                "Transfer-Encoding": "chunked"
            },
            {
                "Content-Length": ["10", "20"]  # Multiple Content-Length headers
            },
            {
                "Transfer-Encoding": ["chunked", "identity"]  # Multiple Transfer-Encoding headers
            }
        ]
        
        for headers in smuggling_headers:
            response = await client.post(
                "/post",
                content=b"test",
                headers=headers
            )
            
            # Should reject smuggling attempts
            assert response.status_code == 400, f"Failed for headers: {headers}"
            assert any(word in response.text.lower() for word in ["invalid", "header"])

async def test_circular_redirects(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of circular redirects."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Configure shorter timeout for this test
        client.timeout = Timeout(5.0, connect=5.0, read=5.0, write=5.0, pool=5.0)

        # Test redirect loop
        response = await client.get(
            "/redirect/5",  # Use 5 redirects instead of too many
            follow_redirects=True
        )
        
        # Should detect redirect loop or too many redirects
        assert response.status_code in (400, 429)
        assert any(word in response.text.lower() for word in ["redirect", "loop", "too many"])

async def test_content_type_mismatch(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of content-type/body mismatches."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Test various content-type mismatches
        test_cases = [
            {
                "content": json.dumps({"test": "data"}).encode(),
                "content_type": "text/plain"
            },
            {
                "content": b"plain text",
                "content_type": "application/json"
            },
            {
                "content": b"<html><body>test</body></html>",
                "content_type": "application/xml"
            }
        ]

        for case in test_cases:
            response = await client.post(
                "/post",
                content=case["content"],
                headers={"Content-Type": case["content_type"]}
            )
            
            # Should detect content-type mismatches
            assert response.status_code == 400, f"Failed for content type: {case['content_type']}"
            assert any(word in response.text.lower() for word in ["content", "type", "mismatch", "invalid"])

async def test_security_headers(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test security headers are properly set."""
    async for client_data in proxy_client:
        client = client_data["client"]
        config = client_data["config"]

        response = await client.get("/headers")
        assert response.status_code == 200

        # Verify all security headers are present
        security_headers = config.get("security_headers", {})
        response_headers = {k.lower(): v for k, v in response.headers.items()}
        
        for name, value in security_headers.items():
            header_name = name.lower()
            assert header_name in response_headers, f"Missing header: {name}"
            assert response_headers[header_name] == value, \
                f"Incorrect value for {name}: {response_headers[header_name]} != {value}"
