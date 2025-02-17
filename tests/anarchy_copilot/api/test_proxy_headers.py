"""Test proxy HTTP security headers."""
import pytest
from typing import Dict, Any, AsyncGenerator
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio

async def test_security_headers_forwarding(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test that security headers are properly forwarded."""
    async for client_data in proxy_client:
        client = client_data["client"]
        config = client_data["config"]

        # Make request through proxy
        response = await client.get("http://httpbin.org/headers")
        assert response.status_code == 200

        # Verify security headers are present from config
        security_headers = config.get("security_headers", {})
        response_headers = response.headers
        for name, value in security_headers.items():
            assert response_headers.get(name) == value, f"Missing or incorrect header: {name}"

async def test_header_sanitization(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test that dangerous headers are sanitized."""
    async for client_data in proxy_client:
        client = client_data["client"]

        dangerous_headers = {
            "X-Forwarded-For": "evil.com",
            "X-Frame-Options": "ALLOW-ALL",
            "Content-Security-Policy": "default-src *",
            "X-XSS-Protection": "0",
        }

        # Make request with dangerous headers
        response = await client.get(
            "http://httpbin.org/headers",
            headers=dangerous_headers
        )
        assert response.status_code == 200

        # Verify headers were sanitized
        response_headers = response.headers
        assert response_headers.get("X-Frame-Options") == "DENY"
        assert "default-src *" not in response_headers.get("Content-Security-Policy", "")
        assert response_headers.get("X-XSS-Protection") == "1; mode=block"

async def test_invalid_header_values(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of invalid header values."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Attempt header injection
        headers = {
            "X-Custom-Header": "\r\nInjected-Header: value",
            "Authorization": "Basic \r\nX-Injected: value",
            "Cookie": "session=123\r\nX-Injected: value"
        }

        response = await client.get(
            "http://httpbin.org/headers",
            headers=headers
        )
        
        # Should reject invalid headers
        assert response.status_code == 400
        assert "invalid" in response.text.lower()

async def test_header_size_limits(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test header size limits."""
    async for client_data in proxy_client:
        client = client_data["client"]
        large_header = "X" * 16384  # 16KB

        # Attempt request with large header
        response = await client.get(
            "http://httpbin.org/headers",
            headers={"X-Large-Header": large_header}
        )
        
        # Should reject oversized headers
        assert response.status_code == 413
        assert "large" in response.text.lower()

async def test_secure_cookie_handling(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test secure cookie handling."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Make request with cookies
        response = await client.get(
            "http://httpbin.org/cookies/set",
            params={"session": "test123"},
            follow_redirects=True
        )
        assert response.status_code == 200

        # Check for Set-Cookie headers
        cookie_header = response.headers.get("Set-Cookie")
        assert cookie_header is not None, "No Set-Cookie header found"
        
        # Verify cookie attributes
        cookie_attrs = cookie_header.lower()
        assert "secure" in cookie_attrs, "Cookie should be secure"
        assert "samesite=strict" in cookie_attrs, "Cookie should be SameSite=Strict"
        assert "httponly" in cookie_attrs, "Cookie should be HttpOnly"

async def test_cors_headers(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test CORS header handling."""
    async for client_data in proxy_client:
        client = client_data["client"]
        base_client = client_data["base_client"]

        # Test CORS preflight
        response = await client.options(
            "http://httpbin.org/get",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        assert response.status_code == 200

        # Verify CORS headers
        cors_headers = response.headers
        assert cors_headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
        assert "GET" in cors_headers["Access-Control-Allow-Methods"]
        assert "Content-Type" in cors_headers["Access-Control-Allow-Headers"]
        assert int(cors_headers["Access-Control-Max-Age"]) <= 86400

async def test_upgrade_insecure_requests(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test upgrade-insecure-requests handling."""
    async for client_data in proxy_client:
        client = client_data["client"]

        response = await client.get(
            "http://httpbin.org/get",
            headers={"Upgrade-Insecure-Requests": "1"}
        )
        assert response.status_code == 200

        csp_header = response.headers.get("Content-Security-Policy", "")
        assert "upgrade-insecure-requests" in csp_header.lower()

async def test_referrer_policy(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test Referrer-Policy enforcement."""
    async for client_data in proxy_client:
        client = client_data["client"]

        response = await client.get(
            "http://httpbin.org/get",
            headers={"Referer": "http://previous-site.com/page"}
        )
        assert response.status_code == 200

        # Verify strict referrer policy
        assert response.headers.get("Referrer-Policy") == \
            "strict-origin-when-cross-origin"
