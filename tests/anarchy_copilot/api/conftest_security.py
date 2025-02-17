"""Test fixtures for security testing."""
import asyncio
from typing import AsyncGenerator, Dict, Any
import pytest
from httpx import AsyncClient, AsyncHTTPTransport, Limits, Timeout
from fastapi import FastAPI

@pytest.fixture
async def secure_proxy_config(app: FastAPI) -> AsyncGenerator[Dict[str, Any], None]:
    """Create and configure a secure proxy instance."""
    transport = AsyncHTTPTransport(
        verify=False,
        retries=1,
        limits=Limits(
            max_connections=5,
            max_keepalive_connections=5,
            keepalive_expiry=5
        )
    )

    # Create client with secure timeouts
    async with AsyncClient(
        transport=transport,
        timeout=Timeout(10.0, connect=5.0),
        follow_redirects=True,
        max_redirects=5,
        base_url="http://testserver",
        verify=False
    ) as client:
        # Start proxy with secure defaults
        response = await client.post("/api/proxy/start", json={
            "host": "127.0.0.1",
            "port": 8080,  # Use secure port
            "interceptRequests": True,
            "interceptResponses": True,
            "allowedHosts": ["httpbin.org"],
            "excludedHosts": ["malicious.com", "evil.com"],
        })
        assert response.status_code == 201

        try:
            # Create proxy transport
            proxy_transport = AsyncHTTPTransport(
                proxy="http://localhost:8080",
                verify=False,
                retries=1,
                limits=Limits(max_connections=5, max_keepalive_connections=5)
            )

            # Create proxy client
            async with AsyncClient(
                transport=proxy_transport,
                base_url="http://httpbin.org",
                follow_redirects=True,
                verify=False,
                timeout=Timeout(10.0, connect=5.0)
            ) as proxy_client:
                yield {
                    "client": proxy_client,
                    "base_client": client,
                    "config": {
                        "max_header_size": 8192,  # 8KB header limit
                        "max_request_size": 1048576,  # 1MB request limit
                        "max_redirects": 5,
                        "validate_content_type": True,
                        "strip_unsafe_headers": True,
                        "security_headers": {
                            "X-Content-Type-Options": "nosniff",
                            "X-Frame-Options": "DENY",
                            "X-XSS-Protection": "1; mode=block",
                            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                            "Content-Security-Policy": "default-src 'self'",
                            "Referrer-Policy": "strict-origin-when-cross-origin"
                        }
                    }
                }
        finally:
            await client.post("/api/proxy/stop")
            await asyncio.sleep(0.1)  # Ensure cleanup completes

@pytest.fixture
def security_headers() -> Dict[str, str]:
    """Get secure default headers."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }

@pytest.fixture
def attack_payloads() -> Dict[str, bytes]:
    """Get test attack payloads."""
    return {
        "xss": b"<script>alert(1)</script>",
        "sqli": b"' OR '1'='1",
        "nullbyte": b"testdata",
        "overflow": b"A" * 65536,  # 64KB
        "format": b"%s%s%s%s%s",
        "shell": b";cat /etc/passwd",
        "path": b"../../../etc/passwd",
        "unicode": "サーバー".encode()
    }

@pytest.fixture
def malformed_headers() -> Dict[str, str]:
    """Get malformed header test cases."""
    return {
        "invalid_chars": "testtest",
        "oversized": "X" * 16384,
        "duplicate": "value1, value1",
        "malformed_encoding": "UTF-7",
        "unicode_escape": "",
        "injection": "test\r\nX-Injected: value"
    }
