"""Test proxy resilience against evasion techniques."""
import pytest
from typing import Dict, Any, AsyncGenerator
from httpx import AsyncClient
from urllib.parse import quote
import base64

pytestmark = pytest.mark.asyncio

async def test_url_encoding_evasion(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of URL encoding evasion attempts."""
    async for client_data in proxy_client:
        client = client_data["client"]
        base_client = client_data["base_client"]

        # Test double encoding
        double_encoded = quote(quote("/../../../etc/passwd"))
        response = await client.get(f"http://httpbin.org/{double_encoded}")
        assert response.status_code == 400
        assert "invalid path" in response.text.lower()

        # Test UTF-8 encoding
        utf8_encoded = "/../%C0%AF..%C0%AF"
        response = await client.get(f"http://httpbin.org/{utf8_encoded}")
        assert response.status_code == 400
        assert "invalid path" in response.text.lower()

        # Test unicode normalization
        unicode_encoded = "/test/./././%E2%80%AE/../../etc/passwd"
        response = await client.get(f"http://httpbin.org/{unicode_encoded}")
        assert response.status_code == 400
        assert "invalid path" in response.text.lower()

async def test_protocol_evasion(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of protocol-based evasion attempts."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Test protocol smuggling
        headers = {
            "X-Forwarded-Proto": "https",
            "Front-End-Https": "on",
            "X-Forwarded-Protocol": "https"
        }
        response = await client.get(
            "http://httpbin.org/get",
            headers=headers
        )
        assert response.status_code == 200
        assert "http" in str(response.url.scheme)  # Should not be affected by headers

        # Test scheme mixing
        response = await client.get("https://httpbin.org\\@evil.com/")
        assert response.status_code == 400
        assert "invalid url" in response.text.lower()

async def test_host_evasion(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of host-based evasion attempts."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Test host header injection variants
        evasion_attempts = [
            {"Host": "evil.com\r\nX-Injected: true"},
            {"Host": "allowed.com@evil.com"},
            {"Host": "allowed.com:80@evil.com"},
            {"Host": "allowed.com\tevil.com"},
            {"Host": f"allowed.com{chr(8)}evil.com"},
        ]

        for headers in evasion_attempts:
            response = await client.get(
                "http://httpbin.org/get",
                headers=headers
            )
            assert response.status_code == 400, f"Failed for headers: {headers}"
            assert "invalid host" in response.text.lower()

async def test_header_evasion(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of header-based evasion attempts."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Test header folding variants
        folded_headers = {
            "X-Test": "value1\r\n \tvalue2",
            "X-Test2": "value1\n\tvalue2",
            "X-Test3": f"value1\r\n{chr(9)}value2"
        }

        response = await client.get(
            "http://httpbin.org/get",
            headers=folded_headers
        )
        assert response.status_code == 400
        assert "invalid header" in response.text.lower()

async def test_content_type_evasion(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of content-type evasion attempts."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Test content-type confusion
        evasion_attempts = [
            {"Content-Type": "application/json; charset=utf-7"},
            {"Content-Type": "application/json;charset=utf-7"},
            {"Content-Type": "application/x-www-form-urlencoded\njson"},
            {"Content-Type": f"text/html;charset=utf-8;charset=utf-7"}
        ]

        data = {"test": "data"}
        for headers in evasion_attempts:
            response = await client.post(
                "http://httpbin.org/post",
                json=data,
                headers=headers
            )
            assert response.status_code == 400, f"Failed for headers: {headers}"
            assert "invalid content type" in response.text.lower()

async def test_encoding_evasion(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of encoding-based evasion attempts."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Test encoding confusion
        headers = {
            "Content-Encoding": "gzip, identity",
            "Accept-Encoding": "*;q=1",
            "Transfer-Encoding": "chunked;utf-7"
        }

        response = await client.get(
            "http://httpbin.org/get",
            headers=headers
        )
        assert response.status_code == 400
        assert "invalid encoding" in response.text.lower()

async def test_path_evasion(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of path-based evasion attempts."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Test various path traversal techniques
        evasion_paths = [
            "/....//....//etc/passwd",
            "/.../...//etc/passwd",
            "/test/../../etc/passwd",
            "/test/%2e%2e/%2e%2e/etc/passwd",
            "/test/..%c0%af../etc/passwd",
            "/%c0%ae%c0%ae/etc/passwd"
        ]

        for path in evasion_paths:
            response = await client.get(f"http://httpbin.org{path}")
            assert response.status_code == 400, f"Failed for path: {path}"
            assert "invalid path" in response.text.lower()

async def test_websocket_evasion(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of WebSocket-based evasion attempts."""
    async for client_data in proxy_client:
        client = client_data["client"]
        base_client = client_data["base_client"]

        # Test WebSocket upgrade evasion
        headers = {
            "Connection": "keep-alive, Upgrade",
            "Upgrade": "websocket",
            "Sec-WebSocket-Version": "13",
            "Sec-WebSocket-Key": base64.b64encode(b"dummy").decode()
        }

        # Attempt WebSocket connection to non-WS endpoint
        response = await client.get(
            "http://httpbin.org/get",
            headers=headers
        )
        assert response.status_code == 400
        assert "invalid upgrade" in response.text.lower()
