"""Test proxy handling of different content encodings."""
import pytest
import gzip
import zlib
import brotli
import json
from typing import Dict, Any, AsyncGenerator
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio

async def test_gzip_compression(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of gzip compressed content."""
    async for client_data in proxy_client:
        client = client_data["client"]
        base_client = client_data["base_client"]

        # Create compressed content
        test_data = {"test": "data"}
        compressed = gzip.compress(json.dumps(test_data).encode())

        # Send compressed request
        response = await client.post(
            "http://httpbin.org/post",
            content=compressed,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip"
            }
        )
        assert response.status_code == 200
        
        request_id = response.headers.get("X-Intercepted-ID")
        assert request_id is not None

        # Verify the proxy decompressed the content
        history_response = await base_client.get(f"/api/proxy/history/{request_id}")
        assert history_response.status_code == 200
        history = history_response.json()
        
        # The stored request body should be decompressed
        assert json.loads(history["request"]["body"]) == test_data

async def test_deflate_compression(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of deflate compressed content."""
    async for client_data in proxy_client:
        client = client_data["client"]
        base_client = client_data["base_client"]

        # Create deflated content
        test_data = {"test": "deflate"}
        compressed = zlib.compress(json.dumps(test_data).encode())

        # Send compressed request
        response = await client.post(
            "http://httpbin.org/post",
            content=compressed,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "deflate"
            }
        )
        assert response.status_code == 200
        
        request_id = response.headers.get("X-Intercepted-ID")
        assert request_id is not None

        # Verify decompression
        history_response = await base_client.get(f"/api/proxy/history/{request_id}")
        assert history_response.status_code == 200
        history = history_response.json()
        assert json.loads(history["request"]["body"]) == test_data

async def test_brotli_compression(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of brotli compressed content."""
    async for client_data in proxy_client:
        client = client_data["client"]
        base_client = client_data["base_client"]

        # Create brotli compressed content
        test_data = {"test": "brotli"}
        compressed = brotli.compress(json.dumps(test_data).encode())

        # Send compressed request
        response = await client.post(
            "http://httpbin.org/post",
            content=compressed,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "br"
            }
        )
        assert response.status_code == 200
        
        request_id = response.headers.get("X-Intercepted-ID")
        assert request_id is not None

        # Verify decompression
        history_response = await base_client.get(f"/api/proxy/history/{request_id}")
        assert history_response.status_code == 200
        history = history_response.json()
        assert json.loads(history["request"]["body"]) == test_data

async def test_multiple_encodings(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of multiple content encodings."""
    async for client_data in proxy_client:
        client = client_data["client"]
        base_client = client_data["base_client"]

        # Create content with multiple encodings (gzip + deflate)
        test_data = {"test": "multiple"}
        compressed = zlib.compress(gzip.compress(json.dumps(test_data).encode()))

        # Send compressed request
        response = await client.post(
            "http://httpbin.org/post",
            content=compressed,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip, deflate"
            }
        )
        assert response.status_code == 200
        
        request_id = response.headers.get("X-Intercepted-ID")
        assert request_id is not None

        # Verify decompression
        history_response = await base_client.get(f"/api/proxy/history/{request_id}")
        assert history_response.status_code == 200
        history = history_response.json()
        assert json.loads(history["request"]["body"]) == test_data

async def test_charset_encoding(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of different character encodings."""
    async for client_data in proxy_client:
        client = client_data["client"]
        base_client = client_data["base_client"]

        # Test with UTF-16 encoded content
        test_data = "测试数据"  # Test data in Chinese
        encoded = test_data.encode('utf-16')

        # Send encoded request
        response = await client.post(
            "http://httpbin.org/post",
            content=encoded,
            headers={
                "Content-Type": "text/plain; charset=utf-16"
            }
        )
        assert response.status_code == 200
        
        request_id = response.headers.get("X-Intercepted-ID")
        assert request_id is not None

        # Verify encoding handling
        history_response = await base_client.get(f"/api/proxy/history/{request_id}")
        assert history_response.status_code == 200
        history = history_response.json()
        
        # The body should be properly decoded
        decoded = history["request"]["body"].encode('utf-8').decode('utf-8')
        assert decoded == test_data
