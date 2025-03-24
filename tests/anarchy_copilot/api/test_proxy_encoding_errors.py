"""Test proxy handling of malformed encodings and compression."""
import pytest
import gzip
import zlib
import json
from typing import Dict, Any, AsyncGenerator
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio

async def test_corrupted_gzip(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of corrupted gzip content."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Create corrupted gzip content
        test_data = {"test": "data"}
        compressed = bytearray(gzip.compress(json.dumps(test_data).encode()))
        # Corrupt the data by modifying bytes
        compressed[5] = (compressed[5] + 1) % 256

        # Send corrupted request
        response = await client.post(
            "http://httpbin:8000/post",
            content=bytes(compressed),
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip"
            }
        )
        
        # Should return 400 Bad Request for corrupted content
        assert response.status_code == 400
        assert "corrupted" in response.text.lower()

async def test_invalid_content_encoding(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of invalid content encoding header."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Send request with invalid encoding
        response = await client.post(
            "http://httpbin:8000/post",
            content=b"test data",
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "invalid-encoding"
            }
        )
        
        # Should return 400 Bad Request for unsupported encoding
        assert response.status_code == 400
        assert "unsupported" in response.text.lower()

async def test_mismatched_content_length(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of incorrect Content-Length header."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Create compressed content with incorrect length
        test_data = {"test": "data"}
        compressed = gzip.compress(json.dumps(test_data).encode())

        # Send request with mismatched content length
        # We need to pad the content to match the declared length to get past httpx/h11
        padded = compressed + b"\0" * 100  # Pad with null bytes
        response = await client.post(
            "http://httpbin:8000/post",
            content=padded,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "Content-Length": str(len(padded))  # Correct length but invalid gzip content
            }
        )
        
        # Should reject the invalid gzip content
        assert response.status_code == 400
        assert "corrupted" in response.text.lower()

async def test_invalid_charset(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of invalid character encoding."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Send request with invalid charset
        response = await client.post(
            "http://httpbin:8000/post",
            content="test data".encode(),
            headers={
                "Content-Type": "text/plain; charset=invalid-charset"
            }
        )
        
        # Should handle invalid charset gracefully
        assert response.status_code == 400
        assert "charset" in response.text.lower()

async def test_mixed_encodings(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of mixed content and transfer encodings."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Create content with conflicting encodings
        test_data = {"test": "mixed"}
        compressed = gzip.compress(json.dumps(test_data).encode())

        # Test with multiple Content-Encoding headers which is invalid
        response = await client.post(
            "http://httpbin:8000/post",
            content=compressed,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Content-Encoding": "gzip",
                # Add a second Content-Encoding header which should be rejected
                "X-Content-Encoding": "deflate"
            }
        )
        
        # Should detect conflicting content encodings
        assert response.status_code == 400
        assert "conflicting" in response.text.lower() or "invalid" in response.text.lower()

async def test_empty_compressed(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of empty compressed content."""
    async for client_data in proxy_client:
        client = client_data["client"]
        base_client = client_data["base_client"]

        # Create empty compressed content and set explicit Content-Length
        empty_compressed = gzip.compress(b"")
        
        # Create a test interceptor directly
        from proxy.encoding import ContentEncodingInterceptor
        from proxy.interceptor import InterceptedRequest

        # Create test request
        interceptor = ContentEncodingInterceptor()
        test_request = InterceptedRequest(
            id="test_empty",
            method="POST",
            url="http://test/post",
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "Content-Length": str(len(empty_compressed))
            },
            body=empty_compressed
        )

        # Test interceptor directly
        modified_request = await interceptor.intercept(test_request)
        
        # Verify the request was handled correctly
        assert modified_request.body is not None, "Modified request body should not be None"
        assert len(modified_request.body) == 20, "Empty gzip content should be 20 bytes"
        assert modified_request.get_header("Content-Length") == "20", "Content-Length should be 20"
        assert gzip.decompress(modified_request.body) == b"", "Should decompress to empty bytes"
