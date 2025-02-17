"""Test proxy interception functionality."""
import pytest
from typing import Dict, Any, AsyncGenerator
import json
import base64
from http import HTTPStatus

pytestmark = pytest.mark.asyncio

async def test_request_interception(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test request interception and modification."""
    async for client_data in proxy_client:
        client = client_data["client"]
        base_client = client_data["base_client"]

        # Send POST request that should be intercepted
        response = await client.post(
            "http://httpbin.org/post",
            json={"original": "data"}
        )
        
        # Get the intercepted request ID from response headers
        request_id = response.headers.get("X-Intercepted-ID")
        assert request_id is not None

        # Modify the intercepted request
        modified_request = {
            "method": "POST",
            "url": "http://httpbin.org/post",
            "headers": {
                "Content-Type": "application/json",
                "X-Modified": "true"
            },
            "body": json.dumps({"modified": "data"})
        }

        # Send modified request
        modify_response = await base_client.post(
            f"/api/proxy/request/intercept/{request_id}",
            json=modified_request
        )
        assert modify_response.status_code == 200

        # Verify the modification in history
        history_response = await base_client.get(f"/api/proxy/history/{request_id}")
        assert history_response.status_code == 200
        history = history_response.json()
        assert history["request"]["headers"].get("X-Modified") == "true"
        assert json.loads(history["request"]["body"]) == {"modified": "data"}

async def test_response_interception(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test response interception and modification."""
    async for client_data in proxy_client:
        client = client_data["client"]
        base_client = client_data["base_client"]

        # Send GET request
        response = await client.get("http://httpbin.org/get")
        
        # Get the intercepted request ID
        request_id = response.headers.get("X-Intercepted-ID")
        assert request_id is not None

        # Modify the response
        modified_response = {
            "statusCode": HTTPStatus.OK,
            "headers": {
                "Content-Type": "application/json",
                "X-Modified-Response": "true"
            },
            "body": json.dumps({"modified": "response"})
        }

        # Send modified response
        modify_response = await base_client.post(
            f"/api/proxy/response/intercept/{request_id}",
            json=modified_response
        )
        assert modify_response.status_code == 200

        # Verify modification in history
        history_response = await base_client.get(f"/api/proxy/history/{request_id}")
        assert history_response.status_code == 200
        history = history_response.json()
        assert history["response"]["headers"].get("X-Modified-Response") == "true"
        assert json.loads(history["response"]["body"]) == {"modified": "response"}

async def test_binary_content_interception(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test interception of binary content."""
    async for client_data in proxy_client:
        client = client_data["client"]
        base_client = client_data["base_client"]

        # Send binary data
        binary_data = b"BinaryData"
        response = await client.post(
            "http://httpbin.org/post",
            content=binary_data,
            headers={"Content-Type": "application/octet-stream"}
        )
        
        request_id = response.headers.get("X-Intercepted-ID")
        assert request_id is not None

        # Modify binary request
        modified_binary = b"Modified"
        modified_request = {
            "method": "POST",
            "url": "http://httpbin.org/post",
            "headers": {
                "Content-Type": "application/octet-stream",
                "X-Binary-Modified": "true"
            },
            "body": base64.b64encode(modified_binary).decode()
        }

        # Send modified request
        modify_response = await base_client.post(
            f"/api/proxy/request/intercept/{request_id}",
            json=modified_request
        )
        assert modify_response.status_code == 200

        # Verify binary modification
        history_response = await base_client.get(f"/api/proxy/history/{request_id}")
        assert history_response.status_code == 200
        history = history_response.json()
        assert history["request"]["headers"].get("X-Binary-Modified") == "true"
        decoded_body = base64.b64decode(history["request"]["body"].encode())
        assert decoded_body == modified_binary

async def test_streaming_content(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of streaming content."""
    async for client_data in proxy_client:
        client = client_data["client"]
        base_client = client_data["base_client"]

        # Test with streaming response
        async with client.stream("GET", "http://httpbin.org/stream/3") as response:
            request_id = response.headers.get("X-Intercepted-ID")
            assert request_id is not None

            chunks = []
            async for chunk in response.aiter_bytes():
                chunks.append(chunk)

            # Verify streaming response was captured
            history_response = await base_client.get(f"/api/proxy/history/{request_id}")
            assert history_response.status_code == 200
            history = history_response.json()
            
            # The proxy should have assembled the full response
            assert len(history["response"]["body"]) > 0
            assembled_response = json.loads(history["response"]["body"])
            assert isinstance(assembled_response, dict)
