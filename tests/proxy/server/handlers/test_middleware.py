"""Tests for proxy middleware and response classes."""
import pytest
from proxy.server.handlers.middleware import ProxyResponse, proxy_middleware

@pytest.fixture
def sample_request():
    """Create a sample request dictionary."""
    return {
        "method": "GET",
        "url": "https://example.com/test",
        "headers": {
            "Host": "example.com",
            "User-Agent": "Test Client",
            "Accept": "*/*"
        },
        "body": b"test body",
        "connection_id": "test-conn-001"
    }

@pytest.fixture
def sample_response():
    """Create a sample response object."""
    return ProxyResponse(
        status_code=200,
        headers={"Content-Type": "text/plain"},
        body=b"Hello World"
    )

def test_proxy_response_initialization():
    """Test ProxyResponse object initialization."""
    # Test with minimal parameters
    response = ProxyResponse(
        status_code=200,
        headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 200
    assert response.headers == {"Content-Type": "text/plain"}
    assert response.body is None
    assert not response.modified
    assert response.intercept_enabled
    assert response.tags == []

    # Test with all parameters
    response = ProxyResponse(
        status_code=404,
        headers={"Content-Type": "application/json"},
        body=b'{"error": "not found"}',
        modified=True,
        intercept_enabled=False,
        tags=["error", "json"]
    )
    assert response.status_code == 404
    assert response.headers == {"Content-Type": "application/json"}
    assert response.body == b'{"error": "not found"}'
    assert response.modified
    assert not response.intercept_enabled
    assert response.tags == ["error", "json"]

def test_proxy_response_tags_default():
    """Test ProxyResponse tags default initialization."""
    response = ProxyResponse(
        status_code=200,
        headers={}
    )
    assert isinstance(response.tags, list)
    assert len(response.tags) == 0

    # Verify tags can be appended
    response.tags.append("test")
    assert "test" in response.tags

def test_proxy_response_immutability():
    """Test ProxyResponse fields cannot be modified after creation."""
    response = ProxyResponse(
        status_code=200,
        headers={"Content-Type": "text/plain"},
        body=b"test"
    )
    
    # Headers should be a new dictionary
    original_headers = response.headers.copy()
    response.headers["New-Header"] = "value"
    assert response.headers == original_headers

@pytest.mark.asyncio
async def test_default_proxy_middleware(sample_request):
    """Test default proxy middleware behavior."""
    # Default middleware should return None (pass-through)
    result = await proxy_middleware(sample_request)
    assert result is None

@pytest.mark.asyncio
async def test_proxy_middleware_with_custom_response():
    """Test proxy middleware with modified request."""
    # Create custom middleware function
    async def custom_middleware(request: dict) -> ProxyResponse:
        if request["url"].endswith("/test"):
            return ProxyResponse(
                status_code=200,
                headers={"Content-Type": "text/plain"},
                body=b"Modified response",
                modified=True,
                tags=["modified"]
            )
        return None

    # Test with matching request
    request = {"url": "https://example.com/test"}
    response = await custom_middleware(request)
    assert response is not None
    assert response.modified
    assert "modified" in response.tags
    assert response.body == b"Modified response"

    # Test with non-matching request
    request = {"url": "https://example.com/other"}
    response = await custom_middleware(request)
    assert response is None

def test_proxy_response_header_case_sensitivity():
    """Test header case sensitivity handling."""
    response = ProxyResponse(
        status_code=200,
        headers={
            "Content-Type": "text/plain",
            "X-Custom-Header": "value"
        }
    )
    
    # Headers should preserve their case
    assert "Content-Type" in response.headers
    assert "content-type" not in response.headers
    assert response.headers["X-Custom-Header"] == "value"

def test_proxy_response_with_empty_body():
    """Test response with empty body handling."""
    # Test with None body
    response = ProxyResponse(
        status_code=204,
        headers={}
    )
    assert response.body is None

    # Test with empty bytes
    response = ProxyResponse(
        status_code=204,
        headers={},
        body=b""
    )
    assert response.body == b""

def test_proxy_response_with_binary_body():
    """Test response with binary body."""
    binary_data = bytes(range(256))
    response = ProxyResponse(
        status_code=200,
        headers={"Content-Type": "application/octet-stream"},
        body=binary_data
    )
    assert response.body == binary_data
    assert len(response.body) == 256

def test_proxy_response_equality():
    """Test ProxyResponse equality comparison."""
    response1 = ProxyResponse(
        status_code=200,
        headers={"Content-Type": "text/plain"},
        body=b"test"
    )
    response2 = ProxyResponse(
        status_code=200,
        headers={"Content-Type": "text/plain"},
        body=b"test"
    )
    response3 = ProxyResponse(
        status_code=404,
        headers={"Content-Type": "text/plain"},
        body=b"test"
    )

    assert response1 == response2
    assert response1 != response3
    assert hash(str(response1)) != hash(str(response3))

@pytest.mark.asyncio
async def test_proxy_middleware_chain():
    """Test chaining multiple middleware functions."""
    async def middleware1(request: dict) -> Optional[ProxyResponse]:
        request["modified_by"] = "middleware1"
        return None

    async def middleware2(request: dict) -> Optional[ProxyResponse]:
        if "modified_by" in request:
            return ProxyResponse(
                status_code=200,
                headers={},
                body=f"Modified by {request['modified_by']}".encode(),
                modified=True
            )
        return None

    # Test middleware chain
    request = {"url": "https://example.com"}
    response = await middleware1(request)
    assert response is None
    
    response = await middleware2(request)
    assert response is not None
    assert response.body == b"Modified by middleware1"
