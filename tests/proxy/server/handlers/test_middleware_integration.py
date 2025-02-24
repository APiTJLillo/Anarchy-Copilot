"""Integration tests for middleware with HTTP handler and proxy server."""
import pytest
import asyncio
import aiohttp
from typing import Optional, Dict, Any

from proxy.server.handlers.middleware import ProxyResponse, proxy_middleware
from proxy.server.handlers.http import HttpRequestHandler
from proxy.server.https_intercept_protocol import HttpsInterceptProtocol
from .test_middleware import sample_request, sample_response

class TestMiddleware:
    """Test middleware that modifies requests/responses."""
    async def __call__(self, request: Dict[str, Any]) -> Optional[ProxyResponse]:
        if request["url"].endswith("/modify"):
            return ProxyResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body=b'{"modified": true}',
                modified=True,
                tags=["test-middleware"]
            )
        return None

@pytest.fixture
async def http_handler_with_middleware(connection_id):
    """Create HTTP handler with test middleware."""
    handler = HttpRequestHandler(connection_id)
    handler.register_middleware(TestMiddleware())
    return handler

@pytest.fixture
async def proxy_server(unused_tcp_port, ca_handler):
    """Create and configure a test proxy server."""
    # Create protocol factory with middleware
    factory = HttpsInterceptProtocol.create_protocol_factory(ca=ca_handler)
    
    # Add test middleware to protocol
    test_middleware = TestMiddleware()
    factory.register_middleware(test_middleware)
    
    # Start server
    server = await asyncio.get_event_loop().create_server(
        factory,
        '127.0.0.1',
        unused_tcp_port
    )
    
    async with server:
        yield ('127.0.0.1', unused_tcp_port)

@pytest.mark.asyncio
async def test_middleware_with_http_handler(http_handler_with_middleware, sample_request):
    """Test middleware integration with HTTP handler."""
    # Test unmodified request
    request = sample_request.copy()
    request["url"] = "https://example.com/normal"
    response = await http_handler_with_middleware.handle_client_data(
        b"GET /normal HTTP/1.1\r\nHost: example.com\r\n\r\n"
    )
    assert response is not None
    assert b"HTTP/1.1" in response
    
    # Test modified request
    request["url"] = "https://example.com/modify"
    response = await http_handler_with_middleware.handle_client_data(
        b"GET /modify HTTP/1.1\r\nHost: example.com\r\n\r\n"
    )
    assert response is not None
    assert b'{"modified": true}' in response
    assert b"Content-Type: application/json" in response

@pytest.mark.asyncio
async def test_middleware_chain_with_handler(http_handler_with_middleware):
    """Test multiple middleware in chain with HTTP handler."""
    async def middleware1(request):
        request["modified_by"] = ["middleware1"]
        return None
    
    async def middleware2(request):
        if "modified_by" in request:
            request["modified_by"].append("middleware2")
            return ProxyResponse(
                status_code=200,
                headers={"X-Modified-By": ",".join(request["modified_by"])},
                body=b"Modified response",
                modified=True
            )
        return None
    
    # Register middlewares
    handler = http_handler_with_middleware
    handler.register_middleware(middleware1)
    handler.register_middleware(middleware2)
    
    # Test request through chain
    response = await handler.handle_client_data(
        b"GET /test HTTP/1.1\r\nHost: example.com\r\n\r\n"
    )
    assert response is not None
    assert b"X-Modified-By: middleware1,middleware2" in response

@pytest.mark.asyncio
async def test_middleware_with_proxy_server(proxy_server):
    """Test middleware integration with full proxy server."""
    host, port = proxy_server
    
    async with aiohttp.ClientSession() as session:
        # Test unmodified request
        async with session.get(
            "http://example.com/normal",
            proxy=f"http://{host}:{port}"
        ) as response:
            assert response.status == 200
        
        # Test modified request
        async with session.get(
            "http://example.com/modify",
            proxy=f"http://{host}:{port}"
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert data["modified"] is True
            assert "test-middleware" in response.headers.get("X-Proxy-Tags", "").split(",")

@pytest.mark.asyncio
async def test_middleware_error_handling(http_handler_with_middleware):
    """Test middleware error handling in HTTP handler."""
    async def error_middleware(request):
        raise ValueError("Test error")
    
    # Register error middleware
    handler = http_handler_with_middleware
    handler.register_middleware(error_middleware)
    
    # Test error handling
    response = await handler.handle_client_data(
        b"GET /test HTTP/1.1\r\nHost: example.com\r\n\r\n"
    )
    assert response is not None
    assert b"500 Internal Server Error" in response

@pytest.mark.asyncio
async def test_middleware_state_persistence(http_handler_with_middleware):
    """Test middleware state persistence across requests."""
    request_count = 0
    
    async def counting_middleware(request):
        nonlocal request_count
        request_count += 1
        return ProxyResponse(
            status_code=200,
            headers={"X-Request-Count": str(request_count)},
            body=b"Test response",
            modified=True
        )
    
    # Register counting middleware
    handler = http_handler_with_middleware
    handler.register_middleware(counting_middleware)
    
    # Send multiple requests
    for i in range(3):
        response = await handler.handle_client_data(
            b"GET /test HTTP/1.1\r\nHost: example.com\r\n\r\n"
        )
        assert f"X-Request-Count: {i+1}".encode() in response

@pytest.mark.asyncio
async def test_middleware_response_modification(http_handler_with_middleware):
    """Test middleware modification of existing responses."""
    async def modifying_middleware(request):
        # Let the request go through
        if "initial_request" not in request:
            request["initial_request"] = True
            return None
        
        # Modify the response
        existing_response = request.get("response")
        if existing_response:
            return ProxyResponse(
                status_code=existing_response.status_code,
                headers={**existing_response.headers, "X-Modified": "true"},
                body=existing_response.body + b" - Modified",
                modified=True
            )
        return None
    
    # Register middleware
    handler = http_handler_with_middleware
    handler.register_middleware(modifying_middleware)
    
    # Test response modification
    response = await handler.handle_client_data(
        b"GET /test HTTP/1.1\r\nHost: example.com\r\n\r\n"
    )
    assert response is not None
    assert b"X-Modified: true" in response
    assert b"Modified" in response
