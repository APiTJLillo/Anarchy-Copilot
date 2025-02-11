"""
Unit tests for the Anarchy Copilot proxy module.

Tests proxy server functionality including request/response interception,
session management, and certificate handling.
"""
import pytest
import aiohttp
import asyncio
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import Mock, patch

from anarchy_copilot.proxy import (
    ProxyConfig,
    ProxyServer,
    InterceptedRequest,
    InterceptedResponse,
    RequestInterceptor,
    ResponseInterceptor
)

class TestRequestInterceptor(RequestInterceptor):
    """Test interceptor that modifies request headers."""
    
    def __init__(self, headers_to_add: Dict[str, str]):
        self.headers_to_add = headers_to_add
    
    async def intercept(self, request: InterceptedRequest) -> InterceptedRequest:
        """Add test headers to the request."""
        for name, value in self.headers_to_add.items():
            request.set_header(name, value)
        return request

class TestResponseInterceptor(ResponseInterceptor):
    """Test interceptor that modifies response headers."""
    
    def __init__(self, headers_to_add: Dict[str, str]):
        self.headers_to_add = headers_to_add
    
    async def intercept(
        self,
        response: InterceptedResponse,
        request: InterceptedRequest
    ) -> InterceptedResponse:
        """Add test headers to the response."""
        for name, value in self.headers_to_add.items():
            response.set_header(name, value)
        return response

@pytest.fixture
def proxy_config():
    """Create a test proxy configuration."""
    return ProxyConfig(
        host="127.0.0.1",
        port=8081,  # Use different port for tests
        ca_cert_path=Path("./test_ca.crt"),
        ca_key_path=Path("./test_ca.key"),
        allowed_hosts=set(),
        history_size=100
    )

@pytest.fixture
def proxy_server(proxy_config):
    """Create a configured proxy server instance."""
    return ProxyServer(proxy_config)

@pytest.mark.asyncio
async def test_proxy_initialization(proxy_server):
    """Test proxy server initialization."""
    assert proxy_server.config is not None
    assert proxy_server.session is not None
    assert len(proxy_server._request_interceptors) == 0
    assert len(proxy_server._response_interceptors) == 0

@pytest.mark.asyncio
async def test_add_interceptors(proxy_server):
    """Test adding interceptors to the proxy."""
    req_interceptor = TestRequestInterceptor({"X-Test": "test"})
    res_interceptor = TestResponseInterceptor({"X-Response": "test"})
    
    proxy_server.add_request_interceptor(req_interceptor)
    proxy_server.add_response_interceptor(res_interceptor)
    
    assert len(proxy_server._request_interceptors) == 1
    assert len(proxy_server._response_interceptors) == 1

@pytest.mark.asyncio
async def test_request_interception():
    """Test request interception and modification."""
    request = InterceptedRequest(
        method="GET",
        url="http://example.com",
        headers={}
    )
    
    interceptor = TestRequestInterceptor({"X-Test": "test-value"})
    modified_request = await interceptor.intercept(request)
    
    assert modified_request.get_header("X-Test") == "test-value"

@pytest.mark.asyncio
async def test_response_interception():
    """Test response interception and modification."""
    request = InterceptedRequest(
        method="GET",
        url="http://example.com",
        headers={}
    )
    
    response = InterceptedResponse(
        status_code=200,
        headers={}
    )
    
    interceptor = TestResponseInterceptor({"X-Response": "test-value"})
    modified_response = await interceptor.intercept(response, request)
    
    assert modified_response.get_header("X-Response") == "test-value"

@pytest.mark.asyncio
async def test_scope_enforcement(proxy_server):
    """Test that the proxy enforces scope restrictions."""
    # Configure proxy to only allow example.com
    proxy_server.config.allowed_hosts = {"example.com"}
    
    # Create test request for out-of-scope domain
    request = Mock(spec=aiohttp.web.Request)
    request.method = "GET"
    request.url = "http://out-of-scope.com"
    request.host = "out-of-scope.com"
    request.headers = {}
    
    response = await proxy_server._handle_request(request)
    assert response.status == 403

@pytest.mark.asyncio
async def test_history_tracking(proxy_server):
    """Test that the proxy tracks request/response history."""
    request = InterceptedRequest(
        method="GET",
        url="http://example.com",
        headers={}
    )
    
    # Create history entry
    entry = proxy_server.session.create_history_entry(request)
    assert entry in proxy_server.session._pending_requests.values()
    
    # Complete history entry
    response = InterceptedResponse(
        status_code=200,
        headers={}
    )
    proxy_server.session.complete_history_entry(entry.id, response)
    
    # Verify history
    history = proxy_server.session.get_history()
    assert len(history) == 1
    assert history[0].request == request
    assert history[0].response == response

@pytest.mark.asyncio
async def test_certificate_generation(proxy_server):
    """Test SSL/TLS certificate generation."""
    if proxy_server._ca:  # Only run if CA is configured
        cert_bytes, key_bytes = proxy_server._ca.generate_certificate("test.com")
        assert cert_bytes is not None
        assert key_bytes is not None
        
        # Verify certificate contents
        from OpenSSL import crypto
        cert = crypto.load_certificate(crypto.FILETYPE_PEM, cert_bytes)
        assert cert.get_subject().CN == "test.com"

def test_proxy_config_validation():
    """Test proxy configuration validation."""
    # Test invalid port
    with pytest.raises(ValueError):
        ProxyConfig(port=-1)
    
    # Test invalid timeout
    with pytest.raises(ValueError):
        ProxyConfig(connection_timeout=-1)
    
    # Test valid configuration
    config = ProxyConfig(
        port=8080,
        connection_timeout=30,
        allowed_hosts={"example.com"}
    )
    assert config.port == 8080
    assert config.connection_timeout == 30
    assert "example.com" in config.allowed_hosts
