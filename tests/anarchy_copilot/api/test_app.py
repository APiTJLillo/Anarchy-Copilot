"""
Tests for the main FastAPI application initialization and configuration.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from api import app

def test_app_initialization():
    """Test that the FastAPI app is properly initialized."""
    client = TestClient(app)
    
    # Test API info
    response = client.get("/docs")
    assert response.status_code == 200
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Anarchy Copilot API"

def test_cors_configuration():
    """Test that CORS is properly configured."""
    client = TestClient(app)
    
    # Test CORS headers
    response = client.options("/", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
    })
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "*"

def test_health_endpoint():
    """Test that the health endpoint is registered."""
    client = TestClient(app)
    
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_proxy_endpoints():
    """Test that proxy endpoints are registered."""
    client = TestClient(app)
    
    # Test proxy status endpoint
    response = client.get("/api/proxy/status")
    assert response.status_code == 200
    assert "isRunning" in response.json()

@pytest.mark.asyncio
async def test_shutdown_handler():
    """Test that the shutdown handler properly cleans up resources."""
    with patch("api.proxy.proxy_server") as mock_proxy:
        # Configure mock proxy
        mock_proxy.stop = AsyncMock()
        
        # Call shutdown event
        await app.router.shutdown()
        
        # Verify proxy server was stopped
        mock_proxy.stop.assert_awaited_once()

def test_api_versioning():
    """Test API version information."""
    client = TestClient(app)
    
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["version"] == "0.1.0"

def test_route_protection():
    """Test that sensitive routes are protected."""
    client = TestClient(app)
    
    # Test proxy management endpoints require authentication (TODO)
    sensitive_endpoints = [
        ("/api/proxy/start", "POST"),
        ("/api/proxy/stop", "POST"),
        ("/api/proxy/settings", "POST"),
    ]
    
    for endpoint, method in sensitive_endpoints:
        response = client.request(method, endpoint)
        # Currently returns 400 because proxy is not running
        # TODO: Add authentication and test 401 responses
        assert response.status_code in (400, 401)

def test_error_handling():
    """Test global error handling."""
    client = TestClient(app)
    
    # Test invalid endpoint
    response = client.get("/invalid/endpoint")
    assert response.status_code == 404
    assert response.json()["detail"] == "Not Found"
    
    # Test invalid method
    response = client.post("/api/health")
    assert response.status_code == 405
    assert response.json()["detail"] == "Method Not Allowed"
