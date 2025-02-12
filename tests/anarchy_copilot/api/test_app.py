"""
Tests for the main FastAPI application initialization and configuration.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

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
    
    with patch("os.system", return_value=0), \
         patch("psutil.virtual_memory", return_value=MagicMock(
             total=16000000000,
             available=8000000000,
             percent=50.0
         )), \
         patch("psutil.disk_usage", return_value=MagicMock(
             total=500000000000,
             free=250000000000,
             percent=50.0
         )):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_proxy_endpoints(test_client, test_port):
    """Test that proxy endpoints are registered."""
    # Test proxy status endpoint
    response = test_client.get("/api/proxy/status")
    assert response.status_code == 200
    assert "isRunning" in response.json()
    
    # Test validation of required fields
    response = test_client.post("/api/proxy/start", json={})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(err["msg"].lower() == "field required" for err in errors)

@pytest.mark.asyncio
async def test_shutdown_handler():
    """Test that the shutdown handler properly cleans up resources."""
    with patch("api.proxy.proxy_server") as mock_proxy:
        # Configure mock proxy
        mock_proxy.stop = AsyncMock()
        mock_proxy.is_running = True
        
        # Call shutdown event
        await app.router.shutdown()
        
        # Verify proxy server was stopped if it was running
        if mock_proxy.is_running:
            mock_proxy.stop.assert_awaited_once()

def test_api_versioning():
    """Test API version information."""
    client = TestClient(app)
    
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["version"] == "0.1.0"

def test_route_validation(test_client):
    """Test input validation on routes."""
    test_cases = [
        # Endpoint, Body, Expected Error Message, Expected Location
        ("/api/proxy/start", {}, "field required", ["body", "host"]),
        ("/api/proxy/history/test-id/tags", {}, "field required", ["body", "tag"]),
        ("/api/proxy/history/test-id/notes", {}, "field required", ["body", "note"]),
    ]
    
    for endpoint, body, expected_error, expected_loc in test_cases:
        response = test_client.post(endpoint, json=body)
        assert response.status_code == 422, \
            f"Expected validation error for {endpoint}"
        
        errors = response.json()["detail"]
        error = next((err for err in errors if expected_error in err["msg"].lower()), None)
        assert error is not None, \
            f"Expected '{expected_error}' in validation errors"
        assert error["loc"] == expected_loc, \
            f"Expected error location {expected_loc}, got {error['loc']}"

def test_route_protection(test_client, mock_proxy_server):
    """Test that proxy operations require an active proxy."""
    post_endpoints = [
        "/api/proxy/stop",
        "/api/proxy/history/clear",
        "/api/proxy/settings"
    ]
    
    # Test POST endpoints
    for endpoint in post_endpoints:
        response = test_client.post(endpoint, json={})
        assert response.status_code == 400, \
            f"Expected proxy not running error for {endpoint}"
        assert "not running" in response.json()["detail"].lower()

    # Test GET endpoints
    response = test_client.get("/api/proxy/history/test-id")
    assert response.status_code == 400
    assert "not running" in response.json()["detail"].lower()

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

    # Test invalid JSON request
    response = client.post(
        "/api/proxy/start",
        headers={"Content-Type": "application/json"},
        content="{invalid-json}"
    )
    assert response.status_code == 422
    error_detail = str(response.json()["detail"]).lower()
    assert "json decode error" in error_detail

@pytest.mark.asyncio
async def test_async_error_handling(test_client, test_port):
    """Test async error handling."""
    with patch('api.proxy.proxy_server') as mock_server:
        mock_server.stop = AsyncMock(side_effect=RuntimeError("Test async error"))
        mock_server.is_running = True
        
        response = test_client.post("/api/proxy/stop")
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()
