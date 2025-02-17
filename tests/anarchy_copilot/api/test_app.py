"""
Tests for the main FastAPI application initialization and configuration.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

def test_app_initialization(client):
    """Test that the FastAPI app is properly initialized."""
    # Test API info
    response = client.get("/docs")
    assert response.status_code == 200
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Anarchy Copilot API"

def test_cors_configuration(client):
    """Test that CORS is properly configured."""
    # Test CORS headers
    response = client.options("/", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
    })
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "*"

def test_health_endpoint(client):
    """Test that the health endpoint is registered."""
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

@pytest.mark.asyncio
async def test_proxy_endpoints(async_client):
    """Test that proxy endpoints are registered."""
    # Test proxy status endpoint
    response = await async_client.get("/api/proxy/status")
    assert response.status_code == 200
    assert "isRunning" in response.json()
    
    # Test validation of required fields
    response = await async_client.post("/api/proxy/start", json={})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(err["msg"].lower() == "field required" for err in errors)

@pytest.mark.asyncio
async def test_shutdown_handler(app):
    """Test that the shutdown handler properly cleans up resources."""
    with patch("api.proxy.proxy_server") as mock_proxy:
        # Configure mock proxy
        mock_proxy.stop = AsyncMock()
        mock_proxy.is_running = True
        
        # Call shutdown event
        for handler in app.router.on_shutdown:
            await handler()
        
        # Verify proxy server was stopped if it was running
        if mock_proxy.is_running:
            mock_proxy.stop.assert_awaited_once()

def test_api_versioning(client):
    """Test API version information."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["version"] == "0.1.0"

@pytest.mark.asyncio
async def test_route_validation(async_client):
    """Test input validation on routes."""
    test_cases = [
        # Endpoint, Body, Expected Error Message, Expected Location
        ("/api/proxy/start", {}, "field required", ["body", "host"]),
        ("/api/proxy/history/test-id/tags", {}, "field required", ["body", "tag"]),
        ("/api/proxy/history/test-id/notes", {}, "field required", ["body", "note"]),
    ]
    
    for endpoint, body, expected_error, expected_loc in test_cases:
        response = await async_client.post(endpoint, json=body)
        assert response.status_code == 422, \
            f"Expected validation error for {endpoint}"
        
        errors = response.json()["detail"]
        error = next((err for err in errors if expected_error in err["msg"].lower()), None)
        assert error is not None, \
            f"Expected '{expected_error}' in validation errors"
        assert error["loc"] == expected_loc, \
            f"Expected error location {expected_loc}, got {error['loc']}"

@pytest.mark.asyncio
async def test_route_protection(async_client):
    """Test that proxy operations require an active proxy."""
    post_endpoints = [
        "/api/proxy/stop",
        "/api/proxy/history/clear",
        "/api/proxy/settings"
    ]
    
    # Test POST endpoints
    for endpoint in post_endpoints:
        response = await async_client.post(endpoint, json={})
        assert response.status_code == 400, \
            f"Expected proxy not running error for {endpoint}"
        assert "not running" in response.json()["detail"].lower()

    # Test GET endpoints
    response = await async_client.get("/api/proxy/history/test-id")
    assert response.status_code == 400
    assert "not running" in response.json()["detail"].lower()

@pytest.mark.asyncio
async def test_error_handling(async_client):
    """Test global error handling."""
    # Test invalid endpoint
    response = await async_client.get("/invalid/endpoint")
    assert response.status_code == 404
    assert response.json()["detail"] == "Not Found"
    
    # Test invalid method
    response = await async_client.post("/api/health")
    assert response.status_code == 405
    assert response.json()["detail"] == "Method Not Allowed"

    # Test invalid JSON request
    response = await async_client.post(
        "/api/proxy/start",
        headers={"Content-Type": "application/json"},
        content="{invalid-json}"
    )
    assert response.status_code == 422
    error_detail = str(response.json()["detail"]).lower()
    assert "json decode error" in error_detail

@pytest.mark.asyncio
async def test_async_error_handling(async_client):
    """Test async error handling."""
    with patch('api.proxy.proxy_server') as mock_server:
        mock_server.stop = AsyncMock(side_effect=RuntimeError("Test async error"))
        mock_server.is_running = True
        
        response = await async_client.post("/api/proxy/stop")
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()
