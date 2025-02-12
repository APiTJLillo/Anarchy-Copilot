"""Tests for API health check endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import psutil
import os
from datetime import datetime, timezone

from api.health import (
    router,
    get_system_status,
    get_application_status,
    check_dependencies
)

@pytest.fixture
def test_client():
    """Create a test client for the API."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)

@pytest.fixture
def mock_system_info():
    """Mock system information."""
    mock_memory = MagicMock()
    mock_memory.total = 16000000000  # 16GB
    mock_memory.available = 8000000000  # 8GB
    mock_memory.percent = 50.0

    mock_disk = MagicMock()
    mock_disk.total = 500000000000  # 500GB
    mock_disk.free = 250000000000  # 250GB
    mock_disk.percent = 50.0

    with patch("psutil.virtual_memory", return_value=mock_memory), \
         patch("psutil.disk_usage", return_value=mock_disk), \
         patch("psutil.cpu_percent", return_value=25.0):
        yield

@pytest.fixture
def mock_process_info():
    """Mock process information."""
    mock_process = MagicMock()
    mock_process.memory_info().rss = 100000000  # 100MB
    mock_process.num_threads.return_value = 4
    mock_process.create_time.return_value = 1612345678  # Example timestamp

    with patch("psutil.Process", return_value=mock_process):
        yield

def test_get_system_status(mock_system_info):
    """Test system status information collection."""
    status = get_system_status()
    
    assert "cpu_percent" in status
    assert "memory" in status
    assert "disk" in status
    assert status["memory"]["percent"] == 50.0
    assert status["disk"]["percent"] == 50.0

def test_get_application_status(mock_process_info):
    """Test application status information collection."""
    status = get_application_status()
    
    assert "version" in status
    assert "pid" in status
    assert "process" in status
    assert "memory" in status["process"]
    assert "threads" in status["process"]
    assert "start_time" in status

def test_check_dependencies():
    """Test external dependency checking."""
    with patch("os.system", return_value=0):  # Simulate all tools available
        deps = check_dependencies()
        assert deps["nuclei"] is True

    with patch("os.system", return_value=1):  # Simulate missing tools
        deps = check_dependencies()
        assert deps["nuclei"] is False

def test_health_check_endpoint(test_client, mock_system_info):
    """Test basic health check endpoint."""
    with patch("os.system", return_value=0):  # Tools available
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["dependencies"]["nuclei"] is True

    with patch("os.system", return_value=1):  # Tools unavailable
        response = test_client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"

def test_detailed_health_check_endpoint(
    test_client,
    mock_system_info,
    mock_process_info
):
    """Test detailed health check endpoint."""
    with patch("os.system", return_value=0):
        response = test_client.get("/health/details")
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "system" in data
        assert "application" in data
        assert "dependencies" in data
        assert "managers" in data
        
        # Check system metrics
        assert "cpu_percent" in data["system"]
        assert "memory" in data["system"]
        assert "disk" in data["system"]
        
        # Check application metrics
        assert "version" in data["application"]
        assert "process" in data["application"]

def test_kubernetes_probe_endpoints(test_client):
    """Test Kubernetes probe endpoints."""
    # Liveness probe should always return 200
    response = test_client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"

    # Readiness probe depends on dependencies
    with patch("os.system", return_value=0):
        response = test_client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    with patch("os.system", return_value=1):
        response = test_client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"

@pytest.mark.asyncio
async def test_health_check_concurrency():
    """Test health check endpoints under concurrent access."""
    import asyncio
    from httpx import AsyncClient
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    async with AsyncClient(app=app, base_url="http://test") as client:
        # Make multiple concurrent requests
        tasks = [
            client.get("/health")
            for _ in range(10)
        ]
        responses = await asyncio.gather(*tasks)

        # All requests should complete successfully
        for response in responses:
            assert response.status_code in (200, 503)

@pytest.mark.parametrize("endpoint", [
    "/health",
    "/health/details",
    "/health/live",
    "/health/ready"
])
def test_health_endpoints_error_handling(test_client, endpoint):
    """Test error handling in health check endpoints."""
    with patch("psutil.virtual_memory", side_effect=Exception("Test error")):
        response = test_client.get(endpoint)
        # Should not crash
        assert response.status_code in (200, 503)
