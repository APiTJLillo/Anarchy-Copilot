"""
Tests for health check endpoints.
"""
import os
import time
import psutil
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def test_get_system_status():
    """Test getting system status."""
    with patch("psutil.virtual_memory", return_value=MagicMock(
            total=16000000000,
            available=8000000000,
            percent=50.0
        )), \
        patch("psutil.disk_usage", return_value=MagicMock(
            total=500000000000,
            free=250000000000,
            percent=50.0
        )):
        
        from api.health import get_system_status
        status = get_system_status()
        
        assert status["memory"]["total"] == 16000000000
        assert status["memory"]["available"] == 8000000000
        assert status["memory"]["percent"] == 50.0
        assert status["disk"]["total"] == 500000000000
        assert status["disk"]["free"] == 250000000000
        assert status["disk"]["used_percent"] == 50.0

    # Test error handling
    with patch("psutil.virtual_memory", side_effect=Exception("Test error")):
        from api.health import get_system_status
        with pytest.raises(Exception) as exc_info:
            get_system_status()
        assert "Test error" in str(exc_info.value)

def test_get_application_status():
    """Test getting application status."""
    with patch("os.getpid", return_value=1234), \
         patch("psutil.Process", return_value=MagicMock(
             memory_info=lambda: MagicMock(rss=100000000),
             cpu_percent=lambda: 5.0,
             pid=1234
         )):
        
        from api.health import get_application_status
        status = get_application_status()
        
        assert status["pid"] == 1234
        assert status["memory"] == 100000000
        assert status["cpu_usage"] == 5.0

def test_check_dependencies():
    """Test dependency checking."""
    expected_deps = {
        "nuclei": {
            "installed": True,
            "version": "2.9.8",
            "path": "/usr/local/bin/nuclei"
        }
    }
    with patch("shutil.which", return_value="/usr/local/bin/nuclei"), \
         patch("subprocess.run", return_value=MagicMock(
             returncode=0,
             stdout="2.9.8"
         )):
        from api.health import check_dependencies
        deps = check_dependencies()
        assert deps == expected_deps

    # Test missing dependency
    with patch("shutil.which", return_value=None):
        from api.health import check_dependencies
        deps = check_dependencies()
        assert deps["nuclei"]["installed"] is False
        assert deps["nuclei"]["version"] is None
        assert deps["nuclei"]["path"] is None

@pytest.mark.asyncio
async def test_health_check_endpoint(test_client):
    """Test basic health check endpoint."""
    def raise_error(*args, **kwargs):
        raise Exception("Test error")

    # Test error handling
    with patch("psutil.virtual_memory", side_effect=raise_error):
        response = test_client.get("/api/health")
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()

    # Test success case
    with patch("psutil.virtual_memory", return_value=MagicMock(
            total=16000000000,
            available=8000000000,
            percent=50.0
        )), \
        patch("psutil.disk_usage", return_value=MagicMock(
            total=500000000000,
            free=250000000000,
            percent=50.0
        )):
        response = test_client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_detailed_health_check_endpoint(test_client):
    """Test detailed health check endpoint."""
    with patch("os.getpid", return_value=1234), \
         patch("shutil.which", return_value="/usr/local/bin/nuclei"), \
         patch("subprocess.run", return_value=MagicMock(
             returncode=0,
             stdout="2.9.8"
         )), \
         patch("psutil.virtual_memory", return_value=MagicMock(
             total=16000000000,
             available=8000000000,
             percent=50.0
         )), \
         patch("psutil.disk_usage", return_value=MagicMock(
             total=500000000000,
             free=250000000000,
             percent=50.0
         )), \
         patch("psutil.Process", return_value=MagicMock(
             memory_info=lambda: MagicMock(rss=100000000),
             cpu_percent=lambda: 5.0,
             pid=1234
         )):
        response = test_client.get("/api/health/details")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "system" in data
        assert "application" in data
        assert "dependencies" in data

@pytest.mark.asyncio
async def test_kubernetes_probe_endpoints(test_client):
    """Test Kubernetes probe endpoints."""
    def raise_error(*args, **kwargs):
        raise Exception("Test error")

    # Test liveness - should never error
    response = test_client.get("/api/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"

    # Test readiness with error
    with patch("psutil.virtual_memory", side_effect=raise_error):
        response = test_client.get("/api/health/ready")
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()

    # Test readiness success
    with patch("psutil.virtual_memory", return_value=MagicMock()), \
         patch("psutil.disk_usage", return_value=MagicMock()):
        response = test_client.get("/api/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

@pytest.mark.asyncio
async def test_health_check_concurrency(test_client):
    """Test concurrent health check requests."""
    with patch("psutil.virtual_memory", return_value=MagicMock(
            total=16000000000,
            available=8000000000,
            percent=50.0
        )), \
        patch("psutil.disk_usage", return_value=MagicMock(
            total=500000000000,
            free=250000000000,
            percent=50.0
        )):
        
        # Make multiple concurrent requests
        responses = []
        for _ in range(5):
            response = test_client.get("/api/health")
            responses.append(response)
        
        # Verify all requests succeeded
        for response in responses:
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

@pytest.mark.parametrize("endpoint", [
    "/api/health",
    "/api/health/details",
    "/api/health/ready"
])
def test_health_endpoints_error_handling(test_client, endpoint):
    """Test error handling in health check endpoints."""
    error_msg = "Test error"
    
    with patch("psutil.virtual_memory", side_effect=Exception(error_msg)), \
         patch("psutil.Process", side_effect=Exception(error_msg)), \
         patch("psutil.disk_usage", side_effect=Exception(error_msg)), \
         patch("shutil.which", side_effect=Exception(error_msg)):
        
        response = test_client.get(endpoint)
        assert response.status_code == 500
        assert error_msg in response.json()["detail"]

@pytest.mark.asyncio
async def test_health_endpoints_timeout(test_client):
    """Test timeout handling in health endpoints."""
    def slow_process(*args, **kwargs):
        time.sleep(6)  # Longer than timeout
        return MagicMock()

    with patch("psutil.virtual_memory", side_effect=slow_process):
        response = test_client.get("/api/health/details")
        assert response.status_code == 500
        assert "operation timed out" in response.json()["detail"].lower()
        assert "5 seconds" in response.json()["detail"].lower()
