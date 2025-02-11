"""
Tests for the proxy API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from anarchy_copilot.api.proxy import router, proxy_server
from anarchy_copilot.proxy.core import ProxyServer
from anarchy_copilot.proxy.config import ProxyConfig

@pytest.fixture
def mock_proxy_server():
    """Create a mock proxy server."""
    server = MagicMock(spec=ProxyServer)
    server.config = MagicMock(spec=ProxyConfig)
    server.session = MagicMock()
    server.is_running = True
    server.start = AsyncMock()
    server.stop = AsyncMock()
    return server

@pytest.fixture
def test_client():
    """Create a test client for the proxy API."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)

def test_get_proxy_status_no_server(test_client):
    """Test getting proxy status when no server is running."""
    global proxy_server
    proxy_server = None
    
    response = test_client.get("/api/proxy/status")
    assert response.status_code == 200
    data = response.json()
    assert data["isRunning"] == False
    assert len(data["history"]) == 0

def test_get_proxy_status_with_server(test_client, mock_proxy_server):
    """Test getting proxy status with a running server."""
    global proxy_server
    proxy_server = mock_proxy_server
    
    # Configure mock server response
    mock_entry = MagicMock()
    mock_entry.id = "test-id"
    mock_entry.timestamp.isoformat.return_value = "2025-02-10T12:00:00"
    mock_entry.request.method = "GET"
    mock_entry.request.url = "http://example.com"
    mock_entry.response.status_code = 200
    mock_entry.duration = 100
    mock_entry.tags = ["test"]
    
    mock_proxy_server.session.get_history.return_value = [mock_entry]
    
    response = test_client.get("/api/proxy/status")
    assert response.status_code == 200
    data = response.json()
    assert data["isRunning"] == True
    assert len(data["history"]) == 1
    assert data["history"][0]["id"] == "test-id"

def test_start_proxy(test_client):
    """Test starting the proxy server."""
    global proxy_server
    proxy_server = None
    
    settings = {
        "host": "127.0.0.1",
        "port": 8080,
        "interceptRequests": True,
        "interceptResponses": True,
        "allowedHosts": [],
        "excludedHosts": []
    }
    
    with patch("anarchy_copilot.api.proxy.ProxyServer") as mock_server:
        response = test_client.post("/api/proxy/start", json=settings)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        mock_server.assert_called_once()

def test_start_proxy_already_running(test_client, mock_proxy_server):
    """Test starting the proxy server when it's already running."""
    global proxy_server
    proxy_server = mock_proxy_server
    
    settings = {
        "host": "127.0.0.1",
        "port": 8080,
        "interceptRequests": True,
        "interceptResponses": True,
        "allowedHosts": [],
        "excludedHosts": []
    }
    
    response = test_client.post("/api/proxy/start", json=settings)
    assert response.status_code == 400
    assert "already running" in response.json()["detail"]

def test_stop_proxy(test_client, mock_proxy_server):
    """Test stopping the proxy server."""
    global proxy_server
    proxy_server = mock_proxy_server
    
    response = test_client.post("/api/proxy/stop")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_proxy_server.stop.assert_awaited_once()

def test_stop_proxy_not_running(test_client):
    """Test stopping the proxy server when it's not running."""
    global proxy_server
    proxy_server = None
    
    response = test_client.post("/api/proxy/stop")
    assert response.status_code == 400
    assert "not running" in response.json()["detail"]

def test_update_settings(test_client, mock_proxy_server):
    """Test updating proxy server settings."""
    global proxy_server
    proxy_server = mock_proxy_server
    
    settings = {
        "host": "127.0.0.1",
        "port": 8080,
        "interceptRequests": False,
        "interceptResponses": True,
        "allowedHosts": ["example.com"],
        "excludedHosts": []
    }
    
    response = test_client.post("/api/proxy/settings", json=settings)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert mock_proxy_server.config.intercept_requests == False
    assert mock_proxy_server.config.allowed_hosts == {"example.com"}

def test_get_history_entry(test_client, mock_proxy_server):
    """Test getting a specific history entry."""
    global proxy_server
    proxy_server = mock_proxy_server
    
    # Configure mock entry
    mock_entry = MagicMock()
    mock_entry.id = "test-id"
    mock_entry.timestamp.isoformat.return_value = "2025-02-10T12:00:00"
    mock_entry.request.to_dict.return_value = {"method": "GET", "url": "http://example.com"}
    mock_entry.response.to_dict.return_value = {"status_code": 200}
    mock_entry.duration = 100
    mock_entry.tags = ["test"]
    mock_entry.notes = "Test notes"
    
    mock_proxy_server.session.find_entry.return_value = mock_entry
    
    response = test_client.get("/api/proxy/history/test-id")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "test-id"
    assert data["request"]["method"] == "GET"
    assert data["response"]["status_code"] == 200

def test_add_entry_tag(test_client, mock_proxy_server):
    """Test adding a tag to a history entry."""
    global proxy_server
    proxy_server = mock_proxy_server
    
    mock_proxy_server.session.add_entry_tag.return_value = True
    
    response = test_client.post("/api/proxy/history/test-id/tags?tag=interesting")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_proxy_server.session.add_entry_tag.assert_called_once_with("test-id", "interesting")

def test_clear_history(test_client, mock_proxy_server):
    """Test clearing the proxy history."""
    global proxy_server
    proxy_server = mock_proxy_server
    
    response = test_client.post("/api/proxy/history/clear")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_proxy_server.session.clear_history.assert_called_once()
