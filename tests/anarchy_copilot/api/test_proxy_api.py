"""Tests for proxy management API endpoints."""
import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient

from api import app
from api.proxy import proxy_server
from proxy.core import ProxyServer
from proxy.config import ProxyConfig
from proxy.session import HistoryEntry
from .conftest import verify_json_response

@pytest.fixture
def mock_proxy_server():
    """Create mock proxy server."""
    mock = Mock(spec=ProxyServer)
    mock.is_running = False
    mock.config = ProxyConfig()
    mock.session = Mock()
    mock.session.get_history = Mock(return_value=[])
    mock.session.find_entry = Mock(return_value=None)
    mock.session.add_entry_tag = Mock(return_value=True)
    mock.session.set_entry_note = Mock(return_value=True)
    mock.session.clear_history = Mock()
    mock.start = AsyncMock()
    mock.stop = AsyncMock()
    return mock

def test_get_proxy_status(test_client):
    """Test getting proxy server status."""
    global proxy_server
    proxy_server = None
    
    response = test_client.get("/api/proxy/status")
    assert response.status_code == 200
    data = response.json()
    assert data["isRunning"] is False

def test_get_proxy_status_with_server(test_client, mock_proxy_server):
    """Test getting proxy status with active server."""
    global proxy_server
    proxy_server = mock_proxy_server
    mock_proxy_server.is_running = True
    
    response = test_client.get("/api/proxy/status")
    assert response.status_code == 200
    data = response.json()
    assert data["isRunning"] is True
    assert "history" in data

@pytest.mark.asyncio
async def test_start_proxy(test_client, mock_proxy_server):
    """Test starting proxy server."""
    global proxy_server
    proxy_server = None
    
    with patch('api.proxy.ProxyServer', return_value=mock_proxy_server):
        response = test_client.post(
            "/api/proxy/start",
            json={
                "host": "127.0.0.1",
                "port": 8080,
                "interceptRequests": True,
                "interceptResponses": True,
                "allowedHosts": [],
                "excludedHosts": []
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        verify_json_response(data)
        assert data["status"] == "success"
        mock_proxy_server.start.assert_awaited_once()

@pytest.mark.asyncio
async def test_stop_proxy(test_client, mock_proxy_server):
    """Test stopping proxy server."""
    global proxy_server
    proxy_server = mock_proxy_server
    
    response = test_client.post("/api/proxy/stop")
    assert response.status_code == 200
    data = response.json()
    verify_json_response(data)
    assert data["status"] == "success"
    mock_proxy_server.stop.assert_awaited_once()

def test_update_settings(test_client, mock_proxy_server):
    """Test updating proxy settings."""
    global proxy_server
    proxy_server = mock_proxy_server
    mock_proxy_server.is_running = True
    
    response = test_client.post(
        "/api/proxy/settings",
        json={
            "host": "127.0.0.1",
            "port": 8080,
            "interceptRequests": True,
            "interceptResponses": True,
            "allowedHosts": ["example.com"],
            "excludedHosts": []
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    verify_json_response(data)
    assert data["status"] == "success"

def test_get_history_entry(test_client, mock_proxy_server):
    """Test getting history entry details."""
    global proxy_server
    proxy_server = mock_proxy_server
    
    mock_entry = HistoryEntry(
        id="test-id",
        timestamp=datetime.now(),
        request={
            "method": "GET",
            "url": "http://example.com",
            "headers": {}
        },
        response={
            "status_code": 200,
            "headers": {},
            "body": b"test"
        }
    )
    mock_proxy_server.session.find_entry.return_value = mock_entry
    
    response = test_client.get("/api/proxy/history/test-id")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "test-id"
    mock_proxy_server.session.find_entry.assert_called_once_with("test-id")

def test_add_entry_tag(test_client, mock_proxy_server):
    """Test adding a tag to a history entry."""
    global proxy_server
    proxy_server = mock_proxy_server
    
    mock_proxy_server.session.add_entry_tag.return_value = True
    
    response = test_client.post("/api/proxy/history/test-id/tags?tag=interesting")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_proxy_server.session.add_entry_tag.assert_called_once_with("test-id", "interesting")

def test_set_entry_note(test_client, mock_proxy_server):
    """Test setting a note on a history entry."""
    global proxy_server
    proxy_server = mock_proxy_server
    
    mock_proxy_server.session.set_entry_note.return_value = True
    
    response = test_client.post(
        "/api/proxy/history/test-id/notes",
        json={"note": "Test note"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_proxy_server.session.set_entry_note.assert_called_once()

def test_clear_history(test_client, mock_proxy_server):
    """Test clearing the proxy history."""
    global proxy_server
    proxy_server = mock_proxy_server
    
    response = test_client.post("/api/proxy/history/clear")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_proxy_server.session.clear_history.assert_called_once()
