"""Tests for proxy management API endpoints."""
import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient
from datetime import datetime, timezone

from api import app
from proxy.interceptor import InterceptedRequest, InterceptedResponse
from api.proxy import ProxySettings
from proxy.core import ProxyServer
from proxy.config import ProxyConfig
from proxy.session import HistoryEntry

def test_get_proxy_status(test_client):
    """Test getting proxy server status."""
    response = test_client.get("/api/proxy/status")
    assert response.status_code == 200
    data = response.json()
    assert data["isRunning"] is False
    assert "history" in data

def test_get_proxy_status_with_server(test_client, mock_proxy_server):
    """Test getting proxy status with active server."""
    import api.proxy
    api.proxy.proxy_server = mock_proxy_server
    
    response = test_client.get("/api/proxy/status")
    assert response.status_code == 200
    data = response.json()
    assert data["isRunning"] is True
    assert data["interceptRequests"] is True
    assert data["interceptResponses"] is True
    assert "history" in data

@pytest.mark.asyncio
async def test_start_proxy(test_client, test_port):
    """Test starting proxy server."""
    # Test without required fields
    response = test_client.post("/api/proxy/start", json={})
    assert response.status_code == 422
    errors = response.json()["detail"]
    
    host_error = next((err for err in errors if err["loc"] == ["body", "host"]), None)
    assert host_error is not None
    assert host_error["msg"].lower() == "field required"

    # Test with invalid host
    response = test_client.post(
        "/api/proxy/start",
        json={
            "host": "",
            "port": test_port,
            "interceptRequests": True,
            "interceptResponses": True
        }
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("min_length" in err["msg"].lower() for err in errors)

    # Test valid config
    mock_server = Mock(spec=ProxyServer)
    mock_server.is_running = True
    mock_server.start = AsyncMock()

    with patch('api.proxy.ProxyServer', return_value=mock_server):
        response = test_client.post(
            "/api/proxy/start",
            json={
                "host": "127.0.0.1",
                "port": test_port,
                "interceptRequests": True,
                "interceptResponses": True,
                "allowedHosts": [],
                "excludedHosts": []
            }
        )
        
        assert response.status_code == 201
        assert response.json()["status"] == "success"
        mock_server.start.assert_awaited_once()

@pytest.mark.asyncio
async def test_stop_proxy(test_client, mock_proxy_server):
    """Test stopping proxy server."""
    # Test without proxy running
    response = test_client.post("/api/proxy/stop")
    assert response.status_code == 400
    assert "not running" in response.json()["detail"].lower()

    # Test with proxy running
    import api.proxy
    api.proxy.proxy_server = mock_proxy_server
    
    response = test_client.post("/api/proxy/stop")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_proxy_server.stop.assert_awaited_once()

def test_get_history_entry(test_client, mock_proxy_server):
    """Test getting history entry details."""
    import api.proxy
    api.proxy.proxy_server = mock_proxy_server

    # Create test data
    request_data = {
        "method": "GET",
        "url": "http://example.com",
        "headers": {},
        "body": ""
    }
    response_data = {
        "status_code": 200,
        "headers": {},
        "body": "test"
    }
    
    # Configure mock objects
    mock_request = Mock(spec=InterceptedRequest)
    mock_request.method = "GET"
    mock_request.url = "http://example.com"
    mock_request.headers = {}
    mock_request.body = b""
    mock_request.to_dict.return_value = request_data

    mock_response = Mock(spec=InterceptedResponse)
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.body = b"test"
    mock_response.to_dict.return_value = response_data
    
    mock_entry = Mock(spec=HistoryEntry)
    mock_entry.id = "test-id"
    mock_entry.timestamp = datetime.now(timezone.utc)
    mock_entry.request = mock_request
    mock_entry.response = mock_response
    mock_entry.tags = []
    mock_entry.notes = None
    mock_entry.duration = 0.1

    mock_proxy_server.session.find_entry.return_value = mock_entry

    response = test_client.get("/api/proxy/history/test-id")
    assert response.status_code == 200
    data = response.json()
    
    assert data["id"] == mock_entry.id
    assert data["request"] == request_data
    assert data["response"] == response_data
    assert data["tags"] == mock_entry.tags
    assert data["notes"] == mock_entry.notes
    assert data["duration"] == mock_entry.duration

def test_get_history_entry_not_found(test_client, mock_proxy_server):
    """Test getting non-existent history entry."""
    import api.proxy
    api.proxy.proxy_server = mock_proxy_server
    mock_proxy_server.session.find_entry.return_value = None
    
    response = test_client.get("/api/proxy/history/invalid-id")
    assert response.status_code == 404

def test_add_entry_tag(test_client, mock_proxy_server):
    """Test adding a tag to a history entry."""
    import api.proxy
    api.proxy.proxy_server = mock_proxy_server
    
    # Test without tag
    response = test_client.post("/api/proxy/history/test-id/tags", json={})
    assert response.status_code == 422
    error = response.json()["detail"]
    assert isinstance(error, list)
    assert error[0]["msg"].lower() == "field required"
    assert error[0]["loc"] == ["body", "tag"]

    # Test valid tag
    response = test_client.post(
        "/api/proxy/history/test-id/tags",
        json={"tag": "interesting"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_proxy_server.session.add_entry_tag.assert_called_once_with("test-id", "interesting")

def test_set_entry_note(test_client, mock_proxy_server):
    """Test setting a note on a history entry."""
    import api.proxy
    api.proxy.proxy_server = mock_proxy_server

    # Test without note    
    response = test_client.post("/api/proxy/history/test-id/notes", json={})
    assert response.status_code == 422
    error = response.json()["detail"]
    assert isinstance(error, list)
    assert error[0]["msg"].lower() == "field required"
    assert error[0]["loc"] == ["body", "note"]

    # Test valid note
    response = test_client.post(
        "/api/proxy/history/test-id/notes",
        json={"note": "Test note"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_proxy_server.session.set_entry_note.assert_called_once_with("test-id", "Test note")

def test_clear_history(test_client, mock_proxy_server):
    """Test clearing the proxy history."""
    import api.proxy
    api.proxy.proxy_server = mock_proxy_server
    
    response = test_client.post("/api/proxy/history/clear")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_proxy_server.session.clear_history.assert_called_once()

def test_validation_errors(test_client):
    """Test validation error handling."""
    # Test without proxy running
    response = test_client.post("/api/proxy/stop")
    assert response.status_code == 400
    assert "not running" in response.json()["detail"].lower()
    
    # Test missing/invalid fields
    test_cases = [
        ("/api/proxy/start", {}, "field required", ["body", "host"]),
        ("/api/proxy/history/test-id/tags", {}, "field required", ["body", "tag"]),
        ("/api/proxy/history/test-id/notes", {}, "field required", ["body", "note"]),
        ("/api/proxy/start", {"host": ""}, "min_length", ["body", "host"])
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
