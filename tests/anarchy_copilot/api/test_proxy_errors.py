"""Test error handling in proxy API endpoints."""
import pytest
from typing import Dict, Any, AsyncGenerator
import json
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio

async def test_start_proxy_errors(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test error handling when starting proxy."""
    async for client_data in proxy_client:
        base_client = client_data["base_client"]

        # Test missing required fields
        response = await base_client.post("/api/proxy/start", json={})
        assert response.status_code == 422
        assert "validation error" in response.text.lower()

        # Test invalid port range
        response = await base_client.post("/api/proxy/start", json={
            "host": "127.0.0.1",
            "port": 99999,  # Invalid port
            "interceptRequests": True,
            "interceptResponses": True,
            "allowedHosts": [],
            "excludedHosts": []
        })
        assert response.status_code == 400
        assert "port" in response.json()["detail"].lower()

        # Test invalid host format
        response = await base_client.post("/api/proxy/start", json={
            "host": "not-a-valid-host!",
            "port": 8443,
            "interceptRequests": True,
            "interceptResponses": True,
            "allowedHosts": [],
            "excludedHosts": []
        })
        assert response.status_code == 400
        assert "host" in response.json()["detail"].lower()

async def test_stop_proxy_errors(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test error handling when stopping proxy."""
    async for client_data in proxy_client:
        base_client = client_data["base_client"]
        
        # First stop the proxy started by the fixture
        await base_client.post("/api/proxy/stop")
        
        # Test stopping when not running
        response = await base_client.post("/api/proxy/stop")
        assert response.status_code == 400
        assert "not running" in response.json()["detail"].lower()

async def test_proxy_host_filtering(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test host filtering validation."""
    async for client_data in proxy_client:
        base_client = client_data["base_client"]

        # Stop existing proxy first
        await base_client.post("/api/proxy/stop")

        # Test invalid host formats in allowed hosts
        response = await base_client.post("/api/proxy/start", json={
            "host": "127.0.0.1",
            "port": 8443,
            "interceptRequests": True,
            "interceptResponses": True,
            "allowedHosts": ["not.valid!", "also.invalid*"],
            "excludedHosts": []
        })
        assert response.status_code == 400
        assert "host" in response.json()["detail"].lower()

        # Test invalid host formats in excluded hosts
        response = await base_client.post("/api/proxy/start", json={
            "host": "127.0.0.1",
            "port": 8443,
            "interceptRequests": True,
            "interceptResponses": True,
            "allowedHosts": [],
            "excludedHosts": ["not.valid!", "also.invalid*"]
        })
        assert response.status_code == 400
        assert "host" in response.json()["detail"].lower()

        # Test conflicting hosts in allowed and excluded
        response = await base_client.post("/api/proxy/start", json={
            "host": "127.0.0.1",
            "port": 8443,
            "interceptRequests": True,
            "interceptResponses": True,
            "allowedHosts": ["example.com"],
            "excludedHosts": ["example.com"]
        })
        assert response.status_code == 400
        assert "conflict" in response.json()["detail"].lower()

async def test_history_errors(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test error handling in history endpoints."""
    async for client_data in proxy_client:
        base_client = client_data["base_client"]

        # Stop proxy first
        await base_client.post("/api/proxy/stop")

        # Test history access when proxy not running
        response = await base_client.get("/api/proxy/history")
        assert response.status_code == 400
        assert "not running" in response.json()["detail"].lower()

        # Start proxy again
        await base_client.post("/api/proxy/start", json={
            "host": "127.0.0.1",
            "port": 8443,
            "interceptRequests": True,
            "interceptResponses": True,
            "allowedHosts": ["httpbin.org"],
            "excludedHosts": []
        })

        # Test invalid history entry ID
        response = await base_client.get("/api/proxy/history/invalid-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

async def test_websocket_errors(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test WebSocket error handling."""
    async for client_data in proxy_client:
        base_client = client_data["base_client"]

        # Stop proxy first
        await base_client.post("/api/proxy/stop")

        # Test WebSocket connection without proxy running
        response = await base_client.get("/api/proxy/ws/intercept")
        assert response.status_code == 400
        assert "not running" in response.text.lower()

        # Test invalid upgrade request
        await base_client.post("/api/proxy/start", json={
            "host": "127.0.0.1",
            "port": 8443,
            "interceptRequests": True,
            "interceptResponses": True,
            "allowedHosts": ["httpbin.org"],
            "excludedHosts": []
        })
        
        response = await base_client.get("/api/proxy/ws/intercept")
        assert response.status_code == 400
        assert "websocket upgrade" in response.text.lower()

async def test_concurrent_operations(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of concurrent operations."""
    async for client_data in proxy_client:
        base_client = client_data["base_client"]

        # Try to start again while running (proxy already started by fixture)
        response = await base_client.post("/api/proxy/start", json={
            "host": "127.0.0.1",
            "port": 8443,
            "interceptRequests": True,
            "interceptResponses": True,
            "allowedHosts": [],
            "excludedHosts": []
        })
        assert response.status_code == 400
        assert "already running" in response.json()["detail"].lower()

async def test_malformed_requests(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of malformed requests."""
    async for client_data in proxy_client:
        base_client = client_data["base_client"]

        # Stop existing proxy first
        await base_client.post("/api/proxy/stop")

        # Test invalid JSON
        response = await base_client.post(
            "/api/proxy/start",
            content=b"not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert "validation error" in response.text.lower()

        # Test wrong content type
        config: Dict[str, Any] = {
            "host": "127.0.0.1",
            "port": 8443,
            "interceptRequests": True,
            "interceptResponses": True,
            "allowedHosts": [],
            "excludedHosts": []
        }
        response = await base_client.post(
            "/api/proxy/start",
            content=json.dumps(config).encode(),
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 415
        assert "unsupported media type" in response.text.lower()
