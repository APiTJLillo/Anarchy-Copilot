"""Test proxy request/response interception functionality."""
import pytest
import aiohttp
from fastapi import FastAPI
from fastapi.testclient import TestClient
import asyncio
from copy import deepcopy

from proxy.core import ProxyServer
from proxy.config import ProxyConfig

def test_proxy_intercept_request(test_app: FastAPI, test_client: TestClient):
    """Test request interception."""
    # Start proxy server with interception enabled
    response = test_client.post("/proxy/start", json={
        "host": "127.0.0.1",
        "port": 8083,
        "interceptRequests": True,
        "interceptResponses": False,
        "allowedHosts": [],
        "excludedHosts": []
    })
    assert response.status_code == 201

    # Make a request that will be intercepted
    request_data = {
        "id": "test-1",
        "method": "GET",
        "url": "http://example.com",
        "headers": [
            {"name": "User-Agent", "value": "AnarchyCopilot/1.0"}
        ],
        "body": None
    }

    # Intercept and modify request
    response = test_client.post("/proxy/request/intercept/test-1", json=request_data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # Check that the request was properly intercepted and stored
    response = test_client.get("/proxy/history/test-1")
    assert response.status_code == 200
    history = response.json()
    assert history["request"]["method"] == "GET"
    assert history["request"]["url"] == "http://example.com"
    assert len(history["request"]["headers"]) == 1
    assert history["request"]["headers"]["User-Agent"] == "AnarchyCopilot/1.0"

def test_proxy_intercept_response(test_app: FastAPI, test_client: TestClient):
    """Test response interception."""
    # Start proxy server with interception enabled
    response = test_client.post("/proxy/start", json={
        "host": "127.0.0.1",
        "port": 8083,
        "interceptRequests": False,
        "interceptResponses": True,
        "allowedHosts": [],
        "excludedHosts": []
    })
    assert response.status_code == 201

    # Make a request and intercept its response
    response_data = {
        "statusCode": 200,
        "headers": [
            {"name": "Content-Type", "value": "application/json"}
        ],
        "body": '{"status": "success"}'
    }

    # Intercept and modify response
    response = test_client.post("/proxy/response/intercept/test-1", json=response_data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # Check that the response was properly intercepted and stored
    response = test_client.get("/proxy/history/test-1")
    assert response.status_code == 200
    history = response.json()
    assert history["response"]["statusCode"] == 200
    assert history["response"]["headers"]["Content-Type"] == "application/json"
    assert history["response"]["body"] == '{"status": "success"}'

def test_proxy_drop_request(test_app: FastAPI, test_client: TestClient):
    """Test dropping an intercepted request."""
    # Start proxy server with interception enabled
    response = test_client.post("/proxy/start", json={
        "host": "127.0.0.1",
        "port": 8083,
        "interceptRequests": True,
        "interceptResponses": False,
        "allowedHosts": [],
        "excludedHosts": []
    })
    assert response.status_code == 201

    # Drop a request via WebSocket
    with test_client.websocket_connect("/proxy/ws/intercept") as websocket:
        websocket.send_json({
            "type": "request",
            "requestId": "test-1",
            "drop": True
        })
        # WebSocket should close without error
        assert websocket.close()

def test_proxy_drop_response(test_app: FastAPI, test_client: TestClient):
    """Test dropping an intercepted response."""
    # Start proxy server with interception enabled
    response = test_client.post("/proxy/start", json={
        "host": "127.0.0.1",
        "port": 8083,
        "interceptRequests": False,
        "interceptResponses": True,
        "allowedHosts": [],
        "excludedHosts": []
    })
    assert response.status_code == 201

    # Drop a response via WebSocket
    with test_client.websocket_connect("/proxy/ws/intercept") as websocket:
        websocket.send_json({
            "type": "response",
            "requestId": "test-1",
            "drop": True
        })
        # WebSocket should close without error
        assert websocket.close()

@pytest.mark.asyncio
async def test_real_http_interception(test_app: FastAPI, test_client: TestClient):
    """Test interception of real HTTP traffic."""
    # Start proxy server with both request and response interception
    response = test_client.post("/proxy/start", json={
        "host": "127.0.0.1",
        "port": 8083,
        "interceptRequests": True,
        "interceptResponses": True,
        "allowedHosts": ["httpbin.org"],
        "excludedHosts": []
    })
    assert response.status_code == 201

    # Make an HTTP request through proxy
    original_headers = {'User-Agent': 'AnarchyCopilot-Test/1.0'}
    modified_headers = {'User-Agent': 'Modified-Test/1.0'}
    
    async with aiohttp.ClientSession() as session:
        # Send request through proxy
        async with session.get(
            'http://httpbin.org/headers',
            headers=original_headers,
            proxy='http://127.0.0.1:8083'
        ) as response:
            assert response.status == 200
            data = await response.json()
            
            # Headers should be modified by proxy
            assert data['headers']['User-Agent'] == 'Modified-Test/1.0'
            
            # Check history entry
            history_resp = test_client.get("/proxy/history")
            assert history_resp.status_code == 200
            history = history_resp.json()
            
            # Should have one entry
            assert len(history) == 1
            entry = history[0]
            
            # Check request details
            assert entry['request']['headers']['User-Agent'] == 'Modified-Test/1.0'
            assert entry['request']['method'] == 'GET'
            assert 'httpbin.org' in entry['request']['url']
            
            # Check response details
            assert entry['response']['statusCode'] == 200
            assert 'application/json' in entry['response']['headers']['Content-Type']
