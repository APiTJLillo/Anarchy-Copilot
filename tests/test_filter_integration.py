"""
Integration tests for the advanced filtering system with bypass mode.

This module provides integration tests for the filter system, testing the interaction
between components and the proxy system.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from proxy.filter import (
    FilterMode, 
    FilterCondition, 
    FilterRule, 
    FilterManager,
    FilterInterceptor
)
from proxy.interceptor import InterceptedRequest, InterceptedResponse
from proxy.interceptors.filter_integration import initialize_filter_system
from api.filter import router as filter_router
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Create test app
app = FastAPI()
app.include_router(filter_router)
client = TestClient(app)

# Test API endpoints
def test_filter_api_endpoints():
    # Test get filter mode
    with patch('api.filter.get_filter_manager') as mock_get_manager:
        mock_manager = AsyncMock()
        mock_manager.get_mode.return_value = FilterMode.ACTIVE
        mock_get_manager.return_value = mock_manager
        
        response = client.get("/api/filter/mode")
        assert response.status_code == 200
        assert response.json() == {"mode": "ACTIVE"}
    
    # Test set filter mode
    with patch('api.filter.get_filter_manager') as mock_get_manager:
        mock_manager = AsyncMock()
        mock_manager.set_mode.return_value = True
        mock_get_manager.return_value = mock_manager
        
        response = client.put("/api/filter/mode", json={"mode": "BYPASS"})
        assert response.status_code == 200
        assert response.json() == {"mode": "BYPASS"}
        mock_manager.set_mode.assert_called_once()
    
    # Test get filter rules
    with patch('api.filter.get_filter_manager') as mock_get_manager:
        mock_manager = AsyncMock()
        mock_rule = FilterRule(
            id="test-id",
            name="Test Rule",
            description="A test rule",
            conditions=[FilterCondition("path", "contains", "/admin")],
            enabled=True,
            priority=10,
            tags=["test", "admin"]
        )
        mock_manager.get_all_rules.return_value = [mock_rule]
        mock_get_manager.return_value = mock_manager
        
        response = client.get("/api/filter/rules")
        assert response.status_code == 200
        rules = response.json()
        assert len(rules) == 1
        assert rules[0]["id"] == "test-id"
        assert rules[0]["name"] == "Test Rule"
    
    # Test create filter rule
    with patch('api.filter.get_filter_manager') as mock_get_manager:
        mock_manager = AsyncMock()
        mock_manager.add_rule.return_value = True
        mock_get_manager.return_value = mock_manager
        
        rule_data = {
            "name": "New Rule",
            "description": "A new rule",
            "conditions": [
                {
                    "field": "path",
                    "operator": "contains",
                    "value": "/admin"
                }
            ],
            "enabled": True,
            "priority": 10,
            "tags": ["test", "admin"]
        }
        
        response = client.post("/api/filter/rules", json=rule_data)
        assert response.status_code == 200
        assert response.json()["name"] == "New Rule"
        mock_manager.add_rule.assert_called_once()

# Test filter interceptor integration
@pytest.mark.asyncio
async def test_filter_interceptor_integration():
    # Mock the proxy system
    with patch('proxy.interceptors.filter_integration.register_interceptor') as mock_register:
        # Initialize filter system
        initialize_filter_system()
        
        # Check if interceptor was registered
        mock_register.assert_called_once()
        
        # Get the registered interceptor
        interceptor = mock_register.call_args[0][0]
        assert isinstance(interceptor, FilterInterceptor)

# Test bypass mode handling
@pytest.mark.asyncio
async def test_bypass_mode_handling():
    # Create mock storage
    mock_storage = AsyncMock()
    mock_storage.get_mode.return_value = FilterMode.BYPASS
    
    # Create manager
    manager = FilterManager(mock_storage)
    
    # Create interceptor
    interceptor = FilterInterceptor(manager)
    
    # Create mock request
    mock_request = MagicMock(spec=InterceptedRequest)
    mock_request.method = "GET"
    mock_request.path = "/admin"
    mock_request.headers = {}
    mock_request.body = b""
    mock_request.query_params = {}
    
    # Test intercept_request in bypass mode
    with patch.object(manager, 'record_traffic') as mock_record:
        result = await interceptor.intercept_request(mock_request)
        
        # Request should not be blocked in bypass mode
        assert not hasattr(result, 'blocked')
        
        # Traffic should be recorded
        mock_record.assert_called_once()
    
    # Create mock response
    mock_response = MagicMock(spec=InterceptedResponse)
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.body = b""
    
    # Test intercept_response in bypass mode
    with patch.object(manager, 'record_traffic') as mock_record:
        result = await interceptor.intercept_response(mock_response, mock_request)
        
        # Response should not be blocked in bypass mode
        assert result is mock_response
        
        # Traffic should be recorded
        mock_record.assert_called_once()

# Test traffic recording
@pytest.mark.asyncio
async def test_traffic_recording():
    # Create manager
    manager = FilterManager(AsyncMock())
    
    # Record request traffic
    await manager.record_traffic({
        "type": "request",
        "method": "GET",
        "path": "/admin",
        "headers": {"User-Agent": "Test"},
        "body": None,
        "timestamp": datetime.now().isoformat()
    })
    
    # Record response traffic
    await manager.record_traffic({
        "type": "response",
        "request_method": "GET",
        "request_path": "/admin",
        "status_code": 200,
        "headers": {"Content-Type": "application/json"},
        "body": '{"success": true}',
        "timestamp": datetime.now().isoformat()
    })
    
    # Check traffic history
    history = manager.get_traffic_history()
    assert len(history) == 2
    assert history[0]["type"] == "request"
    assert history[0]["path"] == "/admin"
    assert history[1]["type"] == "response"
    assert history[1]["status_code"] == 200
    
    # Test create_rule_from_traffic
    rule = manager.create_rule_from_traffic(0)
    assert rule.name.startswith("Rule from request")
    assert len(rule.conditions) > 0
    assert any(c.field == "path" and c.value == "/admin" for c in rule.conditions)
    
    # Test suggest_conditions
    conditions = manager.suggest_conditions([0, 1])
    assert len(conditions) > 0
