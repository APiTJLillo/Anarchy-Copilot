"""
End-to-end tests for the advanced filtering system with bypass mode.

This module provides end-to-end tests for the filter system, testing the complete
flow from frontend to backend and proxy integration.
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
from proxy.analysis.filter_analysis import FilterAnalyzer

# Test end-to-end flow for creating and applying a filter rule
@pytest.mark.asyncio
async def test_create_and_apply_filter():
    # Create mock storage
    mock_storage = AsyncMock()
    mock_storage.get_mode.return_value = FilterMode.ACTIVE
    mock_storage.add_rule.return_value = True
    
    # Create manager
    manager = FilterManager(mock_storage)
    
    # Create interceptor
    interceptor = FilterInterceptor(manager)
    
    # Create a filter rule
    rule = FilterRule(
        name="Block Admin Access",
        description="Block access to admin pages",
        conditions=[
            FilterCondition("path", "contains", "/admin")
        ],
        enabled=True,
        priority=10,
        tags=["admin", "security"]
    )
    
    # Add the rule
    await manager.add_rule(rule)
    mock_storage.add_rule.assert_called_once()
    
    # Create a request that should be blocked
    mock_request = MagicMock(spec=InterceptedRequest)
    mock_request.method = "GET"
    mock_request.path = "/admin/users"
    mock_request.headers = {}
    mock_request.body = b""
    mock_request.query_params = {}
    
    # Mock the evaluate_request method to return our rule
    with patch.object(manager, 'evaluate_request', return_value=(True, [rule.id])):
        # Test that the request is blocked
        result = await interceptor.intercept_request(mock_request)
        assert hasattr(result, 'blocked')
        assert result.blocked is True
        assert hasattr(result, 'blocked_response')
        assert result.blocked_response.status_code == 403
    
    # Create a request that should not be blocked
    mock_request.path = "/public/page"
    
    # Mock the evaluate_request method to return no rules
    with patch.object(manager, 'evaluate_request', return_value=(False, [])):
        # Test that the request is not blocked
        result = await interceptor.intercept_request(mock_request)
        assert not hasattr(result, 'blocked')

# Test end-to-end flow for bypass mode
@pytest.mark.asyncio
async def test_bypass_mode_flow():
    # Create mock storage
    mock_storage = AsyncMock()
    mock_storage.get_mode.return_value = FilterMode.BYPASS
    mock_storage.set_mode.return_value = True
    
    # Create manager
    manager = FilterManager(mock_storage)
    
    # Create interceptor
    interceptor = FilterInterceptor(manager)
    
    # Create a filter rule that would normally block
    rule = FilterRule(
        name="Block Admin Access",
        description="Block access to admin pages",
        conditions=[
            FilterCondition("path", "contains", "/admin")
        ],
        enabled=True,
        priority=10,
        tags=["admin", "security"]
    )
    
    # Add the rule
    await manager.add_rule(rule)
    
    # Create a request that matches the rule
    mock_request = MagicMock(spec=InterceptedRequest)
    mock_request.method = "GET"
    mock_request.path = "/admin/users"
    mock_request.headers = {}
    mock_request.body = b""
    mock_request.query_params = {}
    
    # In bypass mode, the request should not be blocked
    with patch.object(manager, 'record_traffic') as mock_record:
        result = await interceptor.intercept_request(mock_request)
        assert not hasattr(result, 'blocked')
        mock_record.assert_called_once()
    
    # Switch to active mode
    mock_storage.get_mode.return_value = FilterMode.ACTIVE
    
    # Now the request should be blocked
    with patch.object(manager, 'evaluate_request', return_value=(True, [rule.id])):
        result = await interceptor.intercept_request(mock_request)
        assert hasattr(result, 'blocked')
        assert result.blocked is True

# Test end-to-end flow for post-analysis filter addition
@pytest.mark.asyncio
async def test_post_analysis_filter_addition():
    # Create mock storage
    mock_storage = AsyncMock()
    mock_storage.get_mode.return_value = FilterMode.BYPASS
    mock_storage.add_rule.return_value = True
    
    # Create manager
    manager = FilterManager(mock_storage)
    
    # Create analyzer
    analyzer = FilterAnalyzer()
    
    # Record some traffic in bypass mode
    await manager.record_traffic({
        "type": "request",
        "method": "GET",
        "path": "/admin/users",
        "headers": {"User-Agent": "Test"},
        "body": None,
        "timestamp": datetime.now().isoformat(),
        "query_params": {}
    })
    
    await manager.record_traffic({
        "type": "request",
        "method": "POST",
        "path": "/admin/users",
        "headers": {"Content-Type": "application/json"},
        "body": '{"username": "test"}',
        "timestamp": datetime.now().isoformat(),
        "query_params": {}
    })
    
    # Get traffic history
    history = manager.get_traffic_history()
    assert len(history) == 2
    
    # Create a rule from the first traffic item
    rule = manager.create_rule_from_traffic(0, "Block Admin Access", "Created from traffic analysis")
    
    # Add the rule
    await manager.add_rule(rule)
    mock_storage.add_rule.assert_called_once()
    
    # Get suggestions from analyzer
    suggested_rules = analyzer.suggest_filter_rules(history)
    assert len(suggested_rules) > 0
    
    # Switch to active mode
    mock_storage.get_mode.return_value = FilterMode.ACTIVE
    
    # Create interceptor
    interceptor = FilterInterceptor(manager)
    
    # Create a request that matches the rule
    mock_request = MagicMock(spec=InterceptedRequest)
    mock_request.method = "GET"
    mock_request.path = "/admin/users"
    mock_request.headers = {}
    mock_request.body = b""
    mock_request.query_params = {}
    
    # The request should be blocked
    with patch.object(manager, 'evaluate_request', return_value=(True, [rule.id])):
        result = await interceptor.intercept_request(mock_request)
        assert hasattr(result, 'blocked')
        assert result.blocked is True
