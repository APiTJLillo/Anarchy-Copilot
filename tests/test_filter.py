"""
Tests for the advanced filtering system with bypass mode.

This module provides tests for the filter system components, including:
- FilterMode enum
- FilterCondition class
- FilterRule class
- ConditionEvaluator class
- FilterManager class
- FilterInterceptor class
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from proxy.filter import (
    FilterMode, 
    FilterCondition, 
    FilterRule, 
    ConditionEvaluator,
    FilterManager,
    FilterInterceptor,
    FileFilterStorage
)
from proxy.interceptor import InterceptedRequest, InterceptedResponse

# Test FilterMode enum
def test_filter_mode_enum():
    assert FilterMode.ACTIVE is not None
    assert FilterMode.BYPASS is not None
    assert FilterMode.ACTIVE != FilterMode.BYPASS

# Test FilterCondition class
def test_filter_condition():
    # Create a condition
    condition = FilterCondition("path", "contains", "/admin")
    
    # Test to_dict method
    condition_dict = condition.to_dict()
    assert condition_dict["field"] == "path"
    assert condition_dict["operator"] == "contains"
    assert condition_dict["value"] == "/admin"
    
    # Test from_dict method
    new_condition = FilterCondition.from_dict(condition_dict)
    assert new_condition.field == "path"
    assert new_condition.operator == "contains"
    assert new_condition.value == "/admin"

# Test FilterRule class
def test_filter_rule():
    # Create conditions
    condition1 = FilterCondition("path", "contains", "/admin")
    condition2 = FilterCondition("method", "equals", "POST")
    
    # Create a rule
    rule = FilterRule(
        name="Test Rule",
        description="A test rule",
        conditions=[condition1, condition2],
        enabled=True,
        priority=10,
        tags=["test", "admin"]
    )
    
    # Test properties
    assert rule.id is not None
    assert rule.name == "Test Rule"
    assert rule.description == "A test rule"
    assert len(rule.conditions) == 2
    assert rule.enabled is True
    assert rule.priority == 10
    assert rule.tags == ["test", "admin"]
    
    # Test to_dict method
    rule_dict = rule.to_dict()
    assert rule_dict["name"] == "Test Rule"
    assert rule_dict["description"] == "A test rule"
    assert len(rule_dict["conditions"]) == 2
    assert rule_dict["enabled"] is True
    assert rule_dict["priority"] == 10
    assert rule_dict["tags"] == ["test", "admin"]
    
    # Test from_dict method
    new_rule = FilterRule.from_dict(rule_dict)
    assert new_rule.id == rule.id
    assert new_rule.name == "Test Rule"
    assert new_rule.description == "A test rule"
    assert len(new_rule.conditions) == 2
    assert new_rule.enabled is True
    assert new_rule.priority == 10
    assert new_rule.tags == ["test", "admin"]

# Test ConditionEvaluator class
def test_condition_evaluator():
    evaluator = ConditionEvaluator()
    
    # Test data
    data = {
        "method": "POST",
        "path": "/admin/users",
        "headers": {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        },
        "body": '{"username": "admin"}'
    }
    
    # Test different operators
    assert evaluator.evaluate(FilterCondition("method", "equals", "POST"), data) is True
    assert evaluator.evaluate(FilterCondition("method", "equals", "GET"), data) is False
    assert evaluator.evaluate(FilterCondition("method", "not_equals", "GET"), data) is True
    assert evaluator.evaluate(FilterCondition("path", "contains", "admin"), data) is True
    assert evaluator.evaluate(FilterCondition("path", "contains", "login"), data) is False
    assert evaluator.evaluate(FilterCondition("path", "starts_with", "/admin"), data) is True
    assert evaluator.evaluate(FilterCondition("path", "ends_with", "users"), data) is True
    assert evaluator.evaluate(FilterCondition("headers.Content-Type", "equals", "application/json"), data) is True
    assert evaluator.evaluate(FilterCondition("body", "contains", "username"), data) is True
    
    # Test evaluate_all method
    conditions = [
        FilterCondition("method", "equals", "POST"),
        FilterCondition("path", "contains", "admin")
    ]
    assert evaluator.evaluate_all(conditions, data) is True
    
    conditions = [
        FilterCondition("method", "equals", "POST"),
        FilterCondition("path", "contains", "login")
    ]
    assert evaluator.evaluate_all(conditions, data) is False

# Test FileFilterStorage class
@pytest.mark.asyncio
async def test_file_filter_storage(tmp_path):
    # Create temporary files
    rules_file = tmp_path / "filter_rules.json"
    settings_file = tmp_path / "filter_settings.json"
    
    # Create storage
    storage = FileFilterStorage(str(rules_file), str(settings_file))
    
    # Test get_all_rules with empty file
    rules = await storage.get_all_rules()
    assert len(rules) == 0
    
    # Test add_rule
    rule = FilterRule(
        name="Test Rule",
        description="A test rule",
        conditions=[FilterCondition("path", "contains", "/admin")],
        enabled=True,
        priority=10,
        tags=["test", "admin"]
    )
    assert await storage.add_rule(rule) is True
    
    # Test get_rule
    retrieved_rule = await storage.get_rule(rule.id)
    assert retrieved_rule is not None
    assert retrieved_rule.id == rule.id
    assert retrieved_rule.name == "Test Rule"
    
    # Test update_rule
    rule.name = "Updated Rule"
    assert await storage.update_rule(rule) is True
    
    # Test get_rule after update
    retrieved_rule = await storage.get_rule(rule.id)
    assert retrieved_rule is not None
    assert retrieved_rule.name == "Updated Rule"
    
    # Test get_mode (default)
    mode = await storage.get_mode()
    assert mode == FilterMode.ACTIVE
    
    # Test set_mode
    assert await storage.set_mode(FilterMode.BYPASS) is True
    
    # Test get_mode after set
    mode = await storage.get_mode()
    assert mode == FilterMode.BYPASS
    
    # Test delete_rule
    assert await storage.delete_rule(rule.id) is True
    
    # Test get_rule after delete
    retrieved_rule = await storage.get_rule(rule.id)
    assert retrieved_rule is None

# Test FilterManager class
@pytest.mark.asyncio
async def test_filter_manager():
    # Create mock storage
    mock_storage = AsyncMock()
    mock_storage.get_rule.return_value = None
    mock_storage.get_all_rules.return_value = []
    mock_storage.add_rule.return_value = True
    mock_storage.update_rule.return_value = True
    mock_storage.delete_rule.return_value = True
    mock_storage.get_mode.return_value = FilterMode.ACTIVE
    mock_storage.set_mode.return_value = True
    
    # Create manager
    manager = FilterManager(mock_storage)
    
    # Test get_rule
    await manager.get_rule("test-id")
    mock_storage.get_rule.assert_called_once_with("test-id")
    
    # Test get_all_rules
    await manager.get_all_rules()
    mock_storage.get_all_rules.assert_called_once()
    
    # Test add_rule
    rule = FilterRule(name="Test Rule")
    await manager.add_rule(rule)
    mock_storage.add_rule.assert_called_once()
    
    # Test update_rule
    await manager.update_rule(rule)
    mock_storage.update_rule.assert_called_once()
    
    # Test delete_rule
    await manager.delete_rule("test-id")
    mock_storage.delete_rule.assert_called_once_with("test-id")
    
    # Test get_mode
    await manager.get_mode()
    mock_storage.get_mode.assert_called_once()
    
    # Test set_mode
    await manager.set_mode(FilterMode.BYPASS)
    mock_storage.set_mode.assert_called_once_with(FilterMode.BYPASS)
    
    # Test traffic history recording
    assert len(manager.get_traffic_history()) == 0
    
    # Record some traffic
    await manager.record_traffic({"type": "request", "path": "/test"})
    assert len(manager.get_traffic_history()) == 1
    
    # Test clear_traffic_history
    manager.clear_traffic_history()
    assert len(manager.get_traffic_history()) == 0

# Test FilterInterceptor class
@pytest.mark.asyncio
async def test_filter_interceptor():
    # Create mock filter manager
    mock_manager = AsyncMock()
    mock_manager.evaluate_request.return_value = (False, [])
    mock_manager.evaluate_response.return_value = (False, [])
    
    # Create interceptor
    interceptor = FilterInterceptor(mock_manager)
    
    # Create mock request
    mock_request = MagicMock(spec=InterceptedRequest)
    mock_request.method = "GET"
    mock_request.path = "/test"
    mock_request.headers = {}
    mock_request.body = b""
    
    # Test intercept_request
    result = await interceptor.intercept_request(mock_request)
    assert result is mock_request
    mock_manager.evaluate_request.assert_called_once()
    
    # Create mock response
    mock_response = MagicMock(spec=InterceptedResponse)
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.body = b""
    
    # Test intercept_response
    result = await interceptor.intercept_response(mock_response, mock_request)
    assert result is mock_response
    mock_manager.evaluate_response.assert_called_once()
    
    # Test request blocking
    mock_manager.evaluate_request.reset_mock()
    mock_manager.evaluate_request.return_value = (True, ["rule-1"])
    
    result = await interceptor.intercept_request(mock_request)
    assert hasattr(result, 'blocked')
    assert result.blocked is True
    assert hasattr(result, 'blocked_response')
    assert result.blocked_response.status_code == 403
    
    # Test response blocking
    mock_manager.evaluate_response.reset_mock()
    mock_manager.evaluate_response.return_value = (True, ["rule-1"])
    
    result = await interceptor.intercept_response(mock_response, mock_request)
    assert result is not mock_response
    assert result.status_code == 403
