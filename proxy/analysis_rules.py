"""
Analysis Rules System for Anarchy Copilot proxy module.

This module provides a rule-based engine for defining custom analysis rules
that can be applied to proxy traffic for security, performance, and behavior analysis.
"""
import logging
import re
import json
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import yaml

from .interceptor import InterceptedRequest, InterceptedResponse
from .websocket.types import WebSocketMessage
from .analysis import SecurityIssue

logger = logging.getLogger(__name__)

@dataclass
class AnalysisRule:
    """Represents a rule for analyzing traffic."""
    id: str
    name: str
    description: str
    rule_type: str  # "security", "performance", "behavior"
    priority: int  # 1-100, higher is more important
    enabled: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert rule to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "rule_type": self.rule_type,
            "priority": self.priority,
            "enabled": self.enabled,
            "conditions": self.conditions,
            "actions": self.actions,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AnalysisRule':
        """Create rule from dictionary."""
        rule_data = data.copy()
        if "created_at" in rule_data and isinstance(rule_data["created_at"], str):
            rule_data["created_at"] = datetime.fromisoformat(rule_data["created_at"])
        if "updated_at" in rule_data and isinstance(rule_data["updated_at"], str):
            rule_data["updated_at"] = datetime.fromisoformat(rule_data["updated_at"])
        return cls(**rule_data)

class RuleConditionEvaluator:
    """Evaluates rule conditions against traffic data."""
    
    def __init__(self):
        """Initialize the condition evaluator."""
        self._operators = {
            "equals": lambda a, b: a == b,
            "not_equals": lambda a, b: a != b,
            "contains": lambda a, b: b in a if isinstance(a, (str, list, dict)) else False,
            "not_contains": lambda a, b: b not in a if isinstance(a, (str, list, dict)) else True,
            "starts_with": lambda a, b: a.startswith(b) if isinstance(a, str) else False,
            "ends_with": lambda a, b: a.endswith(b) if isinstance(a, str) else False,
            "matches": lambda a, b: bool(re.search(b, a)) if isinstance(a, str) else False,
            "greater_than": lambda a, b: a > b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else False,
            "less_than": lambda a, b: a < b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else False,
            "in_range": lambda a, b: b[0] <= a <= b[1] if isinstance(a, (int, float)) and isinstance(b, list) and len(b) == 2 else False,
            "exists": lambda a, _: a is not None,
            "not_exists": lambda a, _: a is None
        }
    
    def evaluate(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Evaluate a condition against data."""
        if "type" not in condition:
            logger.warning("Condition missing 'type' field")
            return False
            
        condition_type = condition["type"]
        
        if condition_type == "and":
            if "conditions" not in condition or not isinstance(condition["conditions"], list):
                return False
            return all(self.evaluate(subcond, data) for subcond in condition["conditions"])
            
        elif condition_type == "or":
            if "conditions" not in condition or not isinstance(condition["conditions"], list):
                return False
            return any(self.evaluate(subcond, data) for subcond in condition["conditions"])
            
        elif condition_type == "not":
            if "condition" not in condition:
                return False
            return not self.evaluate(condition["condition"], data)
            
        elif condition_type == "field":
            if "field" not in condition or "operator" not in condition or "value" not in condition:
                return False
                
            field_path = condition["field"]
            operator = condition["operator"]
            expected_value = condition["value"]
            
            # Extract field value using path
            field_value = self._get_field_value(data, field_path)
            
            # Apply operator
            if operator in self._operators:
                return self._operators[operator](field_value, expected_value)
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False
                
        else:
            logger.warning(f"Unknown condition type: {condition_type}")
            return False
    
    def _get_field_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get field value using dot notation path."""
        if not field_path:
            return None
            
        parts = field_path.split(".")
        current = data
        
        for part in parts:
            # Handle array indexing
            if "[" in part and part.endswith("]"):
                array_name, index_str = part.split("[", 1)
                index = int(index_str[:-1])
                
                if array_name not in current or not isinstance(current[array_name], list):
                    return None
                    
                if index >= len(current[array_name]):
                    return None
                    
                current = current[array_name][index]
            else:
                if part not in current:
                    return None
                current = current[part]
                
        return current

class RuleActionExecutor:
    """Executes rule actions based on matched conditions."""
    
    def __init__(self):
        """Initialize the action executor."""
        pass
    
    async def execute(self, action: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action based on data."""
        if "type" not in action:
            logger.warning("Action missing 'type' field")
            return {}
            
        action_type = action["type"]
        result = {}
        
        if action_type == "create_security_issue":
            # Create a security issue
            if all(k in action for k in ["severity", "type", "description"]):
                issue = SecurityIssue(
                    severity=action["severity"],
                    type=action["type"],
                    description=action["description"],
                    evidence=action.get("evidence", "Rule-based detection"),
                    request_id=data.get("request_id", "unknown")
                )
                result["security_issue"] = issue
                
        elif action_type == "log_event":
            # Log an event
            if "message" in action:
                level = action.get("level", "info").lower()
                message = action["message"]
                
                if level == "debug":
                    logger.debug(message)
                elif level == "info":
                    logger.info(message)
                elif level == "warning":
                    logger.warning(message)
                elif level == "error":
                    logger.error(message)
                else:
                    logger.info(message)
                    
                result["logged"] = True
                
        elif action_type == "tag_traffic":
            # Add tags to traffic
            if "tags" in action and isinstance(action["tags"], list):
                result["tags"] = action["tags"]
                
        elif action_type == "set_metadata":
            # Set metadata fields
            if "fields" in action and isinstance(action["fields"], dict):
                result["metadata"] = action["fields"]
                
        else:
            logger.warning(f"Unknown action type: {action_type}")
            
        return result

class AnalysisRulesEngine:
    """Rule-based engine for traffic analysis."""
    
    def __init__(self):
        """Initialize the rules engine."""
        self._rules: Dict[str, AnalysisRule] = {}
        self._condition_evaluator = RuleConditionEvaluator()
        self._action_executor = RuleActionExecutor()
    
    def add_rule(self, rule: AnalysisRule) -> bool:
        """Add a rule to the engine."""
        if not rule.id:
            logger.warning("Cannot add rule without ID")
            return False
            
        self._rules[rule.id] = rule
        return True
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the engine."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[AnalysisRule]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)
    
    def get_all_rules(self) -> List[AnalysisRule]:
        """Get all rules."""
        return list(self._rules.values())
    
    def get_rules_by_type(self, rule_type: str) -> List[AnalysisRule]:
        """Get rules by type."""
        return [rule for rule in self._rules.values() if rule.rule_type == rule_type]
    
    async def evaluate_rules(self, data: Dict[str, Any], rule_type: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate rules against data."""
        results = {
            "matched_rules": [],
            "actions": [],
            "security_issues": [],
            "tags": set(),
            "metadata": {}
        }
        
        # Get rules to evaluate
        rules = self.get_rules_by_type(rule_type) if rule_type else self.get_all_rules()
        
        # Sort by priority (higher first)
        rules.sort(key=lambda r: r.priority, reverse=True)
        
        # Evaluate each rule
        for rule in rules:
            if not rule.enabled:
                continue
                
            # Evaluate conditions
            if self._condition_evaluator.evaluate(rule.conditions, data):
                results["matched_rules"].append(rule.id)
                
                # Execute actions
                for action in rule.actions:
                    action_result = await self._action_executor.execute(action, data)
                    results["actions"].append(action_result)
                    
                    # Collect security issues
                    if "security_issue" in action_result:
                        results["security_issues"].append(action_result["security_issue"])
                        
                    # Collect tags
                    if "tags" in action_result:
                        results["tags"].update(action_result["tags"])
                        
                    # Collect metadata
                    if "metadata" in action_result:
                        results["metadata"].update(action_result["metadata"])
        
        # Convert tags to list
        results["tags"] = list(results["tags"])
        
        return results
    
    async def evaluate_request(self, request: InterceptedRequest) -> Dict[str, Any]:
        """Evaluate rules against a request."""
        # Convert request to dictionary
        data = {
            "request_id": request.id,
            "method": request.method,
            "path": request.path,
            "headers": dict(request.headers),
            "query_params": request.query_params,
            "body": request.body,
            "timestamp": datetime.now().isoformat()
        }
        
        return await self.evaluate_rules(data, "security")
    
    async def evaluate_response(self, response: InterceptedResponse, request: InterceptedRequest) -> Dict[str, Any]:
        """Evaluate rules against a response."""
        # Convert response to dictionary
        data = {
            "request_id": request.id,
            "request_method": request.method,
            "request_path": request.path,
            "request_headers": dict(request.headers),
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response.body,
            "timestamp": datetime.now().isoformat()
        }
        
        return await self.evaluate_rules(data, "security")
    
    async def evaluate_websocket_message(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Evaluate rules against a WebSocket message."""
        # Convert message to dictionary
        data = {
            "message_id": message.id,
            "connection_id": message.connection_id,
            "direction": message.direction,
            "content": message.content,
            "timestamp": datetime.now().isoformat()
        }
        
        return await self.evaluate_rules(data, "security")
    
    def load_rules_from_file(self, file_path: str) -> int:
        """Load rules from a YAML or JSON file."""
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    rules_data = yaml.safe_load(f)
                else:
                    rules_data = json.load(f)
                    
            if not isinstance(rules_data, list):
                logger.error("Rules file must contain a list of rules")
                return 0
                
            count = 0
            for rule_data in rules_data:
                try:
                    rule = AnalysisRule.from_dict(rule_data)
                    if self.add_rule(rule):
                        count += 1
                except Exception as e:
                    logger.error(f"Error loading rule: {e}")
                    
            return count
            
        except Exception as e:
            logger.error(f"Error loading rules file: {e}")
            return 0
    
    def save_rules_to_file(self, file_path: str) -> bool:
        """Save rules to a YAML or JSON file."""
        try:
            rules_data = [rule.to_dict() for rule in self.get_all_rules()]
            
            with open(file_path, 'w') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.dump(rules_data, f, default_flow_style=False)
                else:
                    json.dump(rules_data, f, indent=2)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error saving rules file: {e}")
            return False
