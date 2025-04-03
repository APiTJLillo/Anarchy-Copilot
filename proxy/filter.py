"""
Advanced filtering system with bypass mode and post-analysis filter addition capabilities.

This module provides a comprehensive filtering system that allows users to:
1. Filter HTTP requests and responses based on flexible conditions
2. Run in bypass mode to observe traffic without filtering
3. Add specific traffic to filters after analysis
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union, Tuple

import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from proxy.interceptor import ProxyInterceptor, InterceptedRequest, InterceptedResponse
from database import AsyncSessionLocal

logger = logging.getLogger(__name__)

class FilterMode(Enum):
    """Enum representing the filtering mode."""
    ACTIVE = auto()  # Filter is active and will block matching traffic
    BYPASS = auto()  # Filter is in bypass mode, only recording but not blocking traffic


class FilterCondition:
    """Class representing a filter condition."""
    
    def __init__(self, field: str, operator: str, value: Any):
        """Initialize a filter condition.
        
        Args:
            field: The field to match against (e.g., "request.path", "response.status_code")
            operator: The operator to use (e.g., "equals", "contains", "regex")
            value: The value to compare against
        """
        self.field = field
        self.operator = operator
        self.value = value
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the condition to a dictionary."""
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FilterCondition':
        """Create a condition from a dictionary."""
        return cls(
            field=data["field"],
            operator=data["operator"],
            value=data["value"]
        )


class FilterRule:
    """Class representing a filter rule with flexible condition matching."""
    
    def __init__(
        self,
        id: Optional[str] = None,
        name: str = "",
        description: str = "",
        conditions: List[FilterCondition] = None,
        enabled: bool = True,
        priority: int = 0,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        tags: List[str] = None
    ):
        """Initialize a filter rule.
        
        Args:
            id: Unique identifier for the rule
            name: Human-readable name for the rule
            description: Detailed description of the rule
            conditions: List of conditions that must be met for the rule to match
            enabled: Whether the rule is enabled
            priority: Priority of the rule (higher values have higher priority)
            created_at: When the rule was created
            updated_at: When the rule was last updated
            tags: List of tags for categorizing the rule
        """
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.conditions = conditions or []
        self.enabled = enabled
        self.priority = priority
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.tags = tags or []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the rule to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "enabled": self.enabled,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FilterRule':
        """Create a rule from a dictionary."""
        return cls(
            id=data.get("id"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            conditions=[FilterCondition.from_dict(c) for c in data.get("conditions", [])],
            enabled=data.get("enabled", True),
            priority=data.get("priority", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else None,
            tags=data.get("tags", [])
        )


class FilterStorage(ABC):
    """Abstract base class for filter rule storage."""
    
    @abstractmethod
    async def get_rule(self, rule_id: str) -> Optional[FilterRule]:
        """Get a rule by ID."""
        pass
        
    @abstractmethod
    async def get_all_rules(self) -> List[FilterRule]:
        """Get all rules."""
        pass
        
    @abstractmethod
    async def add_rule(self, rule: FilterRule) -> bool:
        """Add a new rule."""
        pass
        
    @abstractmethod
    async def update_rule(self, rule: FilterRule) -> bool:
        """Update an existing rule."""
        pass
        
    @abstractmethod
    async def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by ID."""
        pass
        
    @abstractmethod
    async def get_mode(self) -> FilterMode:
        """Get the current filter mode."""
        pass
        
    @abstractmethod
    async def set_mode(self, mode: FilterMode) -> bool:
        """Set the filter mode."""
        pass


class DatabaseFilterStorage(FilterStorage):
    """Implementation of filter storage using the database."""
    
    async def get_rule(self, rule_id: str) -> Optional[FilterRule]:
        """Get a rule by ID from the database."""
        try:
            async with AsyncSessionLocal() as db:
                async with db.begin():
                    result = await db.execute(
                        text("SELECT * FROM filter_rules WHERE id = :id"),
                        {"id": rule_id}
                    )
                    row = result.fetchone()
                    if row:
                        # Convert row to dict
                        rule_dict = {column: value for column, value in zip(result.keys(), row)}
                        # Parse JSON fields
                        rule_dict["conditions"] = json.loads(rule_dict["conditions"])
                        rule_dict["tags"] = json.loads(rule_dict["tags"])
                        return FilterRule.from_dict(rule_dict)
            return None
        except Exception as e:
            logger.error(f"Error getting filter rule: {e}")
            return None
            
    async def get_all_rules(self) -> List[FilterRule]:
        """Get all rules from the database."""
        try:
            async with AsyncSessionLocal() as db:
                async with db.begin():
                    result = await db.execute(text("SELECT * FROM filter_rules"))
                    rules = []
                    for row in result.fetchall():
                        # Convert row to dict
                        rule_dict = {column: value for column, value in zip(result.keys(), row)}
                        # Parse JSON fields
                        rule_dict["conditions"] = json.loads(rule_dict["conditions"])
                        rule_dict["tags"] = json.loads(rule_dict["tags"])
                        rules.append(FilterRule.from_dict(rule_dict))
            return rules
        except Exception as e:
            logger.error(f"Error getting all filter rules: {e}")
            return []
            
    async def add_rule(self, rule: FilterRule) -> bool:
        """Add a new rule to the database."""
        try:
            async with AsyncSessionLocal() as db:
                async with db.begin():
                    await db.execute(
                        text("""
                            INSERT INTO filter_rules (
                                id, name, description, conditions, enabled,
                                priority, created_at, updated_at, tags
                            ) VALUES (
                                :id, :name, :description, :conditions, :enabled,
                                :priority, :created_at, :updated_at, :tags
                            )
                        """),
                        {
                            "id": rule.id,
                            "name": rule.name,
                            "description": rule.description,
                            "conditions": json.dumps([c.to_dict() for c in rule.conditions]),
                            "enabled": rule.enabled,
                            "priority": rule.priority,
                            "created_at": rule.created_at,
                            "updated_at": rule.updated_at,
                            "tags": json.dumps(rule.tags)
                        }
                    )
            return True
        except Exception as e:
            logger.error(f"Error adding filter rule: {e}")
            return False
            
    async def update_rule(self, rule: FilterRule) -> bool:
        """Update an existing rule in the database."""
        try:
            rule.updated_at = datetime.now()
            async with AsyncSessionLocal() as db:
                async with db.begin():
                    await db.execute(
                        text("""
                            UPDATE filter_rules SET
                                name = :name,
                                description = :description,
                                conditions = :conditions,
                                enabled = :enabled,
                                priority = :priority,
                                updated_at = :updated_at,
                                tags = :tags
                            WHERE id = :id
                        """),
                        {
                            "id": rule.id,
                            "name": rule.name,
                            "description": rule.description,
                            "conditions": json.dumps([c.to_dict() for c in rule.conditions]),
                            "enabled": rule.enabled,
                            "priority": rule.priority,
                            "updated_at": rule.updated_at,
                            "tags": json.dumps(rule.tags)
                        }
                    )
            return True
        except Exception as e:
            logger.error(f"Error updating filter rule: {e}")
            return False
            
    async def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by ID from the database."""
        try:
            async with AsyncSessionLocal() as db:
                async with db.begin():
                    await db.execute(
                        text("DELETE FROM filter_rules WHERE id = :id"),
                        {"id": rule_id}
                    )
            return True
        except Exception as e:
            logger.error(f"Error deleting filter rule: {e}")
            return False
            
    async def get_mode(self) -> FilterMode:
        """Get the current filter mode from the database."""
        try:
            async with AsyncSessionLocal() as db:
                async with db.begin():
                    result = await db.execute(text("SELECT value FROM filter_settings WHERE key = 'mode'"))
                    row = result.fetchone()
                    if row and row[0]:
                        return FilterMode[row[0]]
            # Default to ACTIVE if not found
            return FilterMode.ACTIVE
        except Exception as e:
            logger.error(f"Error getting filter mode: {e}")
            return FilterMode.ACTIVE
            
    async def set_mode(self, mode: FilterMode) -> bool:
        """Set the filter mode in the database."""
        try:
            async with AsyncSessionLocal() as db:
                async with db.begin():
                    # Use upsert pattern
                    await db.execute(
                        text("""
                            INSERT INTO filter_settings (key, value)
                            VALUES ('mode', :mode)
                            ON CONFLICT (key) DO UPDATE SET value = :mode
                        """),
                        {"mode": mode.name}
                    )
            return True
        except Exception as e:
            logger.error(f"Error setting filter mode: {e}")
            return False


class FileFilterStorage(FilterStorage):
    """Implementation of filter storage using files."""
    
    def __init__(self, rules_file: str = "filter_rules.json", settings_file: str = "filter_settings.json"):
        """Initialize file-based filter storage.
        
        Args:
            rules_file: Path to the rules file
            settings_file: Path to the settings file
        """
        self.rules_file = rules_file
        self.settings_file = settings_file
        
    async def get_rule(self, rule_id: str) -> Optional[FilterRule]:
        """Get a rule by ID from the file."""
        rules = await self.get_all_rules()
        for rule in rules:
            if rule.id == rule_id:
                return rule
        return None
        
    async def get_all_rules(self) -> List[FilterRule]:
        """Get all rules from the file."""
        try:
            with open(self.rules_file, 'r') as f:
                if self.rules_file.endswith('.yaml') or self.rules_file.endswith('.yml'):
                    rules_data = yaml.safe_load(f) or []
                else:
                    rules_data = json.load(f) or []
                    
            return [FilterRule.from_dict(rule_data) for rule_data in rules_data]
        except FileNotFoundError:
            # Return empty list if file doesn't exist
            return []
        except Exception as e:
            logger.error(f"Error loading filter rules from file: {e}")
            return []
            
    async def add_rule(self, rule: FilterRule) -> bool:
        """Add a new rule to the file."""
        rules = await self.get_all_rules()
        # Check if rule with same ID already exists
        for i, existing_rule in enumerate(rules):
            if existing_rule.id == rule.id:
                return False
                
        rules.append(rule)
        return await self._save_rules(rules)
            
    async def update_rule(self, rule: FilterRule) -> bool:
        """Update an existing rule in the file."""
        rules = await self.get_all_rules()
        # Find and update the rule
        for i, existing_rule in enumerate(rules):
            if existing_rule.id == rule.id:
                rule.updated_at = datetime.now()
                rules[i] = rule
                return await self._save_rules(rules)
                
        return False
            
    async def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by ID from the file."""
        rules = await self.get_all_rules()
        # Find and remove the rule
        for i, rule in enumerate(rules):
            if rule.id == rule_id:
                rules.pop(i)
                return await self._save_rules(rules)
                
        return False
            
    async def get_mode(self) -> FilterMode:
        """Get the current filter mode from the file."""
        try:
            with open(self.settings_file, 'r') as f:
                if self.settings_file.endswith('.yaml') or self.settings_file.endswith('.yml'):
                    settings = yaml.safe_load(f) or {}
                else:
                    settings = json.load(f) or {}
                    
            mode_str = settings.get("mode", "ACTIVE")
            return FilterMode[mode_str]
        except FileNotFoundError:
            # Default to ACTIVE if file doesn't exist
            return FilterMode.ACTIVE
        except Exception as e:
            logger.error(f"Error loading filter settings from file: {e}")
            return FilterMode.ACTIVE
            
    async def set_mode(self, mode: FilterMode) -> bool:
        """Set the filter mode in the file."""
        try:
            # Load existing settings
            try:
                with open(self.settings_file, 'r') as f:
                    if self.settings_file.endswith('.yaml') or self.settings_file.endswith('.yml'):
                        settings = yaml.safe_load(f) or {}
                    else:
                        settings = json.load(f) or {}
            except FileNotFoundError:
                settings = {}
                
            # Update mode
            settings["mode"] = mode.name
            
            # Save settings
            with open(self.settings_file, 'w') as f:
                if self.settings_file.endswith('.yaml') or self.settings_file.endswith('.yml'):
                    yaml.dump(settings, f, default_flow_style=False)
                else:
                    json.dump(settings, f, indent=2)
                    
            return True
        except Exception as e:
            logger.error(f"Error saving filter settings to file: {e}")
            return False
            
    async def _save_rules(self, rules: List[FilterRule]) -> bool:
        """Save rules to the file."""
        try:
            rules_data = [rule.to_dict() for rule in rules]
            
            with open(self.rules_file, 'w') as f:
                if self.rules_file.endswith('.yaml') or self.rules_file.endswith('.yml'):
                    yaml.dump(rules_data, f, default_flow_style=False)
                else:
                    json.dump(rules_data, f, indent=2)
                    
            return True
        except Exception as e:
            logger.error(f"Error saving filter rules to file: {e}")
            return False


class ConditionEvaluator:
    """Class for evaluating filter conditions."""
    
    def evaluate(self, condition: FilterCondition, data: Dict[str, Any]) -> bool:
        """Evaluate a single condition against data.
        
        Args:
            condition: The condition to evaluate
            data: The data to evaluate against
            
        Returns:
            True if the condition matches, False otherwise
        """
        # Extract field value using dot notation
        field_parts = condition.field.split('.')
        value = data
        for part in field_parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                # Field not found
                return False
                
        # Evaluate based on operator
        if condition.operator == "equals":
            return value == condition.value
        elif condition.operator == "not_equals":
            return value != condition.value
        elif condition.operator == "contains":
            return condition.value in str(value)
        elif condition.operator == "not_contains":
            return condition.value not in str(value)
        elif condition.operator == "starts_with":
            return str(value).startswith(condition.value)
        elif condition.operator == "ends_with":
            return str(value).endswith(condition.value)
        elif condition.operator == "regex":
            import re
            return bool(re.search(condition.value, str(value)))
        elif condition.operator == "greater_than":
            return float(value) > float(condition.value)
        elif condition.operator == "less_than":
            return float(value) < float(condition.value)
        elif condition.operator == "in_list":
            return value in condition.value
        elif condition.operator == "not_in_list":
            return value not in condition.value
        else:
            logger.warning(f"Unknown operator: {condition.operator}")
            return False
            
    def evaluate_all(self, conditions: List[FilterCondition], data: Dict[str, Any]) -> bool:
        """Evaluate all conditions against data (AND logic).
        
        Args:
            conditions: List of conditions to evaluate
            data: The data to evaluate against
            
        Returns:
            True if all conditions match, False otherwise
        """
        if not conditions:
            return False
            
        return all(self.evaluate(condition, data) for condition in conditions)


class FilterManager:
    """Class to centrally manage filtering rules."""
    
    def __init__(self, storage: FilterStorage):
        """Initialize the filter manager.
        
        Args:
            storage: The storage implementation to use
        """
        self.storage = storage
        self.condition_evaluator = ConditionEvaluator()
        self._traffic_history: List[Dict[str, Any]] = []
        self._max_history_size = 1000  # Maximum number of entries to keep in memory
        
    async def get_rule(self, rule_id: str) -> Optional[FilterRule]:
        """Get a rule by ID."""
        return await self.storage.get_rule(rule_id)
        
    async def get_all_rules(self) -> List[FilterRule]:
        """Get all rules."""
        return await self.storage.get_all_rules()
        
    async def add_rule(self, rule: FilterRule) -> bool:
        """Add a new rule."""
        return await self.storage.add_rule(rule)
        
    async def update_rule(self, rule: FilterRule) -> bool:
        """Update an existing rule."""
        return await self.storage.update_rule(rule)
        
    async def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by ID."""
        return await self.storage.delete_rule(rule_id)
        
    async def get_mode(self) -> FilterMode:
        """Get the current filter mode."""
        return await self.storage.get_mode()
        
    async def set_mode(self, mode: FilterMode) -> bool:
        """Set the filter mode."""
        return await self.storage.set_mode(mode)
        
    async def evaluate_request(self, request: InterceptedRequest) -> Tuple[bool, List[str]]:
        """Evaluate if a request should be filtered.
        
        Args:
            request: The intercepted request
            
        Returns:
            Tuple of (should_filter, matched_rule_ids)
        """
        # Check if in bypass mode
        if await self.get_mode() == FilterMode.BYPASS:
            # Record the request for post-analysis
            await self.record_traffic({
                "type": "request",
                "method": request.method,
                "path": request.path,
                "headers": dict(request.headers),
                "query_params": request.query_params,
                "body": request.body.decode('utf-8', errors='ignore') if request.body else None,
                "timestamp": datetime.now().isoformat()
            })
            return False, []
            
        # Get all rules
        rules = await self.get_all_rules()
        
        # Sort by priority (higher first)
        rules.sort(key=lambda r: r.priority, reverse=True)
        
        # Convert request to dictionary for evaluation
        request_data = {
            "method": request.method,
            "path": request.path,
            "headers": dict(request.headers),
            "query_params": request.query_params,
            "body": request.body.decode('utf-8', errors='ignore') if request.body else None
        }
        
        # Evaluate each rule
        matched_rule_ids = []
        for rule in rules:
            if not rule.enabled:
                continue
                
            if self.condition_evaluator.evaluate_all(rule.conditions, request_data):
                matched_rule_ids.append(rule.id)
                
        # Record the request and evaluation result for post-analysis
        await self.record_traffic({
            "type": "request",
            "method": request.method,
            "path": request.path,
            "headers": dict(request.headers),
            "query_params": request.query_params,
            "body": request.body.decode('utf-8', errors='ignore') if request.body else None,
            "timestamp": datetime.now().isoformat(),
            "matched_rules": matched_rule_ids,
            "filtered": bool(matched_rule_ids)
        })
        
        return bool(matched_rule_ids), matched_rule_ids
        
    async def evaluate_response(self, response: InterceptedResponse, request: InterceptedRequest) -> Tuple[bool, List[str]]:
        """Evaluate if a response should be filtered.
        
        Args:
            response: The intercepted response
            request: The original request
            
        Returns:
            Tuple of (should_filter, matched_rule_ids)
        """
        # Check if in bypass mode
        if await self.get_mode() == FilterMode.BYPASS:
            # Record the response for post-analysis
            await self.record_traffic({
                "type": "response",
                "request_method": request.method,
                "request_path": request.path,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.body.decode('utf-8', errors='ignore') if response.body else None,
                "timestamp": datetime.now().isoformat()
            })
            return False, []
            
        # Get all rules
        rules = await self.get_all_rules()
        
        # Sort by priority (higher first)
        rules.sort(key=lambda r: r.priority, reverse=True)
        
        # Convert response to dictionary for evaluation
        response_data = {
            "request_method": request.method,
            "request_path": request.path,
            "request_headers": dict(request.headers),
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response.body.decode('utf-8', errors='ignore') if response.body else None
        }
        
        # Evaluate each rule
        matched_rule_ids = []
        for rule in rules:
            if not rule.enabled:
                continue
                
            if self.condition_evaluator.evaluate_all(rule.conditions, response_data):
                matched_rule_ids.append(rule.id)
                
        # Record the response and evaluation result for post-analysis
        await self.record_traffic({
            "type": "response",
            "request_method": request.method,
            "request_path": request.path,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response.body.decode('utf-8', errors='ignore') if response.body else None,
            "timestamp": datetime.now().isoformat(),
            "matched_rules": matched_rule_ids,
            "filtered": bool(matched_rule_ids)
        })
        
        return bool(matched_rule_ids), matched_rule_ids
        
    async def record_traffic(self, traffic_data: Dict[str, Any]) -> None:
        """Record traffic for post-analysis.
        
        Args:
            traffic_data: The traffic data to record
        """
        # Add to in-memory history
        self._traffic_history.append(traffic_data)
        
        # Trim history if it gets too large
        if len(self._traffic_history) > self._max_history_size:
            self._traffic_history = self._traffic_history[-self._max_history_size:]
            
        # TODO: Optionally persist to database
        
    def get_traffic_history(self) -> List[Dict[str, Any]]:
        """Get the recorded traffic history.
        
        Returns:
            List of traffic history entries
        """
        return self._traffic_history
        
    def clear_traffic_history(self) -> None:
        """Clear the traffic history."""
        self._traffic_history = []
        
    def create_rule_from_traffic(self, traffic_id: int, name: str = "", description: str = "") -> FilterRule:
        """Create a filter rule from a traffic history entry.
        
        Args:
            traffic_id: The index of the traffic history entry
            name: Optional name for the rule
            description: Optional description for the rule
            
        Returns:
            A new FilterRule based on the traffic
            
        Raises:
            IndexError: If traffic_id is out of range
        """
        if traffic_id < 0 or traffic_id >= len(self._traffic_history):
            raise IndexError(f"Traffic ID {traffic_id} is out of range")
            
        traffic = self._traffic_history[traffic_id]
        conditions = []
        
        if traffic["type"] == "request":
            # Create conditions based on request
            conditions.append(FilterCondition("method", "equals", traffic["method"]))
            conditions.append(FilterCondition("path", "equals", traffic["path"]))
            
            # Add some headers as conditions
            for header in ["User-Agent", "Content-Type", "Authorization"]:
                if header in traffic["headers"]:
                    conditions.append(FilterCondition(f"headers.{header}", "equals", traffic["headers"][header]))
                    
        elif traffic["type"] == "response":
            # Create conditions based on response
            conditions.append(FilterCondition("request_method", "equals", traffic["request_method"]))
            conditions.append(FilterCondition("request_path", "equals", traffic["request_path"]))
            conditions.append(FilterCondition("status_code", "equals", traffic["status_code"]))
            
            # Add some headers as conditions
            for header in ["Content-Type", "Server"]:
                if header in traffic["headers"]:
                    conditions.append(FilterCondition(f"headers.{header}", "equals", traffic["headers"][header]))
                    
        # Create the rule
        return FilterRule(
            name=name or f"Rule from {traffic['type']} to {traffic.get('path', traffic.get('request_path', ''))}",
            description=description or f"Automatically generated from {traffic['type']} traffic",
            conditions=conditions,
            tags=[traffic["type"], "auto-generated"]
        )
        
    def suggest_conditions(self, traffic_ids: List[int]) -> List[FilterCondition]:
        """Suggest filter conditions based on multiple traffic history entries.
        
        Args:
            traffic_ids: List of traffic history entry indices
            
        Returns:
            List of suggested FilterCondition objects
            
        Raises:
            IndexError: If any traffic_id is out of range
        """
        if not traffic_ids:
            return []
            
        # Validate traffic IDs
        for traffic_id in traffic_ids:
            if traffic_id < 0 or traffic_id >= len(self._traffic_history):
                raise IndexError(f"Traffic ID {traffic_id} is out of range")
                
        # Get traffic entries
        traffic_entries = [self._traffic_history[i] for i in traffic_ids]
        
        # Group by type
        requests = [t for t in traffic_entries if t["type"] == "request"]
        responses = [t for t in traffic_entries if t["type"] == "response"]
        
        suggested_conditions = []
        
        # Analyze requests
        if requests:
            # Find common method
            methods = set(r["method"] for r in requests)
            if len(methods) == 1:
                suggested_conditions.append(FilterCondition("method", "equals", next(iter(methods))))
                
            # Find common path patterns
            paths = [r["path"] for r in requests]
            if len(set(paths)) == 1:
                # Exact match
                suggested_conditions.append(FilterCondition("path", "equals", paths[0]))
            else:
                # Look for common prefixes
                common_prefix = os.path.commonprefix(paths)
                if common_prefix and len(common_prefix) > 1:
                    suggested_conditions.append(FilterCondition("path", "starts_with", common_prefix))
                    
            # Find common headers
            common_headers = self._find_common_headers(requests, "headers")
            for header, value in common_headers.items():
                suggested_conditions.append(FilterCondition(f"headers.{header}", "equals", value))
                
        # Analyze responses
        if responses:
            # Find common status codes
            status_codes = set(r["status_code"] for r in responses)
            if len(status_codes) == 1:
                suggested_conditions.append(FilterCondition("status_code", "equals", next(iter(status_codes))))
                
            # Find common request paths
            paths = [r["request_path"] for r in responses]
            if len(set(paths)) == 1:
                # Exact match
                suggested_conditions.append(FilterCondition("request_path", "equals", paths[0]))
            else:
                # Look for common prefixes
                common_prefix = os.path.commonprefix(paths)
                if common_prefix and len(common_prefix) > 1:
                    suggested_conditions.append(FilterCondition("request_path", "starts_with", common_prefix))
                    
            # Find common headers
            common_headers = self._find_common_headers(responses, "headers")
            for header, value in common_headers.items():
                suggested_conditions.append(FilterCondition(f"headers.{header}", "equals", value))
                
        return suggested_conditions
        
    def _find_common_headers(self, entries: List[Dict[str, Any]], headers_key: str) -> Dict[str, str]:
        """Find headers that are common across all entries.
        
        Args:
            entries: List of traffic entries
            headers_key: The key for headers in the entries
            
        Returns:
            Dictionary of common headers and their values
        """
        if not entries:
            return {}
            
        # Get all headers from first entry
        common_headers = entries[0][headers_key].copy()
        
        # Intersect with headers from other entries
        for entry in entries[1:]:
            headers = entry[headers_key]
            # Remove headers that don't exist in this entry
            for header in list(common_headers.keys()):
                if header not in headers or headers[header] != common_headers[header]:
                    common_headers.pop(header)
                    
        return common_headers


class FilterInterceptor(ProxyInterceptor):
    """Interceptor that integrates with the proxy system to filter traffic."""
    
    def __init__(self, filter_manager: FilterManager, connection_id: Optional[str] = None):
        """Initialize the filter interceptor.
        
        Args:
            filter_manager: The filter manager to use
            connection_id: Optional unique ID for the connection
        """
        super().__init__(connection_id)
        self.filter_manager = filter_manager
        
    async def intercept_request(self, request: InterceptedRequest) -> InterceptedRequest:
        """Process an intercepted request.
        
        Args:
            request: The intercepted HTTP request
            
        Returns:
            The modified request or a blocked response
        """
        # Evaluate if request should be filtered
        should_filter, matched_rule_ids = await self.filter_manager.evaluate_request(request)
        
        if should_filter:
            # Create a blocked response
            logger.info(f"Request filtered: {request.method} {request.path} (matched rules: {matched_rule_ids})")
            
            # Convert the request to a blocked response
            blocked_response = InterceptedResponse(
                status_code=403,
                status="Forbidden",
                headers={
                    "Content-Type": "application/json",
                    "X-Filtered-By": ",".join(matched_rule_ids)
                },
                body=json.dumps({
                    "error": "Request blocked by filter",
                    "matched_rules": matched_rule_ids
                }).encode('utf-8')
            )
            
            # Set a flag on the request to indicate it should be blocked
            request.blocked = True
            request.blocked_response = blocked_response
            
        return request
        
    async def intercept_response(self, response: InterceptedResponse, request: InterceptedRequest) -> InterceptedResponse:
        """Process an intercepted response.
        
        Args:
            response: The intercepted HTTP response
            request: The original request that generated this response
            
        Returns:
            The modified response
        """
        # Check if request was already blocked
        if hasattr(request, 'blocked') and request.blocked:
            return request.blocked_response
            
        # Evaluate if response should be filtered
        should_filter, matched_rule_ids = await self.filter_manager.evaluate_response(response, request)
        
        if should_filter:
            # Create a blocked response
            logger.info(f"Response filtered: {request.method} {request.path} -> {response.status_code} (matched rules: {matched_rule_ids})")
            
            # Replace with a blocked response
            blocked_response = InterceptedResponse(
                status_code=403,
                status="Forbidden",
                headers={
                    "Content-Type": "application/json",
                    "X-Filtered-By": ",".join(matched_rule_ids)
                },
                body=json.dumps({
                    "error": "Response blocked by filter",
                    "matched_rules": matched_rule_ids
                }).encode('utf-8')
            )
            
            return blocked_response
            
        return response
