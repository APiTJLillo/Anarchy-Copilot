"""Validation for notification rules."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
import re
import jsonschema
from pathlib import Path

from .notification_rules import (
    RuleEngine, NotificationRule, RuleCondition,
    RuleAction, RuleConfig
)

@dataclass
class ValidationConfig:
    """Configuration for rule validation."""
    max_conditions: int = 10
    max_actions: int = 5
    max_rule_size: int = 10_000  # bytes
    enable_syntax_check: bool = True
    enable_schema_check: bool = True
    enable_circular_check: bool = True
    enable_performance_check: bool = True
    schema_file: Optional[Path] = None

@dataclass
class ValidationError:
    """Validation error details."""
    rule_name: str
    error_type: str
    message: str
    path: Optional[str] = None
    severity: str = "error"
    suggestion: Optional[str] = None

@dataclass
class ValidationResult:
    """Result of rule validation."""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    stats: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class RuleValidator:
    """Validator for notification rules."""
    
    # Rule schema for JSON Schema validation
    RULE_SCHEMA = {
        "type": "object",
        "required": ["name", "description", "conditions", "actions"],
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[a-zA-Z0-9_-]+$",
                "minLength": 1,
                "maxLength": 64
            },
            "description": {
                "type": "string",
                "maxLength": 256
            },
            "enabled": {"type": "boolean"},
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["field", "operator", "value"],
                    "properties": {
                        "field": {"type": "string"},
                        "operator": {
                            "type": "string",
                            "enum": [
                                "eq", "ne", "gt", "ge", "lt", "le",
                                "contains", "startswith", "endswith",
                                "matches", "exists", "type"
                            ]
                        },
                        "value": {},
                        "invert": {"type": "boolean"}
                    }
                }
            },
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["type", "params"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["notify", "set_state", "add_context", "route"]
                        },
                        "params": {
                            "type": "object"
                        }
                    }
                }
            },
            "priority": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100
            },
            "stop_processing": {"type": "boolean"}
        }
    }
    
    def __init__(
        self,
        engine: RuleEngine,
        config: ValidationConfig = None
    ):
        self.engine = engine
        self.config = config or ValidationConfig()
        
        # Load custom schema if specified
        if self.config.schema_file and self.config.schema_file.exists():
            with open(self.config.schema_file) as f:
                self.schema = json.load(f)
        else:
            self.schema = self.RULE_SCHEMA
    
    async def validate_rule(
        self,
        rule: NotificationRule
    ) -> ValidationResult:
        """Validate a single rule."""
        errors = []
        warnings = []
        stats = {
            "condition_count": len(rule.conditions),
            "action_count": len(rule.actions),
            "total_size": 0,
            "field_count": 0
        }
        
        # Check rule size
        rule_size = len(str(rule.__dict__))
        stats["total_size"] = rule_size
        if rule_size > self.config.max_rule_size:
            errors.append(ValidationError(
                rule_name=rule.name,
                error_type="size_limit",
                message=f"Rule size ({rule_size} bytes) exceeds limit ({self.config.max_rule_size} bytes)",
                severity="error",
                suggestion="Split into multiple rules or simplify conditions"
            ))
        
        # Check condition count
        if len(rule.conditions) > self.config.max_conditions:
            errors.append(ValidationError(
                rule_name=rule.name,
                error_type="condition_limit",
                message=f"Too many conditions ({len(rule.conditions)})",
                severity="error",
                suggestion=f"Limit to {self.config.max_conditions} conditions"
            ))
        
        # Check action count
        if len(rule.actions) > self.config.max_actions:
            errors.append(ValidationError(
                rule_name=rule.name,
                error_type="action_limit",
                message=f"Too many actions ({len(rule.actions)})",
                severity="error",
                suggestion=f"Limit to {self.config.max_actions} actions"
            ))
        
        # Syntax validation
        if self.config.enable_syntax_check:
            syntax_errors = await self._validate_syntax(rule)
            errors.extend(syntax_errors)
        
        # Schema validation
        if self.config.enable_schema_check:
            schema_errors = await self._validate_schema(rule)
            errors.extend(schema_errors)
        
        # Circular dependency check
        if self.config.enable_circular_check:
            circular_errors = await self._check_circular_deps(rule)
            errors.extend(circular_errors)
        
        # Performance check
        if self.config.enable_performance_check:
            perf_warnings = await self._check_performance(rule)
            warnings.extend(perf_warnings)
        
        # Field analysis
        stats["field_count"] = len(set(
            cond.field for cond in rule.conditions
        ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
    
    async def validate_all_rules(self) -> Dict[str, ValidationResult]:
        """Validate all rules in the engine."""
        results = {}
        for rule in self.engine.rules:
            results[rule.name] = await self.validate_rule(rule)
        return results
    
    async def _validate_syntax(
        self,
        rule: NotificationRule
    ) -> List[ValidationError]:
        """Validate rule syntax."""
        errors = []
        
        # Check condition syntax
        for i, condition in enumerate(rule.conditions):
            # Validate JMESPath syntax
            try:
                jmespath.compile(condition.field)
            except jmespath.exceptions.ParseError as e:
                errors.append(ValidationError(
                    rule_name=rule.name,
                    error_type="invalid_jmespath",
                    message=f"Invalid JMESPath expression: {e}",
                    path=f"conditions[{i}].field",
                    suggestion="Check JMESPath syntax"
                ))
            
            # Validate operator
            if condition.operator not in self.engine.OPERATORS:
                errors.append(ValidationError(
                    rule_name=rule.name,
                    error_type="invalid_operator",
                    message=f"Unknown operator: {condition.operator}",
                    path=f"conditions[{i}].operator",
                    suggestion=f"Use one of: {', '.join(self.engine.OPERATORS.keys())}"
                ))
        
        # Check action syntax
        for i, action in enumerate(rule.actions):
            # Validate action type
            if action.type not in {"notify", "set_state", "add_context", "route"}:
                errors.append(ValidationError(
                    rule_name=rule.name,
                    error_type="invalid_action",
                    message=f"Unknown action type: {action.type}",
                    path=f"actions[{i}].type",
                    suggestion="Use a supported action type"
                ))
            
            # Validate required parameters
            if action.type == "notify":
                if "channels" not in action.params:
                    errors.append(ValidationError(
                        rule_name=rule.name,
                        error_type="missing_param",
                        message="Missing 'channels' parameter for notify action",
                        path=f"actions[{i}].params",
                        suggestion="Add channels list to params"
                    ))
            elif action.type == "set_state":
                if "state" not in action.params:
                    errors.append(ValidationError(
                        rule_name=rule.name,
                        error_type="missing_param",
                        message="Missing 'state' parameter for set_state action",
                        path=f"actions[{i}].params",
                        suggestion="Add state parameter"
                    ))
        
        return errors
    
    async def _validate_schema(
        self,
        rule: NotificationRule
    ) -> List[ValidationError]:
        """Validate rule against JSON schema."""
        errors = []
        
        # Convert rule to dict
        rule_dict = {
            "name": rule.name,
            "description": rule.description,
            "enabled": rule.enabled,
            "conditions": [
                {
                    "field": c.field,
                    "operator": c.operator,
                    "value": c.value,
                    "invert": c.invert
                }
                for c in rule.conditions
            ],
            "actions": [
                {
                    "type": a.type,
                    "params": a.params
                }
                for a in rule.actions
            ],
            "priority": rule.priority,
            "stop_processing": rule.stop_processing
        }
        
        # Validate against schema
        try:
            jsonschema.validate(rule_dict, self.schema)
        except jsonschema.exceptions.ValidationError as e:
            errors.append(ValidationError(
                rule_name=rule.name,
                error_type="schema_error",
                message=str(e),
                path=".".join(str(p) for p in e.path),
                suggestion="Update rule to match schema"
            ))
        
        return errors
    
    async def _check_circular_deps(
        self,
        rule: NotificationRule
    ) -> List[ValidationError]:
        """Check for circular dependencies."""
        errors = []
        
        # Build dependency graph
        deps: Dict[str, Set[str]] = {}
        for action in rule.actions:
            if action.type == "route":
                channels = set(action.params.get("channels", []))
                deps[rule.name] = channels
        
        # Check for cycles
        visited = set()
        path = []
        
        def has_cycle(node: str) -> bool:
            if node in path:
                cycle = path[path.index(node):]
                errors.append(ValidationError(
                    rule_name=rule.name,
                    error_type="circular_dependency",
                    message=f"Circular dependency detected: {' -> '.join(cycle)}",
                    severity="error",
                    suggestion="Remove circular routing"
                ))
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            path.append(node)
            
            for dep in deps.get(node, set()):
                if has_cycle(dep):
                    return True
            
            path.pop()
            return False
        
        has_cycle(rule.name)
        return errors
    
    async def _check_performance(
        self,
        rule: NotificationRule
    ) -> List[ValidationError]:
        """Check for potential performance issues."""
        warnings = []
        
        # Check complex JMESPath expressions
        for i, condition in enumerate(rule.conditions):
            if condition.field.count(".") > 3:
                warnings.append(ValidationError(
                    rule_name=rule.name,
                    error_type="complex_path",
                    message=f"Complex JMESPath expression: {condition.field}",
                    path=f"conditions[{i}].field",
                    severity="warning",
                    suggestion="Consider simplifying expression"
                ))
        
        # Check regex patterns
        for i, condition in enumerate(rule.conditions):
            if condition.operator == "matches":
                pattern = str(condition.value)
                if (
                    len(pattern) > 100 or
                    pattern.count(".*") > 2 or
                    pattern.count(".*?") > 0
                ):
                    warnings.append(ValidationError(
                        rule_name=rule.name,
                        error_type="complex_regex",
                        message=f"Complex regex pattern in condition {i}",
                        path=f"conditions[{i}].value",
                        severity="warning",
                        suggestion="Simplify regex pattern"
                    ))
        
        # Check notification fanout
        for i, action in enumerate(rule.actions):
            if (
                action.type == "notify" and
                len(action.params.get("channels", [])) > 3
            ):
                warnings.append(ValidationError(
                    rule_name=rule.name,
                    error_type="high_fanout",
                    message="High notification fanout",
                    path=f"actions[{i}].params.channels",
                    severity="warning",
                    suggestion="Consider reducing number of notification channels"
                ))
        
        return warnings

def create_rule_validator(
    engine: RuleEngine,
    config: Optional[ValidationConfig] = None
) -> RuleValidator:
    """Create rule validator."""
    return RuleValidator(engine, config)

if __name__ == "__main__":
    from .notification_rules import create_rule_engine
    from .alert_notifications import create_notification_manager
    from .anomaly_alerts import create_alert_manager
    from .anomaly_analysis import create_anomaly_detector
    from .trend_analysis import create_trend_analyzer
    from .adaptation_metrics import create_performance_tracker
    from .preset_adaptation import create_online_adapter
    from .preset_ensemble import create_preset_ensemble
    from .preset_predictions import create_preset_predictor
    from .preset_analytics import create_preset_analytics
    from .mutation_presets import create_preset_manager
    
    async def main():
        # Setup components
        manager = create_preset_manager()
        analytics = create_preset_analytics(manager)
        predictor = create_preset_predictor(analytics)
        ensemble = create_preset_ensemble(predictor)
        adapter = create_online_adapter(ensemble)
        tracker = create_performance_tracker(adapter)
        analyzer = create_trend_analyzer(tracker)
        detector = create_anomaly_detector(tracker, analyzer)
        alert_manager = create_alert_manager(detector)
        notifier = create_notification_manager(alert_manager)
        engine = create_rule_engine(notifier)
        validator = create_rule_validator(engine)
        
        # Create test rules with issues
        rules = [
            NotificationRule(
                name="valid_rule",
                description="Valid test rule",
                conditions=[
                    RuleCondition(
                        field="severity",
                        operator="eq",
                        value="critical"
                    )
                ],
                actions=[
                    RuleAction(
                        type="notify",
                        params={"channels": ["slack"]}
                    )
                ]
            ),
            NotificationRule(
                name="invalid_rule",
                description="Rule with issues",
                conditions=[
                    RuleCondition(
                        field="complex.path[*].deeply.nested[?value > 10]",
                        operator="invalid_op",
                        value=".*complex.*pattern.*with.*backtracking.*"
                    )
                ],
                actions=[
                    RuleAction(
                        type="notify",
                        params={"channels": ["email", "slack", "teams", "sms", "webhook"]}
                    ),
                    RuleAction(
                        type="set_state",
                        params={}  # Missing state parameter
                    )
                ]
            )
        ]
        
        # Validate rules
        for rule in rules:
            result = await validator.validate_rule(rule)
            print(f"\nValidating {rule.name}:")
            print(f"Valid: {result.is_valid}")
            if result.errors:
                print("\nErrors:")
                for error in result.errors:
                    print(f"- {error.message}")
            if result.warnings:
                print("\nWarnings:")
                for warning in result.warnings:
                    print(f"- {warning.message}")
            print("\nStats:", result.stats)
    
    asyncio.run(main())
