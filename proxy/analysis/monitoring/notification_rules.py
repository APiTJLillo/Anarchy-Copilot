"""Rules engine for notification handling."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
from pathlib import Path
import yaml
import jmespath

from .alert_notifications import NotificationManager, AlertGroup, AlertSeverity, AlertState

@dataclass
class RuleCondition:
    """Condition for notification rule."""
    field: str
    operator: str
    value: Any
    invert: bool = False

@dataclass
class RuleAction:
    """Action for notification rule."""
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NotificationRule:
    """Rule for handling notifications."""
    name: str
    description: str
    enabled: bool = True
    conditions: List[RuleCondition] = field(default_factory=list)
    actions: List[RuleAction] = field(default_factory=list)
    priority: int = 0
    stop_processing: bool = False
    created: datetime = field(default_factory=datetime.now)
    modified: datetime = field(default_factory=datetime.now)

@dataclass
class RuleConfig:
    """Configuration for notification rules."""
    rules_dir: Path = Path("notification_rules")
    enable_caching: bool = True
    cache_duration: timedelta = timedelta(minutes=5)
    max_rules: int = 100
    default_priority: int = 50
    enable_stats: bool = True
    auto_reload: bool = True

class RuleEngine:
    """Engine for processing notification rules."""
    
    OPERATORS = {
        "eq": lambda x, y: x == y,
        "ne": lambda x, y: x != y,
        "gt": lambda x, y: x > y,
        "ge": lambda x, y: x >= y,
        "lt": lambda x, y: x < y,
        "le": lambda x, y: x <= y,
        "contains": lambda x, y: y in x if isinstance(x, (str, list, set, dict)) else False,
        "startswith": lambda x, y: x.startswith(y) if isinstance(x, str) else False,
        "endswith": lambda x, y: x.endswith(y) if isinstance(x, str) else False,
        "matches": lambda x, y: bool(re.match(y, x)) if isinstance(x, str) else False,
        "exists": lambda x, _: x is not None,
        "type": lambda x, y: isinstance(x, {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict
        }.get(y, object))
    }
    
    def __init__(
        self,
        notification_manager: NotificationManager,
        config: RuleConfig = None
    ):
        self.notification_manager = notification_manager
        self.config = config or RuleConfig()
        
        # Rule storage
        self.rules: List[NotificationRule] = []
        self.rule_cache: Dict[str, Any] = {}
        self.last_reload: datetime = datetime.min
        self.stats: Dict[str, Dict[str, int]] = {}
        
        # Create rules directory
        self.config.rules_dir.mkdir(parents=True, exist_ok=True)
        
        # Load rules
        self._load_rules()
    
    def _load_rules(self):
        """Load rules from files."""
        if not self.config.auto_reload:
            return
        
        current_time = datetime.now()
        if current_time - self.last_reload < self.config.cache_duration:
            return
        
        rules = []
        for rule_file in self.config.rules_dir.glob("*.yaml"):
            try:
                with open(rule_file) as f:
                    data = yaml.safe_load(f)
                
                if not isinstance(data, list):
                    data = [data]
                
                for rule_data in data:
                    conditions = [
                        RuleCondition(**cond)
                        for cond in rule_data.pop("conditions", [])
                    ]
                    actions = [
                        RuleAction(**action)
                        for action in rule_data.pop("actions", [])
                    ]
                    rules.append(NotificationRule(
                        conditions=conditions,
                        actions=actions,
                        **rule_data
                    ))
            except Exception as e:
                print(f"Failed to load rule file {rule_file}: {e}")
        
        # Sort rules by priority
        rules.sort(key=lambda r: r.priority, reverse=True)
        
        # Trim excess rules
        if len(rules) > self.config.max_rules:
            rules = rules[:self.config.max_rules]
        
        self.rules = rules
        self.last_reload = current_time
    
    async def process_alert(
        self,
        group: AlertGroup
    ) -> bool:
        """Process alert through rules."""
        # Reload rules if needed
        self._load_rules()
        
        # Prepare alert data
        alert_data = {
            "id": group.id,
            "pattern_type": group.pattern_type,
            "severity": group.severity.value,
            "state": group.state.value,
            "count": group.count,
            "first_seen": group.first_seen.isoformat(),
            "last_seen": group.last_seen.isoformat(),
            "correlated_groups": list(group.correlated_groups),
            "context": group.context,
            "alerts": group.alerts
        }
        
        # Cache common expressions
        if self.config.enable_caching:
            cache_key = f"{group.id}_{group.last_seen.timestamp()}"
            if cache_key in self.rule_cache:
                return self.rule_cache[cache_key]
        
        # Process rules
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Check conditions
            if await self._evaluate_conditions(rule.conditions, alert_data):
                # Execute actions
                await self._execute_actions(rule.actions, group, alert_data)
                
                # Update stats
                if self.config.enable_stats:
                    self.stats.setdefault(rule.name, {
                        "matches": 0,
                        "actions": 0
                    })
                    self.stats[rule.name]["matches"] += 1
                    self.stats[rule.name]["actions"] += len(rule.actions)
                
                # Cache result
                if self.config.enable_caching:
                    self.rule_cache[cache_key] = True
                
                if rule.stop_processing:
                    return True
        
        if self.config.enable_caching:
            self.rule_cache[cache_key] = False
        
        return False
    
    async def _evaluate_conditions(
        self,
        conditions: List[RuleCondition],
        data: Dict[str, Any]
    ) -> bool:
        """Evaluate rule conditions."""
        for condition in conditions:
            # Extract field value using JMESPath
            try:
                value = jmespath.search(condition.field, data)
            except Exception:
                value = None
            
            # Get operator function
            operator = self.OPERATORS.get(condition.operator)
            if not operator:
                continue
            
            # Evaluate condition
            try:
                result = operator(value, condition.value)
                if condition.invert:
                    result = not result
                if not result:
                    return False
            except Exception:
                return False
        
        return True
    
    async def _execute_actions(
        self,
        actions: List[RuleAction],
        group: AlertGroup,
        data: Dict[str, Any]
    ):
        """Execute rule actions."""
        for action in actions:
            try:
                if action.type == "notify":
                    # Send notifications
                    channels = action.params.get("channels", [])
                    template_vars = action.params.get("template_vars", {})
                    
                    for channel in channels:
                        if channel in self.notification_manager.channels:
                            await self.notification_manager.channels[channel].send_notification(
                                group,
                                template_vars
                            )
                
                elif action.type == "set_state":
                    # Update alert state
                    new_state = AlertState[action.params["state"]]
                    await self.notification_manager.alert_manager.update_alert_state(
                        group.id,
                        new_state,
                        action.params.get("reason")
                    )
                
                elif action.type == "add_context":
                    # Add context data
                    group.context.update(action.params.get("data", {}))
                
                elif action.type == "route":
                    # Route alert to specific channels
                    channels = set(action.params.get("channels", []))
                    severity = group.severity
                    self.notification_manager.config.severity_channels.setdefault(
                        severity,
                        set()
                    ).update(channels)
                
            except Exception as e:
                print(f"Failed to execute action {action.type}: {e}")
    
    def get_rule_stats(
        self,
        rule_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get rule execution statistics."""
        if not self.config.enable_stats:
            return {}
        
        if rule_name:
            return self.stats.get(rule_name, {})
        
        return {
            "rules": self.stats,
            "total_matches": sum(
                stats["matches"]
                for stats in self.stats.values()
            ),
            "total_actions": sum(
                stats["actions"]
                for stats in self.stats.values()
            )
        }
    
    async def save_rule(
        self,
        rule: NotificationRule
    ):
        """Save rule to file."""
        rule_file = self.config.rules_dir / f"{rule.name}.yaml"
        
        # Convert to dict
        rule_data = {
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
            "stop_processing": rule.stop_processing,
            "created": rule.created.isoformat(),
            "modified": datetime.now().isoformat()
        }
        
        # Save to file
        with open(rule_file, "w") as f:
            yaml.safe_dump(rule_data, f)
        
        # Reload rules
        self._load_rules()
    
    async def delete_rule(
        self,
        rule_name: str
    ) -> bool:
        """Delete rule file."""
        rule_file = self.config.rules_dir / f"{rule_name}.yaml"
        if rule_file.exists():
            rule_file.unlink()
            self._load_rules()
            return True
        return False

def create_rule_engine(
    notification_manager: NotificationManager,
    config: Optional[RuleConfig] = None
) -> RuleEngine:
    """Create rule engine."""
    return RuleEngine(notification_manager, config)

if __name__ == "__main__":
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
        
        # Create test rule
        rule = NotificationRule(
            name="critical_alerts",
            description="Handle critical alerts",
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
                    params={
                        "channels": ["slack", "email"],
                        "template_vars": {
                            "priority": "urgent"
                        }
                    }
                )
            ],
            priority=100
        )
        
        await engine.save_rule(rule)
        
        # Process some alerts
        for i in range(5):
            group = AlertGroup(
                id=f"test_{i}",
                pattern_type="spike",
                severity=AlertSeverity.CRITICAL,
                state=AlertState.NEW,
                alerts=[],
                first_seen=datetime.now(),
                last_seen=datetime.now()
            )
            
            await engine.process_alert(group)
        
        # Print stats
        print("Rule stats:", engine.get_rule_stats())
    
    asyncio.run(main())
