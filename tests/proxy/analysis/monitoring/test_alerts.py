"""Tests for alert generation and management."""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import json
import re
from typing import AsyncGenerator, List
import aiosmtplib
import aiohttp
from unittest.mock import AsyncMock, MagicMock, patch

from proxy.analysis.monitoring.metrics import MetricValue
from proxy.analysis.monitoring.alerts import (
    AlertSeverity,
    AlertRule,
    Alert,
    AlertState,
    AlertHandler,
    EmailAlertHandler,
    SlackAlertHandler,
    AlertManager
)

@pytest.fixture
def sample_rule() -> AlertRule:
    """Create sample alert rule."""
    return AlertRule(
        name="test_rule",
        metric_pattern=r"test\..*",
        condition="value > 100",
        severity=AlertSeverity.WARNING,
        description="Test alert rule",
        cooldown=timedelta(minutes=1)
    )

@pytest.fixture
def sample_metric() -> MetricValue:
    """Create sample metric."""
    return MetricValue(
        name="test.metric",
        value=150.0,
        timestamp=datetime.now(),
        tags={"env": "test"}
    )

@pytest.fixture
async def alert_dir(tmp_path) -> Path:
    """Create temporary alert directory."""
    alert_dir = tmp_path / "alerts"
    alert_dir.mkdir()
    return alert_dir

@pytest.fixture
async def rules_file(tmp_path, sample_rule) -> Path:
    """Create sample rules file."""
    rules_file = tmp_path / "rules.json"
    with rules_file.open("w") as f:
        json.dump([{
            "name": sample_rule.name,
            "metric_pattern": sample_rule.metric_pattern,
            "condition": sample_rule.condition,
            "severity": sample_rule.severity.value,
            "description": sample_rule.description,
            "cooldown_seconds": 60
        }], f)
    return rules_file

@pytest.fixture
async def alert_manager(
    rules_file: Path,
    alert_dir: Path
) -> AsyncGenerator[AlertManager, None]:
    """Create alert manager for testing."""
    manager = AlertManager(rules_file, alert_dir)
    yield manager

class MockEmailHandler(EmailAlertHandler):
    """Mock email handler for testing."""
    
    def __init__(self):
        super().__init__(
            host="localhost",
            port=25,
            username="test",
            password="test",
            from_addr="test@example.com",
            to_addrs=["alerts@example.com"]
        )
        self.sent_alerts: List[Alert] = []
    
    async def handle_alert(self, alert: Alert):
        """Mock alert handling."""
        self.sent_alerts.append(alert)
        return True

class MockSlackHandler(SlackHandler):
    """Mock Slack handler for testing."""
    
    def __init__(self):
        super().__init__(webhook_url="https://hooks.slack.com/test")
        self.sent_alerts: List[Alert] = []
    
    async def handle_alert(self, alert: Alert):
        """Mock alert handling."""
        self.sent_alerts.append(alert)
        return True

class TestAlertRule:
    """Test alert rule functionality."""

    def test_rule_creation(self, sample_rule):
        """Test alert rule creation."""
        assert sample_rule.name == "test_rule"
        assert sample_rule.severity == AlertSeverity.WARNING
        assert sample_rule.cooldown == timedelta(minutes=1)
    
    def test_rule_pattern_matching(self, sample_rule):
        """Test metric pattern matching."""
        assert re.match(sample_rule.metric_pattern, "test.metric")
        assert not re.match(sample_rule.metric_pattern, "other.metric")

class TestAlertState:
    """Test alert state management."""

    def test_can_trigger(self, sample_rule):
        """Test alert triggering conditions."""
        state = AlertState()
        
        # Should allow first trigger
        assert state.can_trigger(sample_rule, "test.metric")
        
        # Mark as triggered
        state.mark_triggered(sample_rule, "test.metric")
        
        # Should not allow immediate retrigger
        assert not state.can_trigger(sample_rule, "test.metric")
    
    def test_cooldown_expiry(self, sample_rule):
        """Test alert cooldown expiry."""
        state = AlertState()
        state.mark_triggered(sample_rule, "test.metric")
        
        # Simulate time passing
        state.last_triggered[f"{sample_rule.name}:test.metric"] -= timedelta(minutes=2)
        
        # Should allow trigger after cooldown
        assert state.can_trigger(sample_rule, "test.metric")

class TestAlertHandlers:
    """Test alert handlers."""

    async def test_email_handler(self):
        """Test email alert handling."""
        handler = MockEmailHandler()
        alert = Alert(
            id="test_1",
            rule_name="test_rule",
            metric_name="test.metric",
            value=150.0,
            threshold=100.0,
            severity=AlertSeverity.WARNING,
            timestamp=datetime.now(),
            description="Test alert"
        )
        
        success = await handler.handle_alert(alert)
        assert success
        assert len(handler.sent_alerts) == 1
        assert handler.sent_alerts[0].id == alert.id
    
    async def test_slack_handler(self):
        """Test Slack alert handling."""
        handler = MockSlackHandler()
        alert = Alert(
            id="test_1",
            rule_name="test_rule",
            metric_name="test.metric",
            value=150.0,
            threshold=100.0,
            severity=AlertSeverity.WARNING,
            timestamp=datetime.now(),
            description="Test alert"
        )
        
        success = await handler.handle_alert(alert)
        assert success
        assert len(handler.sent_alerts) == 1
        assert handler.sent_alerts[0].id == alert.id

class TestAlertManager:
    """Test alert manager functionality."""

    async def test_rule_loading(self, alert_manager, sample_rule):
        """Test loading rules from file."""
        assert len(alert_manager.rules) == 1
        loaded_rule = alert_manager.rules[0]
        assert loaded_rule.name == sample_rule.name
        assert loaded_rule.severity == sample_rule.severity
    
    async def test_metric_evaluation(self, alert_manager, sample_metric):
        """Test metric evaluation against rules."""
        alert = alert_manager.evaluate_metric(sample_metric)
        assert alert is not None
        assert alert.metric_name == sample_metric.name
        assert alert.value == sample_metric.value
        assert alert.severity == AlertSeverity.WARNING
    
    async def test_alert_lifecycle(self, alert_manager):
        """Test full alert lifecycle."""
        # Process metric and generate alert
        metric = MetricValue(
            name="test.metric",
            value=150.0,
            timestamp=datetime.now()
        )
        await alert_manager.process_metric(metric)
        
        # Check active alerts
        active = alert_manager.get_active_alerts()
        assert len(active) == 1
        alert_id = active[0].id
        
        # Acknowledge alert
        success = await alert_manager.acknowledge_alert(alert_id, "test_user")
        assert success
        
        alert = alert_manager.state.active_alerts[alert_id]
        assert alert.acknowledged
        assert alert.acknowledged_by == "test_user"
        
        # Resolve alert
        success = await alert_manager.resolve_alert(alert_id)
        assert success
        
        # Check alert has moved to resolved
        assert alert_id not in alert_manager.state.active_alerts
        assert alert_id in alert_manager.state.resolved_alerts
        
        resolved = alert_manager.get_resolved_alerts()
        assert len(resolved) == 1
        assert resolved[0].id == alert_id
    
    async def test_alert_handlers(self, alert_manager):
        """Test alert handler integration."""
        # Add mock handlers
        email_handler = MockEmailHandler()
        slack_handler = MockSlackHandler()
        alert_manager.handlers = [email_handler, slack_handler]
        
        # Process metric
        metric = MetricValue(
            name="test.metric",
            value=150.0,
            timestamp=datetime.now()
        )
        await alert_manager.process_metric(metric)
        
        # Check handlers received alerts
        assert len(email_handler.sent_alerts) == 1
        assert len(slack_handler.sent_alerts) == 1

async def test_full_alert_pipeline(
    alert_manager,
    sample_metric,
    alert_dir
):
    """Test complete alert pipeline."""
    # Add mock handler
    handler = MockEmailHandler()
    alert_manager.handlers = [handler]
    
    # Process metric
    await alert_manager.process_metric(sample_metric)
    
    # Verify alert was generated
    active = alert_manager.get_active_alerts()
    assert len(active) == 1
    alert = active[0]
    
    # Verify alert was stored
    alert_file = alert_dir / f"{alert.id}.json"
    assert alert_file.exists()
    
    # Verify handler received alert
    assert len(handler.sent_alerts) == 1
    
    # Acknowledge and resolve
    await alert_manager.acknowledge_alert(alert.id, "test_user")
    await alert_manager.resolve_alert(alert.id)
    
    # Verify final state
    assert not alert_manager.get_active_alerts()
    assert len(alert_manager.get_resolved_alerts()) == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
