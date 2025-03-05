"""Alert system for resource monitoring."""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest
import numpy as np
from unittest.mock import Mock

from .test_costbenefit_resources import ResourceMetrics, ResourceProfile, ResourceMonitor

@dataclass
class AlertThreshold:
    """Threshold configuration for resource alerts."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    duration: Optional[float] = None  # Duration threshold must be exceeded
    recovery_threshold: Optional[float] = None  # Threshold for alert recovery
    cooldown_period: float = 60.0  # Minimum time between alerts
    aggregation_func: str = "avg"  # avg, max, min, p95, etc.

@dataclass
class AlertState:
    """Current state of an alert."""
    alert_id: str
    threshold: AlertThreshold
    current_value: float
    start_time: datetime
    last_update: datetime
    duration: float = 0.0
    is_active: bool = False
    severity: str = "none"
    last_notification: Optional[datetime] = None

@dataclass
class Alert:
    """Resource monitoring alert."""
    alert_id: str
    threshold: AlertThreshold
    current_value: float
    threshold_value: float
    severity: str
    start_time: datetime
    duration: float
    message: str

class AlertManager:
    """Manage resource monitoring alerts."""

    def __init__(self):
        self.thresholds: Dict[str, AlertThreshold] = {}
        self.alert_states: Dict[str, AlertState] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.recovery_handlers: List[Callable[[str], None]] = []

    def add_threshold(self, threshold: AlertThreshold) -> None:
        """Add or update an alert threshold."""
        alert_id = f"{threshold.metric_name}_{threshold.aggregation_func}"
        self.thresholds[alert_id] = threshold
        
        # Initialize alert state if needed
        if alert_id not in self.alert_states:
            self.alert_states[alert_id] = AlertState(
                alert_id=alert_id,
                threshold=threshold,
                current_value=0.0,
                start_time=datetime.now(),
                last_update=datetime.now()
            )

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add alert notification handler."""
        self.alert_handlers.append(handler)

    def add_recovery_handler(self, handler: Callable[[str], None]) -> None:
        """Add alert recovery handler."""
        self.recovery_handlers.append(handler)

    def process_metrics(self, metrics: ResourceMetrics) -> List[Alert]:
        """Process new metrics and generate alerts."""
        alerts = []
        current_time = datetime.now()

        for alert_id, threshold in self.thresholds.items():
            state = self.alert_states[alert_id]
            
            # Get metric value
            current_value = getattr(metrics, threshold.metric_name)
            
            # Update state
            state.current_value = current_value
            state.last_update = current_time
            
            # Check thresholds
            if current_value >= threshold.critical_threshold:
                if not state.is_active or state.severity != "critical":
                    alert = self._create_alert(state, "critical", current_value)
                    alerts.append(alert)
                    self._notify_alert(alert)
            
            elif current_value >= threshold.warning_threshold:
                if not state.is_active or state.severity != "warning":
                    alert = self._create_alert(state, "warning", current_value)
                    alerts.append(alert)
                    self._notify_alert(alert)
            
            elif state.is_active:
                # Check recovery
                recovery_threshold = (
                    threshold.recovery_threshold
                    if threshold.recovery_threshold is not None
                    else threshold.warning_threshold
                )
                
                if current_value < recovery_threshold:
                    self._handle_recovery(state)
            
            # Update duration if alert is active
            if state.is_active:
                state.duration = (current_time - state.start_time).total_seconds()

        return alerts

    def _create_alert(
        self,
        state: AlertState,
        severity: str,
        current_value: float
    ) -> Alert:
        """Create new alert."""
        threshold_value = (
            state.threshold.critical_threshold
            if severity == "critical"
            else state.threshold.warning_threshold
        )
        
        state.is_active = True
        state.severity = severity
        state.start_time = datetime.now()
        state.last_notification = datetime.now()
        
        return Alert(
            alert_id=state.alert_id,
            threshold=state.threshold,
            current_value=current_value,
            threshold_value=threshold_value,
            severity=severity,
            start_time=state.start_time,
            duration=0.0,
            message=self._format_alert_message(state, current_value, threshold_value)
        )

    def _handle_recovery(self, state: AlertState) -> None:
        """Handle alert recovery."""
        state.is_active = False
        state.severity = "none"
        state.duration = 0.0
        
        for handler in self.recovery_handlers:
            try:
                handler(state.alert_id)
            except Exception as e:
                print(f"Error in recovery handler: {e}")

    def _notify_alert(self, alert: Alert) -> None:
        """Notify alert handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Error in alert handler: {e}")

    def _format_alert_message(
        self,
        state: AlertState,
        current_value: float,
        threshold_value: float
    ) -> str:
        """Format alert message."""
        return (
            f"{state.alert_id} {state.severity}: "
            f"Current value {current_value:.2f} exceeds threshold {threshold_value:.2f}"
        )

@pytest.fixture
def alert_manager():
    """Create alert manager for testing."""
    manager = AlertManager()
    
    # Add default thresholds
    manager.add_threshold(AlertThreshold(
        metric_name="cpu_percent",
        warning_threshold=70.0,
        critical_threshold=90.0,
        duration=30.0,
        recovery_threshold=60.0
    ))
    
    manager.add_threshold(AlertThreshold(
        metric_name="memory_percent",
        warning_threshold=80.0,
        critical_threshold=95.0,
        duration=60.0,
        recovery_threshold=75.0
    ))
    
    return manager

@pytest.mark.asyncio
async def test_alert_generation(alert_manager):
    """Test alert generation."""
    alerts = []
    
    def alert_handler(alert: Alert):
        alerts.append(alert)
    
    alert_manager.add_alert_handler(alert_handler)
    
    # Test CPU alert
    metrics = ResourceMetrics(
        timestamp=time.time(),
        cpu_percent=85.0,
        memory_percent=50.0,
        memory_used=1000000,
        disk_io_read=0,
        disk_io_write=0,
        net_io_sent=0,
        net_io_recv=0,
        thread_count=10,
        handle_count=100,
        context_switches=1000
    )
    
    new_alerts = alert_manager.process_metrics(metrics)
    
    assert len(new_alerts) == 1
    assert new_alerts[0].severity == "warning"
    assert new_alerts[0].alert_id == "cpu_percent_avg"

@pytest.mark.asyncio
async def test_alert_recovery(alert_manager):
    """Test alert recovery."""
    recoveries = []
    
    def recovery_handler(alert_id: str):
        recoveries.append(alert_id)
    
    alert_manager.add_recovery_handler(recovery_handler)
    
    # Generate alert
    metrics = ResourceMetrics(
        timestamp=time.time(),
        cpu_percent=95.0,
        memory_percent=50.0,
        memory_used=1000000,
        disk_io_read=0,
        disk_io_write=0,
        net_io_sent=0,
        net_io_recv=0,
        thread_count=10,
        handle_count=100,
        context_switches=1000
    )
    
    alert_manager.process_metrics(metrics)
    
    # Test recovery
    metrics.cpu_percent = 50.0
    alert_manager.process_metrics(metrics)
    
    assert len(recoveries) == 1
    assert recoveries[0] == "cpu_percent_avg"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
