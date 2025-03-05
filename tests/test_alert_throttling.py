#!/usr/bin/env python3
"""Integration tests for alert throttling functionality."""

import pytest
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Generator

from scripts.alert_throttling import (
    AlertThrottler,
    ThrottlingConfig,
    AlertKey,
    AlertRecord
)

@pytest.fixture
def temp_storage(tmp_path: Path) -> Path:
    """Provide temporary storage path."""
    return tmp_path / "test_alert_history.json"

@pytest.fixture
def test_config() -> ThrottlingConfig:
    """Provide test throttling configuration."""
    return ThrottlingConfig(
        cooldown_minutes=5,
        max_alerts_per_hour=5,
        min_change_threshold=2.0,
        reset_after_hours=1
    )

@pytest.fixture
def throttler(temp_storage: Path, test_config: ThrottlingConfig) -> AlertThrottler:
    """Provide configured throttler instance."""
    return AlertThrottler(test_config, temp_storage)

@pytest.fixture
def sample_alert() -> Dict[str, Any]:
    """Provide sample alert data."""
    return {
        "title": "Test Alert",
        "severity": "warning",
        "metric": "test_metric",
        "value": 100.0
    }

def test_initial_state(throttler: AlertThrottler) -> None:
    """Test initial throttler state."""
    assert len(throttler.alert_history) == 0
    assert len(throttler.hourly_counts) == 0
    assert not throttler.storage_path.exists()

def test_record_alert(throttler: AlertThrottler, sample_alert: Dict[str, Any]) -> None:
    """Test recording an alert."""
    throttler.record_alert(**sample_alert)
    
    key = AlertKey(
        title=sample_alert["title"],
        severity=sample_alert["severity"],
        metric=sample_alert["metric"]
    )
    
    assert key in throttler.alert_history
    assert throttler.alert_history[key].count == 1
    assert throttler.alert_history[key].last_value == sample_alert["value"]
    assert throttler.storage_path.exists()

def test_throttle_duplicate(throttler: AlertThrottler, sample_alert: Dict[str, Any]) -> None:
    """Test throttling of duplicate alerts."""
    # First alert should not be throttled
    assert not throttler.should_throttle(**sample_alert)
    throttler.record_alert(**sample_alert)
    
    # Immediate duplicate should be throttled
    assert throttler.should_throttle(**sample_alert)

def test_value_change_threshold(throttler: AlertThrottler, sample_alert: Dict[str, Any]) -> None:
    """Test value change threshold."""
    # Record initial alert
    throttler.record_alert(**sample_alert)
    
    # Small change should be throttled
    small_change = sample_alert.copy()
    small_change["value"] = sample_alert["value"] * 1.01  # 1% change
    assert throttler.should_throttle(**small_change)
    
    # Large change should not be throttled
    large_change = sample_alert.copy()
    large_change["value"] = sample_alert["value"] * 1.05  # 5% change
    assert not throttler.should_throttle(**large_change)

def test_rate_limiting(throttler: AlertThrottler) -> None:
    """Test hourly rate limiting."""
    alert_base = {
        "severity": "warning",
        "metric": "test_metric",
        "value": 100.0
    }
    
    # Send max_alerts_per_hour unique alerts
    for i in range(throttler.config.max_alerts_per_hour + 1):
        alert = alert_base.copy()
        alert["title"] = f"Test Alert {i}"
        
        if i < throttler.config.max_alerts_per_hour:
            assert not throttler.should_throttle(**alert)
            throttler.record_alert(**alert)
        else:
            # Should be throttled due to rate limit
            assert throttler.should_throttle(**alert)

@pytest.mark.asyncio
async def test_cooldown_period(throttler: AlertThrottler, sample_alert: Dict[str, Any]) -> None:
    """Test cooldown period between alerts."""
    # Initial alert
    throttler.record_alert(**sample_alert)
    
    # Should be throttled during cooldown
    assert throttler.should_throttle(**sample_alert)
    
    # Simulate waiting for cooldown
    future_time = datetime.now() + timedelta(minutes=throttler.config.cooldown_minutes + 1)
    
    # Mock current time in throttler
    with pytest.MonkeyPatch().context() as m:
        m.setattr("datetime.now", lambda: future_time)
        assert not throttler.should_throttle(**sample_alert)

def test_persistence(temp_storage: Path, test_config: ThrottlingConfig, sample_alert: Dict[str, Any]) -> None:
    """Test alert history persistence."""
    # Create throttler and record alert
    throttler1 = AlertThrottler(test_config, temp_storage)
    throttler1.record_alert(**sample_alert)
    
    # Create new throttler instance
    throttler2 = AlertThrottler(test_config, temp_storage)
    
    # Check if history was loaded
    key = AlertKey(
        title=sample_alert["title"],
        severity=sample_alert["severity"],
        metric=sample_alert["metric"]
    )
    assert key in throttler2.alert_history
    assert throttler2.alert_history[key].last_value == sample_alert["value"]

def test_cleanup(throttler: AlertThrottler, sample_alert: Dict[str, Any]) -> None:
    """Test cleanup of old records."""
    # Record alert
    throttler.record_alert(**sample_alert)
    
    # Simulate future time beyond reset_after_hours
    future_time = datetime.now() + timedelta(hours=throttler.config.reset_after_hours + 1)
    
    with pytest.MonkeyPatch().context() as m:
        m.setattr("datetime.now", lambda: future_time)
        throttler._cleanup_old_records()
        
        # History should be empty
        assert len(throttler.alert_history) == 0
        assert len(throttler.hourly_counts) == 0

def test_alert_stats(throttler: AlertThrottler) -> None:
    """Test alert statistics generation."""
    alerts = [
        {
            "title": "Alert 1",
            "severity": "warning",
            "metric": "metric1",
            "value": 100.0
        },
        {
            "title": "Alert 2",
            "severity": "critical",
            "metric": "metric2",
            "value": 200.0
        },
        {
            "title": "Alert 1",  # Duplicate to test counting
            "severity": "warning",
            "metric": "metric1",
            "value": 150.0
        }
    ]
    
    for alert in alerts:
        if not throttler.should_throttle(**alert):
            throttler.record_alert(**alert)
    
    stats = throttler.get_alert_stats()
    
    assert stats["total_alerts"] == 3
    assert stats["unique_alerts"] == 2
    assert stats["alerts_by_severity"]["warning"] == 2
    assert stats["alerts_by_severity"]["critical"] == 1
    assert len(stats["most_frequent"]) > 0

def test_different_severity_same_metric(throttler: AlertThrottler) -> None:
    """Test handling of same metric with different severities."""
    base_alert = {
        "title": "Test Alert",
        "metric": "test_metric",
        "value": 100.0
    }
    
    # Warning should not be throttled
    warning = base_alert.copy()
    warning["severity"] = "warning"
    assert not throttler.should_throttle(**warning)
    throttler.record_alert(**warning)
    
    # Critical for same metric should not be throttled
    critical = base_alert.copy()
    critical["severity"] = "critical"
    assert not throttler.should_throttle(**critical)

@pytest.mark.parametrize("invalid_value", [
    float('inf'),
    float('nan'),
    None
])
def test_invalid_values(throttler: AlertThrottler, sample_alert: Dict[str, Any], invalid_value: Any) -> None:
    """Test handling of invalid metric values."""
    alert = sample_alert.copy()
    alert["value"] = invalid_value
    
    # Should not raise exception
    throttler.record_alert(**alert)
    assert not throttler.should_throttle(**alert)

def test_concurrent_alerts(throttler: AlertThrottler) -> None:
    """Test handling of concurrent alerts."""
    import threading
    import queue
    
    results = queue.Queue()
    alerts_sent = 0
    
    def send_alert(alert: Dict[str, Any]) -> None:
        if not throttler.should_throttle(**alert):
            throttler.record_alert(**alert)
            results.put(True)
        else:
            results.put(False)
    
    threads = []
    for i in range(10):
        alert = {
            "title": f"Concurrent Alert {i}",
            "severity": "warning",
            "metric": "concurrent_metric",
            "value": float(i)
        }
        thread = threading.Thread(target=send_alert, args=(alert,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Count successful alerts
    while not results.empty():
        if results.get():
            alerts_sent += 1
    
    # Should not exceed rate limit
    assert alerts_sent <= throttler.config.max_alerts_per_hour

if __name__ == '__main__':
    pytest.main([__file__])
