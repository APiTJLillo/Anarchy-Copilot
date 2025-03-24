#!/usr/bin/env python3
"""Alert throttling and rate limiting functionality."""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class AlertKey:
    """Key for identifying similar alerts."""
    title: str
    severity: str
    metric: str

    def __hash__(self) -> int:
        return hash((self.title, self.severity, self.metric))

@dataclass
class AlertRecord:
    """Record of a sent alert."""
    key: AlertKey
    timestamp: datetime
    count: int = 1
    last_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "title": self.key.title,
            "severity": self.key.severity,
            "metric": self.key.metric,
            "timestamp": self.timestamp.isoformat(),
            "count": self.count,
            "last_value": self.last_value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertRecord':
        """Create from dictionary."""
        return cls(
            key=AlertKey(
                title=data["title"],
                severity=data["severity"],
                metric=data["metric"]
            ),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            count=data["count"],
            last_value=data["last_value"]
        )

class ThrottlingConfig:
    """Configuration for alert throttling."""
    def __init__(self, 
                 cooldown_minutes: int = 60,
                 max_alerts_per_hour: int = 10,
                 min_change_threshold: float = 5.0,
                 reset_after_hours: int = 24):
        self.cooldown_minutes = cooldown_minutes
        self.max_alerts_per_hour = max_alerts_per_hour
        self.min_change_threshold = min_change_threshold
        self.reset_after_hours = reset_after_hours

class AlertThrottler:
    """Manages alert throttling and rate limiting."""
    
    def __init__(self, config: ThrottlingConfig, storage_path: Path):
        self.config = config
        self.storage_path = storage_path
        self.alert_history: Dict[AlertKey, AlertRecord] = {}
        self.hourly_counts: Dict[str, int] = defaultdict(int)
        self._load_history()

    def _load_history(self) -> None:
        """Load alert history from storage."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                for record in data["records"]:
                    alert_record = AlertRecord.from_dict(record)
                    self.alert_history[alert_record.key] = alert_record
                
                # Load hourly counts
                self.hourly_counts = defaultdict(int)
                for hour, count in data.get("hourly_counts", {}).items():
                    self.hourly_counts[hour] = count
                
            except Exception as e:
                logger.error(f"Error loading alert history: {e}")

    def _save_history(self) -> None:
        """Save alert history to storage."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "records": [
                    record.to_dict() for record in self.alert_history.values()
                ],
                "hourly_counts": dict(self.hourly_counts),
                "last_update": datetime.now().isoformat()
            }
            self.storage_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Error saving alert history: {e}")

    def _cleanup_old_records(self) -> None:
        """Remove expired records."""
        now = datetime.now()
        cutoff = now - timedelta(hours=self.config.reset_after_hours)
        
        # Clean up alert history
        expired_keys = [
            key for key, record in self.alert_history.items()
            if record.timestamp < cutoff
        ]
        for key in expired_keys:
            del self.alert_history[key]
        
        # Clean up hourly counts
        cutoff_hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=24)
        expired_hours = [
            hour for hour in self.hourly_counts.keys()
            if datetime.fromisoformat(hour) < cutoff_hour
        ]
        for hour in expired_hours:
            del self.hourly_counts[hour]

    def _update_hourly_count(self) -> None:
        """Update hourly alert count."""
        current_hour = datetime.now().replace(
            minute=0, second=0, microsecond=0
        ).isoformat()
        self.hourly_counts[current_hour] += 1

    def _check_rate_limit(self) -> bool:
        """Check if we've exceeded hourly rate limit."""
        current_hour = datetime.now().replace(
            minute=0, second=0, microsecond=0
        )
        total_alerts = sum(
            count for hour, count in self.hourly_counts.items()
            if datetime.fromisoformat(hour) >= current_hour - timedelta(hours=1)
        )
        return total_alerts < self.config.max_alerts_per_hour

    def should_throttle(self, 
                       title: str, 
                       severity: str,
                       metric: str,
                       value: Optional[float] = None) -> bool:
        """Determine if an alert should be throttled."""
        self._cleanup_old_records()
        
        key = AlertKey(title=title, severity=severity, metric=metric)
        now = datetime.now()
        
        # Check if we're under rate limit
        if not self._check_rate_limit():
            logger.warning(f"Rate limit exceeded: {self.config.max_alerts_per_hour} alerts per hour")
            return True
        
        # Check if alert exists in history
        if key in self.alert_history:
            record = self.alert_history[key]
            
            # Check cooldown period
            if now - record.timestamp < timedelta(minutes=self.config.cooldown_minutes):
                logger.debug(f"Alert throttled: in cooldown period for {key}")
                return True
            
            # Check value change threshold
            if value is not None and record.last_value is not None:
                change = abs((value - record.last_value) / record.last_value * 100)
                if change < self.config.min_change_threshold:
                    logger.debug(f"Alert throttled: change {change:.1f}% below threshold")
                    return True
        
        return False

    def record_alert(self, 
                    title: str, 
                    severity: str,
                    metric: str,
                    value: Optional[float] = None) -> None:
        """Record that an alert was sent."""
        key = AlertKey(title=title, severity=severity, metric=metric)
        
        if key in self.alert_history:
            record = self.alert_history[key]
            record.timestamp = datetime.now()
            record.count += 1
            record.last_value = value
        else:
            self.alert_history[key] = AlertRecord(
                key=key,
                timestamp=datetime.now(),
                last_value=value
            )
        
        self._update_hourly_count()
        self._save_history()

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get statistics about alert history."""
        now = datetime.now()
        stats = {
            "total_alerts": sum(r.count for r in self.alert_history.values()),
            "unique_alerts": len(self.alert_history),
            "alerts_last_hour": sum(
                count for hour, count in self.hourly_counts.items()
                if datetime.fromisoformat(hour) >= now.replace(minute=0, second=0, microsecond=0)
            ),
            "alerts_by_severity": defaultdict(int),
            "most_frequent": []
        }
        
        # Count by severity
        for record in self.alert_history.values():
            stats["alerts_by_severity"][record.key.severity] += record.count
        
        # Most frequent alerts
        sorted_alerts = sorted(
            self.alert_history.values(),
            key=lambda r: r.count,
            reverse=True
        )
        stats["most_frequent"] = [
            {
                "title": r.key.title,
                "severity": r.key.severity,
                "metric": r.key.metric,
                "count": r.count,
                "last_seen": r.timestamp.isoformat()
            }
            for r in sorted_alerts[:5]
        ]
        
        return stats

def create_default_throttler() -> AlertThrottler:
    """Create throttler with default configuration."""
    config = ThrottlingConfig(
        cooldown_minutes=60,
        max_alerts_per_hour=10,
        min_change_threshold=5.0,
        reset_after_hours=24
    )
    return AlertThrottler(
        config,
        storage_path=Path("alert_history.json")
    )

def main() -> None:
    """Test the throttling functionality."""
    throttler = create_default_throttler()
    
    # Example usage
    test_alerts = [
        ("Performance Regression", "warning", "processing_speed", 450),
        ("Performance Regression", "warning", "processing_speed", 445),  # Should throttle
        ("Memory Usage", "critical", "memory", 1024),
        ("Type Quality", "warning", "specificity", 0.85)
    ]
    
    for title, severity, metric, value in test_alerts:
        if not throttler.should_throttle(title, severity, metric, value):
            print(f"Sending alert: {title} ({severity})")
            throttler.record_alert(title, severity, metric, value)
        else:
            print(f"Throttled alert: {title} ({severity})")
    
    # Print stats
    print("\nAlert Statistics:")
    print(json.dumps(throttler.get_alert_stats(), indent=2))

if __name__ == "__main__":
    main()
