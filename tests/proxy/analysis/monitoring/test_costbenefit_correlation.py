"""Alert correlation and analysis for cost-benefit monitoring."""

import asyncio
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .test_costbenefit_alerts import Alert, AlertThreshold, AlertManager

@dataclass
class AlertPattern:
    """Pattern of related alerts."""
    pattern_id: str
    alerts: List[Alert]
    frequency: int
    first_seen: datetime
    last_seen: datetime
    avg_duration: float
    root_cause_probability: Dict[str, float]

@dataclass
class CorrelationRule:
    """Rule for correlating alerts."""
    rule_id: str
    metrics: List[str]
    time_window: float
    distance_threshold: float
    min_correlation: float = 0.7
    min_support: int = 3

class AlertCorrelator:
    """Correlate and analyze alert patterns."""

    def __init__(self):
        self.patterns: Dict[str, AlertPattern] = {}
        self.rules: Dict[str, CorrelationRule] = {}
        self.alert_history: List[Alert] = []
        self.max_history_size: int = 10000
        self.correlation_window: float = 300.0  # 5 minutes
        
        # Initialize standard correlation rules
        self._initialize_rules()

    def _initialize_rules(self) -> None:
        """Initialize default correlation rules."""
        self.rules = {
            "resource_exhaustion": CorrelationRule(
                rule_id="resource_exhaustion",
                metrics=["cpu_percent", "memory_percent", "thread_count"],
                time_window=300.0,
                distance_threshold=0.3,
                min_correlation=0.8
            ),
            "io_bottleneck": CorrelationRule(
                rule_id="io_bottleneck",
                metrics=["disk_io_read", "disk_io_write", "net_io_recv"],
                time_window=180.0,
                distance_threshold=0.4
            ),
            "system_stress": CorrelationRule(
                rule_id="system_stress",
                metrics=["cpu_percent", "context_switches", "handle_count"],
                time_window=600.0,
                distance_threshold=0.25,
                min_correlation=0.75
            )
        }

    def add_alert(self, alert: Alert) -> List[AlertPattern]:
        """Add new alert and identify patterns."""
        self.alert_history.append(alert)
        
        # Trim history if needed
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]
        
        # Find correlated patterns
        patterns = self._find_patterns(alert)
        
        # Update existing patterns and create new ones
        updated_patterns = self._update_patterns(patterns)
        
        return list(updated_patterns)

    def _find_patterns(self, alert: Alert) -> List[List[Alert]]:
        """Find alert patterns using correlation rules."""
        patterns = []
        current_time = datetime.now()
        
        # Get relevant alerts within time window
        recent_alerts = [
            a for a in self.alert_history
            if (current_time - a.start_time).total_seconds() <= self.correlation_window
        ]
        
        for rule in self.rules.values():
            # Filter alerts relevant to this rule
            rule_alerts = [
                a for a in recent_alerts
                if a.threshold.metric_name in rule.metrics
            ]
            
            if len(rule_alerts) >= rule.min_support:
                # Prepare feature matrix
                features = self._extract_features(rule_alerts)
                
                # Cluster alerts
                clusters = self._cluster_alerts(features, rule.distance_threshold)
                
                # Extract patterns from clusters
                for cluster_id in set(clusters):
                    if cluster_id != -1:  # -1 is noise in DBSCAN
                        cluster_alerts = [
                            a for i, a in enumerate(rule_alerts)
                            if clusters[i] == cluster_id
                        ]
                        if self._validate_pattern(cluster_alerts, rule):
                            patterns.append(cluster_alerts)
        
        return patterns

    def _extract_features(self, alerts: List[Alert]) -> np.ndarray:
        """Extract numerical features from alerts."""
        features = []
        for alert in alerts:
            feature_vector = [
                alert.current_value / alert.threshold_value,
                alert.duration,
                float(alert.severity == "critical"),
                (datetime.now() - alert.start_time).total_seconds()
            ]
            features.append(feature_vector)
        
        # Normalize features
        scaler = StandardScaler()
        return scaler.fit_transform(features)

    def _cluster_alerts(
        self,
        features: np.ndarray,
        distance_threshold: float
    ) -> np.ndarray:
        """Cluster alerts using DBSCAN."""
        clustering = DBSCAN(
            eps=distance_threshold,
            min_samples=2,
            metric='euclidean'
        )
        return clustering.fit_predict(features)

    def _validate_pattern(
        self,
        alerts: List[Alert],
        rule: CorrelationRule
    ) -> bool:
        """Validate if alerts form a valid pattern."""
        if len(alerts) < rule.min_support:
            return False
        
        # Check temporal proximity
        times = [a.start_time for a in alerts]
        time_span = max(times) - min(times)
        if time_span.total_seconds() > rule.time_window:
            return False
        
        # Check metric correlation
        if len(set(a.threshold.metric_name for a in alerts)) > 1:
            values = np.array([a.current_value for a in alerts])
            correlation = np.corrcoef(values)[0, 1]
            if correlation < rule.min_correlation:
                return False
        
        return True

    def _update_patterns(
        self,
        new_patterns: List[List[Alert]]
    ) -> Set[AlertPattern]:
        """Update pattern database with new patterns."""
        updated_patterns = set()
        
        for alerts in new_patterns:
            # Generate pattern signature
            metrics = sorted(set(a.threshold.metric_name for a in alerts))
            pattern_id = f"pattern_{'_'.join(metrics)}"
            
            # Update existing pattern or create new one
            if pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]
                pattern.frequency += 1
                pattern.last_seen = max(a.start_time for a in alerts)
                pattern.avg_duration = np.mean([a.duration for a in alerts])
                pattern.alerts.extend(alerts)
                
                # Update root cause probabilities
                self._update_root_cause_probabilities(pattern)
                
            else:
                pattern = AlertPattern(
                    pattern_id=pattern_id,
                    alerts=alerts,
                    frequency=1,
                    first_seen=min(a.start_time for a in alerts),
                    last_seen=max(a.start_time for a in alerts),
                    avg_duration=np.mean([a.duration for a in alerts]),
                    root_cause_probability=self._initial_root_cause_probabilities(alerts)
                )
                self.patterns[pattern_id] = pattern
            
            updated_patterns.add(pattern)
        
        return updated_patterns

    def _update_root_cause_probabilities(self, pattern: AlertPattern) -> None:
        """Update root cause probability estimates."""
        total_alerts = len(pattern.alerts)
        metric_counts = {}
        
        # Count alert occurrences by metric
        for alert in pattern.alerts:
            metric = alert.threshold.metric_name
            metric_counts[metric] = metric_counts.get(metric, 0) + 1
        
        # Update probabilities
        for metric, count in metric_counts.items():
            pattern.root_cause_probability[metric] = count / total_alerts

    def _initial_root_cause_probabilities(
        self,
        alerts: List[Alert]
    ) -> Dict[str, float]:
        """Calculate initial root cause probabilities."""
        total = len(alerts)
        probabilities = {}
        
        for alert in alerts:
            metric = alert.threshold.metric_name
            probabilities[metric] = probabilities.get(metric, 0) + 1
        
        return {
            metric: count / total
            for metric, count in probabilities.items()
        }

@pytest.fixture
def alert_correlator():
    """Create alert correlator for testing."""
    return AlertCorrelator()

@pytest.mark.asyncio
async def test_pattern_detection(alert_correlator):
    """Test alert pattern detection."""
    # Create series of related alerts
    alerts = []
    base_time = datetime.now()
    
    for i in range(5):
        alerts.append(Alert(
            alert_id=f"cpu_alert_{i}",
            threshold=AlertThreshold(
                metric_name="cpu_percent",
                warning_threshold=70,
                critical_threshold=90
            ),
            current_value=85 + i,
            threshold_value=70,
            severity="warning",
            start_time=base_time + timedelta(seconds=i * 30),
            duration=float(i * 30),
            message=f"High CPU usage: {85 + i}%"
        ))
    
    # Add alerts and get patterns
    patterns = []
    for alert in alerts:
        new_patterns = alert_correlator.add_alert(alert)
        patterns.extend(new_patterns)
    
    # Verify pattern detection
    assert len(patterns) > 0
    pattern = patterns[0]
    assert pattern.frequency >= 1
    assert "cpu_percent" in pattern.root_cause_probability
    assert pattern.avg_duration > 0

@pytest.mark.asyncio
async def test_correlation_rules(alert_correlator):
    """Test correlation rule application."""
    # Add test rule
    rule = CorrelationRule(
        rule_id="test_rule",
        metrics=["cpu_percent", "memory_percent"],
        time_window=60.0,
        distance_threshold=0.5,
        min_correlation=0.6
    )
    alert_correlator.rules["test_rule"] = rule
    
    # Create correlated alerts
    base_time = datetime.now()
    alerts = [
        Alert(
            alert_id="cpu_alert",
            threshold=AlertThreshold(
                metric_name="cpu_percent",
                warning_threshold=70,
                critical_threshold=90
            ),
            current_value=85,
            threshold_value=70,
            severity="warning",
            start_time=base_time,
            duration=0.0,
            message="High CPU usage"
        ),
        Alert(
            alert_id="memory_alert",
            threshold=AlertThreshold(
                metric_name="memory_percent",
                warning_threshold=80,
                critical_threshold=95
            ),
            current_value=90,
            threshold_value=80,
            severity="warning",
            start_time=base_time + timedelta(seconds=5),
            duration=0.0,
            message="High memory usage"
        )
    ]
    
    # Process alerts
    patterns = []
    for alert in alerts:
        new_patterns = alert_correlator.add_alert(alert)
        patterns.extend(new_patterns)
    
    # Verify correlation
    assert len(patterns) > 0
    assert len(patterns[0].alerts) >= 2
    assert "cpu_percent" in patterns[0].root_cause_probability
    assert "memory_percent" in patterns[0].root_cause_probability

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
