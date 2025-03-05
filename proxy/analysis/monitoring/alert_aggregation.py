"""Alert aggregation and pattern detection."""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
import pandas as pd

from .resource_alerts import AlertConfig, ResourceAlertManager

logger = logging.getLogger(__name__)

@dataclass
class AggregationConfig:
    """Alert aggregation configuration."""
    time_window: timedelta = timedelta(minutes=15)
    min_occurrences: int = 3
    correlation_threshold: float = 0.7
    max_groups: int = 100
    pattern_memory: int = 1000
    similarity_threshold: float = 0.8

class AlertAggregator:
    """Aggregate and analyze alert patterns."""
    
    def __init__(
        self,
        config: AggregationConfig,
        alert_manager: ResourceAlertManager
    ):
        self.config = config
        self.alert_manager = alert_manager
        
        self.alert_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.pattern_store: List[Dict[str, Any]] = []
        self.correlation_cache: Dict[Tuple[str, str], float] = {}
    
    def process_alert(self, alert: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process and aggregate new alert."""
        aggregated = []
        current_time = datetime.now()
        cutoff_time = current_time - self.config.time_window
        
        # Clean old alerts
        self._clean_old_alerts(cutoff_time)
        
        # Add to groups
        group_key = self._get_group_key(alert)
        self.alert_groups[group_key].append(alert)
        
        # Check for patterns
        if len(self.alert_groups[group_key]) >= self.config.min_occurrences:
            pattern = self._analyze_pattern(group_key)
            if pattern:
                aggregated.append(pattern)
                self._store_pattern(pattern)
        
        # Check for correlations
        correlations = self._find_correlations(group_key)
        aggregated.extend(correlations)
        
        return aggregated
    
    def _clean_old_alerts(self, cutoff_time: datetime):
        """Remove alerts older than cutoff time."""
        for group_key in list(self.alert_groups.keys()):
            self.alert_groups[group_key] = [
                alert for alert in self.alert_groups[group_key]
                if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
            ]
            
            if not self.alert_groups[group_key]:
                del self.alert_groups[group_key]
    
    def _get_group_key(self, alert: Dict[str, Any]) -> str:
        """Generate grouping key for alert."""
        return f"{alert['title']}:{alert['operation']}"
    
    def _analyze_pattern(self, group_key: str) -> Optional[Dict[str, Any]]:
        """Analyze alert pattern in group."""
        alerts = self.alert_groups[group_key]
        if not alerts:
            return None
        
        # Calculate metrics
        timestamps = [
            datetime.fromisoformat(a["timestamp"])
            for a in alerts
        ]
        time_diffs = np.diff([t.timestamp() for t in timestamps])
        
        metrics = pd.DataFrame([
            a["metrics"] for a in alerts
        ])
        
        pattern = {
            "group_key": group_key,
            "alert_type": alerts[0]["title"],
            "operation": alerts[0]["operation"],
            "occurrences": len(alerts),
            "first_occurrence": min(timestamps).isoformat(),
            "last_occurrence": max(timestamps).isoformat(),
            "avg_interval": float(np.mean(time_diffs)) if len(time_diffs) > 0 else 0,
            "std_interval": float(np.std(time_diffs)) if len(time_diffs) > 0 else 0,
            "metrics_summary": {
                column: {
                    "mean": metrics[column].mean(),
                    "std": metrics[column].std(),
                    "min": metrics[column].min(),
                    "max": metrics[column].max()
                }
                for column in metrics.columns
            },
            "trend": self._calculate_trend(metrics)
        }
        
        return pattern
    
    def _calculate_trend(self, metrics: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metric trends."""
        trends = {}
        
        for column in metrics.columns:
            values = metrics[column].values
            if len(values) >= 2:
                slope, intercept, r_value, p_value, std_err = \
                    stats.linregress(range(len(values)), values)
                
                trends[column] = {
                    "slope": slope,
                    "r_squared": r_value ** 2,
                    "significant": p_value < 0.05
                }
        
        return trends
    
    def _find_correlations(
        self,
        group_key: str
    ) -> List[Dict[str, Any]]:
        """Find correlations between alert groups."""
        correlations = []
        group = self.alert_groups[group_key]
        
        for other_key, other_group in self.alert_groups.items():
            if other_key == group_key:
                continue
            
            cache_key = tuple(sorted([group_key, other_key]))
            if cache_key in self.correlation_cache:
                correlation = self.correlation_cache[cache_key]
            else:
                correlation = self._calculate_correlation(group, other_group)
                self.correlation_cache[cache_key] = correlation
            
            if correlation > self.config.correlation_threshold:
                correlations.append({
                    "type": "correlation",
                    "groups": [group_key, other_key],
                    "correlation": correlation,
                    "timestamp": datetime.now().isoformat()
                })
        
        return correlations
    
    def _calculate_correlation(
        self,
        group1: List[Dict[str, Any]],
        group2: List[Dict[str, Any]]
    ) -> float:
        """Calculate correlation between two alert groups."""
        if not group1 or not group2:
            return 0.0
        
        # Create time series
        times1 = set(
            datetime.fromisoformat(a["timestamp"])
            for a in group1
        )
        times2 = set(
            datetime.fromisoformat(a["timestamp"])
            for a in group2
        )
        
        # Calculate time overlap
        all_times = sorted(times1 | times2)
        if not all_times:
            return 0.0
        
        window = (max(all_times) - min(all_times)).total_seconds() / len(all_times)
        
        # Create binary signals
        signal1 = [1 if self._is_near_any(t, times1, window) else 0 for t in all_times]
        signal2 = [1 if self._is_near_any(t, times2, window) else 0 for t in all_times]
        
        # Calculate correlation
        if len(signal1) <= 1 or len(set(signal1)) <= 1 or len(set(signal2)) <= 1:
            return 0.0
        
        correlation = np.corrcoef(signal1, signal2)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _is_near_any(
        self,
        time: datetime,
        times: Set[datetime],
        window: float
    ) -> bool:
        """Check if time is near any time in set."""
        return any(
            abs((t - time).total_seconds()) <= window
            for t in times
        )
    
    def _store_pattern(self, pattern: Dict[str, Any]):
        """Store detected pattern."""
        self.pattern_store.append(pattern)
        
        # Trim pattern store
        if len(self.pattern_store) > self.config.pattern_memory:
            self.pattern_store = self.pattern_store[-self.config.pattern_memory:]
    
    def find_similar_patterns(
        self,
        pattern: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find patterns similar to given pattern."""
        if not pattern or not self.pattern_store:
            return []
        
        similar = []
        for stored in self.pattern_store:
            if stored["group_key"] == pattern["group_key"]:
                continue
            
            similarity = self._calculate_pattern_similarity(pattern, stored)
            if similarity >= self.config.similarity_threshold:
                similar.append({
                    "pattern": stored,
                    "similarity": similarity
                })
        
        return sorted(
            similar,
            key=lambda x: x["similarity"],
            reverse=True
        )
    
    def _calculate_pattern_similarity(
        self,
        pattern1: Dict[str, Any],
        pattern2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between patterns."""
        scores = []
        
        # Compare intervals
        if pattern1["avg_interval"] > 0 and pattern2["avg_interval"] > 0:
            interval_ratio = min(
                pattern1["avg_interval"],
                pattern2["avg_interval"]
            ) / max(
                pattern1["avg_interval"],
                pattern2["avg_interval"]
            )
            scores.append(interval_ratio)
        
        # Compare metrics
        common_metrics = set(pattern1["metrics_summary"]) & \
                        set(pattern2["metrics_summary"])
        
        for metric in common_metrics:
            stats1 = pattern1["metrics_summary"][metric]
            stats2 = pattern2["metrics_summary"][metric]
            
            # Compare means
            mean_ratio = min(stats1["mean"], stats2["mean"]) / \
                        max(stats1["mean"], stats2["mean"])
            scores.append(mean_ratio)
            
            # Compare standard deviations
            if stats1["std"] > 0 and stats2["std"] > 0:
                std_ratio = min(stats1["std"], stats2["std"]) / \
                           max(stats1["std"], stats2["std"])
                scores.append(std_ratio)
        
        # Compare trends
        if pattern1.get("trend") and pattern2.get("trend"):
            common_trends = set(pattern1["trend"]) & set(pattern2["trend"])
            for metric in common_trends:
                trend1 = pattern1["trend"][metric]
                trend2 = pattern2["trend"][metric]
                
                if trend1["significant"] and trend2["significant"]:
                    slope_similarity = 1 - min(
                        1,
                        abs(
                            (trend1["slope"] - trend2["slope"]) /
                            max(abs(trend1["slope"]), abs(trend2["slope"]))
                        )
                    )
                    scores.append(slope_similarity)
        
        return np.mean(scores) if scores else 0.0

def create_aggregator(
    alert_manager: ResourceAlertManager,
    time_window: timedelta = timedelta(minutes=15),
    min_occurrences: int = 3
) -> AlertAggregator:
    """Create alert aggregator with configuration."""
    config = AggregationConfig(
        time_window=time_window,
        min_occurrences=min_occurrences
    )
    return AlertAggregator(config, alert_manager)

if __name__ == "__main__":
    # Example usage
    from .resource_alerts import create_alert_manager
    
    alert_manager = create_alert_manager(
        email_recipients=["admin@example.com"]
    )
    aggregator = create_aggregator(alert_manager)
    
    # Process some alerts
    alert = {
        "title": "CPU Usage Alert",
        "operation": "test_operation",
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "cpu_percent": 90.0,
            "memory_percent": 50.0
        }
    }
    
    patterns = aggregator.process_alert(alert)
    print(f"Detected patterns: {len(patterns)}")
