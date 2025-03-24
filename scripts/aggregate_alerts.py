#!/usr/bin/env python3
"""Alert aggregation and filtering for test performance alerts."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AlertGroup:
    """Group of related alerts."""
    key: str  # Grouping key (e.g., "coverage_drop")
    alerts: List[Dict[str, Any]]
    first_seen: datetime
    last_seen: datetime
    severity: str
    count: int
    metrics: Set[str]
    summary: str

class AlertAggregator:
    """Aggregate and filter test performance alerts."""

    def __init__(
        self,
        history_dir: Path,
        config_file: Optional[Path] = None,
        lookback_days: int = 7
    ):
        self.history_dir = history_dir
        self.config = self._load_config(config_file)
        self.lookback_days = lookback_days
        self.alert_groups: Dict[str, AlertGroup] = {}

    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load aggregation configuration."""
        default_config = {
            "grouping": {
                "time_window": "1h",  # Group alerts within time window
                "max_group_size": 10,  # Max alerts per group
                "similarity_threshold": 0.8  # Text similarity threshold
            },
            "filtering": {
                "min_severity": "warning",
                "min_occurrence": 2,  # Minimum occurrences to report
                "ignore_flapping": True,  # Ignore metrics that flip-flop
                "cooldown_minutes": 30  # Minimum time between similar alerts
            },
            "reporting": {
                "max_groups": 5,  # Max groups to report at once
                "summary_metrics": True,  # Include metric summaries
                "trend_detection": True  # Include trend analysis
            }
        }

        if config_file and config_file.exists():
            custom_config = json.loads(config_file.read_text())
            default_config.update(custom_config)

        return default_config

    def _load_recent_alerts(self) -> List[Dict[str, Any]]:
        """Load recent alerts from history."""
        cutoff = datetime.now() - timedelta(days=self.lookback_days)
        alerts = []

        try:
            alert_file = self.history_dir / "alert_history.json"
            if alert_file.exists():
                data = json.loads(alert_file.read_text())
                for alert in data:
                    timestamp = datetime.fromisoformat(alert["timestamp"])
                    if timestamp >= cutoff:
                        alert["timestamp"] = timestamp
                        alerts.append(alert)
        except Exception as e:
            logger.error(f"Error loading alerts: {e}")

        return alerts

    def _create_alert_key(self, alert: Dict[str, Any]) -> str:
        """Create grouping key for alert."""
        components = [
            alert["severity"],
            alert["metric"].split("_")[0],  # Base metric type
            "decrease" if alert.get("comparison") == "below" else "increase"
        ]
        return "_".join(components)

    def _alerts_are_similar(self, alert1: Dict[str, Any], alert2: Dict[str, Any]) -> bool:
        """Check if alerts are similar enough to group."""
        # Check basic properties
        if alert1["severity"] != alert2["severity"]:
            return False

        if alert1["metric"] != alert2["metric"]:
            return False

        # Check timing
        time1 = alert1["timestamp"]
        time2 = alert2["timestamp"]
        window = pd.Timedelta(self.config["grouping"]["time_window"])
        if abs(time1 - time2) > window:
            return False

        # Check values
        if "value" in alert1 and "value" in alert2:
            diff = abs(alert1["value"] - alert2["value"])
            threshold = max(alert1["value"], alert2["value"]) * 0.1  # 10% tolerance
            if diff > threshold:
                return False

        return True

    def group_alerts(self, alerts: List[Dict[str, Any]]) -> List[AlertGroup]:
        """Group similar alerts together."""
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Sort alerts by timestamp
        sorted_alerts = sorted(alerts, key=lambda x: x["timestamp"])
        
        for alert in sorted_alerts:
            # Find matching group or create new one
            key = self._create_alert_key(alert)
            
            # Check if alert should join existing group
            grouped = False
            if key in groups:
                last_alert = groups[key][-1]
                if self._alerts_are_similar(last_alert, alert):
                    if len(groups[key]) < self.config["grouping"]["max_group_size"]:
                        groups[key].append(alert)
                        grouped = True

            if not grouped:
                # Start new group
                groups[key] = [alert]

        # Convert groups to AlertGroup objects
        result = []
        for key, alerts in groups.items():
            if len(alerts) >= self.config["filtering"]["min_occurrence"]:
                result.append(AlertGroup(
                    key=key,
                    alerts=alerts,
                    first_seen=alerts[0]["timestamp"],
                    last_seen=alerts[-1]["timestamp"],
                    severity=alerts[0]["severity"],
                    count=len(alerts),
                    metrics=set(a["metric"] for a in alerts),
                    summary=self._create_group_summary(alerts)
                ))

        return sorted(
            result,
            key=lambda g: (g.severity == "critical", g.count),
            reverse=True
        )

    def _create_group_summary(self, alerts: List[Dict[str, Any]]) -> str:
        """Create summary for alert group."""
        metric_type = alerts[0]["metric"].split("_")[0]
        comparison = alerts[0].get("comparison", "changed")
        
        values = [a["value"] for a in alerts]
        avg_value = sum(values) / len(values)
        
        duration = alerts[-1]["timestamp"] - alerts[0]["timestamp"]
        
        return (
            f"{metric_type.title()} performance {comparison} by average of {avg_value:.1f} "
            f"over {duration.total_seconds() / 3600:.1f} hours "
            f"({len(alerts)} occurrences)"
        )

    def detect_trends(self, groups: List[AlertGroup]) -> Dict[str, Any]:
        """Detect trends in alert groups."""
        trends = {
            "increasing": [],
            "decreasing": [],
            "oscillating": [],
            "persistent": []
        }

        for group in groups:
            if len(group.alerts) < 3:
                continue

            # Extract values and times
            values = [a["value"] for a in group.alerts]
            times = [a["timestamp"] for a in group.alerts]

            # Calculate changes
            changes = [b - a for a, b in zip(values[:-1], values[1:])]
            
            # Detect patterns
            if all(c > 0 for c in changes):
                trends["increasing"].append(group.key)
            elif all(c < 0 for c in changes):
                trends["decreasing"].append(group.key)
            elif any(a * b < 0 for a, b in zip(changes[:-1], changes[1:])):
                trends["oscillating"].append(group.key)
            else:
                trends["persistent"].append(group.key)

        return trends

    def filter_flapping(self, groups: List[AlertGroup]) -> List[AlertGroup]:
        """Filter out flapping alerts."""
        if not self.config["filtering"]["ignore_flapping"]:
            return groups

        result = []
        for group in groups:
            if len(group.alerts) < 3:
                result.append(group)
                continue

            # Check for alternating patterns
            values = [a["value"] for a in group.alerts]
            changes = [b - a for a, b in zip(values[:-1], values[1:])]
            
            # If changes alternate sign frequently, skip group
            sign_changes = sum(1 for a, b in zip(changes[:-1], changes[1:]) if a * b < 0)
            if sign_changes > len(changes) * 0.5:
                logger.info(f"Filtering flapping alerts for {group.key}")
                continue

            result.append(group)

        return result

    def apply_cooldown(self, groups: List[AlertGroup]) -> List[AlertGroup]:
        """Apply cooldown period to similar alerts."""
        cooldown = timedelta(minutes=self.config["filtering"]["cooldown_minutes"])
        
        result = []
        last_alert_time: Dict[str, datetime] = {}

        for group in groups:
            key = f"{group.severity}_{next(iter(group.metrics))}"
            
            if key in last_alert_time:
                if group.first_seen - last_alert_time[key] < cooldown:
                    continue

            result.append(group)
            last_alert_time[key] = group.last_seen

        return result

    def generate_report(self, groups: List[AlertGroup]) -> str:
        """Generate aggregated alert report."""
        if not groups:
            return "No significant alert patterns detected."

        trends = self.detect_trends(groups)
        
        lines = ["# Aggregated Alert Report", ""]
        
        # Summary
        lines.extend([
            "## Summary",
            f"- Total Alert Groups: {len(groups)}",
            f"- Critical Groups: {sum(1 for g in groups if g.severity == 'critical')}",
            f"- Warning Groups: {sum(1 for g in groups if g.severity == 'warning')}",
            f"- Total Alerts: {sum(g.count for g in groups)}",
            ""
        ])

        # Trends
        lines.extend([
            "## Trends",
            f"- Increasing: {', '.join(trends['increasing']) or 'None'}",
            f"- Decreasing: {', '.join(trends['decreasing']) or 'None'}",
            f"- Oscillating: {', '.join(trends['oscillating']) or 'None'}",
            f"- Persistent: {', '.join(trends['persistent']) or 'None'}",
            ""
        ])

        # Groups
        lines.append("## Alert Groups")
        for group in groups:
            lines.extend([
                f"### {group.key}",
                f"- Severity: {group.severity}",
                f"- Count: {group.count}",
                f"- First Seen: {group.first_seen.isoformat()}",
                f"- Last Seen: {group.last_seen.isoformat()}",
                f"- Metrics: {', '.join(sorted(group.metrics))}",
                f"- Summary: {group.summary}",
                ""
            ])

        return "\n".join(lines)

def main() -> int:
    """Main entry point."""
    try:
        history_dir = Path("benchmark_results/performance_history")
        if not history_dir.exists():
            logger.error("No performance history directory found")
            return 1

        # Create aggregator
        aggregator = AlertAggregator(history_dir)
        
        # Load and process alerts
        alerts = aggregator.load_recent_alerts()
        if not alerts:
            logger.info("No recent alerts found")
            return 0
        
        # Group alerts
        groups = aggregator.group_alerts(alerts)
        
        # Apply filters
        groups = aggregator.filter_flapping(groups)
        groups = aggregator.apply_cooldown(groups)
        
        # Generate report
        report = aggregator.generate_report(groups)
        
        # Save report
        output_file = history_dir / "alert_summary.md"
        output_file.write_text(report)
        
        logger.info(f"Alert summary written to {output_file}")
        
        # Exit with error if critical groups present
        return 1 if any(g.severity == "critical" for g in groups) else 0

    except Exception as e:
        logger.error(f"Error aggregating alerts: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
