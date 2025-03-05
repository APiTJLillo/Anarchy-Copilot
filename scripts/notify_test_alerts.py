#!/usr/bin/env python3
"""Alert system for test performance monitoring."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Test performance alert."""
    severity: str  # "critical", "warning", "info"
    metric: str
    value: float
    threshold: float
    comparison: str  # "above", "below"
    message: str
    context: Dict[str, Any]

class TestAlertManager:
    """Manage test performance alerts."""
    
    def __init__(
        self, 
        history_dir: Path,
        config_file: Optional[Path] = None,
        alert_history_file: Optional[Path] = None
    ):
        self.history_dir = history_dir
        self.config = self._load_config(config_file)
        self.alert_history_file = alert_history_file or (history_dir / "alert_history.json")
        self.alert_history = self._load_alert_history()
        
    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load alert configuration."""
        default_config = {
            "thresholds": {
                "critical": {
                    "pass_rate_drop": 5.0,  # Percentage points
                    "execution_time_increase": 50.0,  # Percentage
                    "coverage_drop": 2.0  # Percentage points
                },
                "warning": {
                    "pass_rate_drop": 2.0,
                    "execution_time_increase": 20.0,
                    "coverage_drop": 1.0
                }
            },
            "notification": {
                "email": {
                    "enabled": bool(os.getenv("SMTP_SERVER")),
                    "server": os.getenv("SMTP_SERVER", ""),
                    "port": int(os.getenv("SMTP_PORT", "587")),
                    "username": os.getenv("SMTP_USERNAME", ""),
                    "password": os.getenv("SMTP_PASSWORD", ""),
                    "recipients": os.getenv("ALERT_EMAILS", "").split(",")
                },
                "slack": {
                    "enabled": bool(os.getenv("SLACK_WEBHOOK")),
                    "webhook_url": os.getenv("SLACK_WEBHOOK", "")
                },
                "github": {
                    "enabled": bool(os.getenv("GITHUB_TOKEN")),
                    "token": os.getenv("GITHUB_TOKEN", ""),
                    "repository": os.getenv("GITHUB_REPOSITORY", "")
                }
            },
            "analysis": {
                "window_size": "7d",
                "min_samples": 5,
                "trend_threshold": 0.05
            }
        }
        
        if config_file and config_file.exists():
            custom_config = json.loads(config_file.read_text())
            default_config.update(custom_config)
        
        return default_config

    def _load_alert_history(self) -> List[Dict[str, Any]]:
        """Load alert history."""
        if self.alert_history_file.exists():
            return json.loads(self.alert_history_file.read_text())
        return []

    def _save_alert_history(self) -> None:
        """Save alert history."""
        self.alert_history_file.write_text(json.dumps(self.alert_history, indent=2))

    def analyze_test_results(self, results_file: Path) -> List[Alert]:
        """Analyze test results for alerts."""
        alerts = []
        
        # Load current results
        current = json.loads(results_file.read_text())
        
        # Load historical data
        historical_data = self._load_historical_data()
        if not historical_data.empty:
            # Check pass rates
            for test_type in ["unit", "stress", "load"]:
                current_rate = current["pass_rate"][test_type]
                historical_mean = historical_data[f"{test_type}_pass_rate"].mean()
                
                drop = historical_mean - current_rate
                if drop > self.config["thresholds"]["critical"]["pass_rate_drop"]:
                    alerts.append(Alert(
                        severity="critical",
                        metric=f"{test_type}_pass_rate",
                        value=current_rate,
                        threshold=historical_mean,
                        comparison="below",
                        message=f"Critical drop in {test_type} test pass rate",
                        context={
                            "historical_mean": historical_mean,
                            "drop": drop
                        }
                    ))
                elif drop > self.config["thresholds"]["warning"]["pass_rate_drop"]:
                    alerts.append(Alert(
                        severity="warning",
                        metric=f"{test_type}_pass_rate",
                        value=current_rate,
                        threshold=historical_mean,
                        comparison="below",
                        message=f"Pass rate drop in {test_type} tests",
                        context={
                            "historical_mean": historical_mean,
                            "drop": drop
                        }
                    ))
            
            # Check execution times
            for test_type in ["unit", "stress", "load"]:
                current_time = current["execution_time"][test_type]
                historical_mean = historical_data[f"{test_type}_time"].mean()
                
                increase = ((current_time - historical_mean) / historical_mean) * 100
                if increase > self.config["thresholds"]["critical"]["execution_time_increase"]:
                    alerts.append(Alert(
                        severity="critical",
                        metric=f"{test_type}_time",
                        value=current_time,
                        threshold=historical_mean,
                        comparison="above",
                        message=f"Critical increase in {test_type} test execution time",
                        context={
                            "historical_mean": historical_mean,
                            "increase": increase
                        }
                    ))
                elif increase > self.config["thresholds"]["warning"]["execution_time_increase"]:
                    alerts.append(Alert(
                        severity="warning",
                        metric=f"{test_type}_time",
                        value=current_time,
                        threshold=historical_mean,
                        comparison="above",
                        message=f"Execution time increase in {test_type} tests",
                        context={
                            "historical_mean": historical_mean,
                            "increase": increase
                        }
                    ))
            
            # Check coverage
            current_coverage = current["coverage"]
            historical_coverage = historical_data["coverage"].mean()
            
            drop = historical_coverage - current_coverage
            if drop > self.config["thresholds"]["critical"]["coverage_drop"]:
                alerts.append(Alert(
                    severity="critical",
                    metric="coverage",
                    value=current_coverage,
                    threshold=historical_coverage,
                    comparison="below",
                    message="Critical drop in test coverage",
                    context={
                        "historical_coverage": historical_coverage,
                        "drop": drop
                    }
                ))
            elif drop > self.config["thresholds"]["warning"]["coverage_drop"]:
                alerts.append(Alert(
                    severity="warning",
                    metric="coverage",
                    value=current_coverage,
                    threshold=historical_coverage,
                    comparison="below",
                    message="Test coverage drop detected",
                    context={
                        "historical_coverage": historical_coverage,
                        "drop": drop
                    }
                ))
        
        return alerts

    def _load_historical_data(self) -> pd.DataFrame:
        """Load historical test data."""
        records = []
        for test_type in ["unit", "stress", "load"]:
            history_dir = self.history_dir / test_type
            for file in history_dir.glob("*.json"):
                data = json.loads(file.read_text())
                metrics = data["metrics"]
                records.append({
                    "timestamp": pd.Timestamp(data["timestamp"]),
                    f"{test_type}_pass_rate": 100 * (
                        1 - (metrics["failures"] + metrics["errors"]) / metrics["tests"]
                    ),
                    f"{test_type}_time": metrics["time"],
                    "coverage": metrics.get("line_rate", 0) * 100 if test_type == "unit" else None
                })
        
        return pd.DataFrame(records)

    def _format_alert_message(self, alert: Alert) -> str:
        """Format alert message for notifications."""
        return f"""
        {alert.severity.upper()}: {alert.message}
        
        Metric: {alert.metric}
        Current Value: {alert.value:.2f}
        Threshold: {alert.threshold:.2f}
        
        Context:
        {json.dumps(alert.context, indent=2)}
        """

    def send_email_alert(self, alert: Alert) -> None:
        """Send email alert."""
        if not self.config["notification"]["email"]["enabled"]:
            return
        
        try:
            msg = MIMEMultipart()
            msg["Subject"] = f"[{alert.severity.upper()}] Test Performance Alert"
            msg["From"] = self.config["notification"]["email"]["username"]
            msg["To"] = ", ".join(self.config["notification"]["email"]["recipients"])
            
            msg.attach(MIMEText(self._format_alert_message(alert)))
            
            with smtplib.SMTP(
                self.config["notification"]["email"]["server"],
                self.config["notification"]["email"]["port"]
            ) as server:
                server.starttls()
                server.login(
                    self.config["notification"]["email"]["username"],
                    self.config["notification"]["email"]["password"]
                )
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def send_slack_alert(self, alert: Alert) -> None:
        """Send Slack alert."""
        if not self.config["notification"]["slack"]["enabled"]:
            return
        
        try:
            message = {
                "text": f"*{alert.severity.upper()}*: {alert.message}",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": self._format_alert_message(alert)
                        }
                    }
                ]
            }
            
            requests.post(
                self.config["notification"]["slack"]["webhook_url"],
                json=message
            )
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    def create_github_issue(self, alert: Alert) -> None:
        """Create GitHub issue for alert."""
        if not self.config["notification"]["github"]["enabled"]:
            return
        
        try:
            url = f"https://api.github.com/repos/{self.config['notification']['github']['repository']}/issues"
            headers = {
                "Authorization": f"token {self.config['notification']['github']['token']}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            data = {
                "title": f"[{alert.severity.upper()}] {alert.message}",
                "body": self._format_alert_message(alert),
                "labels": ["test-performance", alert.severity]
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to create GitHub issue: {e}")

    def handle_alerts(self, alerts: List[Alert]) -> None:
        """Handle all alerts."""
        for alert in alerts:
            # Record alert
            self.alert_history.append({
                "timestamp": datetime.now().isoformat(),
                "severity": alert.severity,
                "metric": alert.metric,
                "message": alert.message,
                "context": alert.context
            })
            
            # Send notifications based on severity
            if alert.severity == "critical":
                self.send_email_alert(alert)
                self.send_slack_alert(alert)
                self.create_github_issue(alert)
            elif alert.severity == "warning":
                self.send_slack_alert(alert)
        
        # Save updated history
        self._save_alert_history()

def main() -> int:
    """Main entry point."""
    try:
        results_dir = Path("test-results")
        if not results_dir.exists():
            logger.error("No test results directory found")
            return 1
        
        results_file = results_dir / "summary.json"
        if not results_file.exists():
            logger.error("No test summary found")
            return 1
        
        history_dir = Path("benchmark_results/performance_history")
        alert_manager = TestAlertManager(history_dir)
        
        alerts = alert_manager.analyze_test_results(results_file)
        if alerts:
            alert_manager.handle_alerts(alerts)
            logger.info(f"Processed {len(alerts)} alerts")
            
            # Exit with error if critical alerts
            if any(a.severity == "critical" for a in alerts):
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error processing alerts: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
