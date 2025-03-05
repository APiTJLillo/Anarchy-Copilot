#!/usr/bin/env python3
"""Alert handlers for performance regression notifications."""

import os
import json
import smtplib
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AlertConfig:
    """Configuration for alert handlers."""
    def __init__(self, config_file: Optional[Path] = None):
        self.config = self._load_config(config_file)
    
    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load alert configuration."""
        if config_file and config_file.exists():
            return json.loads(config_file.read_text())
        return {
            "email": {
                "enabled": bool(os.getenv("SMTP_HOST")),
                "smtp_host": os.getenv("SMTP_HOST", ""),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "username": os.getenv("SMTP_USERNAME", ""),
                "password": os.getenv("SMTP_PASSWORD", ""),
                "from_address": os.getenv("ALERT_FROM_EMAIL", ""),
                "to_addresses": os.getenv("ALERT_TO_EMAILS", "").split(",")
            },
            "slack": {
                "enabled": bool(os.getenv("SLACK_WEBHOOK_URL")),
                "webhook_url": os.getenv("SLACK_WEBHOOK_URL", ""),
                "channel": os.getenv("SLACK_CHANNEL", "#alerts")
            },
            "github": {
                "enabled": bool(os.getenv("GITHUB_TOKEN")),
                "token": os.getenv("GITHUB_TOKEN", ""),
                "repo": os.getenv("GITHUB_REPOSITORY", ""),
                "issue_labels": ["performance", "regression"]
            },
            "thresholds": {
                "critical": {
                    "speed_regression": float(os.getenv("CRITICAL_SPEED_THRESHOLD", "-15")),
                    "memory_increase": float(os.getenv("CRITICAL_MEMORY_THRESHOLD", "25")),
                    "quality_decrease": float(os.getenv("CRITICAL_QUALITY_THRESHOLD", "-15"))
                },
                "warning": {
                    "speed_regression": float(os.getenv("WARNING_SPEED_THRESHOLD", "-8")),
                    "memory_increase": float(os.getenv("WARNING_MEMORY_THRESHOLD", "15")),
                    "quality_decrease": float(os.getenv("WARNING_QUALITY_THRESHOLD", "-8"))
                }
            }
        }

class AlertHandler:
    """Base class for alert handlers."""
    def __init__(self, config: AlertConfig):
        self.config = config

    async def send_alert(self, 
                        title: str, 
                        message: str, 
                        severity: str,
                        metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Send alert through the handler."""
        raise NotImplementedError

class EmailAlertHandler(AlertHandler):
    """Handles email alerts."""
    
    def _create_html_message(self, title: str, message: str, metrics: Optional[Dict[str, Any]]) -> str:
        """Create HTML formatted message."""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .critical {{ color: #e74c3c; }}
                .warning {{ color: #f39c12; }}
                .metric {{ margin: 10px 0; }}
                .change-negative {{ color: #e74c3c; }}
                .change-positive {{ color: #2ecc71; }}
            </style>
        </head>
        <body>
            <h2>{title}</h2>
            <pre>{message}</pre>
            
            {self._format_metrics(metrics) if metrics else ''}
            
            <p>
            View full report: <a href="{os.getenv('CI_PIPELINE_URL', '#')}">CI Pipeline</a>
            </p>
        </body>
        </html>
        """
        return html

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for email."""
        if not metrics:
            return ""
        
        sections = ["<h3>Metrics Summary</h3>"]
        for name, value in metrics.items():
            if isinstance(value, dict) and 'change' in value:
                change_class = 'change-negative' if value['change'] < 0 else 'change-positive'
                sections.append(
                    f"<div class='metric'>"
                    f"<strong>{name}:</strong> {value['current']:.2f} "
                    f"<span class='{change_class}'>({value['change']:+.1f}%)</span>"
                    "</div>"
                )
        
        return "\n".join(sections)

    async def send_alert(self, 
                        title: str, 
                        message: str, 
                        severity: str,
                        metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Send email alert."""
        if not self.config.config["email"]["enabled"]:
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{severity.upper()}] {title}"
            msg["From"] = self.config.config["email"]["from_address"]
            msg["To"] = ", ".join(self.config.config["email"]["to_addresses"])
            
            html_content = self._create_html_message(title, message, metrics)
            msg.attach(MIMEText(html_content, "html"))
            
            with smtplib.SMTP(
                self.config.config["email"]["smtp_host"],
                self.config.config["email"]["smtp_port"]
            ) as server:
                server.starttls()
                server.login(
                    self.config.config["email"]["username"],
                    self.config.config["email"]["password"]
                )
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

class SlackAlertHandler(AlertHandler):
    """Handles Slack alerts."""
    
    def _create_slack_payload(self, 
                            title: str, 
                            message: str, 
                            severity: str,
                            metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create Slack message payload."""
        color = "#e74c3c" if severity == "critical" else "#f39c12"
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"[{severity.upper()}] {title}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"```{message}```"
                }
            }
        ]
        
        if metrics:
            metrics_text = []
            for name, value in metrics.items():
                if isinstance(value, dict) and 'change' in value:
                    change_icon = "🔴" if value['change'] < 0 else "🟢"
                    metrics_text.append(
                        f"{name}: {value['current']:.2f} "
                        f"{change_icon} ({value['change']:+.1f}%)"
                    )
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Metrics Summary*\n" + "\n".join(metrics_text)
                }
            })
        
        return {
            "channel": self.config.config["slack"]["channel"],
            "attachments": [{
                "color": color,
                "blocks": blocks,
                "footer": f"Performance Monitor • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }]
        }

    async def send_alert(self, 
                        title: str, 
                        message: str, 
                        severity: str,
                        metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Send Slack alert."""
        if not self.config.config["slack"]["enabled"]:
            return False

        try:
            payload = self._create_slack_payload(title, message, severity, metrics)
            response = requests.post(
                self.config.config["slack"]["webhook_url"],
                json=payload
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

class GitHubAlertHandler(AlertHandler):
    """Handles GitHub alerts."""
    
    def _create_issue_body(self, 
                          title: str, 
                          message: str,
                          metrics: Optional[Dict[str, Any]]) -> str:
        """Create GitHub issue body."""
        sections = [
            "# Performance Regression Alert",
            "",
            message,
            ""
        ]
        
        if metrics:
            sections.extend([
                "## Metrics Summary",
                "",
                "| Metric | Current | Change |",
                "| ------ | ------- | ------ |"
            ])
            
            for name, value in metrics.items():
                if isinstance(value, dict) and 'change' in value:
                    change_icon = "🔴" if value['change'] < 0 else "🟢"
                    sections.append(
                        f"| {name} | {value['current']:.2f} | "
                        f"{change_icon} {value['change']:+.1f}% |"
                    )
        
        sections.extend([
            "",
            "## Additional Information",
            f"- Detected at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- CI Pipeline: {os.getenv('CI_PIPELINE_URL', 'N/A')}",
            f"- Commit: {os.getenv('GITHUB_SHA', 'N/A')}"
        ])
        
        return "\n".join(sections)

    async def send_alert(self, 
                        title: str, 
                        message: str, 
                        severity: str,
                        metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Create GitHub issue for alert."""
        if not self.config.config["github"]["enabled"]:
            return False

        try:
            headers = {
                "Authorization": f"token {self.config.config['github']['token']}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            payload = {
                "title": f"[{severity.upper()}] {title}",
                "body": self._create_issue_body(title, message, metrics),
                "labels": self.config.config["github"]["issue_labels"]
            }
            
            owner, repo = self.config.config["github"]["repo"].split("/")
            response = requests.post(
                f"https://api.github.com/repos/{owner}/{repo}/issues",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to create GitHub issue: {e}")
            return False

class AlertManager:
    """Manages alert distribution through multiple handlers."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config = AlertConfig(config_file)
        self.handlers = [
            EmailAlertHandler(self.config),
            SlackAlertHandler(self.config),
            GitHubAlertHandler(self.config)
        ]
    
    async def send_alert(self, 
                        title: str, 
                        message: str, 
                        severity: str = "warning",
                        metrics: Optional[Dict[str, Any]] = None) -> None:
        """Send alert through all configured handlers."""
        results = []
        for handler in self.handlers:
            try:
                result = await handler.send_alert(title, message, severity, metrics)
                results.append(result)
            except Exception as e:
                logger.error(f"Handler {handler.__class__.__name__} failed: {e}")
                results.append(False)
        
        if not any(results):
            logger.error("All alert handlers failed")
            raise RuntimeError("Failed to send alert through any handler")

async def main() -> int:
    """Main entry point for testing."""
    if os.getenv("TEST_ALERTS"):
        manager = AlertManager()
        await manager.send_alert(
            "Test Performance Alert",
            "This is a test alert from the alert system.",
            "warning",
            {
                "Processing Speed": {"current": 450, "change": -12.5},
                "Memory Usage": {"current": 256, "change": 15.3},
                "Type Quality": {"current": 0.85, "change": -5.2}
            }
        )
    return 0

if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
