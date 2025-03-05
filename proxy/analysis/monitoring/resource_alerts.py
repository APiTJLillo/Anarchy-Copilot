"""Resource monitoring alerts and notifications."""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import queue
import threading
import requests
from collections import defaultdict

from .resource_monitor import ResourceMetrics, ResourceMonitor

logger = logging.getLogger(__name__)

@dataclass
class AlertThreshold:
    """Resource alert threshold configuration."""
    cpu_percent: float = 80.0
    memory_percent: float = 80.0
    disk_io_mbps: float = 100.0
    thread_count: int = 100
    connection_count: int = 50
    file_handles: int = 1000
    duration_seconds: float = 300.0

@dataclass
class AlertConfig:
    """Alert configuration."""
    enabled: bool = True
    thresholds: AlertThreshold = AlertThreshold()
    cooldown: timedelta = timedelta(minutes=5)
    email_recipients: List[str] = None
    slack_webhook: Optional[str] = None
    teams_webhook: Optional[str] = None
    custom_handler: Optional[Callable[[str, Dict[str, Any]], None]] = None

class ResourceAlertManager:
    """Manage resource monitoring alerts."""
    
    def __init__(
        self,
        config: AlertConfig,
        alert_history_file: Optional[Path] = None
    ):
        self.config = config
        self.alert_history_file = alert_history_file or Path("alert_history.json")
        
        self.alert_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.last_alert: Dict[str, datetime] = {}
        self.alert_queue: queue.Queue = queue.Queue()
        
        self._alert_thread = threading.Thread(
            target=self._process_alerts,
            daemon=True
        )
        self._alert_thread.start()
        
        self._load_history()
    
    def check_metrics(
        self,
        metrics: ResourceMetrics,
        operation: str
    ) -> List[Dict[str, Any]]:
        """Check metrics against thresholds."""
        alerts = []
        current_time = datetime.now()
        
        # Check CPU usage
        if metrics.cpu_percent > self.config.thresholds.cpu_percent:
            if self._should_alert("cpu", current_time):
                alerts.append(self._create_alert(
                    "CPU Usage Alert",
                    f"CPU usage at {metrics.cpu_percent:.1f}% exceeds threshold "
                    f"of {self.config.thresholds.cpu_percent}%",
                    metrics,
                    operation
                ))
        
        # Check memory usage
        if metrics.memory_percent > self.config.thresholds.memory_percent:
            if self._should_alert("memory", current_time):
                alerts.append(self._create_alert(
                    "Memory Usage Alert",
                    f"Memory usage at {metrics.memory_percent:.1f}% exceeds threshold "
                    f"of {self.config.thresholds.memory_percent}%",
                    metrics,
                    operation
                ))
        
        # Check disk I/O
        disk_io_mbps = (metrics.disk_io_read + metrics.disk_io_write) / (1024 * 1024)
        if disk_io_mbps > self.config.thresholds.disk_io_mbps:
            if self._should_alert("disk_io", current_time):
                alerts.append(self._create_alert(
                    "Disk I/O Alert",
                    f"Disk I/O at {disk_io_mbps:.1f} MB/s exceeds threshold "
                    f"of {self.config.thresholds.disk_io_mbps} MB/s",
                    metrics,
                    operation
                ))
        
        # Check thread count
        if metrics.thread_count > self.config.thresholds.thread_count:
            if self._should_alert("threads", current_time):
                alerts.append(self._create_alert(
                    "Thread Count Alert",
                    f"Thread count of {metrics.thread_count} exceeds threshold "
                    f"of {self.config.thresholds.thread_count}",
                    metrics,
                    operation
                ))
        
        # Check connection count
        if metrics.connection_count > self.config.thresholds.connection_count:
            if self._should_alert("connections", current_time):
                alerts.append(self._create_alert(
                    "Connection Count Alert",
                    f"Connection count of {metrics.connection_count} exceeds threshold "
                    f"of {self.config.thresholds.connection_count}",
                    metrics,
                    operation
                ))
        
        # Check file handles
        if metrics.open_files > self.config.thresholds.file_handles:
            if self._should_alert("files", current_time):
                alerts.append(self._create_alert(
                    "File Handle Alert",
                    f"Open file handles count of {metrics.open_files} exceeds threshold "
                    f"of {self.config.thresholds.file_handles}",
                    metrics,
                    operation
                ))
        
        # Queue alerts for processing
        for alert in alerts:
            self.alert_queue.put(alert)
            self._record_alert(alert)
        
        return alerts
    
    def _should_alert(self, alert_type: str, current_time: datetime) -> bool:
        """Check if alert should be sent based on cooldown."""
        last_alert_time = self.last_alert.get(alert_type)
        if last_alert_time is None:
            return True
        
        return current_time - last_alert_time > self.config.cooldown
    
    def _create_alert(
        self,
        title: str,
        message: str,
        metrics: ResourceMetrics,
        operation: str
    ) -> Dict[str, Any]:
        """Create alert data structure."""
        return {
            "title": title,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_io_read": metrics.disk_io_read,
                "disk_io_write": metrics.disk_io_write,
                "thread_count": metrics.thread_count,
                "connection_count": metrics.connection_count,
                "open_files": metrics.open_files
            }
        }
    
    def _process_alerts(self):
        """Process queued alerts."""
        while True:
            try:
                alert = self.alert_queue.get()
                
                if self.config.enabled:
                    # Send email alerts
                    if self.config.email_recipients:
                        self._send_email_alert(alert)
                    
                    # Send Slack alerts
                    if self.config.slack_webhook:
                        self._send_slack_alert(alert)
                    
                    # Send Teams alerts
                    if self.config.teams_webhook:
                        self._send_teams_alert(alert)
                    
                    # Call custom handler
                    if self.config.custom_handler:
                        try:
                            self.config.custom_handler(
                                alert["title"],
                                alert
                            )
                        except Exception as e:
                            logger.error(f"Custom alert handler failed: {e}")
                
                self.alert_queue.task_done()
                
            except Exception as e:
                logger.error(f"Alert processing failed: {e}")
    
    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send email alert."""
        try:
            msg = MIMEMultipart()
            msg["Subject"] = f"Resource Alert: {alert['title']}"
            msg["From"] = "resource-monitor@localhost"
            msg["To"] = ", ".join(self.config.email_recipients)
            
            body = f"""
            {alert['message']}
            
            Operation: {alert['operation']}
            Time: {alert['timestamp']}
            
            Resource Metrics:
            - CPU: {alert['metrics']['cpu_percent']:.1f}%
            - Memory: {alert['metrics']['memory_percent']:.1f}%
            - Disk I/O Read: {alert['metrics']['disk_io_read'] / (1024*1024):.1f} MB/s
            - Disk I/O Write: {alert['metrics']['disk_io_write'] / (1024*1024):.1f} MB/s
            - Threads: {alert['metrics']['thread_count']}
            - Connections: {alert['metrics']['connection_count']}
            - Open Files: {alert['metrics']['open_files']}
            """
            
            msg.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP("localhost") as server:
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_slack_alert(self, alert: Dict[str, Any]):
        """Send Slack alert."""
        try:
            message = {
                "text": f"*{alert['title']}*\n"
                        f"{alert['message']}\n\n"
                        f"Operation: {alert['operation']}\n"
                        f"Time: {alert['timestamp']}\n\n"
                        f"*Resource Metrics:*\n"
                        f"• CPU: {alert['metrics']['cpu_percent']:.1f}%\n"
                        f"• Memory: {alert['metrics']['memory_percent']:.1f}%\n"
                        f"• Disk I/O: {(alert['metrics']['disk_io_read'] + alert['metrics']['disk_io_write']) / (1024*1024):.1f} MB/s\n"
                        f"• Threads: {alert['metrics']['thread_count']}\n"
                        f"• Connections: {alert['metrics']['connection_count']}\n"
                        f"• Open Files: {alert['metrics']['open_files']}"
            }
            
            requests.post(
                self.config.slack_webhook,
                json=message,
                timeout=5
            )
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_teams_alert(self, alert: Dict[str, Any]):
        """Send Microsoft Teams alert."""
        try:
            message = {
                "type": "message",
                "attachments": [{
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "type": "AdaptiveCard",
                        "body": [
                            {
                                "type": "TextBlock",
                                "size": "Large",
                                "weight": "Bolder",
                                "text": alert["title"]
                            },
                            {
                                "type": "TextBlock",
                                "text": alert["message"]
                            },
                            {
                                "type": "FactSet",
                                "facts": [
                                    {
                                        "title": "Operation",
                                        "value": alert["operation"]
                                    },
                                    {
                                        "title": "Time",
                                        "value": alert["timestamp"]
                                    }
                                ]
                            }
                        ]
                    }
                }]
            }
            
            requests.post(
                self.config.teams_webhook,
                json=message,
                timeout=5
            )
            
        except Exception as e:
            logger.error(f"Failed to send Teams alert: {e}")
    
    def _record_alert(self, alert: Dict[str, Any]):
        """Record alert in history."""
        alert_type = alert["title"].split()[0].lower()
        self.last_alert[alert_type] = datetime.now()
        
        self.alert_history[alert_type].append(alert)
        self._trim_history(alert_type)
        self._save_history()
    
    def _trim_history(self, alert_type: str, max_entries: int = 1000):
        """Trim alert history to prevent excessive growth."""
        if len(self.alert_history[alert_type]) > max_entries:
            self.alert_history[alert_type] = \
                self.alert_history[alert_type][-max_entries:]
    
    def _load_history(self):
        """Load alert history from file."""
        try:
            if self.alert_history_file.exists():
                with open(self.alert_history_file) as f:
                    self.alert_history = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load alert history: {e}")
    
    def _save_history(self):
        """Save alert history to file."""
        try:
            with open(self.alert_history_file, "w") as f:
                json.dump(self.alert_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alert history: {e}")

def create_alert_manager(
    email_recipients: Optional[List[str]] = None,
    slack_webhook: Optional[str] = None,
    teams_webhook: Optional[str] = None,
    custom_handler: Optional[Callable] = None
) -> ResourceAlertManager:
    """Create alert manager with configuration."""
    config = AlertConfig(
        email_recipients=email_recipients,
        slack_webhook=slack_webhook,
        teams_webhook=teams_webhook,
        custom_handler=custom_handler
    )
    return ResourceAlertManager(config)

if __name__ == "__main__":
    # Example usage
    alert_manager = create_alert_manager(
        email_recipients=["admin@example.com"],
        slack_webhook="https://hooks.slack.com/services/..."
    )
    
    # Simulate high CPU metrics
    metrics = ResourceMetrics(
        timestamp=datetime.now(),
        cpu_percent=90.0,
        memory_percent=50.0,
        disk_io_read=1000000,
        disk_io_write=500000,
        open_files=100,
        thread_count=50,
        connection_count=20,
        operation="test_operation",
        duration=1.0
    )
    
    alerts = alert_manager.check_metrics(metrics, "test_operation")
    print(f"Generated alerts: {len(alerts)}")
