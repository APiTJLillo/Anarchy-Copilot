"""Alert generation and management for monitoring system."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
import asyncio
import logging
from enum import Enum
import json
from pathlib import Path
import aiosmtplib
from email.message import EmailMessage
import aiohttp
from .metrics import MetricValue, TimeseriesMetric

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AlertRule:
    """Rule for generating alerts."""
    name: str
    metric_pattern: str  # Regex pattern for metric names
    condition: str  # Python expression for evaluation
    severity: AlertSeverity
    description: str
    cooldown: timedelta = timedelta(minutes=5)
    aggregation_window: timedelta = timedelta(minutes=1)
    tags_filter: Optional[Dict[str, str]] = None

@dataclass
class Alert:
    """Alert instance."""
    id: str
    rule_name: str
    metric_name: str
    value: float
    threshold: float
    severity: AlertSeverity
    timestamp: datetime
    description: str
    tags: Optional[Dict[str, str]] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class AlertState:
    """Track alert state."""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.resolved_alerts: Dict[str, Alert] = {}
        self.last_triggered: Dict[str, datetime] = {}
    
    def can_trigger(self, rule: AlertRule, metric_name: str) -> bool:
        """Check if alert can be triggered based on cooldown."""
        key = f"{rule.name}:{metric_name}"
        if key not in self.last_triggered:
            return True
        
        now = datetime.now()
        return (now - self.last_triggered[key]) >= rule.cooldown
    
    def mark_triggered(self, rule: AlertRule, metric_name: str):
        """Mark alert as triggered."""
        self.last_triggered[f"{rule.name}:{metric_name}"] = datetime.now()

class AlertHandler:
    """Base class for alert handlers."""
    
    async def handle_alert(self, alert: Alert):
        """Handle an alert."""
        raise NotImplementedError

class EmailAlertHandler(AlertHandler):
    """Send alerts via email."""
    
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
        use_tls: bool = True
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls
    
    async def handle_alert(self, alert: Alert):
        """Send alert via email."""
        message = EmailMessage()
        message.set_content(
            f"Alert: {alert.rule_name}\n"
            f"Severity: {alert.severity.value}\n"
            f"Metric: {alert.metric_name}\n"
            f"Value: {alert.value}\n"
            f"Threshold: {alert.threshold}\n"
            f"Time: {alert.timestamp}\n"
            f"Description: {alert.description}\n"
        )
        
        message["Subject"] = f"[{alert.severity.value.upper()}] {alert.rule_name}"
        message["From"] = self.from_addr
        message["To"] = ", ".join(self.to_addrs)
        
        try:
            async with aiosmtplib.SMTP(
                hostname=self.host,
                port=self.port,
                use_tls=self.use_tls
            ) as smtp:
                await smtp.login(self.username, self.password)
                await smtp.send_message(message)
            
            logger.info(f"Sent email alert: {alert.id}")
            return True
        
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False

class SlackAlertHandler(AlertHandler):
    """Send alerts to Slack."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.session = aiohttp.ClientSession()
    
    async def handle_alert(self, alert: Alert):
        """Send alert to Slack."""
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#7b0000"
        }[alert.severity]
        
        message = {
            "attachments": [
                {
                    "color": color,
                    "title": f"Alert: {alert.rule_name}",
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value,
                            "short": True
                        },
                        {
                            "title": "Metric",
                            "value": alert.metric_name,
                            "short": True
                        },
                        {
                            "title": "Value",
                            "value": str(alert.value),
                            "short": True
                        },
                        {
                            "title": "Threshold",
                            "value": str(alert.threshold),
                            "short": True
                        }
                    ],
                    "text": alert.description,
                    "footer": alert.timestamp.isoformat()
                }
            ]
        }
        
        try:
            async with self.session.post(
                self.webhook_url,
                json=message
            ) as response:
                if response.status < 400:
                    logger.info(f"Sent Slack alert: {alert.id}")
                    return True
                else:
                    logger.error(
                        f"Error sending Slack alert: {response.status}"
                    )
                    return False
        
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False

class AlertManager:
    """Manage alert rules and handle alert lifecycle."""
    
    def __init__(
        self,
        rules_file: Path,
        alert_dir: Path,
        handlers: Optional[List[AlertHandler]] = None
    ):
        self.rules_file = rules_file
        self.alert_dir = alert_dir
        self.handlers = handlers or []
        self.rules: List[AlertRule] = []
        self.state = AlertState()
        
        self.alert_dir.mkdir(parents=True, exist_ok=True)
        self._load_rules()
    
    def _load_rules(self):
        """Load alert rules from file."""
        try:
            with self.rules_file.open() as f:
                rules_data = json.load(f)
            
            self.rules = [
                AlertRule(
                    name=r["name"],
                    metric_pattern=r["metric_pattern"],
                    condition=r["condition"],
                    severity=AlertSeverity(r["severity"]),
                    description=r["description"],
                    cooldown=timedelta(seconds=r.get("cooldown_seconds", 300)),
                    aggregation_window=timedelta(
                        seconds=r.get("aggregation_window_seconds", 60)
                    ),
                    tags_filter=r.get("tags_filter")
                )
                for r in rules_data
            ]
            
        except Exception as e:
            logger.error(f"Error loading alert rules: {e}")
            self.rules = []
    
    def evaluate_metric(
        self,
        metric: MetricValue
    ) -> Optional[Alert]:
        """Evaluate metric against rules."""
        from re import match
        
        for rule in self.rules:
            # Check metric name pattern
            if not match(rule.metric_pattern, metric.name):
                continue
            
            # Check tags filter
            if rule.tags_filter:
                if not metric.tags or not all(
                    metric.tags.get(k) == v
                    for k, v in rule.tags_filter.items()
                ):
                    continue
            
            # Check if alert can be triggered
            if not self.state.can_trigger(rule, metric.name):
                continue
            
            # Evaluate condition
            try:
                threshold = float(eval(
                    rule.condition,
                    {"value": metric.value}
                ))
                if metric.value > threshold:
                    alert = Alert(
                        id=f"{rule.name}_{datetime.now().timestamp()}",
                        rule_name=rule.name,
                        metric_name=metric.name,
                        value=metric.value,
                        threshold=threshold,
                        severity=rule.severity,
                        timestamp=metric.timestamp,
                        description=rule.description,
                        tags=metric.tags
                    )
                    
                    self.state.mark_triggered(rule, metric.name)
                    return alert
                
            except Exception as e:
                logger.error(
                    f"Error evaluating rule {rule.name}: {e}"
                )
        
        return None
    
    async def process_metric(self, metric: MetricValue):
        """Process metric and trigger alerts if needed."""
        alert = self.evaluate_metric(metric)
        if alert:
            await self._handle_alert(alert)
    
    async def _handle_alert(self, alert: Alert):
        """Handle generated alert."""
        # Store alert
        alert_file = self.alert_dir / f"{alert.id}.json"
        with alert_file.open("w") as f:
            json.dump(asdict(alert), f, indent=2)
        
        # Add to active alerts
        self.state.active_alerts[alert.id] = alert
        
        # Send to handlers
        for handler in self.handlers:
            try:
                await handler.handle_alert(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        user: str
    ) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.state.active_alerts:
            return False
        
        alert = self.state.active_alerts[alert_id]
        alert.acknowledged = True
        alert.acknowledged_by = user
        alert.acknowledged_at = datetime.now()
        
        # Update stored alert
        alert_file = self.alert_dir / f"{alert_id}.json"
        with alert_file.open("w") as f:
            json.dump(asdict(alert), f, indent=2)
        
        return True
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id not in self.state.active_alerts:
            return False
        
        alert = self.state.active_alerts.pop(alert_id)
        alert.resolved = True
        alert.resolved_at = datetime.now()
        
        # Move to resolved alerts
        self.state.resolved_alerts[alert_id] = alert
        
        # Update stored alert
        alert_file = self.alert_dir / f"{alert_id}.json"
        with alert_file.open("w") as f:
            json.dump(asdict(alert), f, indent=2)
        
        return True
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get active alerts with optional severity filter."""
        alerts = list(self.state.active_alerts.values())
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_resolved_alerts(
        self,
        since: Optional[datetime] = None
    ) -> List[Alert]:
        """Get resolved alerts since given time."""
        alerts = list(self.state.resolved_alerts.values())
        if since:
            alerts = [
                a for a in alerts
                if a.resolved_at and a.resolved_at >= since
            ]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
