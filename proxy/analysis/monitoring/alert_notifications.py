"""Notification channels for anomaly alerts."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import jinja2
import markdown
from pathlib import Path

from .anomaly_alerts import AlertManager, AlertGroup, AlertSeverity, AlertState

class NotificationChannel(Protocol):
    """Protocol for notification channels."""
    async def send_notification(
        self,
        group: AlertGroup,
        template_vars: Dict[str, Any]
    ) -> bool:
        """Send notification through channel."""
        ...

@dataclass
class EmailConfig:
    """Configuration for email notifications."""
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    use_tls: bool = True
    from_address: str = "alerts@example.com"
    to_addresses: List[str] = field(default_factory=list)
    subject_prefix: str = "[Mutation Alert]"
    template_path: Optional[str] = None

@dataclass
class SlackConfig:
    """Configuration for Slack notifications."""
    webhook_url: str
    channel: str
    username: str = "Mutation Monitor"
    icon_emoji: str = ":warning:"
    template_path: Optional[str] = None

@dataclass
class WebhookConfig:
    """Configuration for webhook notifications."""
    url: str
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    template_path: Optional[str] = None
    verify_ssl: bool = True

@dataclass
class NotificationConfig:
    """Configuration for notifications."""
    enabled_channels: Set[str] = field(default_factory=lambda: {"email", "slack"})
    severity_channels: Dict[AlertSeverity, Set[str]] = field(default_factory=dict)
    template_dir: Path = Path("notification_templates")
    batch_interval: timedelta = timedelta(minutes=5)
    max_batch_size: int = 10
    enable_html: bool = True
    enable_markdown: bool = True
    alert_link_base: str = "http://localhost:8080/alerts"

class EmailNotifier:
    """Email notification channel."""
    
    def __init__(
        self,
        config: EmailConfig
    ):
        self.config = config
        self._template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(config.template_path)
            if config.template_path
            else jinja2.PackageLoader(__package__, "templates")
        )
    
    async def send_notification(
        self,
        group: AlertGroup,
        template_vars: Dict[str, Any]
    ) -> bool:
        """Send email notification."""
        try:
            # Render templates
            subject = self._template_env.get_template("email_subject.jinja2").render(
                group=group,
                **template_vars
            )
            
            text_body = self._template_env.get_template("email_body.txt.jinja2").render(
                group=group,
                **template_vars
            )
            
            html_body = self._template_env.get_template("email_body.html.jinja2").render(
                group=group,
                **template_vars
            )
            
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"{self.config.subject_prefix} {subject}"
            msg["From"] = self.config.from_address
            msg["To"] = ", ".join(self.config.to_addresses)
            
            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))
            
            # Send email
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls()
                
                if self.config.username and self.config.password:
                    server.login(self.config.username, self.config.password)
                
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            print(f"Failed to send email notification: {e}")
            return False

class SlackNotifier:
    """Slack notification channel."""
    
    def __init__(
        self,
        config: SlackConfig
    ):
        self.config = config
        self._template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(config.template_path)
            if config.template_path
            else jinja2.PackageLoader(__package__, "templates")
        )
    
    async def send_notification(
        self,
        group: AlertGroup,
        template_vars: Dict[str, Any]
    ) -> bool:
        """Send Slack notification."""
        try:
            # Render template
            message = self._template_env.get_template("slack_message.jinja2").render(
                group=group,
                **template_vars
            )
            
            # Create payload
            payload = {
                "channel": self.config.channel,
                "username": self.config.username,
                "icon_emoji": self.config.icon_emoji,
                "text": message,
                "attachments": [{
                    "color": {
                        AlertSeverity.CRITICAL: "danger",
                        AlertSeverity.HIGH: "warning",
                        AlertSeverity.MEDIUM: "#daa520",
                        AlertSeverity.LOW: "#3498db",
                        AlertSeverity.INFO: "good"
                    }[group.severity],
                    "fields": [
                        {
                            "title": "Pattern Type",
                            "value": group.pattern_type,
                            "short": True
                        },
                        {
                            "title": "Alert Count",
                            "value": str(group.count),
                            "short": True
                        }
                    ]
                }]
            }
            
            # Send notification
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=payload
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            print(f"Failed to send Slack notification: {e}")
            return False

class WebhookNotifier:
    """Webhook notification channel."""
    
    def __init__(
        self,
        config: WebhookConfig
    ):
        self.config = config
        self._template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(config.template_path)
            if config.template_path
            else jinja2.PackageLoader(__package__, "templates")
        )
    
    async def send_notification(
        self,
        group: AlertGroup,
        template_vars: Dict[str, Any]
    ) -> bool:
        """Send webhook notification."""
        try:
            # Render template
            payload = self._template_env.get_template("webhook_payload.jinja2").render(
                group=group,
                **template_vars
            )
            
            # Send notification
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    self.config.method,
                    self.config.url,
                    json=json.loads(payload),
                    headers=self.config.headers,
                    ssl=self.config.verify_ssl
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            print(f"Failed to send webhook notification: {e}")
            return False

class NotificationManager:
    """Manage alert notifications across channels."""
    
    def __init__(
        self,
        alert_manager: AlertManager,
        config: NotificationConfig = None
    ):
        self.alert_manager = alert_manager
        self.config = config or NotificationConfig()
        
        # Notification channels
        self.channels: Dict[str, NotificationChannel] = {}
        
        # Notification state
        self.last_notification: Dict[str, datetime] = {}
        self.notification_batch: Dict[str, List[AlertGroup]] = {}
    
    def add_channel(
        self,
        name: str,
        channel: NotificationChannel
    ):
        """Add notification channel."""
        self.channels[name] = channel
        self.last_notification[name] = datetime.min
        self.notification_batch[name] = []
    
    async def process_notifications(
        self,
        template_vars: Optional[Dict[str, Any]] = None
    ):
        """Process pending notifications."""
        template_vars = template_vars or {}
        current_time = datetime.now()
        
        # Group alerts by channel
        for group in self.alert_manager.active_alerts.values():
            if group.state != AlertState.NEW:
                continue
            
            # Get channels for severity
            channels = self.config.severity_channels.get(
                group.severity,
                self.config.enabled_channels
            )
            
            # Add to channel batches
            for channel_name in channels:
                if channel_name not in self.channels:
                    continue
                
                batch = self.notification_batch[channel_name]
                batch.append(group)
                
                # Check if batch should be sent
                if (
                    len(batch) >= self.config.max_batch_size or
                    current_time - self.last_notification[channel_name] >= self.config.batch_interval
                ):
                    await self._send_batch(
                        channel_name,
                        batch,
                        template_vars
                    )
                    batch.clear()
                    self.last_notification[channel_name] = current_time
    
    async def _send_batch(
        self,
        channel_name: str,
        batch: List[AlertGroup],
        template_vars: Dict[str, Any]
    ):
        """Send batch of notifications through channel."""
        if not batch:
            return
        
        channel = self.channels[channel_name]
        template_vars = {
            **template_vars,
            "base_url": self.config.alert_link_base,
            "batch_size": len(batch)
        }
        
        # Send each alert
        for group in batch:
            success = await channel.send_notification(group, template_vars)
            if success:
                await self.alert_manager.acknowledge_alert(group.id)

def create_notification_manager(
    alert_manager: AlertManager,
    config: Optional[NotificationConfig] = None
) -> NotificationManager:
    """Create notification manager."""
    return NotificationManager(alert_manager, config)

if __name__ == "__main__":
    from .anomaly_alerts import create_alert_manager
    from .anomaly_analysis import create_anomaly_detector
    from .trend_analysis import create_trend_analyzer
    from .adaptation_metrics import create_performance_tracker
    from .preset_adaptation import create_online_adapter
    from .preset_ensemble import create_preset_ensemble
    from .preset_predictions import create_preset_predictor
    from .preset_analytics import create_preset_analytics
    from .mutation_presets import create_preset_manager
    
    async def main():
        # Setup components
        manager = create_preset_manager()
        analytics = create_preset_analytics(manager)
        predictor = create_preset_predictor(analytics)
        ensemble = create_preset_ensemble(predictor)
        adapter = create_online_adapter(ensemble)
        tracker = create_performance_tracker(adapter)
        analyzer = create_trend_analyzer(tracker)
        detector = create_anomaly_detector(tracker, analyzer)
        alert_manager = create_alert_manager(detector)
        notifier = create_notification_manager(alert_manager)
        
        # Add notification channels
        notifier.add_channel(
            "email",
            EmailNotifier(
                EmailConfig(
                    smtp_host="localhost",
                    smtp_port=25,
                    username="test",
                    password="test",
                    to_addresses=["alerts@example.com"]
                )
            )
        )
        
        notifier.add_channel(
            "slack",
            SlackNotifier(
                SlackConfig(
                    webhook_url="https://hooks.slack.com/services/xxx/yyy/zzz",
                    channel="#alerts"
                )
            )
        )
        
        # Generate test alerts
        for i in range(10):
            await alert_manager.process_anomalies("test_preset")
            await notifier.process_notifications({
                "environment": "test",
                "version": "1.0.0"
            })
            await asyncio.sleep(1)
    
    asyncio.run(main())
