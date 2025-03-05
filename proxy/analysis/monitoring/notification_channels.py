"""Multi-channel notification system for validation updates."""

import aiohttp
import asyncio
from typing import Dict, List, Any, Optional, Union, Protocol
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import json
import jinja2
from datetime import datetime

from .validation_notifications import ValidationNotifier, EmailConfig

logger = logging.getLogger(__name__)

class NotificationChannel(ABC):
    """Base class for notification channels."""
    
    @abstractmethod
    async def send_notification(
        self,
        title: str,
        content: Dict[str, Any],
        attachments: Optional[List[Path]] = None,
        **kwargs
    ):
        """Send notification through channel."""
        pass

@dataclass
class SlackConfig:
    """Slack notification configuration."""
    webhook_url: str
    channel: Optional[str] = None
    username: Optional[str] = None
    icon_emoji: Optional[str] = None

class SlackChannel(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(
        self,
        config: SlackConfig
    ):
        self.config = config
    
    async def send_notification(
        self,
        title: str,
        content: Dict[str, Any],
        attachments: Optional[List[Path]] = None,
        **kwargs
    ):
        """Send Slack notification."""
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": title
                }
            }
        ]
        
        # Add content sections
        for key, value in content.items():
            if isinstance(value, (list, dict)):
                text = f"*{key}*\n```{json.dumps(value, indent=2)}```"
            else:
                text = f"*{key}*\n{value}"
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": text
                }
            })
        
        # Prepare message
        message = {
            "blocks": blocks,
            "text": title  # Fallback text
        }
        
        if self.config.channel:
            message["channel"] = self.config.channel
        if self.config.username:
            message["username"] = self.config.username
        if self.config.icon_emoji:
            message["icon_emoji"] = self.config.icon_emoji
        
        # Send message
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.webhook_url,
                json=message
            ) as response:
                if response.status not in (200, 201):
                    raise ValueError(
                        f"Slack notification failed: {await response.text()}"
                    )

@dataclass
class TeamsConfig:
    """Microsoft Teams notification configuration."""
    webhook_url: str
    theme_color: Optional[str] = "#0078D7"

class TeamsChannel(NotificationChannel):
    """Microsoft Teams notification channel."""
    
    def __init__(
        self,
        config: TeamsConfig
    ):
        self.config = config
    
    async def send_notification(
        self,
        title: str,
        content: Dict[str, Any],
        attachments: Optional[List[Path]] = None,
        **kwargs
    ):
        """Send Teams notification."""
        sections = []
        
        for key, value in content.items():
            if isinstance(value, (list, dict)):
                text = f"**{key}**\n\n```json\n{json.dumps(value, indent=2)}\n```"
            else:
                text = f"**{key}**\n\n{value}"
            
            sections.append({
                "text": text
            })
        
        card = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": title,
            "themeColor": self.config.theme_color,
            "title": title,
            "sections": sections
        }
        
        # Send card
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.webhook_url,
                json=card
            ) as response:
                if response.status not in (200, 201):
                    raise ValueError(
                        f"Teams notification failed: {await response.text()}"
                    )

@dataclass
class WebhookConfig:
    """Generic webhook notification configuration."""
    url: str
    headers: Optional[Dict[str, str]] = None
    method: str = "POST"

class WebhookChannel(NotificationChannel):
    """Generic webhook notification channel."""
    
    def __init__(
        self,
        config: WebhookConfig
    ):
        self.config = config
    
    async def send_notification(
        self,
        title: str,
        content: Dict[str, Any],
        attachments: Optional[List[Path]] = None,
        **kwargs
    ):
        """Send webhook notification."""
        payload = {
            "title": title,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        headers = self.config.headers or {}
        
        async with aiohttp.ClientSession() as session:
            async with session.request(
                self.config.method,
                self.config.url,
                json=payload,
                headers=headers
            ) as response:
                if response.status not in (200, 201):
                    raise ValueError(
                        f"Webhook notification failed: {await response.text()}"
                    )

class NotificationManager:
    """Manage multiple notification channels."""
    
    def __init__(self):
        self.channels: Dict[str, NotificationChannel] = {}
    
    def add_channel(
        self,
        name: str,
        channel: NotificationChannel
    ):
        """Add notification channel."""
        self.channels[name] = channel
    
    def remove_channel(
        self,
        name: str
    ):
        """Remove notification channel."""
        self.channels.pop(name, None)
    
    async def notify(
        self,
        title: str,
        content: Dict[str, Any],
        channels: Optional[List[str]] = None,
        attachments: Optional[List[Path]] = None,
        **kwargs
    ):
        """Send notification to specified channels."""
        channels = channels or list(self.channels.keys())
        tasks = []
        
        for channel_name in channels:
            if channel_name not in self.channels:
                logger.warning(f"Channel {channel_name} not found")
                continue
            
            channel = self.channels[channel_name]
            tasks.append(
                channel.send_notification(
                    title,
                    content,
                    attachments,
                    **kwargs
                )
            )
        
        # Send notifications concurrently
        results = await asyncio.gather(
            *tasks,
            return_exceptions=True
        )
        
        # Check for errors
        for channel_name, result in zip(channels, results):
            if isinstance(result, Exception):
                logger.error(
                    f"Notification failed for channel {channel_name}: {result}"
                )

def create_notification_manager() -> NotificationManager:
    """Create notification manager."""
    return NotificationManager()

if __name__ == "__main__":
    # Example usage
    from .validation_notifications import create_validation_notifier
    
    async def main():
        # Create notification manager
        manager = create_notification_manager()
        
        # Add email channel
        email_notifier = create_validation_notifier(
            "smtp.example.com",
            587,
            "user@example.com",
            "password"
        )
        manager.add_channel("email", email_notifier)
        
        # Add Slack channel
        slack_config = SlackConfig(
            webhook_url="https://hooks.slack.com/services/xxx",
            channel="#monitoring"
        )
        manager.add_channel("slack", SlackChannel(slack_config))
        
        # Add Teams channel
        teams_config = TeamsConfig(
            webhook_url="https://outlook.office.com/webhook/xxx"
        )
        manager.add_channel("teams", TeamsChannel(teams_config))
        
        # Send multi-channel notification
        await manager.notify(
            "Validation Report",
            {
                "Status": "Success",
                "Metrics": {
                    "silhouette": 0.82,
                    "cluster_count": 5
                },
                "Issues": ["Low cohesion in cluster 2"],
                "Actions": ["Review cluster parameters"]
            },
            channels=["email", "slack", "teams"]
        )
    
    asyncio.run(main())
