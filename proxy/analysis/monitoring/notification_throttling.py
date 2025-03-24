"""Rate limiting and throttling for notifications."""

import asyncio
from typing import Dict, List, Any, Optional, Set, DefaultDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import json
from pathlib import Path

from .notification_channels import NotificationChannel, NotificationManager

logger = logging.getLogger(__name__)

@dataclass
class ThrottleConfig:
    """Configuration for notification throttling."""
    max_rate: int = 10  # Max notifications per window
    window_seconds: int = 60  # Time window in seconds
    burst_limit: int = 3  # Max consecutive notifications
    cooldown_seconds: int = 300  # Cooldown period after burst
    batch_size: int = 5  # Max notifications to batch together
    batch_window_seconds: int = 30  # Time window for batching

@dataclass
class ChannelState:
    """Track notification state for a channel."""
    notifications: List[datetime] = field(default_factory=list)
    last_notification: Optional[datetime] = None
    burst_count: int = 0
    cooldown_until: Optional[datetime] = None
    batched_notifications: List[Dict[str, Any]] = field(default_factory=list)
    batch_timer: Optional[asyncio.Task] = None

class NotificationThrottler:
    """Rate limit and throttle notifications."""
    
    def __init__(
        self,
        manager: NotificationManager,
        config: ThrottleConfig = None
    ):
        self.manager = manager
        self.config = config or ThrottleConfig()
        self.channel_states: DefaultDict[str, ChannelState] = defaultdict(ChannelState)
    
    def _check_rate_limit(
        self,
        channel: str
    ) -> bool:
        """Check if channel has exceeded rate limit."""
        state = self.channel_states[channel]
        now = datetime.now()
        
        # Remove old notifications outside window
        window_start = now - timedelta(seconds=self.config.window_seconds)
        state.notifications = [
            t for t in state.notifications
            if t >= window_start
        ]
        
        return len(state.notifications) < self.config.max_rate
    
    def _check_burst_limit(
        self,
        channel: str
    ) -> bool:
        """Check if channel has exceeded burst limit."""
        state = self.channel_states[channel]
        now = datetime.now()
        
        # Check cooldown
        if state.cooldown_until and now < state.cooldown_until:
            return False
        
        # Reset burst count if enough time has passed
        if (
            state.last_notification and
            (now - state.last_notification) > timedelta(seconds=self.config.window_seconds)
        ):
            state.burst_count = 0
            state.cooldown_until = None
        
        return state.burst_count < self.config.burst_limit
    
    def _update_channel_state(
        self,
        channel: str
    ):
        """Update channel state after notification."""
        state = self.channel_states[channel]
        now = datetime.now()
        
        state.notifications.append(now)
        state.last_notification = now
        state.burst_count += 1
        
        # Start cooldown if burst limit reached
        if state.burst_count >= self.config.burst_limit:
            state.cooldown_until = now + timedelta(seconds=self.config.cooldown_seconds)
            state.burst_count = 0
    
    async def _send_batch(
        self,
        channel: str
    ):
        """Send batched notifications."""
        state = self.channel_states[channel]
        if not state.batched_notifications:
            return
        
        # Combine notifications
        combined = {
            "title": "Batched Notifications",
            "content": {
                "notifications": state.batched_notifications,
                "count": len(state.batched_notifications),
                "timestamp_range": {
                    "start": state.batched_notifications[0].get("timestamp"),
                    "end": state.batched_notifications[-1].get("timestamp")
                }
            }
        }
        
        try:
            await self.manager.notify(
                combined["title"],
                combined["content"],
                channels=[channel]
            )
            logger.info(
                f"Sent batched notification to {channel} "
                f"({len(state.batched_notifications)} items)"
            )
        except Exception as e:
            logger.error(f"Failed to send batched notification: {e}")
        
        # Clear batch
        state.batched_notifications = []
        state.batch_timer = None
    
    def _schedule_batch(
        self,
        channel: str
    ):
        """Schedule batch notification."""
        state = self.channel_states[channel]
        
        if state.batch_timer:
            state.batch_timer.cancel()
        
        state.batch_timer = asyncio.create_task(
            self._send_batch_later(channel)
        )
    
    async def _send_batch_later(
        self,
        channel: str
    ):
        """Send batch after delay."""
        await asyncio.sleep(self.config.batch_window_seconds)
        await self._send_batch(channel)
    
    async def notify(
        self,
        title: str,
        content: Dict[str, Any],
        channels: Optional[List[str]] = None,
        force: bool = False,
        **kwargs
    ):
        """Send throttled notification."""
        channels = channels or list(self.manager.channels.keys())
        now = datetime.now()
        
        for channel in channels:
            if channel not in self.manager.channels:
                logger.warning(f"Channel {channel} not found")
                continue
            
            # Check limits unless forced
            if not force:
                if not self._check_rate_limit(channel):
                    logger.warning(
                        f"Rate limit exceeded for channel {channel}"
                    )
                    continue
                
                if not self._check_burst_limit(channel):
                    logger.warning(
                        f"Burst limit exceeded for channel {channel}"
                    )
                    continue
            
            state = self.channel_states[channel]
            
            # Add to batch if not forced
            if not force and len(state.batched_notifications) < self.config.batch_size:
                notification = {
                    "title": title,
                    "content": content,
                    "timestamp": now.isoformat(),
                    **kwargs
                }
                state.batched_notifications.append(notification)
                
                if len(state.batched_notifications) == 1:
                    # Start batch timer for first notification
                    self._schedule_batch(channel)
                
                elif len(state.batched_notifications) >= self.config.batch_size:
                    # Send immediately if batch is full
                    await self._send_batch(channel)
                
                continue
            
            # Send immediately if forced or can't batch
            try:
                await self.manager.notify(
                    title,
                    content,
                    channels=[channel],
                    **kwargs
                )
                self._update_channel_state(channel)
                
            except Exception as e:
                logger.error(
                    f"Failed to send notification to {channel}: {e}"
                )
    
    def get_channel_metrics(
        self,
        channel: str
    ) -> Dict[str, Any]:
        """Get metrics for channel."""
        state = self.channel_states[channel]
        now = datetime.now()
        
        return {
            "notification_count": len(state.notifications),
            "burst_count": state.burst_count,
            "in_cooldown": bool(
                state.cooldown_until and
                now < state.cooldown_until
            ),
            "cooldown_remaining": (
                (state.cooldown_until - now).total_seconds()
                if state.cooldown_until and now < state.cooldown_until
                else 0
            ),
            "batched_count": len(state.batched_notifications),
            "time_since_last": (
                (now - state.last_notification).total_seconds()
                if state.last_notification
                else None
            )
        }
    
    async def flush_batches(self):
        """Send all batched notifications immediately."""
        tasks = []
        
        for channel, state in self.channel_states.items():
            if state.batched_notifications:
                tasks.append(self._send_batch(channel))
        
        if tasks:
            await asyncio.gather(*tasks)

def create_throttled_manager(
    manager: NotificationManager,
    config: Optional[ThrottleConfig] = None
) -> NotificationThrottler:
    """Create throttled notification manager."""
    return NotificationThrottler(manager, config)

if __name__ == "__main__":
    # Example usage
    from .notification_channels import (
        create_notification_manager,
        SlackConfig,
        SlackChannel
    )
    
    async def main():
        # Create notification system
        manager = create_notification_manager()
        
        # Add Slack channel
        slack_config = SlackConfig(
            webhook_url="https://hooks.slack.com/services/xxx",
            channel="#monitoring"
        )
        manager.add_channel("slack", SlackChannel(slack_config))
        
        # Create throttled manager
        throttler = create_throttled_manager(
            manager,
            ThrottleConfig(
                max_rate=5,
                window_seconds=60,
                burst_limit=2,
                batch_size=3,
                batch_window_seconds=10
            )
        )
        
        # Send test notifications
        for i in range(10):
            await throttler.notify(
                f"Test Notification {i}",
                {"value": i}
            )
            await asyncio.sleep(1)
        
        # Check metrics
        metrics = throttler.get_channel_metrics("slack")
        print("Channel Metrics:", json.dumps(metrics, indent=2))
        
        # Flush remaining batches
        await throttler.flush_batches()
    
    asyncio.run(main())
