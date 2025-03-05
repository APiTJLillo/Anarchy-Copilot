"""Priority-based notification routing and queuing."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import heapq
import json
from pathlib import Path

from .notification_throttling import NotificationThrottler, ThrottleConfig
from .notification_channels import NotificationManager

logger = logging.getLogger(__name__)

class Priority(Enum):
    """Notification priority levels."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    DEBUG = 4

@dataclass
class PriorityConfig:
    """Configuration for priority-based routing."""
    queue_size: int = 1000  # Max items in queue
    high_water_mark: float = 0.9  # Queue utilization threshold
    low_water_mark: float = 0.5  # Queue cleanup threshold
    ttl_seconds: Dict[Priority, int] = field(default_factory=lambda: {
        Priority.CRITICAL: 3600,  # 1 hour
        Priority.HIGH: 1800,      # 30 minutes
        Priority.MEDIUM: 900,     # 15 minutes
        Priority.LOW: 300,        # 5 minutes
        Priority.DEBUG: 60        # 1 minute
    })
    channel_priorities: Dict[str, Set[Priority]] = field(default_factory=dict)
    retry_attempts: Dict[Priority, int] = field(default_factory=lambda: {
        Priority.CRITICAL: 5,
        Priority.HIGH: 3,
        Priority.MEDIUM: 2,
        Priority.LOW: 1,
        Priority.DEBUG: 0
    })

class PrioritizedNotification(NamedTuple):
    """Notification with priority information."""
    priority: Priority
    timestamp: datetime
    title: str
    content: Dict[str, Any]
    channels: Optional[List[str]]
    attempt: int = 0
    metadata: Dict[str, Any] = {}

class NotificationQueue:
    """Priority queue for notifications."""
    
    def __init__(self):
        self.queue: List[tuple[int, int, PrioritizedNotification]] = []
        self._counter = 0  # For stable sorting
    
    def put(
        self,
        notification: PrioritizedNotification
    ):
        """Add notification to queue."""
        # Priority tuple: (priority value, timestamp, counter, notification)
        entry = (
            notification.priority.value,
            int(notification.timestamp.timestamp()),
            self._counter,
            notification
        )
        self._counter += 1
        heapq.heappush(self.queue, entry)
    
    def get(self) -> Optional[PrioritizedNotification]:
        """Get next notification from queue."""
        if not self.queue:
            return None
        return heapq.heappop(self.queue)[-1]
    
    def peek(self) -> Optional[PrioritizedNotification]:
        """Peek at next notification without removing."""
        if not self.queue:
            return None
        return self.queue[0][-1]
    
    def __len__(self) -> int:
        return len(self.queue)

class PriorityRouter:
    """Route notifications based on priority."""
    
    def __init__(
        self,
        throttler: NotificationThrottler,
        config: PriorityConfig = None
    ):
        self.throttler = throttler
        self.config = config or PriorityConfig()
        self.queue = NotificationQueue()
        self.processing = False
        self._process_task: Optional[asyncio.Task] = None
    
    def _should_accept(
        self,
        notification: PrioritizedNotification
    ) -> bool:
        """Check if notification should be accepted."""
        # Check queue capacity
        if len(self.queue) >= self.config.queue_size:
            queue_ratio = len(self.queue) / self.config.queue_size
            
            # Only accept higher priority when queue is full
            return (
                queue_ratio > self.config.high_water_mark and
                notification.priority.value <= Priority.HIGH.value
            )
        
        return True
    
    def _should_drop(
        self,
        notification: PrioritizedNotification
    ) -> bool:
        """Check if notification should be dropped."""
        ttl = self.config.ttl_seconds.get(
            notification.priority,
            60  # Default TTL
        )
        
        age = (datetime.now() - notification.timestamp).total_seconds()
        return age > ttl
    
    def _cleanup_queue(self):
        """Remove expired notifications."""
        if not self.queue:
            return
        
        queue_ratio = len(self.queue) / self.config.queue_size
        if queue_ratio < self.config.low_water_mark:
            return
        
        # Remove expired notifications
        valid_entries = []
        while self.queue.queue:
            entry = heapq.heappop(self.queue.queue)
            notification = entry[-1]
            
            if not self._should_drop(notification):
                valid_entries.append(entry)
        
        # Restore valid entries
        self.queue.queue = valid_entries
        heapq.heapify(self.queue.queue)
    
    def _get_channels_for_priority(
        self,
        priority: Priority,
        requested_channels: Optional[List[str]] = None
    ) -> List[str]:
        """Get channels that accept given priority."""
        valid_channels = []
        
        for channel, priorities in self.config.channel_priorities.items():
            if priority in priorities:
                if (
                    not requested_channels or
                    channel in requested_channels
                ):
                    valid_channels.append(channel)
        
        return valid_channels or requested_channels or []
    
    async def notify(
        self,
        title: str,
        content: Dict[str, Any],
        priority: Priority,
        channels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Queue notification with priority."""
        notification = PrioritizedNotification(
            priority=priority,
            timestamp=datetime.now(),
            title=title,
            content=content,
            channels=channels,
            metadata=metadata or {}
        )
        
        if self._should_accept(notification):
            self.queue.put(notification)
            self._cleanup_queue()
            
            # Start processing if not already running
            if not self.processing:
                self._start_processing()
        else:
            logger.warning(
                f"Dropped {priority.name} notification due to queue constraints"
            )
    
    def _start_processing(self):
        """Start processing queue."""
        if not self._process_task:
            self._process_task = asyncio.create_task(self._process_queue())
    
    async def _process_queue(self):
        """Process queued notifications."""
        self.processing = True
        
        try:
            while True:
                notification = self.queue.get()
                if not notification:
                    break
                
                if self._should_drop(notification):
                    continue
                
                channels = self._get_channels_for_priority(
                    notification.priority,
                    notification.channels
                )
                
                if not channels:
                    logger.warning(
                        f"No channels available for {notification.priority.name}"
                    )
                    continue
                
                try:
                    await self.throttler.notify(
                        notification.title,
                        notification.content,
                        channels=channels,
                        force=notification.priority <= Priority.HIGH,
                        metadata=notification.metadata
                    )
                    
                except Exception as e:
                    # Handle retry logic
                    max_attempts = self.config.retry_attempts.get(
                        notification.priority,
                        0
                    )
                    
                    if notification.attempt < max_attempts:
                        # Requeue with incremented attempt counter
                        self.queue.put(notification._replace(
                            attempt=notification.attempt + 1
                        ))
                        logger.warning(
                            f"Retrying {notification.priority.name} notification "
                            f"(attempt {notification.attempt + 1}): {e}"
                        )
                    else:
                        logger.error(
                            f"Failed to send {notification.priority.name} "
                            f"notification after {max_attempts} attempts: {e}"
                        )
                
                # Pause between notifications
                await asyncio.sleep(0.1)
        
        finally:
            self.processing = False
            self._process_task = None
    
    def get_queue_metrics(self) -> Dict[str, Any]:
        """Get queue metrics."""
        priority_counts = defaultdict(int)
        for entry in self.queue.queue:
            notification = entry[-1]
            priority_counts[notification.priority.name] += 1
        
        return {
            "queue_size": len(self.queue),
            "utilization": len(self.queue) / self.config.queue_size,
            "priority_distribution": dict(priority_counts),
            "processing": self.processing
        }

def create_priority_router(
    throttler: NotificationThrottler,
    config: Optional[PriorityConfig] = None
) -> PriorityRouter:
    """Create priority router."""
    return PriorityRouter(throttler, config)

if __name__ == "__main__":
    # Example usage
    from .notification_throttling import create_throttled_manager
    from .notification_channels import (
        create_notification_manager,
        SlackConfig,
        SlackChannel
    )
    
    async def main():
        # Create notification stack
        manager = create_notification_manager()
        
        # Add Slack channel
        slack_config = SlackConfig(
            webhook_url="https://hooks.slack.com/services/xxx",
            channel="#monitoring"
        )
        manager.add_channel("slack", SlackChannel(slack_config))
        
        # Create throttled manager
        throttler = create_throttled_manager(manager)
        
        # Create priority router
        config = PriorityConfig(
            channel_priorities={
                "slack": {Priority.HIGH, Priority.MEDIUM}
            }
        )
        router = create_priority_router(throttler, config)
        
        # Send test notifications
        priorities = [
            Priority.HIGH,
            Priority.MEDIUM,
            Priority.LOW,
            Priority.HIGH,
            Priority.CRITICAL
        ]
        
        for i, priority in enumerate(priorities):
            await router.notify(
                f"{priority.name} Notification {i}",
                {"value": i},
                priority=priority
            )
            await asyncio.sleep(1)
        
        # Check metrics
        metrics = router.get_queue_metrics()
        print("Queue Metrics:", json.dumps(metrics, indent=2))
        
        # Wait for processing to complete
        await asyncio.sleep(5)
    
    asyncio.run(main())
