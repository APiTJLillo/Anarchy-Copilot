"""Scheduler for animation events."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
import heapq
from functools import partial

from .animation_events import AnimationEventManager, AnimationEvent, EventConfig

logger = logging.getLogger(__name__)

@dataclass
class SchedulerConfig:
    """Configuration for event scheduler."""
    max_scheduled: int = 1000
    min_interval: float = 0.01
    max_delay: float = 3600
    cleanup_interval: float = 60
    check_interval: float = 0.1
    persist: bool = True
    output_path: Optional[Path] = None

class ScheduledEvent:
    """Scheduled animation event."""
    
    def __init__(
        self,
        event: Union[str, AnimationEvent],
        trigger_time: datetime,
        data: Optional[Dict[str, Any]] = None,
        condition: Optional[Callable[[], bool]] = None,
        repeat: bool = False,
        interval: Optional[float] = None
    ):
        self.event = (
            event if isinstance(event, AnimationEvent)
            else AnimationEvent(event, 0, data)
        )
        self.trigger_time = trigger_time
        self.condition = condition
        self.repeat = repeat
        self.interval = interval
        self.executed = False
        self.attempts = 0
        self.last_execution = None
    
    def should_trigger(self, current_time: datetime) -> bool:
        """Check if event should trigger."""
        if self.executed and not self.repeat:
            return False
        
        if current_time < self.trigger_time:
            return False
        
        if self.condition and not self.condition():
            return False
        
        if self.repeat and self.last_execution:
            next_trigger = self.last_execution + timedelta(seconds=self.interval or 0)
            if current_time < next_trigger:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scheduled event to dictionary."""
        return {
            "event": self.event.to_dict(),
            "trigger_time": self.trigger_time.isoformat(),
            "repeat": self.repeat,
            "interval": self.interval,
            "executed": self.executed,
            "attempts": self.attempts,
            "last_execution": (
                self.last_execution.isoformat()
                if self.last_execution else None
            )
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduledEvent":
        """Create scheduled event from dictionary."""
        return cls(
            event=AnimationEvent.from_dict(data["event"]),
            trigger_time=datetime.fromisoformat(data["trigger_time"]),
            repeat=data.get("repeat", False),
            interval=data.get("interval"),
            data=data.get("event", {}).get("data")
        )

class EventScheduler:
    """Schedule and manage animation events."""
    
    def __init__(
        self,
        event_manager: AnimationEventManager,
        config: SchedulerConfig
    ):
        self.event_manager = event_manager
        self.config = config
        self.scheduled_events: List[ScheduledEvent] = []
        self.running = False
        self.event_heap: List[Tuple[datetime, int, ScheduledEvent]] = []
        self.event_counter = 0
        self.event_types: Dict[str, List[ScheduledEvent]] = defaultdict(list)
        
        # Load persisted events
        if config.persist:
            self.load_events()
    
    def schedule_event(
        self,
        event: Union[str, AnimationEvent],
        delay: Optional[float] = None,
        trigger_time: Optional[datetime] = None,
        data: Optional[Dict[str, Any]] = None,
        condition: Optional[Callable[[], bool]] = None,
        repeat: bool = False,
        interval: Optional[float] = None
    ) -> ScheduledEvent:
        """Schedule new event."""
        if len(self.scheduled_events) >= self.config.max_scheduled:
            raise ValueError("Maximum number of scheduled events reached")
        
        if delay is not None:
            if not 0 <= delay <= self.config.max_delay:
                raise ValueError(f"Delay must be between 0 and {self.config.max_delay}")
            trigger_time = datetime.now() + timedelta(seconds=delay)
        elif trigger_time is None:
            trigger_time = datetime.now()
        
        if interval is not None and interval < self.config.min_interval:
            raise ValueError(f"Interval must be at least {self.config.min_interval}")
        
        scheduled = ScheduledEvent(
            event=event,
            trigger_time=trigger_time,
            data=data,
            condition=condition,
            repeat=repeat,
            interval=interval
        )
        
        self.scheduled_events.append(scheduled)
        heapq.heappush(
            self.event_heap,
            (trigger_time, self.event_counter, scheduled)
        )
        self.event_counter += 1
        
        if isinstance(event, str):
            self.event_types[event].append(scheduled)
        else:
            self.event_types[event.name].append(scheduled)
        
        return scheduled
    
    def cancel_event(
        self,
        scheduled: ScheduledEvent
    ):
        """Cancel scheduled event."""
        if scheduled in self.scheduled_events:
            self.scheduled_events.remove(scheduled)
            
            # Remove from type mapping
            event_name = (
                scheduled.event if isinstance(scheduled.event, str)
                else scheduled.event.name
            )
            if event_name in self.event_types:
                self.event_types[event_name].remove(scheduled)
            
            # Note: Event remains in heap but will be skipped during execution
    
    def cancel_events_by_type(
        self,
        event_type: str
    ):
        """Cancel all events of a specific type."""
        if event_type in self.event_types:
            for event in self.event_types[event_type][:]:
                self.cancel_event(event)
    
    async def start(self):
        """Start event scheduler."""
        self.running = True
        await self._schedule_loop()
    
    async def stop(self):
        """Stop event scheduler."""
        self.running = False
        
        # Persist events if configured
        if self.config.persist:
            self.save_events()
    
    async def _schedule_loop(self):
        """Main scheduling loop."""
        last_cleanup = datetime.now()
        
        while self.running:
            current_time = datetime.now()
            
            # Check for cleanup
            if (current_time - last_cleanup).total_seconds() >= self.config.cleanup_interval:
                self._cleanup_events()
                last_cleanup = current_time
            
            # Process due events
            while self.event_heap and self.event_heap[0][0] <= current_time:
                _, _, scheduled = heapq.heappop(self.event_heap)
                
                if scheduled not in self.scheduled_events:
                    continue  # Event was cancelled
                
                if scheduled.should_trigger(current_time):
                    # Trigger event
                    try:
                        await self.event_manager.trigger_event(
                            scheduled.event,
                            scheduled.event.data
                        )
                        scheduled.last_execution = current_time
                        scheduled.attempts += 1
                        scheduled.executed = True
                        
                        # Reschedule if repeating
                        if scheduled.repeat:
                            next_time = current_time + timedelta(
                                seconds=scheduled.interval or 0
                            )
                            heapq.heappush(
                                self.event_heap,
                                (next_time, self.event_counter, scheduled)
                            )
                            self.event_counter += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to trigger event: {e}")
            
            await asyncio.sleep(self.config.check_interval)
    
    def _cleanup_events(self):
        """Clean up completed non-repeating events."""
        self.scheduled_events = [
            event for event in self.scheduled_events
            if not event.executed or event.repeat
        ]
    
    def save_events(self):
        """Save scheduled events to disk."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            events_file = output_path / "scheduled_events.json"
            with open(events_file, "w") as f:
                json.dump(
                    [event.to_dict() for event in self.scheduled_events],
                    f,
                    indent=2
                )
            
            logger.info(f"Saved scheduled events to {events_file}")
            
        except Exception as e:
            logger.error(f"Failed to save events: {e}")
    
    def load_events(self):
        """Load scheduled events from disk."""
        if not self.config.output_path:
            return
        
        try:
            events_file = self.config.output_path / "scheduled_events.json"
            if not events_file.exists():
                return
            
            with open(events_file) as f:
                events_data = json.load(f)
            
            for data in events_data:
                try:
                    event = ScheduledEvent.from_dict(data)
                    self.scheduled_events.append(event)
                    heapq.heappush(
                        self.event_heap,
                        (event.trigger_time, self.event_counter, event)
                    )
                    self.event_counter += 1
                    
                    event_name = (
                        event.event if isinstance(event.event, str)
                        else event.event.name
                    )
                    self.event_types[event_name].append(event)
                    
                except Exception as e:
                    logger.error(f"Failed to load event: {e}")
            
            logger.info(f"Loaded scheduled events from {events_file}")
            
        except Exception as e:
            logger.error(f"Failed to load events: {e}")

def create_event_scheduler(
    event_manager: AnimationEventManager,
    output_path: Optional[Path] = None
) -> EventScheduler:
    """Create event scheduler."""
    config = SchedulerConfig(output_path=output_path)
    return EventScheduler(event_manager, config)

if __name__ == "__main__":
    # Example usage
    from .animation_events import create_event_manager
    from .animation_controls import create_animation_controls
    from .interactive_easing import create_interactive_easing
    from .easing_visualization import create_easing_visualizer
    from .easing_transitions import create_easing_functions
    
    async def main():
        # Create components
        easing = create_easing_functions()
        visualizer = create_easing_visualizer(easing)
        interactive = create_interactive_easing(visualizer)
        controls = create_animation_controls(interactive)
        events = create_event_manager(controls)
        scheduler = create_event_scheduler(
            events,
            output_path=Path("event_scheduler")
        )
        
        # Schedule events
        scheduler.schedule_event(
            "animation:start",
            delay=1.0,
            data={"reason": "initial"}
        )
        
        scheduler.schedule_event(
            "progress:update",
            delay=2.0,
            repeat=True,
            interval=0.5,
            data={"progress": 0}
        )
        
        # Run scheduler
        await scheduler.start()
        await asyncio.sleep(5)
        await scheduler.stop()
    
    asyncio.run(main())
