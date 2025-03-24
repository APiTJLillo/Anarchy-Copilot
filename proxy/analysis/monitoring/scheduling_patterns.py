"""Common scheduling patterns for events."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import asyncio
from functools import partial

from .event_scheduler import EventScheduler, ScheduledEvent, SchedulerConfig

logger = logging.getLogger(__name__)

@dataclass
class PatternConfig:
    """Configuration for scheduling patterns."""
    cascade_delay: float = 0.1
    stagger_delay: float = 0.05
    batch_size: int = 10
    max_iterations: int = 1000
    random_seed: Optional[int] = None
    output_path: Optional[Path] = None

class SchedulingPattern:
    """Base class for scheduling patterns."""
    
    def __init__(
        self,
        scheduler: EventScheduler,
        config: PatternConfig
    ):
        self.scheduler = scheduler
        self.config = config
        
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    def sequence(
        self,
        events: List[Union[str, AnimationEvent]],
        delay: float = 0.0,
        interval: Optional[float] = None
    ) -> List[ScheduledEvent]:
        """Schedule sequence of events."""
        scheduled = []
        current_delay = delay
        
        for event in events:
            scheduled.append(
                self.scheduler.schedule_event(
                    event,
                    delay=current_delay
                )
            )
            current_delay += interval or self.config.cascade_delay
        
        return scheduled
    
    def parallel(
        self,
        events: List[Union[str, AnimationEvent]],
        delay: float = 0.0
    ) -> List[ScheduledEvent]:
        """Schedule events in parallel."""
        return [
            self.scheduler.schedule_event(event, delay=delay)
            for event in events
        ]
    
    def staggered(
        self,
        events: List[Union[str, AnimationEvent]],
        delay: float = 0.0,
        stagger: Optional[float] = None
    ) -> List[ScheduledEvent]:
        """Schedule events with staggered delays."""
        scheduled = []
        stagger_delay = stagger or self.config.stagger_delay
        
        for i, event in enumerate(events):
            scheduled.append(
                self.scheduler.schedule_event(
                    event,
                    delay=delay + (i * stagger_delay)
                )
            )
        
        return scheduled
    
    def batched(
        self,
        events: List[Union[str, AnimationEvent]],
        batch_size: Optional[int] = None,
        delay: float = 0.0,
        batch_interval: Optional[float] = None
    ) -> List[List[ScheduledEvent]]:
        """Schedule events in batches."""
        size = batch_size or self.config.batch_size
        interval = batch_interval or self.config.cascade_delay
        batches = []
        
        for i in range(0, len(events), size):
            batch = events[i:i + size]
            batches.append(
                self.parallel(
                    batch,
                    delay=delay + (i // size) * interval
                )
            )
        
        return batches
    
    def cascading(
        self,
        event: Union[str, AnimationEvent],
        count: int,
        delay: float = 0.0,
        cascade_delay: Optional[float] = None
    ) -> List[ScheduledEvent]:
        """Schedule cascading repeating events."""
        scheduled = []
        interval = cascade_delay or self.config.cascade_delay
        
        for i in range(count):
            scheduled.append(
                self.scheduler.schedule_event(
                    event,
                    delay=delay + (i * interval),
                    repeat=True,
                    interval=interval * count
                )
            )
        
        return scheduled
    
    def alternating(
        self,
        events: List[Union[str, AnimationEvent]],
        count: int,
        delay: float = 0.0,
        interval: Optional[float] = None
    ) -> List[ScheduledEvent]:
        """Schedule alternating events."""
        scheduled = []
        alt_interval = interval or self.config.cascade_delay
        
        for i in range(count):
            event = events[i % len(events)]
            scheduled.append(
                self.scheduler.schedule_event(
                    event,
                    delay=delay + (i * alt_interval)
                )
            )
        
        return scheduled
    
    def random_sequence(
        self,
        events: List[Union[str, AnimationEvent]],
        count: int,
        delay: float = 0.0,
        min_interval: Optional[float] = None,
        max_interval: Optional[float] = None
    ) -> List[ScheduledEvent]:
        """Schedule random sequence of events."""
        scheduled = []
        current_delay = delay
        
        min_int = min_interval or self.config.cascade_delay
        max_int = max_interval or min_int * 3
        
        for _ in range(count):
            event = np.random.choice(events)
            interval = np.random.uniform(min_int, max_int)
            
            scheduled.append(
                self.scheduler.schedule_event(
                    event,
                    delay=current_delay
                )
            )
            current_delay += interval
        
        return scheduled
    
    def cyclic(
        self,
        events: List[Union[str, AnimationEvent]],
        period: float,
        count: int,
        delay: float = 0.0
    ) -> List[ScheduledEvent]:
        """Schedule events in cyclic pattern."""
        scheduled = []
        interval = period / len(events)
        
        for i in range(count):
            cycle_start = delay + (i * period)
            for j, event in enumerate(events):
                scheduled.append(
                    self.scheduler.schedule_event(
                        event,
                        delay=cycle_start + (j * interval)
                    )
                )
        
        return scheduled
    
    def exponential(
        self,
        event: Union[str, AnimationEvent],
        base: float = 2.0,
        count: int = 5,
        delay: float = 0.0
    ) -> List[ScheduledEvent]:
        """Schedule events with exponential delays."""
        scheduled = []
        current_delay = delay
        
        for i in range(count):
            scheduled.append(
                self.scheduler.schedule_event(
                    event,
                    delay=current_delay
                )
            )
            current_delay += pow(base, i) * self.config.cascade_delay
        
        return scheduled
    
    def fibonacci(
        self,
        event: Union[str, AnimationEvent],
        count: int = 5,
        delay: float = 0.0
    ) -> List[ScheduledEvent]:
        """Schedule events with Fibonacci sequence delays."""
        scheduled = []
        current_delay = delay
        a, b = 1, 1
        
        for _ in range(count):
            scheduled.append(
                self.scheduler.schedule_event(
                    event,
                    delay=current_delay
                )
            )
            current_delay += a * self.config.cascade_delay
            a, b = b, a + b
        
        return scheduled
    
    def conditional_sequence(
        self,
        events: List[Tuple[Union[str, AnimationEvent], Callable[[], bool]]],
        delay: float = 0.0,
        interval: Optional[float] = None
    ) -> List[ScheduledEvent]:
        """Schedule sequence of conditional events."""
        scheduled = []
        current_delay = delay
        
        for event, condition in events:
            scheduled.append(
                self.scheduler.schedule_event(
                    event,
                    delay=current_delay,
                    condition=condition
                )
            )
            current_delay += interval or self.config.cascade_delay
        
        return scheduled
    
    def adaptive_sequence(
        self,
        events: List[Union[str, AnimationEvent]],
        interval_func: Callable[[int], float],
        count: int,
        delay: float = 0.0
    ) -> List[ScheduledEvent]:
        """Schedule sequence with adaptive intervals."""
        scheduled = []
        current_delay = delay
        
        for i in range(count):
            event = events[i % len(events)]
            interval = interval_func(i)
            
            scheduled.append(
                self.scheduler.schedule_event(
                    event,
                    delay=current_delay
                )
            )
            current_delay += interval
        
        return scheduled

def create_scheduling_pattern(
    scheduler: EventScheduler,
    output_path: Optional[Path] = None
) -> SchedulingPattern:
    """Create scheduling pattern."""
    config = PatternConfig(output_path=output_path)
    return SchedulingPattern(scheduler, config)

if __name__ == "__main__":
    # Example usage
    from .event_scheduler import create_event_scheduler
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
        scheduler = create_event_scheduler(events)
        patterns = create_scheduling_pattern(
            scheduler,
            output_path=Path("scheduling_patterns")
        )
        
        # Example patterns
        events_list = [
            "animation:start",
            "progress:update",
            "animation:complete"
        ]
        
        # Sequence pattern
        sequence = patterns.sequence(events_list, delay=1.0)
        
        # Parallel pattern
        parallel = patterns.parallel(events_list, delay=2.0)
        
        # Staggered pattern
        staggered = patterns.staggered(events_list, delay=3.0)
        
        # Run scheduler
        await scheduler.start()
        await asyncio.sleep(5)
        await scheduler.stop()
    
    asyncio.run(main())
