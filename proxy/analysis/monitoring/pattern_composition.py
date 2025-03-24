"""Composition tools for scheduling patterns."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import asyncio

from .scheduling_patterns import SchedulingPattern, PatternConfig
from .event_scheduler import ScheduledEvent, AnimationEvent

logger = logging.getLogger(__name__)

@dataclass
class CompositionConfig:
    """Configuration for pattern composition."""
    max_depth: int = 5
    max_branches: int = 10
    auto_adjust: bool = True
    normalize_delays: bool = True
    merge_similar: bool = True
    output_path: Optional[Path] = None

class PatternComposer:
    """Compose scheduling patterns."""
    
    def __init__(
        self,
        pattern: SchedulingPattern,
        config: CompositionConfig
    ):
        self.pattern = pattern
        self.config = config
        self.compositions: Dict[str, Callable] = {}
        self.register_default_compositions()
    
    def register_default_compositions(self):
        """Register default pattern compositions."""
        self.compositions.update({
            "chain": self.chain_patterns,
            "merge": self.merge_patterns,
            "alternate": self.alternate_patterns,
            "nest": self.nest_patterns,
            "interleave": self.interleave_patterns,
            "pipeline": self.pipeline_patterns
        })
    
    def register_composition(
        self,
        name: str,
        func: Callable
    ):
        """Register custom pattern composition."""
        self.compositions[name] = func
    
    def chain_patterns(
        self,
        patterns: List[List[ScheduledEvent]],
        delay: float = 0.0,
        gap: Optional[float] = None
    ) -> List[ScheduledEvent]:
        """Chain patterns in sequence."""
        result = []
        current_delay = delay
        
        for pattern in patterns:
            # Adjust delays relative to chain start
            adjusted = self._adjust_delays(pattern, current_delay)
            result.extend(adjusted)
            
            # Calculate next pattern start
            if pattern:
                max_delay = max(
                    event.trigger_time.timestamp()
                    for event in adjusted
                )
                current_delay = max_delay + (gap or self.pattern.config.cascade_delay)
        
        return result
    
    def merge_patterns(
        self,
        patterns: List[List[ScheduledEvent]],
        delay: float = 0.0,
        normalize: bool = None
    ) -> List[ScheduledEvent]:
        """Merge patterns preserving relative timing."""
        result = []
        should_normalize = (
            normalize if normalize is not None
            else self.config.normalize_delays
        )
        
        if should_normalize:
            # Normalize delays across patterns
            min_delays = [
                min(
                    event.trigger_time.timestamp()
                    for event in pattern
                )
                for pattern in patterns if pattern
            ]
            if min_delays:
                offset = min(min_delays)
                delay_adjust = delay - offset
                
                for pattern in patterns:
                    adjusted = self._adjust_delays(pattern, delay_adjust)
                    result.extend(adjusted)
        else:
            # Preserve original timing
            for pattern in patterns:
                adjusted = self._adjust_delays(pattern, delay)
                result.extend(adjusted)
        
        if self.config.merge_similar:
            result = self._merge_similar_events(result)
        
        return result
    
    def alternate_patterns(
        self,
        patterns: List[List[ScheduledEvent]],
        count: int,
        delay: float = 0.0,
        interval: Optional[float] = None
    ) -> List[ScheduledEvent]:
        """Alternate between patterns."""
        result = []
        current_delay = delay
        alt_interval = interval or self.pattern.config.cascade_delay
        
        for i in range(count):
            pattern = patterns[i % len(patterns)]
            adjusted = self._adjust_delays(pattern, current_delay)
            result.extend(adjusted)
            current_delay += alt_interval
        
        return result
    
    def nest_patterns(
        self,
        outer: List[ScheduledEvent],
        inner: List[ScheduledEvent],
        delay: float = 0.0,
        relative: bool = True
    ) -> List[ScheduledEvent]:
        """Nest one pattern within another."""
        if not outer or not inner:
            return []
        
        result = self._adjust_delays(outer, delay)
        
        if relative:
            # Insert inner pattern relative to outer events
            for i, outer_event in enumerate(outer):
                inner_delay = outer_event.trigger_time.timestamp()
                if i < len(outer) - 1:
                    next_delay = outer[i + 1].trigger_time.timestamp()
                    available_time = next_delay - inner_delay
                    
                    # Scale inner pattern to fit
                    scaled = self._scale_pattern_duration(
                        inner,
                        available_time
                    )
                    adjusted = self._adjust_delays(scaled, inner_delay)
                    result.extend(adjusted)
        else:
            # Insert complete inner pattern at each outer event
            for outer_event in outer:
                inner_delay = outer_event.trigger_time.timestamp()
                adjusted = self._adjust_delays(inner, inner_delay)
                result.extend(adjusted)
        
        return result
    
    def interleave_patterns(
        self,
        patterns: List[List[ScheduledEvent]],
        delay: float = 0.0,
        spacing: Optional[float] = None
    ) -> List[ScheduledEvent]:
        """Interleave multiple patterns."""
        if not patterns:
            return []
        
        result = []
        current_delay = delay
        space = spacing or self.pattern.config.stagger_delay
        
        # Find longest pattern
        max_len = max(len(pattern) for pattern in patterns)
        
        # Interleave events
        for i in range(max_len):
            for pattern in patterns:
                if i < len(pattern):
                    event = pattern[i]
                    new_time = datetime.fromtimestamp(current_delay)
                    event.trigger_time = new_time
                    result.append(event)
                    current_delay += space
        
        return result
    
    def pipeline_patterns(
        self,
        stages: List[List[ScheduledEvent]],
        delay: float = 0.0,
        stage_gap: Optional[float] = None
    ) -> List[ScheduledEvent]:
        """Create pipeline of patterns."""
        result = []
        current_delay = delay
        gap = stage_gap or self.pattern.config.cascade_delay
        
        for i, stage in enumerate(stages):
            # Each stage starts after previous stage's first event
            stage_delay = current_delay + (i * gap)
            adjusted = self._adjust_delays(stage, stage_delay)
            result.extend(adjusted)
        
        return result
    
    def compose(
        self,
        composition: Union[str, Callable],
        patterns: List[List[ScheduledEvent]],
        **kwargs
    ) -> List[ScheduledEvent]:
        """Compose patterns using named composition or custom function."""
        if isinstance(composition, str):
            if composition not in self.compositions:
                raise ValueError(f"Unknown composition: {composition}")
            composition_func = self.compositions[composition]
        else:
            composition_func = composition
        
        return composition_func(patterns, **kwargs)
    
    def _adjust_delays(
        self,
        events: List[ScheduledEvent],
        delay: float
    ) -> List[ScheduledEvent]:
        """Adjust event delays relative to new start time."""
        if not events:
            return []
        
        # Find current minimum delay
        min_time = min(
            event.trigger_time.timestamp()
            for event in events
        )
        
        # Calculate adjustment
        adjustment = delay - min_time
        
        # Create new events with adjusted times
        return [
            ScheduledEvent(
                event=event.event,
                trigger_time=datetime.fromtimestamp(
                    event.trigger_time.timestamp() + adjustment
                ),
                data=event.event.data,
                condition=event.condition,
                repeat=event.repeat,
                interval=event.interval
            )
            for event in events
        ]
    
    def _scale_pattern_duration(
        self,
        events: List[ScheduledEvent],
        target_duration: float
    ) -> List[ScheduledEvent]:
        """Scale pattern to fit target duration."""
        if not events:
            return []
        
        # Calculate current duration
        start_time = min(
            event.trigger_time.timestamp()
            for event in events
        )
        end_time = max(
            event.trigger_time.timestamp()
            for event in events
        )
        current_duration = end_time - start_time
        
        if current_duration == 0:
            return events
        
        # Calculate scaling factor
        scale = target_duration / current_duration
        
        # Create new events with scaled times
        return [
            ScheduledEvent(
                event=event.event,
                trigger_time=datetime.fromtimestamp(
                    start_time + (
                        (event.trigger_time.timestamp() - start_time) * scale
                    )
                ),
                data=event.event.data,
                condition=event.condition,
                repeat=event.repeat,
                interval=event.interval * scale if event.interval else None
            )
            for event in events
        ]
    
    def _merge_similar_events(
        self,
        events: List[ScheduledEvent]
    ) -> List[ScheduledEvent]:
        """Merge similar events occurring at same time."""
        if not events:
            return []
        
        # Group by time and event name
        grouped: Dict[Tuple[float, str], List[ScheduledEvent]] = {}
        for event in events:
            key = (
                event.trigger_time.timestamp(),
                event.event.name if isinstance(event.event, AnimationEvent)
                else event.event
            )
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(event)
        
        # Merge groups
        result = []
        for events_group in grouped.values():
            if len(events_group) == 1:
                result.extend(events_group)
            else:
                # Merge event data
                merged_data = {}
                for event in events_group:
                    if isinstance(event.event, AnimationEvent):
                        merged_data.update(event.event.data)
                
                # Create merged event
                base_event = events_group[0]
                result.append(
                    ScheduledEvent(
                        event=base_event.event,
                        trigger_time=base_event.trigger_time,
                        data=merged_data,
                        condition=base_event.condition,
                        repeat=base_event.repeat,
                        interval=base_event.interval
                    )
                )
        
        return sorted(
            result,
            key=lambda x: x.trigger_time.timestamp()
        )

def create_pattern_composer(
    pattern: SchedulingPattern,
    output_path: Optional[Path] = None
) -> PatternComposer:
    """Create pattern composer."""
    config = CompositionConfig(output_path=output_path)
    return PatternComposer(pattern, config)

if __name__ == "__main__":
    # Example usage
    from .scheduling_patterns import create_scheduling_pattern
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
        pattern = create_scheduling_pattern(scheduler)
        composer = create_pattern_composer(pattern)
        
        # Example event lists
        events_a = ["animation:start", "progress:update"]
        events_b = ["animation:pause", "animation:resume"]
        
        # Create basic patterns
        sequence_a = pattern.sequence(events_a)
        sequence_b = pattern.sequence(events_b)
        
        # Compose patterns
        result = composer.compose(
            "chain",
            [sequence_a, sequence_b],
            delay=1.0,
            gap=0.5
        )
        
        # Run scheduler
        await scheduler.start()
        await asyncio.sleep(5)
        await scheduler.stop()
    
    asyncio.run(main())
