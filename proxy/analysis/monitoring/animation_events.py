"""Event hooks for animation controls."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .animation_controls import AnimationControls, ControlConfig

logger = logging.getLogger(__name__)

@dataclass
class EventConfig:
    """Configuration for animation events."""
    async_hooks: bool = True
    max_workers: int = 4
    timeout: float = 5.0
    retry_count: int = 3
    queue_size: int = 100
    log_events: bool = True
    output_path: Optional[Path] = None

class AnimationEvent:
    """Base class for animation events."""
    
    def __init__(
        self,
        name: str,
        time: float,
        data: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.time = time
        self.data = data or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "name": self.name,
            "time": self.time,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnimationEvent":
        """Create event from dictionary."""
        return cls(
            name=data["name"],
            time=data["time"],
            data=data.get("data", {})
        )

class AnimationEventManager:
    """Manage animation events and hooks."""
    
    def __init__(
        self,
        controls: AnimationControls,
        config: EventConfig
    ):
        self.controls = controls
        self.config = config
        self.hooks: Dict[str, List[Callable]] = {}
        self.event_queue: List[AnimationEvent] = []
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Register default events
        self.register_default_events()
    
    def register_default_events(self):
        """Register default animation events."""
        # Animation lifecycle events
        self.register_hook("animation:start", self._on_animation_start)
        self.register_hook("animation:pause", self._on_animation_pause)
        self.register_hook("animation:stop", self._on_animation_stop)
        self.register_hook("animation:complete", self._on_animation_complete)
        self.register_hook("animation:loop", self._on_animation_loop)
        
        # Progress events
        self.register_hook("progress:update", self._on_progress_update)
        self.register_hook("progress:quarter", self._on_quarter_complete)
        self.register_hook("progress:half", self._on_half_complete)
        self.register_hook("progress:threequarter", self._on_three_quarter_complete)
        
        # Error events
        self.register_hook("error:animation", self._on_animation_error)
        self.register_hook("error:hook", self._on_hook_error)
    
    def register_hook(
        self,
        event_name: str,
        callback: Callable
    ):
        """Register event hook."""
        if event_name not in self.hooks:
            self.hooks[event_name] = []
        
        if callback not in self.hooks[event_name]:
            self.hooks[event_name].append(callback)
    
    def unregister_hook(
        self,
        event_name: str,
        callback: Callable
    ):
        """Unregister event hook."""
        if event_name in self.hooks and callback in self.hooks[event_name]:
            self.hooks[event_name].remove(callback)
    
    async def trigger_event(
        self,
        event: Union[str, AnimationEvent],
        data: Optional[Dict[str, Any]] = None
    ):
        """Trigger event and execute hooks."""
        if isinstance(event, str):
            event = AnimationEvent(event, self.controls.get_current_time(), data)
        
        if self.config.log_events:
            self._log_event(event)
        
        # Add event to queue
        if len(self.event_queue) < self.config.queue_size:
            self.event_queue.append(event)
        else:
            logger.warning("Event queue full, dropping oldest event")
            self.event_queue.pop(0)
            self.event_queue.append(event)
        
        # Execute hooks
        if event.name in self.hooks:
            if self.config.async_hooks:
                await self._execute_async_hooks(event)
            else:
                self._execute_sync_hooks(event)
    
    async def _execute_async_hooks(
        self,
        event: AnimationEvent
    ):
        """Execute hooks asynchronously."""
        tasks = []
        
        for hook in self.hooks[event.name]:
            task = asyncio.create_task(
                self._execute_hook_with_retry(hook, event)
            )
            tasks.append(task)
        
        # Wait for all hooks with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=self.config.timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Hooks for {event.name} timed out")
    
    def _execute_sync_hooks(
        self,
        event: AnimationEvent
    ):
        """Execute hooks synchronously."""
        for hook in self.hooks[event.name]:
            try:
                hook(event)
            except Exception as e:
                logger.error(f"Hook execution failed: {e}")
                self.trigger_event("error:hook", {"error": str(e)})
    
    async def _execute_hook_with_retry(
        self,
        hook: Callable,
        event: AnimationEvent
    ):
        """Execute hook with retry logic."""
        for attempt in range(self.config.retry_count):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(event)
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        hook,
                        event
                    )
                break
            except Exception as e:
                logger.error(f"Hook attempt {attempt + 1} failed: {e}")
                if attempt == self.config.retry_count - 1:
                    self.trigger_event("error:hook", {"error": str(e)})
                await asyncio.sleep(0.1 * (attempt + 1))
    
    def _log_event(
        self,
        event: AnimationEvent
    ):
        """Log animation event."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            log_file = output_path / "animation_events.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
            
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
    
    # Default event handlers
    async def _on_animation_start(
        self,
        event: AnimationEvent
    ):
        """Handle animation start."""
        logger.info("Animation started")
    
    async def _on_animation_pause(
        self,
        event: AnimationEvent
    ):
        """Handle animation pause."""
        logger.info("Animation paused")
    
    async def _on_animation_stop(
        self,
        event: AnimationEvent
    ):
        """Handle animation stop."""
        logger.info("Animation stopped")
    
    async def _on_animation_complete(
        self,
        event: AnimationEvent
    ):
        """Handle animation completion."""
        logger.info("Animation completed")
    
    async def _on_animation_loop(
        self,
        event: AnimationEvent
    ):
        """Handle animation loop."""
        logger.info("Animation looped")
    
    async def _on_progress_update(
        self,
        event: AnimationEvent
    ):
        """Handle progress update."""
        progress = event.data.get("progress", 0)
        logger.debug(f"Progress: {progress:.2%}")
    
    async def _on_quarter_complete(
        self,
        event: AnimationEvent
    ):
        """Handle 25% completion."""
        logger.info("Animation 25% complete")
    
    async def _on_half_complete(
        self,
        event: AnimationEvent
    ):
        """Handle 50% completion."""
        logger.info("Animation 50% complete")
    
    async def _on_three_quarter_complete(
        self,
        event: AnimationEvent
    ):
        """Handle 75% completion."""
        logger.info("Animation 75% complete")
    
    async def _on_animation_error(
        self,
        event: AnimationEvent
    ):
        """Handle animation error."""
        error = event.data.get("error", "Unknown error")
        logger.error(f"Animation error: {error}")
    
    async def _on_hook_error(
        self,
        event: AnimationEvent
    ):
        """Handle hook error."""
        error = event.data.get("error", "Unknown error")
        logger.error(f"Hook error: {error}")

def create_event_manager(
    controls: AnimationControls,
    output_path: Optional[Path] = None
) -> AnimationEventManager:
    """Create animation event manager."""
    config = EventConfig(output_path=output_path)
    return AnimationEventManager(controls, config)

if __name__ == "__main__":
    # Example usage
    from .animation_controls import create_animation_controls
    from .interactive_easing import create_interactive_easing
    from .easing_visualization import create_easing_visualizer
    from .easing_transitions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    visualizer = create_easing_visualizer(easing)
    interactive = create_interactive_easing(visualizer)
    controls = create_animation_controls(interactive)
    events = create_event_manager(
        controls,
        output_path=Path("animation_events")
    )
    
    # Example custom event handler
    async def on_custom_event(event: AnimationEvent):
        print(f"Custom event triggered: {event.data}")
    
    # Register custom event
    events.register_hook("custom:event", on_custom_event)
    
    # Trigger custom event
    asyncio.run(events.trigger_event(
        "custom:event",
        {"message": "Hello from custom event!"}
    ))
