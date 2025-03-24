"""Real-time updates and transitions for interactive visualizations."""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

import pytest
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .test_costbenefit_interactive_viz import (
    InteractiveVisualizer,
    InteractiveControlConfig
)

@dataclass
class TransitionConfig:
    """Configuration for visualization transitions."""
    enable_transitions: bool = True
    duration: int = 750
    easing: str = "cubic-in-out"
    fade_in: bool = True
    fade_out: bool = True
    fade_duration: int = 200
    smooth_scroll: bool = True
    scroll_duration: int = 500
    highlight_changes: bool = True
    highlight_duration: int = 1000
    highlight_color: str = "#ffeb3b"

@dataclass
class RealtimeConfig:
    """Configuration for real-time updates."""
    enable_updates: bool = True
    update_interval: float = 1.0  # seconds
    max_points: int = 1000
    buffer_size: int = 100
    batch_updates: bool = True
    batch_size: int = 10
    debounce_delay: float = 0.1
    throttle_updates: bool = True
    min_update_interval: float = 0.1
    interpolate_missing: bool = True

class RealtimeVisualizer:
    """Add real-time updates and transitions to visualizations."""

    def __init__(
        self,
        base_visualizer: InteractiveVisualizer,
        transition_config: TransitionConfig,
        realtime_config: RealtimeConfig
    ):
        self.base_visualizer = base_visualizer
        self.transition_config = transition_config
        self.realtime_config = realtime_config
        
        # State management
        self.update_queue: asyncio.Queue = asyncio.Queue()
        self.last_update = datetime.min
        self.update_task: Optional[asyncio.Task] = None
        self.data_buffer: Dict[str, List[Any]] = {}
        self.update_handlers: Dict[str, List[Callable]] = {}

    async def start_updates(self) -> None:
        """Start real-time update processing."""
        if self.update_task is None:
            self.update_task = asyncio.create_task(self._process_updates())

    async def stop_updates(self) -> None:
        """Stop real-time update processing."""
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
            self.update_task = None

    async def _process_updates(self) -> None:
        """Process queued updates."""
        while True:
            try:
                # Batch updates if enabled
                updates = []
                if self.realtime_config.batch_updates:
                    try:
                        while len(updates) < self.realtime_config.batch_size:
                            update = await asyncio.wait_for(
                                self.update_queue.get(),
                                timeout=self.realtime_config.debounce_delay
                            )
                            updates.append(update)
                    except asyncio.TimeoutError:
                        pass
                else:
                    update = await self.update_queue.get()
                    updates = [update]

                if updates:
                    # Apply throttling if enabled
                    current_time = datetime.now()
                    if self.realtime_config.throttle_updates:
                        time_since_last = (current_time - self.last_update).total_seconds()
                        if time_since_last < self.realtime_config.min_update_interval:
                            await asyncio.sleep(
                                self.realtime_config.min_update_interval - time_since_last
                            )

                    # Process updates
                    await self._apply_updates(updates)
                    self.last_update = current_time

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing updates: {e}")
                await asyncio.sleep(1.0)

    async def _apply_updates(self, updates: List[Any]) -> None:
        """Apply updates to visualizations."""
        for update in updates:
            plot_type = update.get("type")
            if plot_type in self.update_handlers:
                for handler in self.update_handlers[plot_type]:
                    try:
                        await handler(update)
                    except Exception as e:
                        print(f"Error in update handler: {e}")

    def add_transition_effects(self, fig: go.Figure) -> go.Figure:
        """Add transition effects to figure."""
        if not self.transition_config.enable_transitions:
            return fig
        
        # Add transition configuration
        fig.update_layout(
            transition={
                "duration": self.transition_config.duration,
                "easing": self.transition_config.easing
            }
        )
        
        # Add fade effects
        if self.transition_config.fade_in or self.transition_config.fade_out:
            fig.update_traces(
                opacity=1.0,
                transforms=[{
                    "type": "fade",
                    "duration": self.transition_config.fade_duration
                }]
            )
        
        # Add smooth scrolling
        if self.transition_config.smooth_scroll:
            fig.update_xaxes(
                rangeslider={
                    "visible": True,
                    "thickness": 0.1
                },
                range=[0, 100]
            )
        
        return fig

    async def update_feature_importance(
        self,
        fig: go.Figure,
        new_data: Dict[str, Any]
    ) -> go.Figure:
        """Update feature importance visualization."""
        # Buffer new data
        for feature, value in new_data.items():
            if feature not in self.data_buffer:
                self.data_buffer[feature] = []
            self.data_buffer[feature].append(value)
            
            # Trim buffer if needed
            if len(self.data_buffer[feature]) > self.realtime_config.buffer_size:
                self.data_buffer[feature] = self.data_buffer[feature][
                    -self.realtime_config.buffer_size:
                ]
        
        # Update plot data with transitions
        with fig.batch_update():
            for trace in fig.data:
                feature = trace.name
                if feature in self.data_buffer:
                    new_values = self.data_buffer[feature]
                    
                    # Interpolate missing values if enabled
                    if (
                        self.realtime_config.interpolate_missing and
                        len(new_values) < self.realtime_config.buffer_size
                    ):
                        new_values = self._interpolate_values(new_values)
                    
                    trace.y = new_values
                    
                    # Add highlight effect for changes
                    if self.transition_config.highlight_changes:
                        trace.marker.color = self.transition_config.highlight_color
                        await asyncio.sleep(
                            self.transition_config.highlight_duration / 1000
                        )
                        trace.marker.color = None
        
        return fig

    def _interpolate_values(self, values: List[float]) -> List[float]:
        """Interpolate missing values."""
        full_size = self.realtime_config.buffer_size
        if len(values) >= full_size:
            return values
        
        # Create evenly spaced indices
        x = np.linspace(0, full_size - 1, len(values))
        x_new = np.arange(full_size)
        
        # Interpolate
        return np.interp(x_new, x, values).tolist()

@pytest.fixture
def realtime_visualizer(interactive_visualizer):
    """Create realtime visualizer for testing."""
    transition_config = TransitionConfig()
    realtime_config = RealtimeConfig()
    return RealtimeVisualizer(
        interactive_visualizer,
        transition_config,
        realtime_config
    )

@pytest.mark.asyncio
async def test_realtime_updates(realtime_visualizer):
    """Test real-time visualization updates."""
    # Create test figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name="test_feature",
        x=list(range(10)),
        y=list(range(10))
    ))
    
    # Add transition effects
    fig = realtime_visualizer.add_transition_effects(fig)
    
    # Verify transition configuration
    assert "transition" in fig.layout
    assert fig.layout.transition.duration == realtime_visualizer.transition_config.duration
    
    # Start updates
    await realtime_visualizer.start_updates()
    
    # Add test update
    update = {
        "type": "feature_importance",
        "test_feature": list(range(10, 20))
    }
    await realtime_visualizer.update_queue.put(update)
    
    # Allow time for update processing
    await asyncio.sleep(0.5)
    
    # Stop updates
    await realtime_visualizer.stop_updates()
    
    # Verify data buffer
    assert "test_feature" in realtime_visualizer.data_buffer
    assert len(realtime_visualizer.data_buffer["test_feature"]) <= (
        realtime_visualizer.realtime_config.buffer_size
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
