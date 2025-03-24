"""Real-time updates for prediction visualization."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import plotly.graph_objects as go
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
import threading
import queue
import time
import asyncio
from websockets import connect
import websockets

from .prediction_controls import InteractiveControls, ControlConfig
from .prediction_visualization import PredictionVisualizer

logger = logging.getLogger(__name__)

@dataclass
class RealtimeConfig:
    """Configuration for real-time updates."""
    update_interval: float = 1.0  # seconds
    batch_size: int = 10
    buffer_size: int = 1000
    websocket_url: Optional[str] = None
    auto_reconnect: bool = True
    max_retries: int = 3
    output_path: Optional[Path] = None

class RealtimePrediction:
    """Real-time prediction updates."""
    
    def __init__(
        self,
        controls: InteractiveControls,
        config: RealtimeConfig
    ):
        self.controls = controls
        self.config = config
        self.running = False
        self.update_thread: Optional[threading.Thread] = None
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.data_queue = queue.Queue()
        self.update_callbacks: List[Callable] = []
        self.buffer: Dict[str, List[Any]] = defaultdict(list)
    
    async def start_streaming(self):
        """Start real-time data streaming."""
        if self.running:
            return
        
        self.running = True
        
        # Start update thread
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()
        
        # Connect to websocket if configured
        if self.config.websocket_url:
            await self._connect_websocket()
    
    async def stop_streaming(self):
        """Stop real-time streaming."""
        self.running = False
        
        # Stop update thread
        if self.update_thread:
            self.update_thread.join()
            self.update_thread = None
        
        # Close websocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
    
    def register_update_callback(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ):
        """Register callback for data updates."""
        self.update_callbacks.append(callback)
    
    async def send_update(
        self,
        data: Dict[str, Any]
    ):
        """Send update to visualization."""
        self.data_queue.put(data)
        
        # Trim buffer if needed
        for key in self.buffer:
            if len(self.buffer[key]) > self.config.buffer_size:
                self.buffer[key] = self.buffer[key][-self.config.buffer_size:]
    
    def get_recent_data(
        self,
        key: str,
        window: Optional[int] = None
    ) -> List[Any]:
        """Get recent data from buffer."""
        if key not in self.buffer:
            return []
        
        if window:
            return self.buffer[key][-window:]
        return self.buffer[key]
    
    async def update_visualization(
        self,
        fig: go.Figure,
        data: Dict[str, Any]
    ):
        """Update visualization with new data."""
        # Update traces
        for trace in fig.data:
            if trace.name in data:
                new_data = data[trace.name]
                
                # Extend trace data
                if isinstance(new_data, (list, np.ndarray)):
                    trace.x = np.append(trace.x, range(len(new_data)))
                    trace.y = np.append(trace.y, new_data)
                else:
                    trace.x = np.append(trace.x, datetime.now())
                    trace.y = np.append(trace.y, new_data)
        
        # Update layout if needed
        if "layout" in data:
            fig.update_layout(**data["layout"])
        
        # Trigger callbacks
        for callback in self.update_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def _connect_websocket(self):
        """Connect to websocket data source."""
        retries = 0
        
        while (
            self.running and
            retries < self.config.max_retries
        ):
            try:
                async with connect(self.config.websocket_url) as websocket:
                    self.websocket = websocket
                    await self._handle_websocket_messages()
                    retries = 0
                    
            except Exception as e:
                logger.error(f"Websocket error: {e}")
                retries += 1
                
                if self.config.auto_reconnect:
                    await asyncio.sleep(2 ** retries)
                else:
                    break
    
    async def _handle_websocket_messages(self):
        """Handle incoming websocket messages."""
        while self.running:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                await self.send_update(data)
                
            except websockets.ConnectionClosed:
                logger.warning("Websocket connection closed")
                break
                
            except Exception as e:
                logger.error(f"Message handling error: {e}")
    
    def _update_loop(self):
        """Background update loop."""
        batch = []
        last_update = time.time()
        
        while self.running:
            try:
                # Collect data batch
                while (
                    len(batch) < self.config.batch_size and
                    not self.data_queue.empty()
                ):
                    data = self.data_queue.get_nowait()
                    batch.append(data)
                
                # Process batch
                if batch and (
                    len(batch) >= self.config.batch_size or
                    time.time() - last_update >= self.config.update_interval
                ):
                    self._process_batch(batch)
                    batch = []
                    last_update = time.time()
                
                # Small sleep to prevent busy loop
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Update loop error: {e}")
                batch = []
    
    def _process_batch(
        self,
        batch: List[Dict[str, Any]]
    ):
        """Process batch of updates."""
        # Merge batch data
        merged = defaultdict(list)
        for data in batch:
            for key, value in data.items():
                merged[key].append(value)
        
        # Average numerical values
        processed = {}
        for key, values in merged.items():
            if all(isinstance(v, (int, float)) for v in values):
                processed[key] = np.mean(values)
            else:
                processed[key] = values[-1]
        
        # Update buffer
        for key, value in processed.items():
            self.buffer[key].append(value)
        
        # Create event loop for async updates
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Update main figure
            main_fig = self.controls.visualizer.fig
            loop.run_until_complete(
                self.update_visualization(main_fig, processed)
            )
            
        finally:
            loop.close()
    
    def save_buffer(self):
        """Save buffer contents."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            buffer_file = output_path / "data_buffer.json"
            with open(buffer_file, "w") as f:
                json.dump(self.buffer, f, indent=2)
            
            logger.info(f"Saved buffer to {buffer_file}")
            
        except Exception as e:
            logger.error(f"Failed to save buffer: {e}")
    
    def load_buffer(self):
        """Load saved buffer."""
        if not self.config.output_path:
            return
        
        try:
            buffer_file = self.config.output_path / "data_buffer.json"
            if not buffer_file.exists():
                return
            
            with open(buffer_file) as f:
                self.buffer = json.load(f)
            
            logger.info(f"Loaded buffer from {buffer_file}")
            
        except Exception as e:
            logger.error(f"Failed to load buffer: {e}")

def create_realtime_prediction(
    controls: InteractiveControls,
    websocket_url: Optional[str] = None,
    output_path: Optional[Path] = None
) -> RealtimePrediction:
    """Create realtime prediction updater."""
    config = RealtimeConfig(
        websocket_url=websocket_url,
        output_path=output_path
    )
    return RealtimePrediction(controls, config)

if __name__ == "__main__":
    # Example usage
    from .prediction_controls import create_interactive_controls
    from .prediction_visualization import create_prediction_visualizer
    from .easing_prediction import create_easing_predictor
    from .easing_statistics import create_easing_statistics
    from .easing_metrics import create_easing_metrics
    from .easing_functions import create_easing_functions
    
    async def main():
        # Create components
        easing = create_easing_functions()
        metrics = create_easing_metrics(easing)
        stats = create_easing_statistics(metrics)
        predictor = create_easing_predictor(stats)
        visualizer = create_prediction_visualizer(predictor)
        controls = create_interactive_controls(visualizer)
        
        # Create realtime updater
        realtime = create_realtime_prediction(
            controls,
            websocket_url="ws://localhost:8765",
            output_path=Path("realtime_data")
        )
        
        # Start streaming
        await realtime.start_streaming()
        
        # Simulate some updates
        for _ in range(10):
            await realtime.send_update({
                "value": np.random.random(),
                "timestamp": datetime.now().isoformat()
            })
            await asyncio.sleep(1)
        
        # Stop streaming
        await realtime.stop_streaming()
        
        # Save buffer
        realtime.save_buffer()
    
    asyncio.run(main())
