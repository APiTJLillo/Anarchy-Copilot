"""Performance metrics collection and analysis."""

import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging
from collections import defaultdict
import queue
import threading
import psutil
import plotly.graph_objects as go
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    """Configuration for performance metrics."""
    sampling_interval: float = 0.1  # seconds
    history_size: int = 1000
    alert_threshold: float = 0.9  # 90% threshold
    smoothing_window: int = 5
    fps_target: float = 60.0
    memory_threshold_mb: float = 1024
    io_threshold_mb: float = 100
    cpu_threshold: float = 80.0
    log_metrics: bool = True
    metrics_file: Optional[Path] = None

@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    timestamp: datetime
    frame_time: float
    fps: float
    cpu_percent: float
    memory_mb: float
    io_read_mb: float
    io_write_mb: float
    gpu_utilization: Optional[float] = None
    frame_size: Optional[int] = None
    queue_size: int = 0
    dropped_frames: int = 0

class PerformanceMonitor:
    """Monitor and analyze animation performance."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.metrics: List[PerformanceMetric] = []
        self.alerts: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.monitoring = False
        
        self._frame_times: List[float] = []
        self._metric_queue = queue.Queue()
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._stats: Dict[str, Any] = defaultdict(float)
    
    def start(self):
        """Start performance monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = datetime.now()
        self._monitor_thread = threading.Thread(
            target=self._monitor_performance,
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
            self._monitor_thread = None
        
        self._save_metrics()
    
    def record_frame(
        self,
        frame_time: float,
        frame_size: Optional[int] = None
    ):
        """Record frame render timing."""
        self._frame_times.append(frame_time)
        if len(self._frame_times) > self.config.smoothing_window:
            self._frame_times = self._frame_times[-self.config.smoothing_window:]
        
        fps = 1.0 / np.mean(self._frame_times) if self._frame_times else 0
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            frame_time=frame_time,
            fps=fps,
            cpu_percent=psutil.cpu_percent(),
            memory_mb=psutil.Process().memory_info().rss / (1024 * 1024),
            io_read_mb=0,  # Updated by monitor thread
            io_write_mb=0,
            frame_size=frame_size,
            queue_size=self._metric_queue.qsize()
        )
        
        self._metric_queue.put(metric)
        self._check_alerts(metric)
    
    def _monitor_performance(self):
        """Monitor system performance."""
        process = psutil.Process()
        last_io = process.io_counters()
        last_time = time.time()
        
        while self.monitoring:
            try:
                # Process queued metrics
                while not self._metric_queue.empty():
                    metric = self._metric_queue.get()
                    self._process_metric(metric)
                
                # Update IO stats
                current_time = time.time()
                current_io = process.io_counters()
                
                time_diff = current_time - last_time
                if time_diff > 0:
                    read_mb = (current_io.read_bytes - last_io.read_bytes) / (1024 * 1024 * time_diff)
                    write_mb = (current_io.write_bytes - last_io.write_bytes) / (1024 * 1024 * time_diff)
                    
                    self._stats["io_read_rate"] = read_mb
                    self._stats["io_write_rate"] = write_mb
                
                last_io = current_io
                last_time = current_time
                
                time.sleep(self.config.sampling_interval)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    def _process_metric(self, metric: PerformanceMetric):
        """Process and store performance metric."""
        self.metrics.append(metric)
        
        # Trim history if needed
        if len(self.metrics) > self.config.history_size:
            self.metrics = self.metrics[-self.config.history_size:]
        
        # Update statistics
        self._update_stats(metric)
        
        # Log metrics if enabled
        if self.config.log_metrics:
            self._log_metric(metric)
    
    def _update_stats(self, metric: PerformanceMetric):
        """Update running statistics."""
        self._stats["total_frames"] += 1
        self._stats["avg_frame_time"] = (
            self._stats["avg_frame_time"] * 0.95 +
            metric.frame_time * 0.05
        )
        self._stats["min_fps"] = min(
            self._stats.get("min_fps", float("inf")),
            metric.fps
        )
        self._stats["max_fps"] = max(
            self._stats.get("max_fps", 0),
            metric.fps
        )
        self._stats["peak_memory_mb"] = max(
            self._stats.get("peak_memory_mb", 0),
            metric.memory_mb
        )
    
    def _check_alerts(self, metric: PerformanceMetric):
        """Check for performance alerts."""
        alerts = []
        
        # Check FPS
        if metric.fps < self.config.fps_target * self.config.alert_threshold:
            alerts.append({
                "type": "fps",
                "message": f"Low FPS: {metric.fps:.1f}",
                "value": metric.fps,
                "threshold": self.config.fps_target * self.config.alert_threshold
            })
        
        # Check memory
        if metric.memory_mb > self.config.memory_threshold_mb:
            alerts.append({
                "type": "memory",
                "message": f"High memory usage: {metric.memory_mb:.1f} MB",
                "value": metric.memory_mb,
                "threshold": self.config.memory_threshold_mb
            })
        
        # Check CPU
        if metric.cpu_percent > self.config.cpu_threshold:
            alerts.append({
                "type": "cpu",
                "message": f"High CPU usage: {metric.cpu_percent:.1f}%",
                "value": metric.cpu_percent,
                "threshold": self.config.cpu_threshold
            })
        
        # Add alerts and trigger callbacks
        for alert in alerts:
            self.alerts.append({
                **alert,
                "timestamp": datetime.now().isoformat()
            })
            self._trigger_callbacks("alert", alert)
    
    def add_callback(
        self,
        event_type: str,
        callback: Callable
    ):
        """Add callback for performance events."""
        self._callbacks[event_type].append(callback)
    
    def _trigger_callbacks(
        self,
        event_type: str,
        data: Any
    ):
        """Trigger registered callbacks."""
        for callback in self._callbacks[event_type]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.metrics:
            return {}
        
        recent = self.metrics[-self.config.smoothing_window:]
        
        return {
            "current_fps": recent[-1].fps if recent else 0,
            "avg_fps": np.mean([m.fps for m in recent]),
            "frame_time_ms": np.mean([m.frame_time * 1000 for m in recent]),
            "memory_mb": recent[-1].memory_mb if recent else 0,
            "cpu_percent": recent[-1].cpu_percent if recent else 0,
            "io_read_mb": self._stats["io_read_rate"],
            "io_write_mb": self._stats["io_write_rate"],
            "total_frames": self._stats["total_frames"],
            "dropped_frames": sum(m.dropped_frames for m in self.metrics),
            "queue_size": recent[-1].queue_size if recent else 0,
            "alerts_count": len(self.alerts),
            "duration": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }
    
    def plot_performance(self) -> Dict[str, go.Figure]:
        """Create performance visualization plots."""
        if not self.metrics:
            return {}
        
        df = pd.DataFrame([
            {
                "timestamp": m.timestamp,
                "fps": m.fps,
                "frame_time": m.frame_time * 1000,  # Convert to ms
                "memory_mb": m.memory_mb,
                "cpu_percent": m.cpu_percent,
                "queue_size": m.queue_size
            }
            for m in self.metrics
        ])
        
        plots = {}
        
        # FPS/Frame Time plot
        fps_fig = go.Figure()
        fps_fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["fps"],
            name="FPS",
            mode="lines"
        ))
        fps_fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["frame_time"],
            name="Frame Time (ms)",
            mode="lines",
            yaxis="y2"
        ))
        fps_fig.update_layout(
            title="FPS and Frame Time",
            yaxis=dict(title="FPS"),
            yaxis2=dict(title="Frame Time (ms)", overlaying="y", side="right")
        )
        plots["fps"] = fps_fig
        
        # Resource usage plot
        resource_fig = go.Figure()
        resource_fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["memory_mb"],
            name="Memory (MB)",
            mode="lines"
        ))
        resource_fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["cpu_percent"],
            name="CPU %",
            mode="lines",
            yaxis="y2"
        ))
        resource_fig.update_layout(
            title="Resource Usage",
            yaxis=dict(title="Memory (MB)"),
            yaxis2=dict(title="CPU %", overlaying="y", side="right")
        )
        plots["resources"] = resource_fig
        
        return plots
    
    def _save_metrics(self):
        """Save metrics to file."""
        if self.config.metrics_file and self.metrics:
            data = {
                "summary": self.get_metrics_summary(),
                "alerts": self.alerts,
                "metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "fps": m.fps,
                        "frame_time": m.frame_time,
                        "memory_mb": m.memory_mb,
                        "cpu_percent": m.cpu_percent,
                        "queue_size": m.queue_size
                    }
                    for m in self.metrics
                ]
            }
            
            with open(self.config.metrics_file, "w") as f:
                json.dump(data, f, indent=2)
    
    def _log_metric(self, metric: PerformanceMetric):
        """Log performance metric."""
        logger.debug(
            f"Performance: FPS={metric.fps:.1f}, "
            f"Frame Time={metric.frame_time*1000:.1f}ms, "
            f"Memory={metric.memory_mb:.1f}MB, "
            f"CPU={metric.cpu_percent:.1f}%"
        )

def monitor_performance(
    config: Optional[MetricsConfig] = None
) -> PerformanceMonitor:
    """Create and start performance monitor."""
    monitor = PerformanceMonitor(config or MetricsConfig())
    monitor.start()
    return monitor

if __name__ == "__main__":
    # Example usage
    monitor = monitor_performance(
        MetricsConfig(
            metrics_file=Path("performance_metrics.json")
        )
    )
    
    # Simulate frame rendering
    for _ in range(100):
        time.sleep(1/60)  # Simulate 60 FPS
        monitor.record_frame(time.perf_counter())
    
    monitor.stop()
    
    # Create visualizations
    plots = monitor.plot_performance()
    for name, fig in plots.items():
        fig.write_html(f"performance_{name}.html")
