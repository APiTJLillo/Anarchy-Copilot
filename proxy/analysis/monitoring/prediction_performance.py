"""Performance monitoring for real-time prediction."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
import time
import psutil
import tracemalloc
from collections import deque
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go

from .realtime_prediction import RealtimePrediction, RealtimeConfig

logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    sampling_interval: float = 0.1  # seconds
    history_length: int = 1000
    alert_threshold: float = 0.9
    memory_threshold: float = 0.85
    thread_pool_size: int = 4
    profiling_enabled: bool = True
    output_path: Optional[Path] = None

class PredictionPerformance:
    """Performance monitoring for predictions."""
    
    def __init__(
        self,
        prediction: RealtimePrediction,
        config: PerformanceConfig
    ):
        self.prediction = prediction
        self.config = config
        self.metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.history_length)
        )
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.alert_callbacks: List[Callable] = []
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.thread_pool_size
        )
        
        # Initialize tracemalloc if profiling enabled
        if self.config.profiling_enabled:
            tracemalloc.start()
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Started performance monitoring")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join()
            self.monitor_thread = None
        
        self.executor.shutdown(wait=True)
        
        if self.config.profiling_enabled:
            tracemalloc.stop()
        
        logger.info("Stopped performance monitoring")
    
    def register_alert_callback(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ):
        """Register callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            key: list(values)[-1] if values else None
            for key, values in self.metrics.items()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get statistical summary of performance metrics."""
        summary = {}
        
        for key, values in self.metrics.items():
            if not values:
                continue
            
            values_array = np.array(values)
            summary[key] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "percentiles": {
                    "25": float(np.percentile(values_array, 25)),
                    "50": float(np.percentile(values_array, 50)),
                    "75": float(np.percentile(values_array, 75)),
                    "90": float(np.percentile(values_array, 90)),
                    "95": float(np.percentile(values_array, 95)),
                    "99": float(np.percentile(values_array, 99))
                }
            }
        
        return summary
    
    def create_performance_dashboard(self) -> go.Figure:
        """Create performance monitoring dashboard."""
        fig = go.Figure()
        
        # Add CPU usage
        cpu_usage = list(self.metrics["cpu_usage"])
        if cpu_usage:
            fig.add_trace(
                go.Scatter(
                    y=cpu_usage,
                    name="CPU Usage",
                    line=dict(color="blue")
                )
            )
        
        # Add memory usage
        memory_usage = list(self.metrics["memory_usage"])
        if memory_usage:
            fig.add_trace(
                go.Scatter(
                    y=memory_usage,
                    name="Memory Usage",
                    line=dict(color="red")
                )
            )
        
        # Add latency
        latency = list(self.metrics["latency"])
        if latency:
            fig.add_trace(
                go.Scatter(
                    y=latency,
                    name="Latency",
                    line=dict(color="green")
                )
            )
        
        # Add throughput
        throughput = list(self.metrics["throughput"])
        if throughput:
            fig.add_trace(
                go.Scatter(
                    y=throughput,
                    name="Throughput",
                    line=dict(color="purple")
                )
            )
        
        # Update layout
        fig.update_layout(
            title="Prediction Performance Metrics",
            xaxis_title="Sample",
            yaxis_title="Value",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        last_sample = time.time()
        samples = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                if current_time - last_sample >= self.config.sampling_interval:
                    # Collect metrics
                    metrics = self._collect_metrics()
                    
                    # Update metrics history
                    for key, value in metrics.items():
                        self.metrics[key].append(value)
                    
                    # Check for alerts
                    self._check_alerts(metrics)
                    
                    # Update counters
                    last_sample = current_time
                    samples += 1
                
                # Small sleep to prevent busy loop
                time.sleep(max(0, self.config.sampling_interval / 10))
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics."""
        metrics = {}
        
        # System metrics
        metrics["cpu_usage"] = psutil.cpu_percent() / 100.0
        metrics["memory_usage"] = (
            psutil.Process().memory_info().rss /
            psutil.virtual_memory().total
        )
        
        # Prediction metrics
        queue_size = self.prediction.data_queue.qsize()
        metrics["queue_size"] = queue_size
        metrics["throughput"] = len(
            self.prediction.get_recent_data(
                "processed_count",
                window=self.config.sampling_interval
            )
        ) / self.config.sampling_interval
        
        # Latency metrics
        recent_updates = self.prediction.get_recent_data(
            "update_time",
            window=10
        )
        if recent_updates:
            metrics["latency"] = np.mean([
                t["end"] - t["start"]
                for t in recent_updates
            ])
        
        # Memory profiling
        if self.config.profiling_enabled:
            current, peak = tracemalloc.get_traced_memory()
            metrics["memory_current"] = current / 1024 / 1024  # MB
            metrics["memory_peak"] = peak / 1024 / 1024  # MB
        
        return metrics
    
    def _check_alerts(self, metrics: Dict[str, float]):
        """Check metrics for alert conditions."""
        alerts = []
        
        # CPU usage alert
        if metrics["cpu_usage"] > self.config.alert_threshold:
            alerts.append({
                "type": "high_cpu",
                "value": metrics["cpu_usage"],
                "threshold": self.config.alert_threshold,
                "timestamp": datetime.now().isoformat()
            })
        
        # Memory usage alert
        if metrics["memory_usage"] > self.config.memory_threshold:
            alerts.append({
                "type": "high_memory",
                "value": metrics["memory_usage"],
                "threshold": self.config.memory_threshold,
                "timestamp": datetime.now().isoformat()
            })
        
        # Queue backlog alert
        if metrics["queue_size"] > self.config.history_length / 2:
            alerts.append({
                "type": "queue_backlog",
                "value": metrics["queue_size"],
                "threshold": self.config.history_length / 2,
                "timestamp": datetime.now().isoformat()
            })
        
        # Trigger callbacks for alerts
        if alerts:
            for callback in self.alert_callbacks:
                try:
                    callback({"alerts": alerts, "metrics": metrics})
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    def save_metrics(self):
        """Save collected metrics."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save metrics history
            metrics_file = output_path / "performance_metrics.json"
            metrics_data = {
                key: list(values)
                for key, values in self.metrics.items()
            }
            
            with open(metrics_file, "w") as f:
                json.dump(metrics_data, f, indent=2)
            
            # Save summary
            summary_file = output_path / "performance_summary.json"
            with open(summary_file, "w") as f:
                json.dump(self.get_performance_summary(), f, indent=2)
            
            logger.info(f"Saved performance data to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def load_metrics(self):
        """Load saved metrics."""
        if not self.config.output_path:
            return
        
        try:
            metrics_file = self.config.output_path / "performance_metrics.json"
            if not metrics_file.exists():
                return
            
            with open(metrics_file) as f:
                metrics_data = json.load(f)
            
            for key, values in metrics_data.items():
                self.metrics[key] = deque(
                    values,
                    maxlen=self.config.history_length
                )
            
            logger.info(f"Loaded metrics from {metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")

def create_prediction_performance(
    prediction: RealtimePrediction,
    output_path: Optional[Path] = None
) -> PredictionPerformance:
    """Create prediction performance monitor."""
    config = PerformanceConfig(output_path=output_path)
    return PredictionPerformance(prediction, config)

if __name__ == "__main__":
    # Example usage
    from .realtime_prediction import create_realtime_prediction
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
        realtime = create_realtime_prediction(controls)
        
        # Create performance monitor
        performance = create_prediction_performance(
            realtime,
            output_path=Path("performance_data")
        )
        
        # Start monitoring
        performance.start_monitoring()
        
        # Run some predictions
        await realtime.start_streaming()
        
        for _ in range(100):
            await realtime.send_update({
                "value": np.random.random(),
                "timestamp": datetime.now().isoformat()
            })
            await asyncio.sleep(0.1)
        
        # Stop everything
        await realtime.stop_streaming()
        performance.stop_monitoring()
        
        # Save metrics
        performance.save_metrics()
        
        # Create dashboard
        dashboard = performance.create_performance_dashboard()
        dashboard.write_html("performance_dashboard.html")
    
    asyncio.run(main())
