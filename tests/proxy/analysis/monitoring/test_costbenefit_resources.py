"""Resource monitoring for cost-benefit analysis load tests."""

import asyncio
import time
import psutil
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import pytest
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@dataclass
class ResourceMetrics:
    """Container for resource utilization metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used: float
    disk_io_read: float
    disk_io_write: float
    net_io_sent: float
    net_io_recv: float
    thread_count: int
    handle_count: int
    context_switches: int

@dataclass
class ResourceProfile:
    """Profile of resource utilization over time."""
    metrics: List[ResourceMetrics] = field(default_factory=list)
    sampling_interval: float = 0.1  # 100ms
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def add_metric(self, metric: ResourceMetrics) -> None:
        """Add a metric snapshot."""
        self.metrics.append(metric)
        if not self.start_time:
            self.start_time = metric.timestamp

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate resource utilization statistics."""
        if not self.metrics:
            return {}

        cpu_usage = np.array([m.cpu_percent for m in self.metrics])
        mem_usage = np.array([m.memory_percent for m in self.metrics])
        thread_counts = np.array([m.thread_count for m in self.metrics])

        return {
            "duration": self.end_time - self.start_time if self.end_time else 0,
            "cpu_avg": np.mean(cpu_usage),
            "cpu_peak": np.max(cpu_usage),
            "cpu_p95": np.percentile(cpu_usage, 95),
            "memory_avg": np.mean(mem_usage),
            "memory_peak": np.max(mem_usage),
            "memory_p95": np.percentile(mem_usage, 95),
            "thread_avg": np.mean(thread_counts),
            "thread_peak": np.max(thread_counts)
        }

    def plot_metrics(self, output_path: Optional[str] = None) -> go.Figure:
        """Create visualization of resource metrics."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('CPU & Memory Usage', 'I/O Activity', 'Thread & Handle Count'),
            vertical_spacing=0.1
        )

        times = [(m.timestamp - self.start_time) for m in self.metrics]
        
        # CPU and Memory
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[m.cpu_percent for m in self.metrics],
                name='CPU %',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[m.memory_percent for m in self.metrics],
                name='Memory %',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # I/O Activity
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[m.disk_io_read / 1024 / 1024 for m in self.metrics],
                name='Disk Read (MB/s)',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[m.net_io_recv / 1024 / 1024 for m in self.metrics],
                name='Network Recv (MB/s)',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # Thread and Handle Count
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[m.thread_count for m in self.metrics],
                name='Threads',
                line=dict(color='orange')
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[m.handle_count for m in self.metrics],
                name='Handles',
                line=dict(color='brown')
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="Resource Utilization Profile"
        )
        
        if output_path:
            fig.write_html(output_path)
        
        return fig

class ResourceMonitor:
    """Monitor system resources during tests."""

    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.profile = ResourceProfile(sampling_interval=sampling_interval)
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self.process = psutil.Process()
        self.last_disk_io = psutil.disk_io_counters()
        self.last_net_io = psutil.net_io_counters()
        self.last_timestamp = time.time()

    def start(self) -> None:
        """Start resource monitoring."""
        if self._monitor_thread is None:
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_resources)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()

    def stop(self) -> ResourceProfile:
        """Stop resource monitoring and return profile."""
        if self._monitor_thread:
            self._stop_event.set()
            self._monitor_thread.join()
            self._monitor_thread = None
            self.profile.end_time = time.time()
        return self.profile

    def _monitor_resources(self) -> None:
        """Monitor resource utilization."""
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                
                # Get current metrics
                disk_io = psutil.disk_io_counters()
                net_io = psutil.net_io_counters()
                interval = current_time - self.last_timestamp
                
                metric = ResourceMetrics(
                    timestamp=current_time,
                    cpu_percent=self.process.cpu_percent(),
                    memory_percent=self.process.memory_percent(),
                    memory_used=self.process.memory_info().rss,
                    disk_io_read=(disk_io.read_bytes - self.last_disk_io.read_bytes) / interval,
                    disk_io_write=(disk_io.write_bytes - self.last_disk_io.write_bytes) / interval,
                    net_io_sent=(net_io.bytes_sent - self.last_net_io.bytes_sent) / interval,
                    net_io_recv=(net_io.bytes_recv - self.last_net_io.bytes_recv) / interval,
                    thread_count=self.process.num_threads(),
                    handle_count=self.process.num_handles() if hasattr(self.process, 'num_handles') else 0,
                    context_switches=self.process.num_ctx_switches().voluntary
                )
                
                self.profile.add_metric(metric)
                
                # Update last values
                self.last_disk_io = disk_io
                self.last_net_io = net_io
                self.last_timestamp = current_time
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                print(f"Error monitoring resources: {e}")
                time.sleep(1)

@pytest.fixture
def resource_monitor():
    """Provide resource monitor for tests."""
    monitor = ResourceMonitor(sampling_interval=0.1)
    monitor.start()
    yield monitor
    profile = monitor.stop()
    
    # Save resource plots
    profile.plot_metrics("resource_profile.html")
    
    # Print resource summary
    stats = profile.get_statistics()
    print("\nResource Utilization Summary:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")

def test_with_resources(resource_monitor):
    """Example test using resource monitor."""
    # Your test code here
    pass
