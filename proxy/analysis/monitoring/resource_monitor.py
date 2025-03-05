"""Resource monitoring for migrations and system operations."""

import psutil
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import json
from contextlib import contextmanager
import sqlite3
import os
import queue

logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_io_read: int
    disk_io_write: int
    open_files: int
    thread_count: int
    connection_count: int
    operation: str
    duration: float

class ResourceMonitor:
    """Monitor system resources during operations."""
    
    def __init__(
        self,
        sample_interval: float = 0.1,
        history_size: int = 1000,
        metrics_file: Optional[Path] = None
    ):
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.metrics_file = metrics_file or Path("resource_metrics.json")
        
        self.metrics: List[ResourceMetrics] = []
        self.monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._metrics_queue: queue.Queue = queue.Queue()
        self._operation_start: Optional[datetime] = None
        self._current_operation: str = ""
    
    @contextmanager
    def monitor_operation(self, operation: str):
        """Context manager to monitor an operation."""
        try:
            self.start_monitoring(operation)
            yield
        finally:
            self.stop_monitoring()
    
    def start_monitoring(self, operation: str):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._operation_start = datetime.now()
        self._current_operation = operation
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
            self._monitor_thread = None
    
    def _monitor_resources(self):
        """Monitor system resources."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # Collect metrics
                metrics = ResourceMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=process.cpu_percent(),
                    memory_percent=process.memory_percent(),
                    disk_io_read=process.io_counters().read_bytes if hasattr(process, 'io_counters') else 0,
                    disk_io_write=process.io_counters().write_bytes if hasattr(process, 'io_counters') else 0,
                    open_files=len(process.open_files()),
                    thread_count=process.num_threads(),
                    connection_count=len(process.connections()),
                    operation=self._current_operation,
                    duration=(datetime.now() - self._operation_start).total_seconds()
                )
                
                self._metrics_queue.put(metrics)
                
                # Process queue
                self._process_metrics_queue()
                
                time.sleep(self.sample_interval)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.warning("Failed to collect process metrics")
                break
    
    def _process_metrics_queue(self):
        """Process collected metrics."""
        while not self._metrics_queue.empty():
            metrics = self._metrics_queue.get()
            self.metrics.append(metrics)
            
            # Trim history if needed
            if len(self.metrics) > self.history_size:
                self.metrics = self.metrics[-self.history_size:]
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage."""
        if not self.metrics:
            return {}
        
        cpu_usage = [m.cpu_percent for m in self.metrics]
        memory_usage = [m.memory_percent for m in self.metrics]
        disk_reads = [m.disk_io_read for m in self.metrics]
        disk_writes = [m.disk_io_write for m in self.metrics]
        
        return {
            "operation": self._current_operation,
            "duration": (datetime.now() - self._operation_start).total_seconds(),
            "cpu": {
                "avg": sum(cpu_usage) / len(cpu_usage),
                "max": max(cpu_usage),
                "min": min(cpu_usage)
            },
            "memory": {
                "avg": sum(memory_usage) / len(memory_usage),
                "max": max(memory_usage),
                "min": min(memory_usage)
            },
            "disk_io": {
                "total_reads": max(disk_reads) - min(disk_reads),
                "total_writes": max(disk_writes) - min(disk_writes)
            },
            "threads": {
                "max": max(m.thread_count for m in self.metrics),
                "min": min(m.thread_count for m in self.metrics)
            },
            "connections": {
                "max": max(m.connection_count for m in self.metrics),
                "min": min(m.connection_count for m in self.metrics)
            },
            "files": {
                "max": max(m.open_files for m in self.metrics),
                "min": min(m.open_files for m in self.metrics)
            }
        }
    
    def get_resource_timeline(self) -> List[Dict[str, Any]]:
        """Get timeline of resource usage."""
        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "cpu": m.cpu_percent,
                "memory": m.memory_percent,
                "disk_read": m.disk_io_read,
                "disk_write": m.disk_io_write,
                "threads": m.thread_count,
                "connections": m.connection_count,
                "files": m.open_files,
                "operation": m.operation,
                "duration": m.duration
            }
            for m in self.metrics
        ]
    
    def save_metrics(self):
        """Save metrics to file."""
        data = {
            "summary": self.get_resource_summary(),
            "timeline": self.get_resource_timeline()
        }
        
        with open(self.metrics_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze resource bottlenecks."""
        if not self.metrics:
            return []
        
        bottlenecks = []
        
        # CPU bottlenecks
        cpu_threshold = 80
        high_cpu = [
            m for m in self.metrics
            if m.cpu_percent > cpu_threshold
        ]
        if high_cpu:
            bottlenecks.append({
                "type": "CPU",
                "threshold": cpu_threshold,
                "occurrences": len(high_cpu),
                "max_value": max(m.cpu_percent for m in high_cpu),
                "duration": sum(m.duration for m in high_cpu)
            })
        
        # Memory bottlenecks
        memory_threshold = 80
        high_memory = [
            m for m in self.metrics
            if m.memory_percent > memory_threshold
        ]
        if high_memory:
            bottlenecks.append({
                "type": "Memory",
                "threshold": memory_threshold,
                "occurrences": len(high_memory),
                "max_value": max(m.memory_percent for m in high_memory),
                "duration": sum(m.duration for m in high_memory)
            })
        
        # IO bottlenecks
        io_values = [
            (m.disk_io_read + m.disk_io_write) / (1024 * 1024)  # MB/s
            for m in self.metrics
        ]
        if io_values:
            avg_io = sum(io_values) / len(io_values)
            high_io = [v for v in io_values if v > avg_io * 2]
            if high_io:
                bottlenecks.append({
                    "type": "Disk I/O",
                    "threshold": f"{avg_io:.1f} MB/s",
                    "occurrences": len(high_io),
                    "max_value": max(high_io),
                    "duration": len(high_io) * self.sample_interval
                })
        
        return bottlenecks

@contextmanager
def monitor_resources(
    operation: str,
    metrics_file: Optional[Path] = None
) -> ResourceMonitor:
    """Context manager for resource monitoring."""
    monitor = ResourceMonitor(metrics_file=metrics_file)
    try:
        monitor.start_monitoring(operation)
        yield monitor
    finally:
        monitor.stop_monitoring()
        monitor.save_metrics()

def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": {
            path: psutil.disk_usage(path).percent
            for path in psutil.disk_partitions()
            if "rw" in path.opts
        },
        "network": psutil.net_io_counters()._asdict(),
        "load_avg": psutil.getloadavg(),
        "process_count": len(psutil.pids())
    }

if __name__ == "__main__":
    # Example usage
    with monitor_resources("example_operation") as monitor:
        # Simulate work
        time.sleep(5)
        
        # Get resource summary
        summary = monitor.get_resource_summary()
        print(json.dumps(summary, indent=2))
