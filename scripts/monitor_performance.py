#!/usr/bin/env python3
"""Real-time performance monitoring for analysis components."""

import sys
import time
import psutil
import threading
from pathlib import Path
import json
import queue
from datetime import datetime
import logging
from typing import Dict, Any, List, Optional, Set
import numpy as np
from dataclasses import dataclass, asdict
import websockets
import asyncio
import signal
from concurrent.futures import ThreadPoolExecutor
import socket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetricSnapshot:
    """Snapshot of performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_usage: float
    io_read_bytes: int
    io_write_bytes: int
    thread_count: int
    open_files: int
    function_name: str
    duration: Optional[float] = None

class PerformanceMonitor:
    """Monitor and collect performance metrics."""
    
    def __init__(
        self,
        output_dir: Path,
        sample_interval: float = 0.1,
        max_history: int = 1000
    ):
        self.output_dir = output_dir
        self.sample_interval = sample_interval
        self.max_history = max_history
        
        self.metric_queue = queue.Queue()
        self.snapshot_history: List[MetricSnapshot] = []
        self.active_functions: Set[str] = set()
        self.start_times: Dict[str, float] = {}
        
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.websocket_server: Optional[websockets.WebSocketServer] = None
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self._setup_output_dir()

    def _setup_output_dir(self):
        """Setup output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "live_metrics.jsonl"
        self.alert_file = self.output_dir / "performance_alerts.jsonl"

    async def start_monitoring(self, port: int = 8765):
        """Start monitoring with websocket server."""
        self.running = True
        
        # Start metric collection thread
        self.monitor_thread = threading.Thread(target=self._collect_metrics)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start websocket server
        start_server = websockets.serve(self._handle_client, "localhost", port)
        self.websocket_server = await start_server
        
        logger.info(f"Performance monitor started on port {port}")
        
        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

    def _collect_metrics(self):
        """Collect system metrics continuously."""
        process = psutil.Process()
        last_io = process.io_counters()
        
        while self.running:
            try:
                # Collect system metrics
                snapshot = MetricSnapshot(
                    timestamp=time.time(),
                    cpu_percent=process.cpu_percent(),
                    memory_usage=process.memory_info().rss / 1024 / 1024,  # MB
                    io_read_bytes=process.io_counters().read_bytes - last_io.read_bytes,
                    io_write_bytes=process.io_counters().write_bytes - last_io.write_bytes,
                    thread_count=process.num_threads(),
                    open_files=len(process.open_files()),
                    function_name="system"  # Default to system metrics
                )
                
                last_io = process.io_counters()
                
                # Add to history and queue
                self.snapshot_history.append(snapshot)
                if len(self.snapshot_history) > self.max_history:
                    self.snapshot_history.pop(0)
                
                self.metric_queue.put(snapshot)
                
                # Check for alerts
                self._check_alerts(snapshot)
                
                # Save metrics
                self._save_metrics(snapshot)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(1)  # Prevent rapid error loops

    def _check_alerts(self, snapshot: MetricSnapshot):
        """Check for performance alerts."""
        alerts = []
        
        # CPU usage alerts
        if snapshot.cpu_percent > 80:
            alerts.append({
                "type": "high_cpu",
                "value": snapshot.cpu_percent,
                "threshold": 80,
                "timestamp": snapshot.timestamp
            })
        
        # Memory usage alerts
        if snapshot.memory_usage > 1024:  # More than 1GB
            alerts.append({
                "type": "high_memory",
                "value": snapshot.memory_usage,
                "threshold": 1024,
                "timestamp": snapshot.timestamp
            })
        
        # IO alerts
        total_io = snapshot.io_read_bytes + snapshot.io_write_bytes
        if total_io > 100 * 1024 * 1024:  # More than 100MB
            alerts.append({
                "type": "high_io",
                "value": total_io,
                "threshold": 100 * 1024 * 1024,
                "timestamp": snapshot.timestamp
            })
        
        # Duration alerts
        if snapshot.duration and snapshot.duration > 10:  # More than 10 seconds
            alerts.append({
                "type": "long_duration",
                "value": snapshot.duration,
                "threshold": 10,
                "timestamp": snapshot.timestamp
            })
        
        # Save alerts
        if alerts:
            with self.alert_file.open("a") as f:
                for alert in alerts:
                    f.write(json.dumps(alert) + "\n")

    def _save_metrics(self, snapshot: MetricSnapshot):
        """Save metrics to file."""
        with self.metrics_file.open("a") as f:
            f.write(json.dumps(asdict(snapshot)) + "\n")

    async def _handle_client(
        self,
        websocket: websockets.WebSocketServerProtocol,
        path: str
    ):
        """Handle websocket client connection."""
        self.clients.add(websocket)
        try:
            while self.running:
                # Get latest metrics
                try:
                    snapshot = self.metric_queue.get_nowait()
                    await websocket.send(json.dumps(asdict(snapshot)))
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                
                # Check for client commands
                try:
                    message = await asyncio.wait_for(websocket.recv(), 0.1)
                    await self._handle_command(message)
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    pass
                
        finally:
            self.clients.remove(websocket)

    async def _handle_command(self, message: str):
        """Handle client commands."""
        try:
            command = json.loads(message)
            if command["type"] == "start_function":
                self.start_function_monitoring(command["name"])
            elif command["type"] == "stop_function":
                self.stop_function_monitoring(command["name"])
            elif command["type"] == "clear_history":
                self.clear_history()
        except Exception as e:
            logger.error(f"Error handling command: {e}")

    def start_function_monitoring(self, function_name: str):
        """Start monitoring a specific function."""
        self.active_functions.add(function_name)
        self.start_times[function_name] = time.time()

    def stop_function_monitoring(self, function_name: str):
        """Stop monitoring a specific function."""
        if function_name in self.active_functions:
            self.active_functions.remove(function_name)
            if function_name in self.start_times:
                duration = time.time() - self.start_times[function_name]
                # Create final snapshot for function
                snapshot = MetricSnapshot(
                    timestamp=time.time(),
                    cpu_percent=0,  # Final values not relevant
                    memory_usage=0,
                    io_read_bytes=0,
                    io_write_bytes=0,
                    thread_count=0,
                    open_files=0,
                    function_name=function_name,
                    duration=duration
                )
                self._save_metrics(snapshot)
                del self.start_times[function_name]

    def clear_history(self):
        """Clear metric history."""
        self.snapshot_history.clear()
        self.metric_queue.queue.clear()

    async def stop(self):
        """Stop monitoring."""
        self.running = False
        
        # Close websocket connections
        if self.clients:
            await asyncio.gather(*[
                client.close()
                for client in self.clients
            ])
        
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        # Wait for monitor thread
        if self.monitor_thread:
            self.monitor_thread.join()
        
        logger.info("Performance monitor stopped")

class PerformanceMetricCollector:
    """Collect and aggregate performance metrics."""
    
    def __init__(self, monitor_url: str = "ws://localhost:8765"):
        self.monitor_url = monitor_url
        self.metrics: Dict[str, List[MetricSnapshot]] = {}
        self.running = False
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self):
        """Connect to performance monitor."""
        self.websocket = await websockets.connect(self.monitor_url)
        self.running = True

    async def collect_metrics(self):
        """Collect metrics from monitor."""
        if not self.websocket:
            raise RuntimeError("Not connected to monitor")
        
        while self.running:
            try:
                message = await self.websocket.recv()
                snapshot = MetricSnapshot(**json.loads(message))
                
                if snapshot.function_name not in self.metrics:
                    self.metrics[snapshot.function_name] = []
                
                self.metrics[snapshot.function_name].append(snapshot)
                
            except websockets.exceptions.ConnectionClosed:
                logger.error("Connection to monitor lost")
                break
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(1)

    def get_function_metrics(
        self,
        function_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[MetricSnapshot]:
        """Get metrics for a specific function."""
        if function_name not in self.metrics:
            return []
        
        metrics = self.metrics[function_name]
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]
        
        return metrics

    def get_summary_statistics(
        self,
        function_name: str
    ) -> Dict[str, Any]:
        """Get summary statistics for a function."""
        metrics = self.get_function_metrics(function_name)
        if not metrics:
            return {}
        
        cpu_usage = [m.cpu_percent for m in metrics]
        memory_usage = [m.memory_usage for m in metrics]
        io_read = [m.io_read_bytes for m in metrics]
        io_write = [m.io_write_bytes for m in metrics]
        
        return {
            "cpu": {
                "mean": np.mean(cpu_usage),
                "max": np.max(cpu_usage),
                "std": np.std(cpu_usage)
            },
            "memory": {
                "mean": np.mean(memory_usage),
                "max": np.max(memory_usage),
                "std": np.std(memory_usage)
            },
            "io": {
                "total_read": sum(io_read),
                "total_write": sum(io_write),
                "read_rate": np.mean(io_read),
                "write_rate": np.mean(io_write)
            }
        }

    async def stop(self):
        """Stop collecting metrics."""
        self.running = False
        if self.websocket:
            await self.websocket.close()

async def main() -> int:
    """Main entry point."""
    try:
        output_dir = Path("performance_monitoring")
        
        # Start performance monitor
        monitor = PerformanceMonitor(output_dir)
        await monitor.start_monitoring()
        
        # Start metric collector
        collector = PerformanceMetricCollector()
        await collector.connect()
        collection_task = asyncio.create_task(collector.collect_metrics())
        
        try:
            # Run until interrupted
            await asyncio.gather(collection_task)
        except KeyboardInterrupt:
            pass
        finally:
            await collector.stop()
            await monitor.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in performance monitoring: {e}")
        return 1

if __name__ == "__main__":
    asyncio.run(main())
