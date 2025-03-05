"""Metric collectors for various data sources."""

import psutil
import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
import logging
from pathlib import Path
from .metrics import MetricValue, TimeseriesMetric

logger = logging.getLogger(__name__)

class MetricCollector(ABC):
    """Base class for metric collectors."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.subscribers: Set[str] = set()

    @abstractmethod
    async def collect(self) -> List[MetricValue]:
        """Collect metrics."""
        pass

    async def start(self):
        """Start collecting metrics."""
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._collection_loop())
        logger.info(f"Started {self.__class__.__name__}")

    async def stop(self):
        """Stop collecting metrics."""
        if not self.running:
            return
        
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped {self.__class__.__name__}")

    async def _collection_loop(self):
        """Main collection loop."""
        while self.running:
            try:
                metrics = await self.collect()
                await self._publish_metrics(metrics)
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(1)  # Back off on error

    async def _publish_metrics(self, metrics: List[MetricValue]):
        """Publish metrics to subscribers."""
        # Implementation depends on storage/notification system
        pass

class SystemMetrics(MetricCollector):
    """Collect system-level metrics."""

    def __init__(
        self,
        interval: float = 1.0,
        include_network: bool = True,
        include_disk: bool = True
    ):
        super().__init__(interval)
        self.include_network = include_network
        self.include_disk = include_disk
        self.process = psutil.Process()
        self._last_cpu_times = psutil.cpu_times()
        self._last_net_io = psutil.net_io_counters()
        self._last_disk_io = psutil.disk_io_counters()
        self._last_time = time.time()

    async def collect(self) -> List[MetricValue]:
        """Collect system metrics."""
        now = datetime.now()
        current_time = time.time()
        elapsed = current_time - self._last_time
        metrics = []

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_times = psutil.cpu_times()
        
        metrics.extend([
            MetricValue(
                name="system.cpu.percent",
                value=cpu_percent,
                timestamp=now,
                unit="percent"
            ),
            MetricValue(
                name="system.cpu.user",
                value=(cpu_times.user - self._last_cpu_times.user) / elapsed * 100,
                timestamp=now,
                unit="percent"
            ),
            MetricValue(
                name="system.cpu.system",
                value=(cpu_times.system - self._last_cpu_times.system) / elapsed * 100,
                timestamp=now,
                unit="percent"
            )
        ])
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.extend([
            MetricValue(
                name="system.memory.total",
                value=memory.total,
                timestamp=now,
                unit="bytes"
            ),
            MetricValue(
                name="system.memory.available",
                value=memory.available,
                timestamp=now,
                unit="bytes"
            ),
            MetricValue(
                name="system.memory.used",
                value=memory.used,
                timestamp=now,
                unit="bytes"
            ),
            MetricValue(
                name="system.memory.percent",
                value=memory.percent,
                timestamp=now,
                unit="percent"
            )
        ])
        
        # Network metrics
        if self.include_network:
            net_io = psutil.net_io_counters()
            metrics.extend([
                MetricValue(
                    name="system.network.bytes_sent",
                    value=(net_io.bytes_sent - self._last_net_io.bytes_sent) / elapsed,
                    timestamp=now,
                    unit="bytes/sec"
                ),
                MetricValue(
                    name="system.network.bytes_recv",
                    value=(net_io.bytes_recv - self._last_net_io.bytes_recv) / elapsed,
                    timestamp=now,
                    unit="bytes/sec"
                ),
                MetricValue(
                    name="system.network.packets_sent",
                    value=(net_io.packets_sent - self._last_net_io.packets_sent) / elapsed,
                    timestamp=now,
                    unit="packets/sec"
                ),
                MetricValue(
                    name="system.network.packets_recv",
                    value=(net_io.packets_recv - self._last_net_io.packets_recv) / elapsed,
                    timestamp=now,
                    unit="packets/sec"
                )
            ])
            self._last_net_io = net_io
        
        # Disk metrics
        if self.include_disk:
            disk_io = psutil.disk_io_counters()
            metrics.extend([
                MetricValue(
                    name="system.disk.read_bytes",
                    value=(disk_io.read_bytes - self._last_disk_io.read_bytes) / elapsed,
                    timestamp=now,
                    unit="bytes/sec"
                ),
                MetricValue(
                    name="system.disk.write_bytes",
                    value=(disk_io.write_bytes - self._last_disk_io.write_bytes) / elapsed,
                    timestamp=now,
                    unit="bytes/sec"
                ),
                MetricValue(
                    name="system.disk.read_count",
                    value=(disk_io.read_count - self._last_disk_io.read_count) / elapsed,
                    timestamp=now,
                    unit="ops/sec"
                ),
                MetricValue(
                    name="system.disk.write_count",
                    value=(disk_io.write_count - self._last_disk_io.write_count) / elapsed,
                    timestamp=now,
                    unit="ops/sec"
                )
            ])
            self._last_disk_io = disk_io
        
        self._last_cpu_times = cpu_times
        self._last_time = current_time
        
        return metrics

class ProxyMetrics(MetricCollector):
    """Collect proxy-specific metrics."""
    
    def __init__(
        self,
        interval: float = 1.0,
        connection_tracker: Any = None,
        request_tracker: Any = None
    ):
        super().__init__(interval)
        self.connection_tracker = connection_tracker
        self.request_tracker = request_tracker
        self._last_request_count = 0
        self._last_time = time.time()

    async def collect(self) -> List[MetricValue]:
        """Collect proxy metrics."""
        now = datetime.now()
        current_time = time.time()
        elapsed = current_time - self._last_time
        metrics = []
        
        # Connection metrics
        if self.connection_tracker:
            conn_stats = self.connection_tracker.get_stats()
            metrics.extend([
                MetricValue(
                    name="proxy.connections.active",
                    value=conn_stats["active_connections"],
                    timestamp=now,
                    unit="connections"
                ),
                MetricValue(
                    name="proxy.connections.total",
                    value=conn_stats["total_connections"],
                    timestamp=now,
                    unit="connections"
                ),
                MetricValue(
                    name="proxy.connections.errors",
                    value=conn_stats["connection_errors"],
                    timestamp=now,
                    unit="errors"
                )
            ])
        
        # Request metrics
        if self.request_tracker:
            req_stats = self.request_tracker.get_stats()
            request_count = req_stats["total_requests"]
            request_rate = (request_count - self._last_request_count) / elapsed
            
            metrics.extend([
                MetricValue(
                    name="proxy.requests.rate",
                    value=request_rate,
                    timestamp=now,
                    unit="requests/sec"
                ),
                MetricValue(
                    name="proxy.requests.errors",
                    value=req_stats["error_count"],
                    timestamp=now,
                    unit="errors"
                ),
                MetricValue(
                    name="proxy.requests.latency_avg",
                    value=req_stats["average_latency"],
                    timestamp=now,
                    unit="seconds"
                ),
                MetricValue(
                    name="proxy.requests.latency_p95",
                    value=req_stats["p95_latency"],
                    timestamp=now,
                    unit="seconds"
                )
            ])
            
            self._last_request_count = request_count
        
        self._last_time = current_time
        return metrics

class RequestMetrics(MetricCollector):
    """Collect detailed request-level metrics."""
    
    def __init__(
        self,
        interval: float = 1.0,
        request_store: Any = None,
        max_requests: int = 1000
    ):
        super().__init__(interval)
        self.request_store = request_store
        self.max_requests = max_requests
        self._request_buffer: List[Dict[str, Any]] = []

    async def collect(self) -> List[MetricValue]:
        """Collect request metrics."""
        if not self.request_store:
            return []
        
        now = datetime.now()
        metrics = []
        
        # Get recent requests
        recent_requests = self.request_store.get_recent_requests(self.max_requests)
        
        # Group by path
        path_stats: Dict[str, Dict[str, List[float]]] = {}
        
        for req in recent_requests:
            path = req["path"]
            if path not in path_stats:
                path_stats[path] = {
                    "latency": [],
                    "size": [],
                    "status": []
                }
            
            path_stats[path]["latency"].append(req["latency"])
            path_stats[path]["size"].append(req["response_size"])
            path_stats[path]["status"].append(req["status_code"])
        
        # Generate metrics for each path
        for path, stats in path_stats.items():
            tags = {"path": path}
            
            metrics.extend([
                MetricValue(
                    name="proxy.path.latency.avg",
                    value=np.mean(stats["latency"]),
                    timestamp=now,
                    tags=tags,
                    unit="seconds"
                ),
                MetricValue(
                    name="proxy.path.latency.p95",
                    value=np.percentile(stats["latency"], 95),
                    timestamp=now,
                    tags=tags,
                    unit="seconds"
                ),
                MetricValue(
                    name="proxy.path.size.avg",
                    value=np.mean(stats["size"]),
                    timestamp=now,
                    tags=tags,
                    unit="bytes"
                ),
                MetricValue(
                    name="proxy.path.error_rate",
                    value=sum(1 for s in stats["status"] if s >= 400) / len(stats["status"]),
                    timestamp=now,
                    tags=tags,
                    unit="percent"
                )
            ])
        
        return metrics
