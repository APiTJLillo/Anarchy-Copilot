"""Performance monitoring for distributed tuning."""

import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import psutil
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from .distributed_tuning import DistributedTuner, DistributedConfig

logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    """Configuration for tuning monitoring."""
    sampling_interval: float = 1.0  # Seconds between samples
    history_window: int = 3600  # Samples to keep in history
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_percent": 90.0,
        "memory_percent": 85.0,
        "disk_percent": 90.0,
        "trial_failure_rate": 0.2
    })
    export_path: Optional[Path] = None
    enable_profiling: bool = False

@dataclass
class ResourceMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    per_worker_cpu: Dict[str, float]
    per_worker_memory: Dict[str, float]

@dataclass
class TuningMetrics:
    """Tuning performance metrics."""
    timestamp: datetime
    active_trials: int
    completed_trials: int
    failed_trials: int
    mean_trial_duration: float
    trial_throughput: float
    best_score: Optional[float]
    worker_utilization: Dict[str, float]

class TuningMonitor:
    """Monitor distributed tuning performance."""
    
    def __init__(
        self,
        tuner: DistributedTuner,
        config: MonitoringConfig = None
    ):
        self.tuner = tuner
        self.config = config or MonitoringConfig()
        
        # Metric storage
        self.resource_history: List[ResourceMetrics] = []
        self.tuning_history: List[TuningMetrics] = []
        self.alerts: List[Dict[str, Any]] = []
        
        # Monitoring state
        self.monitoring: bool = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Started tuning performance monitoring")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self._monitor_task = None
        logger.info("Stopped tuning performance monitoring")
    
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect system resource metrics."""
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        net_io = psutil.net_io_counters()._asdict()
        
        # Get per-worker metrics
        per_worker_cpu = {}
        per_worker_memory = {}
        
        if self.tuner.config.cluster_mode == "dask":
            for worker in self.tuner.client.scheduler_info()["workers"]:
                info = self.tuner.client.scheduler_info()["workers"][worker]
                per_worker_cpu[worker] = info.get("cpu_percent", 0)
                per_worker_memory[worker] = info.get("memory_percent", 0)
        
        elif self.tuner.config.cluster_mode == "ray":
            for node in ray.nodes():
                resources = node["Resources"]
                worker_id = node["NodeID"]
                per_worker_cpu[worker_id] = resources.get("CPU", 0) * 100
                per_worker_memory[worker_id] = (
                    resources.get("memory", 0) /
                    resources.get("memory_total", 1) * 100
                )
        
        return ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            network_io=net_io,
            per_worker_cpu=per_worker_cpu,
            per_worker_memory=per_worker_memory
        )
    
    def _collect_tuning_metrics(self) -> TuningMetrics:
        """Collect tuning performance metrics."""
        metrics = defaultdict(float)
        
        for study_name, study in self.tuner.studies.items():
            status = self.tuner.get_study_status(study_name)
            
            metrics["active_trials"] += status.get("running_trials", 0)
            metrics["completed_trials"] += status.get("complete_trials", 0)
            metrics["failed_trials"] += status.get("failed_trials", 0)
            
            if status.get("study_duration"):
                duration = status["study_duration"].total_seconds()
                throughput = status["complete_trials"] / duration
                metrics["trial_throughput"] += throughput
            
            if status.get("best_score"):
                metrics["best_score"] = max(
                    metrics["best_score"],
                    status["best_score"]
                )
        
        # Calculate worker utilization
        utilization = {}
        total_trials = (
            metrics["active_trials"] +
            metrics["completed_trials"]
        )
        
        if total_trials > 0:
            for i in range(self.tuner.config.n_workers):
                worker_trials = total_trials / self.tuner.config.n_workers
                utilization[f"worker_{i}"] = worker_trials / total_trials
        
        return TuningMetrics(
            timestamp=datetime.now(),
            active_trials=int(metrics["active_trials"]),
            completed_trials=int(metrics["completed_trials"]),
            failed_trials=int(metrics["failed_trials"]),
            mean_trial_duration=metrics.get("mean_duration", 0),
            trial_throughput=metrics["trial_throughput"],
            best_score=metrics.get("best_score"),
            worker_utilization=utilization
        )
    
    def _check_alerts(
        self,
        resource_metrics: ResourceMetrics,
        tuning_metrics: TuningMetrics
    ):
        """Check for alert conditions."""
        alerts = []
        
        # Resource alerts
        if resource_metrics.cpu_percent > self.config.alert_thresholds["cpu_percent"]:
            alerts.append({
                "type": "resource",
                "severity": "warning",
                "message": f"High CPU usage: {resource_metrics.cpu_percent:.1f}%"
            })
        
        if resource_metrics.memory_percent > self.config.alert_thresholds["memory_percent"]:
            alerts.append({
                "type": "resource",
                "severity": "warning",
                "message": f"High memory usage: {resource_metrics.memory_percent:.1f}%"
            })
        
        # Performance alerts
        total_trials = tuning_metrics.completed_trials + tuning_metrics.failed_trials
        if total_trials > 0:
            failure_rate = tuning_metrics.failed_trials / total_trials
            if failure_rate > self.config.alert_thresholds["trial_failure_rate"]:
                alerts.append({
                    "type": "performance",
                    "severity": "error",
                    "message": f"High trial failure rate: {failure_rate:.1%}"
                })
        
        # Worker alerts
        for worker, util in tuning_metrics.worker_utilization.items():
            if util < 0.5:  # Less than 50% utilization
                alerts.append({
                    "type": "worker",
                    "severity": "warning",
                    "message": f"Low worker utilization for {worker}: {util:.1%}"
                })
        
        # Add alerts to history
        for alert in alerts:
            alert["timestamp"] = datetime.now()
            self.alerts.append(alert)
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect metrics
                resource_metrics = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self._collect_resource_metrics
                )
                tuning_metrics = self._collect_tuning_metrics()
                
                # Update history
                self.resource_history.append(resource_metrics)
                self.tuning_history.append(tuning_metrics)
                
                # Trim history
                if len(self.resource_history) > self.config.history_window:
                    self.resource_history = self.resource_history[-self.config.history_window:]
                if len(self.tuning_history) > self.config.history_window:
                    self.tuning_history = self.tuning_history[-self.config.history_window:]
                
                # Check alerts
                self._check_alerts(resource_metrics, tuning_metrics)
                
                # Export metrics
                await self._export_metrics()
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            await asyncio.sleep(self.config.sampling_interval)
    
    async def _export_metrics(self):
        """Export monitoring metrics."""
        if not self.config.export_path:
            return
        
        try:
            export_dir = Path(self.config.export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export resource metrics
            resource_df = pd.DataFrame([
                {
                    "timestamp": m.timestamp,
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "disk_percent": m.disk_percent,
                    **m.network_io,
                    **{f"worker_{k}_cpu": v for k, v in m.per_worker_cpu.items()},
                    **{f"worker_{k}_memory": v for k, v in m.per_worker_memory.items()}
                }
                for m in self.resource_history
            ])
            resource_df.to_csv(export_dir / f"resource_metrics_{timestamp}.csv")
            
            # Export tuning metrics
            tuning_df = pd.DataFrame([
                {
                    "timestamp": m.timestamp,
                    "active_trials": m.active_trials,
                    "completed_trials": m.completed_trials,
                    "failed_trials": m.failed_trials,
                    "trial_throughput": m.trial_throughput,
                    "best_score": m.best_score,
                    **m.worker_utilization
                }
                for m in self.tuning_history
            ])
            tuning_df.to_csv(export_dir / f"tuning_metrics_{timestamp}.csv")
            
            # Export alerts
            if self.alerts:
                alerts_df = pd.DataFrame(self.alerts)
                alerts_df.to_csv(export_dir / f"alerts_{timestamp}.csv")
        
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        if not self.resource_history or not self.tuning_history:
            return {}
        
        latest_resource = self.resource_history[-1]
        latest_tuning = self.tuning_history[-1]
        
        return {
            "resources": {
                "cpu_percent": latest_resource.cpu_percent,
                "memory_percent": latest_resource.memory_percent,
                "disk_percent": latest_resource.disk_percent,
                "network_io": latest_resource.network_io
            },
            "tuning": {
                "active_trials": latest_tuning.active_trials,
                "completed_trials": latest_tuning.completed_trials,
                "failed_trials": latest_tuning.failed_trials,
                "trial_throughput": latest_tuning.trial_throughput,
                "best_score": latest_tuning.best_score
            },
            "workers": {
                "utilization": latest_tuning.worker_utilization,
                "cpu": latest_resource.per_worker_cpu,
                "memory": latest_resource.per_worker_memory
            },
            "alerts": len(self.alerts)
        }

def create_tuning_monitor(
    tuner: DistributedTuner,
    config: Optional[MonitoringConfig] = None
) -> TuningMonitor:
    """Create tuning monitor."""
    return TuningMonitor(tuner, config)

if __name__ == "__main__":
    # Example usage
    from .distributed_tuning import create_distributed_tuner
    from .priority_tuning import create_priority_tuner
    from .priority_validation import create_priority_validator
    from .adaptive_priority import create_priority_learner
    from .notification_priority import create_priority_router
    from .notification_throttling import create_throttled_manager
    from .notification_channels import create_notification_manager
    
    async def main():
        # Create tuning stack
        manager = create_notification_manager()
        throttler = create_throttled_manager(manager)
        router = create_priority_router(throttler)
        learner = create_priority_learner(router)
        validator = create_priority_validator(learner)
        tuner = create_priority_tuner(validator)
        dist_tuner = create_distributed_tuner(tuner)
        
        # Create monitor
        monitor = create_tuning_monitor(
            dist_tuner,
            MonitoringConfig(
                sampling_interval=5.0,
                export_path=Path("tuning_metrics")
            )
        )
        
        # Start monitoring
        await monitor.start_monitoring()
        
        try:
            # Run distributed tuning
            result = await dist_tuner.tune_distributed()
            
            # Get performance summary
            summary = monitor.get_performance_summary()
            print("Performance Summary:", json.dumps(summary, indent=2))
            
        finally:
            await monitor.stop_monitoring()
    
    asyncio.run(main())
