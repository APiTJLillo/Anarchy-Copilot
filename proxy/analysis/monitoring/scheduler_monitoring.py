"""Monitoring hooks for report scheduler."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Callable, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from collections import deque
import psutil
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .report_scheduler import ReportScheduler, ReportSchedule, ScheduleConfig

@dataclass
class MonitoringConfig:
    """Configuration for scheduler monitoring."""
    enabled: bool = True
    history_size: int = 1000
    metrics_interval: float = 1.0  # seconds
    log_level: str = "INFO"
    enable_profiling: bool = True
    enable_alerting: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_percent": 80.0,
        "memory_percent": 80.0,
        "error_rate": 0.1,
        "schedule_delay": 300.0  # seconds
    })
    visualization_dir: Path = Path("monitoring_visualizations")

class MonitoringHook(Protocol):
    """Protocol for monitoring hooks."""
    async def on_schedule_start(
        self,
        schedule: ReportSchedule
    ):
        """Called when a schedule starts."""
        ...
    
    async def on_schedule_complete(
        self,
        schedule: ReportSchedule,
        duration: float,
        success: bool
    ):
        """Called when a schedule completes."""
        ...
    
    async def on_error(
        self,
        schedule: Optional[ReportSchedule],
        error: Exception,
        context: Dict[str, Any]
    ):
        """Called when an error occurs."""
        ...
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect monitoring metrics."""
        ...

@dataclass
class SchedulerMetrics:
    """Scheduler performance metrics."""
    timestamp: datetime
    active_schedules: int
    running_schedules: int
    pending_schedules: int
    total_runs: int
    total_errors: int
    cpu_percent: float
    memory_percent: float
    avg_duration: float
    success_rate: float
    context: Dict[str, Any] = field(default_factory=dict)

class SchedulerMonitor:
    """Monitor scheduler performance and health."""
    
    def __init__(
        self,
        scheduler: ReportScheduler,
        config: MonitoringConfig = None
    ):
        self.scheduler = scheduler
        self.config = config or MonitoringConfig()
        
        # Metric storage
        self.metrics_history: deque[SchedulerMetrics] = deque(
            maxlen=self.config.history_size
        )
        
        # Hook registry
        self.hooks: List[MonitoringHook] = []
        
        # State tracking
        self.start_times: Dict[str, datetime] = {}
        self.schedule_stats: Dict[str, Dict[str, Any]] = {}
        
        # Process monitoring
        self.process = psutil.Process()
        
        # Create output directory
        if self.config.visualization_dir:
            self.config.visualization_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("scheduler_monitor.log")
            ]
        )
    
    def add_hook(
        self,
        hook: MonitoringHook
    ):
        """Add monitoring hook."""
        self.hooks.append(hook)
    
    async def start_monitoring(self):
        """Start monitoring tasks."""
        if not self.config.enabled:
            return
        
        # Patch scheduler methods
        self._patch_scheduler()
        
        # Start metrics collection
        asyncio.create_task(self._collect_metrics())
    
    def _patch_scheduler(self):
        """Patch scheduler methods with monitoring."""
        original_generate = self.scheduler._generate_report
        
        async def monitored_generate(name: str):
            """Monitored report generation."""
            schedule = self.scheduler.schedules[name]
            start_time = datetime.now()
            self.start_times[name] = start_time
            
            # Notify hooks of start
            for hook in self.hooks:
                try:
                    await hook.on_schedule_start(schedule)
                except Exception as e:
                    logging.error(f"Hook error on schedule start: {e}")
            
            try:
                # Run original method
                await original_generate(name)
                success = True
            except Exception as e:
                success = False
                # Notify hooks of error
                for hook in self.hooks:
                    try:
                        await hook.on_error(schedule, e, {
                            "phase": "generation",
                            "duration": (datetime.now() - start_time).total_seconds()
                        })
                    except Exception as hook_e:
                        logging.error(f"Hook error on error: {hook_e}")
                raise
            finally:
                duration = (datetime.now() - start_time).total_seconds()
                
                # Update stats
                stats = self.schedule_stats.setdefault(name, {
                    "runs": 0,
                    "errors": 0,
                    "total_duration": 0,
                    "min_duration": None,
                    "max_duration": None,
                    "last_success": None
                })
                
                stats["runs"] += 1
                stats["total_duration"] += duration
                
                if not stats["min_duration"] or duration < stats["min_duration"]:
                    stats["min_duration"] = duration
                if not stats["max_duration"] or duration > stats["max_duration"]:
                    stats["max_duration"] = duration
                
                if not success:
                    stats["errors"] += 1
                else:
                    stats["last_success"] = datetime.now()
                
                # Notify hooks of completion
                for hook in self.hooks:
                    try:
                        await hook.on_schedule_complete(schedule, duration, success)
                    except Exception as e:
                        logging.error(f"Hook error on schedule complete: {e}")
        
        self.scheduler._generate_report = monitored_generate
    
    async def _collect_metrics(self):
        """Collect scheduler metrics."""
        while True:
            try:
                # Collect basic metrics
                metrics = SchedulerMetrics(
                    timestamp=datetime.now(),
                    active_schedules=len(self.scheduler.schedules),
                    running_schedules=len(self.scheduler.running),
                    pending_schedules=len(self.scheduler.pending),
                    total_runs=sum(s["runs"] for s in self.schedule_stats.values()),
                    total_errors=sum(s["errors"] for s in self.schedule_stats.values()),
                    cpu_percent=self.process.cpu_percent(),
                    memory_percent=self.process.memory_percent(),
                    avg_duration=(
                        sum(s["total_duration"] for s in self.schedule_stats.values()) /
                        max(sum(s["runs"] for s in self.schedule_stats.values()), 1)
                    ),
                    success_rate=(
                        1 - sum(s["errors"] for s in self.schedule_stats.values()) /
                        max(sum(s["runs"] for s in self.schedule_stats.values()), 1)
                    )
                )
                
                # Collect hook metrics
                for hook in self.hooks:
                    try:
                        hook_metrics = await hook.collect_metrics()
                        metrics.context.update(hook_metrics)
                    except Exception as e:
                        logging.error(f"Hook metrics error: {e}")
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Check thresholds
                if self.config.enable_alerting:
                    await self._check_thresholds(metrics)
                
                await asyncio.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logging.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.config.metrics_interval)
    
    async def _check_thresholds(
        self,
        metrics: SchedulerMetrics
    ):
        """Check metric thresholds."""
        alerts = []
        
        # CPU usage
        if (
            metrics.cpu_percent >
            self.config.alert_thresholds["cpu_percent"]
        ):
            alerts.append(
                f"High CPU usage: {metrics.cpu_percent:.1f}%"
            )
        
        # Memory usage
        if (
            metrics.memory_percent >
            self.config.alert_thresholds["memory_percent"]
        ):
            alerts.append(
                f"High memory usage: {metrics.memory_percent:.1f}%"
            )
        
        # Error rate
        if (
            1 - metrics.success_rate >
            self.config.alert_thresholds["error_rate"]
        ):
            alerts.append(
                f"High error rate: {(1 - metrics.success_rate)*100:.1f}%"
            )
        
        # Schedule delays
        for name, schedule in self.scheduler.schedules.items():
            if (
                schedule.next_run and
                datetime.now() - schedule.next_run >
                timedelta(seconds=self.config.alert_thresholds["schedule_delay"])
            ):
                alerts.append(
                    f"Schedule delay for {name}: "
                    f"{(datetime.now() - schedule.next_run).total_seconds():.0f}s"
                )
        
        if alerts:
            logging.warning("Scheduler alerts:\n" + "\n".join(alerts))
    
    def get_schedule_stats(
        self,
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get schedule statistics."""
        if name:
            return self.schedule_stats.get(name, {})
        
        return {
            name: stats for name, stats in self.schedule_stats.items()
        }
    
    async def create_monitoring_plots(
        self) -> Dict[str, go.Figure]:
        """Create monitoring visualization plots."""
        plots = {}
        
        if not self.metrics_history:
            return plots
        
        # Convert metrics to dataframe
        df = pd.DataFrame([
            {
                "timestamp": m.timestamp,
                "active_schedules": m.active_schedules,
                "running_schedules": m.running_schedules,
                "pending_schedules": m.pending_schedules,
                "cpu_percent": m.cpu_percent,
                "memory_percent": m.memory_percent,
                "success_rate": m.success_rate,
                "avg_duration": m.avg_duration
            }
            for m in self.metrics_history
        ])
        
        # Resource usage plot
        resource_fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["CPU Usage", "Memory Usage"]
        )
        
        resource_fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["cpu_percent"],
                name="CPU %",
                line=dict(color="blue")
            ),
            row=1,
            col=1
        )
        
        resource_fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["memory_percent"],
                name="Memory %",
                line=dict(color="green")
            ),
            row=2,
            col=1
        )
        
        # Add threshold lines
        resource_fig.add_hline(
            y=self.config.alert_thresholds["cpu_percent"],
            line_dash="dash",
            line_color="red",
            row=1,
            col=1
        )
        
        resource_fig.add_hline(
            y=self.config.alert_thresholds["memory_percent"],
            line_dash="dash",
            line_color="red",
            row=2,
            col=1
        )
        
        resource_fig.update_layout(
            height=600,
            showlegend=True,
            title="Resource Usage"
        )
        plots["resources"] = resource_fig
        
        # Schedule metrics plot
        schedule_fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Active Schedules",
                "Success Rate",
                "Average Duration",
                "Queue Size"
            ]
        )
        
        schedule_fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["active_schedules"],
                name="Active"
            ),
            row=1,
            col=1
        )
        
        schedule_fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["success_rate"],
                name="Success %"
            ),
            row=1,
            col=2
        )
        
        schedule_fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["avg_duration"],
                name="Avg Duration"
            ),
            row=2,
            col=1
        )
        
        schedule_fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["pending_schedules"],
                name="Queue Size"
            ),
            row=2,
            col=2
        )
        
        schedule_fig.update_layout(
            height=800,
            showlegend=True,
            title="Schedule Metrics"
        )
        plots["schedules"] = schedule_fig
        
        # Save plots
        if self.config.visualization_dir:
            for name, fig in plots.items():
                fig.write_html(
                    self.config.visualization_dir /
                    f"scheduler_monitor_{name}.html"
                )
        
        return plots

def create_scheduler_monitor(
    scheduler: ReportScheduler,
    config: Optional[MonitoringConfig] = None
) -> SchedulerMonitor:
    """Create scheduler monitor."""
    return SchedulerMonitor(scheduler, config)

if __name__ == "__main__":
    from .report_scheduler import create_report_scheduler
    from .validation_reports import create_validation_reporter
    from .rule_validation import create_rule_validator
    from .notification_rules import create_rule_engine
    from .alert_notifications import create_notification_manager
    from .anomaly_alerts import create_alert_manager
    from .anomaly_analysis import create_anomaly_detector
    from .trend_analysis import create_trend_analyzer
    from .adaptation_metrics import create_performance_tracker
    from .preset_adaptation import create_online_adapter
    from .preset_ensemble import create_preset_ensemble
    from .preset_predictions import create_preset_predictor
    from .preset_analytics import create_preset_analytics
    from .mutation_presets import create_preset_manager
    
    async def main():
        # Setup components
        manager = create_preset_manager()
        analytics = create_preset_analytics(manager)
        predictor = create_preset_predictor(analytics)
        ensemble = create_preset_ensemble(predictor)
        adapter = create_online_adapter(ensemble)
        tracker = create_performance_tracker(adapter)
        analyzer = create_trend_analyzer(tracker)
        detector = create_anomaly_detector(tracker, analyzer)
        alert_manager = create_alert_manager(detector)
        notifier = create_notification_manager(alert_manager)
        engine = create_rule_engine(notifier)
        validator = create_rule_validator(engine)
        reporter = create_validation_reporter(validator)
        scheduler = create_report_scheduler(reporter)
        monitor = create_scheduler_monitor(scheduler)
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Add test schedule
        await scheduler.add_schedule(
            ReportSchedule(
                name="test_schedule",
                cron="*/5 * * * *",  # Every 5 minutes
                notification_targets=[
                    {
                        "type": "email",
                        "to": ["admin@example.com"]
                    }
                ]
            )
        )
        
        # Run scheduler
        await scheduler.start()
        
        try:
            while True:
                # Create monitoring plots
                plots = await monitor.create_monitoring_plots()
                
                # Print stats
                stats = monitor.get_schedule_stats()
                print("\nSchedule stats:")
                for name, schedule_stats in stats.items():
                    print(f"\n{name}:")
                    for key, value in schedule_stats.items():
                        print(f"  {key}: {value}")
                
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            await scheduler.stop()
    
    asyncio.run(main())
