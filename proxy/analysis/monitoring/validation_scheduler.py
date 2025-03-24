"""Scheduler for automated validation exports."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from croniter import croniter
import pytz

from .validation_export import ValidationExporter, create_validation_exporter
from .interactive_validation import InteractiveValidationControls

logger = logging.getLogger(__name__)

@dataclass
class ExportSchedule:
    """Schedule configuration for automated exports."""
    cron_expression: str  # Cron expression for scheduling
    formats: List[str]  # Export formats
    window: Optional[int]  # Time window for metrics
    report_type: str  # Type of export (metrics/dashboard/comparison/full)
    metadata: Dict[str, Any]  # Additional metadata

class ValidationScheduler:
    """Schedule and manage automated validation exports."""
    
    def __init__(
        self,
        exporter: ValidationExporter,
        schedules_file: Optional[Path] = None
    ):
        self.exporter = exporter
        self.schedules_file = schedules_file or Path("validation_schedules.json")
        self.schedules: Dict[str, ExportSchedule] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.timezone = pytz.timezone("UTC")
        
        # Load existing schedules
        self._load_schedules()
    
    def _load_schedules(self):
        """Load schedules from file."""
        if self.schedules_file.exists():
            try:
                with open(self.schedules_file) as f:
                    data = json.load(f)
                
                for schedule_id, config in data.items():
                    self.schedules[schedule_id] = ExportSchedule(**config)
                
                logger.info(f"Loaded {len(self.schedules)} export schedules")
                
            except Exception as e:
                logger.error(f"Failed to load schedules: {e}")
    
    def _save_schedules(self):
        """Save schedules to file."""
        try:
            data = {
                id: {
                    "cron_expression": s.cron_expression,
                    "formats": s.formats,
                    "window": s.window,
                    "report_type": s.report_type,
                    "metadata": s.metadata
                }
                for id, s in self.schedules.items()
            }
            
            with open(self.schedules_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.schedules)} export schedules")
            
        except Exception as e:
            logger.error(f"Failed to save schedules: {e}")
    
    def add_schedule(
        self,
        schedule_id: str,
        cron_expression: str,
        report_type: str,
        formats: List[str] = ["html"],
        window: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add new export schedule."""
        if schedule_id in self.schedules:
            raise ValueError(f"Schedule {schedule_id} already exists")
        
        if not croniter.is_valid(cron_expression):
            raise ValueError(f"Invalid cron expression: {cron_expression}")
        
        if report_type not in ["metrics", "dashboard", "comparison", "full"]:
            raise ValueError(f"Invalid report type: {report_type}")
        
        schedule = ExportSchedule(
            cron_expression=cron_expression,
            formats=formats,
            window=window,
            report_type=report_type,
            metadata=metadata or {}
        )
        
        self.schedules[schedule_id] = schedule
        self._save_schedules()
        
        # Start schedule if scheduler is running
        if self.active_tasks:
            self._start_schedule(schedule_id, schedule)
    
    def remove_schedule(
        self,
        schedule_id: str
    ):
        """Remove export schedule."""
        if schedule_id not in self.schedules:
            raise ValueError(f"Schedule {schedule_id} not found")
        
        # Stop running task if exists
        if schedule_id in self.active_tasks:
            self.active_tasks[schedule_id].cancel()
            del self.active_tasks[schedule_id]
        
        del self.schedules[schedule_id]
        self._save_schedules()
    
    def update_schedule(
        self,
        schedule_id: str,
        **updates
    ):
        """Update existing export schedule."""
        if schedule_id not in self.schedules:
            raise ValueError(f"Schedule {schedule_id} not found")
        
        schedule = self.schedules[schedule_id]
        
        # Update schedule fields
        for field, value in updates.items():
            if not hasattr(schedule, field):
                raise ValueError(f"Invalid field: {field}")
            setattr(schedule, field, value)
        
        # Validate cron expression if updated
        if "cron_expression" in updates:
            if not croniter.is_valid(schedule.cron_expression):
                raise ValueError(
                    f"Invalid cron expression: {schedule.cron_expression}"
                )
        
        self._save_schedules()
        
        # Restart schedule if running
        if schedule_id in self.active_tasks:
            self.active_tasks[schedule_id].cancel()
            self._start_schedule(schedule_id, schedule)
    
    async def start(self):
        """Start all export schedules."""
        logger.info("Starting validation scheduler")
        
        # Start all schedules
        for schedule_id, schedule in self.schedules.items():
            self._start_schedule(schedule_id, schedule)
    
    def _start_schedule(
        self,
        schedule_id: str,
        schedule: ExportSchedule
    ):
        """Start single export schedule."""
        task = asyncio.create_task(
            self._run_schedule(schedule_id, schedule)
        )
        self.active_tasks[schedule_id] = task
    
    async def _run_schedule(
        self,
        schedule_id: str,
        schedule: ExportSchedule
    ):
        """Run scheduled exports."""
        try:
            while True:
                now = datetime.now(self.timezone)
                cron = croniter(schedule.cron_expression, now)
                next_run = cron.get_next(datetime)
                
                # Sleep until next run
                await asyncio.sleep(
                    (next_run - now).total_seconds()
                )
                
                # Perform export
                try:
                    if schedule.report_type == "metrics":
                        for format in schedule.formats:
                            await self._export_metrics(
                                format,
                                schedule.window,
                                schedule.metadata
                            )
                    
                    elif schedule.report_type == "dashboard":
                        for format in schedule.formats:
                            await self._export_dashboard(
                                format,
                                schedule.metadata
                            )
                    
                    elif schedule.report_type == "comparison":
                        for format in schedule.formats:
                            await self._export_comparison(
                                format,
                                schedule.metadata
                            )
                    
                    elif schedule.report_type == "full":
                        await self._export_full_report(
                            schedule.window,
                            schedule.metadata
                        )
                    
                    logger.info(
                        f"Completed scheduled export {schedule_id} at {datetime.now()}"
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Failed scheduled export {schedule_id}: {e}"
                    )
                
        except asyncio.CancelledError:
            logger.info(f"Cancelled schedule {schedule_id}")
    
    async def _export_metrics(
        self,
        format: str,
        window: Optional[int],
        metadata: Dict[str, Any]
    ):
        """Export metrics asynchronously."""
        return await asyncio.to_thread(
            self.exporter.export_metrics,
            format=format,
            window=window
        )
    
    async def _export_dashboard(
        self,
        format: str,
        metadata: Dict[str, Any]
    ):
        """Export dashboard asynchronously."""
        return await asyncio.to_thread(
            self.exporter.export_dashboard,
            format=format
        )
    
    async def _export_comparison(
        self,
        format: str,
        metadata: Dict[str, Any]
    ):
        """Export comparison asynchronously."""
        return await asyncio.to_thread(
            self.exporter.export_comparison,
            format=format
        )
    
    async def _export_full_report(
        self,
        window: Optional[int],
        metadata: Dict[str, Any]
    ):
        """Export full report asynchronously."""
        return await asyncio.to_thread(
            self.exporter.export_full_report,
            window=window
        )
    
    async def stop(self):
        """Stop all export schedules."""
        logger.info("Stopping validation scheduler")
        
        for schedule_id, task in self.active_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.active_tasks.clear()

def create_validation_scheduler(
    exporter: ValidationExporter,
    schedules_file: Optional[Path] = None
) -> ValidationScheduler:
    """Create validation scheduler."""
    return ValidationScheduler(exporter, schedules_file)

if __name__ == "__main__":
    # Example usage
    from .validation_export import create_validation_exporter
    from .interactive_validation import create_interactive_controls
    from .validation_visualization import create_validation_visualizer
    from .cluster_validation import create_cluster_validator
    from .alert_clustering import create_alert_clusterer
    from .alert_management import create_alert_manager
    from .realtime_anomalies import create_realtime_detector
    from .anomaly_detection import create_anomaly_detector
    from .exploration_trends import create_trend_analyzer
    
    async def main():
        # Create components
        analyzer = create_trend_analyzer()
        detector = create_anomaly_detector(analyzer)
        realtime = create_realtime_detector(detector)
        manager = create_alert_manager(realtime)
        clusterer = create_alert_clusterer(manager)
        validator = create_cluster_validator(clusterer)
        visualizer = create_validation_visualizer(validator)
        controls = create_interactive_controls(visualizer)
        exporter = create_validation_exporter(controls)
        scheduler = create_validation_scheduler(exporter)
        
        # Add schedules
        scheduler.add_schedule(
            "daily_metrics",
            "0 0 * * *",  # Daily at midnight
            "metrics",
            formats=["csv", "json"],
            window=24,
            metadata={"description": "Daily metrics export"}
        )
        
        scheduler.add_schedule(
            "weekly_report",
            "0 0 * * 0",  # Weekly on Sunday
            "full",
            window=168,
            metadata={"description": "Weekly validation report"}
        )
        
        # Start scheduler
        await scheduler.start()
        
        try:
            # Run for demonstration
            await asyncio.sleep(60)
        finally:
            # Stop scheduler
            await scheduler.stop()
    
    # Run example
    asyncio.run(main())
