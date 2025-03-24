"""Scheduling for validation reports."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
import pytz
import croniter
from collections import defaultdict

from .validation_reports import (
    ValidationReporter, ReportConfig, ReportSummary
)

@dataclass
class ScheduleConfig:
    """Configuration for report scheduling."""
    enabled: bool = True
    timezone: str = "UTC"
    report_retention: timedelta = timedelta(days=30)
    archive_dir: Optional[Path] = None
    max_concurrent: int = 3
    enable_recovery: bool = True
    recovery_window: timedelta = timedelta(hours=1)
    backup_schedule: str = "0 0 * * *"  # Daily at midnight

@dataclass
class ReportSchedule:
    """Schedule for report generation."""
    name: str
    cron: str
    config: Optional[ReportConfig] = None
    notification_targets: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    error_count: int = 0
    created: datetime = field(default_factory=datetime.now)

class ReportScheduler:
    """Schedule and manage report generation."""
    
    def __init__(
        self,
        reporter: ValidationReporter,
        config: ScheduleConfig = None
    ):
        self.reporter = reporter
        self.config = config or ScheduleConfig()
        
        # Initialize timezone
        try:
            self.timezone = pytz.timezone(self.config.timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            logging.warning(f"Unknown timezone {self.config.timezone}, using UTC")
            self.timezone = pytz.UTC
        
        # Schedule storage
        self.schedules: Dict[str, ReportSchedule] = {}
        self.running: Set[str] = set()
        self.pending: List[str] = []
        
        # State management
        self.is_running: bool = False
        self.last_backup: Optional[datetime] = None
        self.last_cleanup: Optional[datetime] = None
        
        # Stats tracking
        self.stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "runs": 0,
            "errors": 0,
            "last_duration": None,
            "avg_duration": None
        })
    
    async def start(self):
        """Start the scheduler."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start management tasks
        asyncio.create_task(self._run_scheduler())
        asyncio.create_task(self._run_backup())
        asyncio.create_task(self._run_cleanup())
        
        logging.info("Report scheduler started")
    
    async def stop(self):
        """Stop the scheduler."""
        self.is_running = False
        
        # Wait for running tasks
        while self.running:
            await asyncio.sleep(1)
        
        logging.info("Report scheduler stopped")
    
    async def add_schedule(
        self,
        schedule: ReportSchedule
    ) -> bool:
        """Add a report schedule."""
        try:
            # Validate cron expression
            cron = croniter.croniter(schedule.cron)
            
            # Set next run time
            now = datetime.now(self.timezone)
            schedule.next_run = cron.get_next(datetime)
            
            self.schedules[schedule.name] = schedule
            await self._save_state()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to add schedule {schedule.name}: {e}")
            return False
    
    async def remove_schedule(
        self,
        name: str
    ) -> bool:
        """Remove a report schedule."""
        if name in self.schedules:
            del self.schedules[name]
            await self._save_state()
            return True
        return False
    
    async def get_schedule(
        self,
        name: str
    ) -> Optional[ReportSchedule]:
        """Get schedule details."""
        return self.schedules.get(name)
    
    async def list_schedules(self) -> List[Dict[str, Any]]:
        """List all schedules."""
        return [
            {
                "name": name,
                "cron": schedule.cron,
                "enabled": schedule.enabled,
                "last_run": schedule.last_run.isoformat() if schedule.last_run else None,
                "next_run": schedule.next_run.isoformat() if schedule.next_run else None,
                "error_count": schedule.error_count,
                "stats": self.stats[name]
            }
            for name, schedule in self.schedules.items()
        ]
    
    async def _run_scheduler(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                now = datetime.now(self.timezone)
                
                # Check schedules
                for name, schedule in self.schedules.items():
                    if (
                        schedule.enabled and
                        schedule.next_run and
                        now >= schedule.next_run and
                        name not in self.running
                    ):
                        # Add to pending queue
                        self.pending.append(name)
                
                # Process pending reports
                while (
                    self.pending and
                    len(self.running) < self.config.max_concurrent
                ):
                    name = self.pending.pop(0)
                    self.running.add(name)
                    asyncio.create_task(self._generate_report(name))
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logging.error(f"Scheduler error: {e}")
                await asyncio.sleep(5)
    
    async def _generate_report(
        self,
        name: str
    ):
        """Generate scheduled report."""
        schedule = self.schedules[name]
        start_time = datetime.now()
        
        try:
            # Generate report
            results = await self.reporter.validator.validate_all_rules()
            summary = await self.reporter.generate_report(results)
            
            # Update schedule
            schedule.last_run = start_time
            schedule.next_run = croniter.croniter(
                schedule.cron,
                start_time
            ).get_next(datetime)
            
            # Send notifications
            await self._send_notifications(name, summary)
            
            # Update stats
            duration = (datetime.now() - start_time).total_seconds()
            stats = self.stats[name]
            stats["runs"] += 1
            stats["last_duration"] = duration
            
            if stats["avg_duration"] is None:
                stats["avg_duration"] = duration
            else:
                stats["avg_duration"] = (
                    0.9 * stats["avg_duration"] +
                    0.1 * duration
                )
            
        except Exception as e:
            logging.error(f"Report generation error for {name}: {e}")
            schedule.error_count += 1
            self.stats[name]["errors"] += 1
        
        finally:
            self.running.remove(name)
            await self._save_state()
    
    async def _send_notifications(
        self,
        name: str,
        summary: ReportSummary
    ):
        """Send report notifications."""
        schedule = self.schedules[name]
        
        for target in schedule.notification_targets:
            try:
                if target["type"] == "email":
                    await self._send_email_notification(target, name, summary)
                elif target["type"] == "slack":
                    await self._send_slack_notification(target, name, summary)
                elif target["type"] == "webhook":
                    await self._send_webhook_notification(target, name, summary)
            except Exception as e:
                logging.error(f"Notification error for {name}: {e}")
    
    async def _send_email_notification(
        self,
        target: Dict[str, Any],
        name: str,
        summary: ReportSummary
    ):
        """Send email notification."""
        # Implementation would connect to email service
        pass
    
    async def _send_slack_notification(
        self,
        target: Dict[str, Any],
        name: str,
        summary: ReportSummary
    ):
        """Send Slack notification."""
        # Implementation would connect to Slack API
        pass
    
    async def _send_webhook_notification(
        self,
        target: Dict[str, Any],
        name: str,
        summary: ReportSummary
    ):
        """Send webhook notification."""
        # Implementation would send HTTP request
        pass
    
    async def _run_backup(self):
        """Backup scheduler state."""
        while self.is_running:
            try:
                now = datetime.now(self.timezone)
                
                if (
                    not self.last_backup or
                    croniter.croniter(
                        self.config.backup_schedule,
                        self.last_backup
                    ).get_next(datetime) <= now
                ):
                    await self._save_state(backup=True)
                    self.last_backup = now
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logging.error(f"Backup error: {e}")
                await asyncio.sleep(300)
    
    async def _run_cleanup(self):
        """Clean up old reports."""
        while self.is_running:
            try:
                now = datetime.now(self.timezone)
                cutoff = now - self.config.report_retention
                
                # Clean output directory
                for path in self.reporter.config.output_dir.glob("*"):
                    if path.stat().st_mtime < cutoff.timestamp():
                        path.unlink()
                
                # Clean archive directory
                if self.config.archive_dir:
                    for path in self.config.archive_dir.glob("*"):
                        if path.stat().st_mtime < cutoff.timestamp():
                            path.unlink()
                
                self.last_cleanup = now
                await asyncio.sleep(3600)  # Check hourly
                
            except Exception as e:
                logging.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def _save_state(
        self,
        backup: bool = False
    ):
        """Save scheduler state."""
        state = {
            "schedules": {
                name: {
                    "name": schedule.name,
                    "cron": schedule.cron,
                    "enabled": schedule.enabled,
                    "last_run": schedule.last_run.isoformat() if schedule.last_run else None,
                    "next_run": schedule.next_run.isoformat() if schedule.next_run else None,
                    "error_count": schedule.error_count,
                    "notification_targets": schedule.notification_targets,
                    "created": schedule.created.isoformat()
                }
                for name, schedule in self.schedules.items()
            },
            "stats": self.stats,
            "last_backup": self.last_backup.isoformat() if self.last_backup else None,
            "last_cleanup": self.last_cleanup.isoformat() if self.last_cleanup else None
        }
        
        # Save state file
        state_file = (
            self.config.archive_dir / f"scheduler_state_{datetime.now():%Y%m%d_%H%M%S}.json"
            if backup and self.config.archive_dir
            else Path("scheduler_state.json")
        )
        
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    async def _load_state(self):
        """Load scheduler state."""
        try:
            with open("scheduler_state.json") as f:
                state = json.load(f)
            
            # Restore schedules
            self.schedules = {
                name: ReportSchedule(
                    name=data["name"],
                    cron=data["cron"],
                    enabled=data["enabled"],
                    last_run=datetime.fromisoformat(data["last_run"])
                    if data["last_run"] else None,
                    next_run=datetime.fromisoformat(data["next_run"])
                    if data["next_run"] else None,
                    error_count=data["error_count"],
                    notification_targets=data["notification_targets"],
                    created=datetime.fromisoformat(data["created"])
                )
                for name, data in state["schedules"].items()
            }
            
            # Restore stats
            self.stats.update(state["stats"])
            
            # Restore timestamps
            if state["last_backup"]:
                self.last_backup = datetime.fromisoformat(state["last_backup"])
            if state["last_cleanup"]:
                self.last_cleanup = datetime.fromisoformat(state["last_cleanup"])
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to load state: {e}")
            return False

def create_report_scheduler(
    reporter: ValidationReporter,
    config: Optional[ScheduleConfig] = None
) -> ReportScheduler:
    """Create report scheduler."""
    return ReportScheduler(reporter, config)

if __name__ == "__main__":
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
        
        # Add test schedule
        await scheduler.add_schedule(
            ReportSchedule(
                name="hourly_validation",
                cron="0 * * * *",  # Every hour
                notification_targets=[
                    {
                        "type": "email",
                        "to": ["admin@example.com"],
                        "subject": "Hourly Validation Report"
                    }
                ]
            )
        )
        
        # Start scheduler
        await scheduler.start()
        
        try:
            while True:
                schedules = await scheduler.list_schedules()
                print("\nActive schedules:")
                for schedule in schedules:
                    print(f"- {schedule['name']}: Next run at {schedule['next_run']}")
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            await scheduler.stop()
    
    asyncio.run(main())
