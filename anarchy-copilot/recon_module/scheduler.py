"""Scheduler for recurring reconnaissance tasks."""

from typing import List, Dict, Any, Optional, TypedDict, cast, TypeVar, Union
from datetime import datetime, timedelta
import asyncio
import json

from .orchestrator import ScanOrchestrator
from .database import ReconDatabase
from .models import ReconResult

T = TypeVar('T')

class ScanDict(TypedDict, total=False):
    """Type definition for scan result dictionaries."""
    domains: List[str]
    open_ports: List[int]
    findings: List[Dict[str, Any]]
    type: str
    details: Any

class ReconSchedule:
    """Configuration for a scheduled reconnaissance task."""
    def __init__(
        self,
        domain: str,
        interval_hours: int,
        enabled: bool = True,
        last_run: Optional[datetime] = None,
        next_run: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.domain = domain
        self.interval_hours = interval_hours
        self.enabled = enabled
        self.last_run = last_run
        self.next_run = next_run or datetime.now()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert schedule to dictionary for storage."""
        return {
            "domain": self.domain,
            "interval_hours": self.interval_hours,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReconSchedule':
        """Create schedule from dictionary."""
        return cls(
            domain=data["domain"],
            interval_hours=data["interval_hours"],
            enabled=data.get("enabled", True),
            last_run=datetime.fromisoformat(data["last_run"]) if data.get("last_run") else None,
            next_run=datetime.fromisoformat(data["next_run"]) if data.get("next_run") else None,
            metadata=data.get("metadata", {})
        )

class ReconScheduler:
    """Manages scheduled reconnaissance tasks."""

    def __init__(self, orchestrator: ScanOrchestrator, db: ReconDatabase):
        """Initialize scheduler with orchestrator and database."""
        self.orchestrator = orchestrator
        self.db = db
        self.schedules: Dict[str, ReconSchedule] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._schedule_file = "recon_schedules.json"
        self._load_schedules()

    def _load_schedules(self) -> None:
        """Load saved schedules from file."""
        try:
            with open(self._schedule_file, 'r') as f:
                data = json.load(f)
                self.schedules = {
                    domain: ReconSchedule.from_dict(schedule_data)
                    for domain, schedule_data in data.items()
                }
        except FileNotFoundError:
            self.schedules = {}

    def _save_schedules(self) -> None:
        """Save current schedules to file."""
        data = {
            domain: schedule.to_dict()
            for domain, schedule in self.schedules.items()
        }
        with open(self._schedule_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_schedule(
        self,
        domain: str,
        interval_hours: int,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new scheduled task."""
        schedule = ReconSchedule(
            domain=domain,
            interval_hours=interval_hours,
            enabled=enabled,
            metadata=metadata
        )
        self.schedules[domain] = schedule
        self._save_schedules()

    def remove_schedule(self, domain: str) -> None:
        """Remove a scheduled task."""
        if domain in self.schedules:
            del self.schedules[domain]
            self._save_schedules()

    def update_schedule(
        self,
        domain: str,
        interval_hours: Optional[int] = None,
        enabled: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update an existing schedule."""
        if domain not in self.schedules:
            raise ValueError(f"No schedule exists for domain: {domain}")

        schedule = self.schedules[domain]
        if interval_hours is not None:
            schedule.interval_hours = interval_hours
        if enabled is not None:
            schedule.enabled = enabled
        if metadata is not None:
            schedule.metadata.update(metadata)

        self._save_schedules()

    def get_schedule(self, domain: str) -> Optional[ReconSchedule]:
        """Get schedule for a domain."""
        return self.schedules.get(domain)

    def list_schedules(self) -> List[Dict[str, Any]]:
        """List all scheduled tasks."""
        return [
            {
                "domain": domain,
                **schedule.to_dict()
            }
            for domain, schedule in self.schedules.items()
        ]

    async def _run_scheduled_scan(self, domain: str, schedule: ReconSchedule) -> None:
        """Execute a scheduled scan."""
        try:
            # Run the scan
            results = await self.orchestrator.full_scan(domain)
            
            # Compare with previous results
            if schedule.last_run:
                previous_results = await self.db.get_results_since(
                    domain, schedule.last_run
                )
                changes = self._analyze_changes(previous_results, results)
                # Store changes in metadata
                for result in results:
                    if not result.metadata:
                        result.metadata = {}
                    result.metadata["changes"] = changes.get(result.scan_type, {})
            
            # Update schedule tracking
            schedule.last_run = datetime.now()
            schedule.next_run = schedule.last_run + timedelta(hours=schedule.interval_hours)
            
            # Save updated schedule
            self._save_schedules()
            
        except Exception as e:
            # Log error but don't stop scheduler
            print(f"Error in scheduled scan for {domain}: {str(e)}")
            if schedule.metadata is None:
                schedule.metadata = {}
            schedule.metadata["last_error"] = str(e)
            schedule.metadata["error_time"] = datetime.now().isoformat()

    def _analyze_changes(
        self,
        previous: List[ReconResult],
        current: List[ReconResult]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze changes between scan results."""
        changes: Dict[str, Dict[str, Any]] = {}
        
        # Group results by scan type
        prev_by_type = {r.scan_type: r for r in previous}
        curr_by_type = {r.scan_type: r for r in current}
        
        # Compare each scan type
        for scan_type in set(prev_by_type.keys()) | set(curr_by_type.keys()):
            prev_result = prev_by_type.get(scan_type)
            curr_result = curr_by_type.get(scan_type)
            
            if not prev_result:
                changes[scan_type] = {
                    "type": "new",
                    "details": curr_result.results if curr_result else None
                }
            elif not curr_result:
                changes[scan_type] = {
                    "type": "removed",
                    "details": prev_result.results
                }
            else:
                # Compare results based on scan type
                if scan_type == "subdomain_scan":
                    changes[scan_type] = self._compare_subdomains(
                        cast(ScanDict, prev_result.results),
                        cast(ScanDict, curr_result.results)
                    )
                elif scan_type == "port_scan":
                    changes[scan_type] = self._compare_ports(
                        cast(ScanDict, prev_result.results),
                        cast(ScanDict, curr_result.results)
                    )
                elif scan_type in ["web_scan", "vuln_scan"]:
                    changes[scan_type] = self._compare_endpoints(
                        cast(ScanDict, prev_result.results),
                        cast(ScanDict, curr_result.results)
                    )
        
        return changes

    def _compare_subdomains(
        self,
        prev: ScanDict,
        curr: ScanDict
    ) -> Dict[str, Any]:
        """Compare subdomain scan results."""
        prev_domains = set(cast(List[str], prev.get("domains", [])))
        curr_domains = set(cast(List[str], curr.get("domains", [])))
        
        return {
            "new": list(curr_domains - prev_domains),
            "removed": list(prev_domains - curr_domains),
            "total": len(curr_domains)
        }

    def _compare_ports(
        self,
        prev: ScanDict,
        curr: ScanDict
    ) -> Dict[str, Any]:
        """Compare port scan results."""
        prev_ports = set(cast(List[int], prev.get("open_ports", [])))
        curr_ports = set(cast(List[int], curr.get("open_ports", [])))
        
        return {
            "new": list(curr_ports - prev_ports),
            "removed": list(prev_ports - curr_ports),
            "total": len(curr_ports)
        }

    def _compare_endpoints(
        self,
        prev: ScanDict,
        curr: ScanDict
    ) -> Dict[str, Any]:
        """Compare web/vuln scan results."""
        prev_findings = cast(List[Dict[str, Any]], prev.get("findings", []))
        curr_findings = cast(List[Dict[str, Any]], curr.get("findings", []))
        
        # Convert to sets for comparison
        prev_set = {json.dumps(f, sort_keys=True) for f in prev_findings}
        curr_set = {json.dumps(f, sort_keys=True) for f in curr_findings}
        
        return {
            "new": [json.loads(f) for f in curr_set - prev_set],
            "removed": [json.loads(f) for f in prev_set - curr_set],
            "total": len(curr_findings)
        }

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            now = datetime.now()
            
            # Check each schedule
            for domain, schedule in self.schedules.items():
                if (
                    schedule.enabled and 
                    schedule.next_run and 
                    now >= schedule.next_run
                ):
                    # Run scan in background
                    asyncio.create_task(
                        self._run_scheduled_scan(domain, schedule)
                    )
            
            # Wait before next check
            await asyncio.sleep(60)  # Check every minute

    def start(self) -> None:
        """Start the scheduler."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._scheduler_loop())

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
