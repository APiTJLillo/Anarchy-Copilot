"""Rate limiting for scanning tools to prevent overwhelming targets."""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import os

@dataclass
class ScanRecord:
    """Record of a scan against a target."""
    target: str
    tool: str
    timestamp: datetime
    scan_duration: float
    scan_size: int  # Number of requests/ports/etc.

class RateLimiter:
    """Rate limiter for scanning tools."""
    
    # Default limits for different tools
    DEFAULT_LIMITS = {
        # Tool: (requests_per_minute, min_interval_between_scans_seconds, max_concurrent)
        "masscan": (1000, 300, 1),    # 1000 ports/min, 5 min between scans, 1 concurrent
        "nmap": (100, 300, 2),        # 100 ports/min, 5 min between scans, 2 concurrent
        "httpx": (60, 60, 3),         # 60 requests/min, 1 min between scans, 3 concurrent
        "nuclei": (30, 120, 2),       # 30 requests/min, 2 min between scans, 2 concurrent
        "default": (30, 60, 1)        # Default conservative limits
    }

    def __init__(self, history_file: str = "scan_history.json"):
        """Initialize rate limiter."""
        self.scan_history: List[ScanRecord] = []
        self.active_scans: Dict[str, List[str]] = {}  # tool -> list of targets
        self.history_file = history_file
        self._load_history()

    def _load_history(self) -> None:
        """Load scan history from file."""
        if not os.path.exists(self.history_file):
            return

        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self.scan_history = [
                    ScanRecord(
                        target=record['target'],
                        tool=record['tool'],
                        timestamp=datetime.fromisoformat(record['timestamp']),
                        scan_duration=record['scan_duration'],
                        scan_size=record['scan_size']
                    )
                    for record in data
                ]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading scan history: {e}")
            self.scan_history = []

    def _save_history(self) -> None:
        """Save scan history to file."""
        try:
            # Convert to serializable format
            data = [
                {
                    'target': record.target,
                    'tool': record.tool,
                    'timestamp': record.timestamp.isoformat(),
                    'scan_duration': record.scan_duration,
                    'scan_size': record.scan_size
                }
                for record in self.scan_history
            ]
            
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving scan history: {e}")

    def _cleanup_history(self) -> None:
        """Remove old scan records (older than 24 hours)."""
        cutoff = datetime.now() - timedelta(hours=24)
        self.scan_history = [
            record for record in self.scan_history
            if record.timestamp > cutoff
        ]

    def can_scan(self, target: str, tool: str) -> "tuple[bool, Optional[str]]":
        """Check if a scan can be started."""
        tool_limits = self.DEFAULT_LIMITS.get(tool, self.DEFAULT_LIMITS["default"])
        requests_per_minute, min_interval, max_concurrent = tool_limits

        # Check concurrent scans
        active_scans = self.active_scans.get(tool, [])
        if len(active_scans) >= max_concurrent:
            return False, f"Too many concurrent {tool} scans (max {max_concurrent})"

        # Check if target is already being scanned
        if target in active_scans:
            return False, f"Target {target} is already being scanned by {tool}"

        # Find most recent scan of this target with this tool
        recent_scans = [
            record for record in self.scan_history
            if record.target == target and record.tool == tool
        ]
        
        if recent_scans:
            most_recent = max(recent_scans, key=lambda x: x.timestamp)
            time_since_scan = (datetime.now() - most_recent.timestamp).total_seconds()
            
            if time_since_scan < min_interval:
                return False, (
                    f"Must wait {min_interval - time_since_scan:.0f} more seconds "
                    f"before scanning {target} with {tool} again"
                )

        # Count recent requests in the last minute
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        recent_requests = sum(
            record.scan_size for record in self.scan_history
            if record.tool == tool and record.timestamp > one_minute_ago
        )

        if recent_requests >= requests_per_minute:
            return False, f"Rate limit exceeded for {tool} ({requests_per_minute} requests/minute)"

        return True, None

    def start_scan(self, target: str, tool: str) -> None:
        """Record the start of a scan."""
        if tool not in self.active_scans:
            self.active_scans[tool] = []
        self.active_scans[tool].append(target)

    def end_scan(self, target: str, tool: str, scan_size: int, duration: float) -> None:
        """Record the end of a scan."""
        # Remove from active scans
        if tool in self.active_scans:
            self.active_scans[tool] = [t for t in self.active_scans[tool] if t != target]

        # Add to history
        self.scan_history.append(ScanRecord(
            target=target,
            tool=tool,
            timestamp=datetime.now(),
            scan_duration=duration,
            scan_size=scan_size
        ))

        # Cleanup and save
        self._cleanup_history()
        self._save_history()

    def get_tool_limits(self, tool: str) -> tuple[int, int, int]:
        """Get rate limits for a tool."""
        return self.DEFAULT_LIMITS.get(tool, self.DEFAULT_LIMITS["default"])

    def get_active_scans(self, tool: Optional[str] = None) -> Dict[str, List[str]]:
        """Get currently active scans."""
        if tool:
            return {tool: self.active_scans.get(tool, [])}
        return self.active_scans

    def get_scan_history(self, 
                        tool: Optional[str] = None,
                        target: Optional[str] = None,
                        hours: int = 24) -> List[ScanRecord]:
        """Get scan history filtered by tool, target, and time window."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        return [
            record for record in self.scan_history
            if record.timestamp > cutoff
            and (tool is None or record.tool == tool)
            and (target is None or record.target == target)
        ]
