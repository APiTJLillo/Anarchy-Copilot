"""Vulnerability scanning models."""

from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


class VulnSeverity(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ScanConfig:
    """Configuration for vulnerability scans."""
    target: str
    templates: Optional[List[str]] = None
    severity: Optional[List[VulnSeverity]] = None
    threads: int = 10
    timeout: int = 30
    rate_limit: Optional[int] = None


@dataclass
class VulnResult:
    """Represents a vulnerability finding."""
    name: str
    description: str
    severity: VulnSeverity
    target: str
    url: str
    timestamp: datetime
    scanner: str
    template_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    verified: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "target": self.target,
            "url": self.url,
            "timestamp": self.timestamp.isoformat(),
            "scanner": self.scanner,
            "template_id": self.template_id,
            "details": self.details,
            "verified": self.verified
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VulnResult":
        """Create from dictionary format."""
        data = data.copy()
        data["severity"] = VulnSeverity(data["severity"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
