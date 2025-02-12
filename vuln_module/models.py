"""Data models for vulnerability discovery module."""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum, auto

class VulnSeverity(Enum):
    """Vulnerability severity levels."""
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

class PayloadType(Enum):
    """Types of vulnerability payloads."""
    XSS = auto()
    SQLI = auto()
    COMMAND_INJECTION = auto()
    PATH_TRAVERSAL = auto()
    SSRF = auto()
    TEMPLATE_INJECTION = auto()
    XXE = auto()
    DESERIALIZATION = auto()
    FILE_UPLOAD = auto()
    CUSTOM = auto()

@dataclass
class Payload:
    """Represents a vulnerability testing payload."""
    content: str
    type: PayloadType
    encoding: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_by: Optional[str] = None  # AI model or tool that generated this
    effectiveness_score: Optional[float] = None  # Historical success rate

    def __post_init__(self):
        """Validate the payload type."""
        if isinstance(self.type, str):
            try:
                self.type = PayloadType[self.type.upper()]
            except KeyError:
                raise ValueError(f"Invalid payload type: {self.type}")
        elif not isinstance(self.type, PayloadType):
            raise ValueError(f"Invalid payload type: {self.type}")

@dataclass
class PayloadResult:
    """Result of a payload execution."""
    payload: Payload
    success: bool
    response_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class VulnResult:
    """Represents a discovered vulnerability."""
    name: str
    type: str
    severity: VulnSeverity
    description: str
    endpoint: str
    payloads: List[PayloadResult]
    found_at: datetime = field(default_factory=datetime.utcnow)
    confirmed: bool = False
    false_positive: bool = False
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    proof_of_concept: Optional[str] = None
    remediation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert vulnerability to dictionary format."""
        return {
            "name": self.name,
            "type": self.type,
            "severity": self.severity.name,
            "description": self.description,
            "endpoint": self.endpoint,
            "payloads": [
                {
                    "content": p.payload.content,
                    "type": p.payload.type.name,
                    "success": p.success,
                    "timestamp": p.timestamp.isoformat(),
                    "error": p.error
                } for p in self.payloads
            ],
            "found_at": self.found_at.isoformat(),
            "confirmed": self.confirmed,
            "false_positive": self.false_positive,
            "cwe_id": self.cwe_id,
            "cvss_score": self.cvss_score,
            "references": self.references,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "proof_of_concept": self.proof_of_concept,
            "remediation": self.remediation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VulnResult':
        """Create vulnerability from dictionary data."""
        payloads = [
            PayloadResult(
                payload=Payload(
                    content=p["content"],
                    type=PayloadType[p["type"]]
                ),
                success=p["success"],
                response_data={},  # Historical data might not have full response
                timestamp=datetime.fromisoformat(p["timestamp"]),
                error=p.get("error")
            ) for p in data["payloads"]
        ]

        return cls(
            name=data["name"],
            type=data["type"],
            severity=VulnSeverity[data["severity"]],
            description=data["description"],
            endpoint=data["endpoint"],
            payloads=payloads,
            found_at=datetime.fromisoformat(data["found_at"]),
            confirmed=data["confirmed"],
            false_positive=data["false_positive"],
            cwe_id=data.get("cwe_id"),
            cvss_score=data.get("cvss_score"),
            references=data.get("references", []),
            metadata=data.get("metadata", {}),
            tags=set(data.get("tags", [])),
            proof_of_concept=data.get("proof_of_concept"),
            remediation=data.get("remediation")
        )

@dataclass
class ScanConfig:
    """Configuration for vulnerability scanning."""
    target: str
    payload_types: Set[PayloadType]
    max_depth: int = 3
    threads: int = 10
    timeout: int = 30
    custom_headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)
    proxy: Optional[str] = None
    verify_ssl: bool = True
    follow_redirects: bool = True
    scope_constraints: Dict[str, Any] = field(default_factory=dict)
    rate_limit: Optional[int] = None  # requests per second
    ai_assistance: bool = True  # whether to use AI for payload generation
