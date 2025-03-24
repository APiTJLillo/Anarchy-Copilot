"""API models for proxy functionality."""

__all__ = [
    "ConditionBase",
    "ModificationBase",
    "InterceptionRuleBase",
    "InterceptionRuleCreate",
    "InterceptionRuleUpdate",
    "InterceptionRule",
    "ProxyHistoryBase",
    "ProxyHistory",
    "ProxyAnalysisBase",
    "ProxyAnalysis",
    "ProxySessionBase",
    "CreateProxySession",
    "ProxySession",
    "ProxySettings",
    "ConnectionInfo",
    "ConnectionEventInfo",
]
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class ConditionBase(BaseModel):
    """Base model for an interception condition."""
    field: str
    operator: str
    value: str
    use_regex: bool = False

class ModificationBase(BaseModel):
    """Base model for request/response modifications."""
    headers: Optional[Dict[str, str]] = None
    body: Optional[str] = None
    status_code: Optional[int] = None

class InterceptionRuleBase(BaseModel):
    """Base model for interception rules."""
    name: str
    enabled: bool = True
    conditions: List[ConditionBase]
    action: str
    modification: Optional[ModificationBase] = None

class InterceptionRuleCreate(InterceptionRuleBase):
    """Model for creating a new interception rule."""
    pass

class InterceptionRuleUpdate(BaseModel):
    """Model for updating an existing interception rule."""
    name: Optional[str] = None
    enabled: Optional[bool] = None
    conditions: Optional[List[ConditionBase]] = None
    action: Optional[str] = None
    modification: Optional[ModificationBase] = None

class InterceptionRule(InterceptionRuleBase):
    """Model for a complete interception rule."""
    id: int
    session_id: int
    priority: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ProxyHistoryBase(BaseModel):
    """Base model for proxy history entries."""
    method: str
    url: str
    request_headers: Dict[str, str]
    request_body: Optional[str] = None
    response_status: Optional[int] = None
    response_headers: Optional[Dict[str, str]] = None
    response_body: Optional[str] = None
    duration: Optional[float] = None
    is_intercepted: bool = False
    applied_rules: Optional[List[Dict[str, Any]]] = None
    tags: List[str] = []
    notes: Optional[str] = None

class ProxyHistory(ProxyHistoryBase):
    """Model for a complete history entry."""
    id: int
    session_id: int
    timestamp: datetime

    class Config:
        from_attributes = True

class ProxyAnalysisBase(BaseModel):
    """Base model for proxy analysis results."""
    analysis_type: str
    findings: Dict[str, Any]
    severity: Optional[str] = None
    analysis_metadata: Optional[Dict[str, Any]] = None

class ProxyAnalysis(ProxyAnalysisBase):
    """Model for a complete analysis result."""
    id: int
    session_id: int
    history_entry_id: Optional[int]
    timestamp: datetime

    class Config:
        from_attributes = True

class ProxySessionBase(BaseModel):
    """Base model for proxy sessions."""
    name: str
    description: Optional[str] = None
    settings: Dict[str, Any]

class CreateProxySession(BaseModel):
    """Model for creating a new proxy session."""
    name: str
    project_id: int
    user_id: int
    settings: Dict[str, Any]

class ProxySession(ProxySessionBase):
    """Model for a complete proxy session."""
    id: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    is_active: bool
    project_id: Optional[int]
    created_by: Optional[int]

    class Config:
        from_attributes = True

class ConnectionEventInfo(BaseModel):
    """Model for connection event information."""
    type: str
    direction: str
    timestamp: float
    status: str
    bytes_transferred: Optional[int] = None

class ConnectionInfo(BaseModel):
    """Model for active connection information."""
    id: str
    host: str
    port: int
    start_time: float
    end_time: Optional[float] = None
    status: str
    events: List[ConnectionEventInfo]
    bytes_received: int
    bytes_sent: int
    requests_processed: int
    error: Optional[str] = None

class ProxySettings(BaseModel):
    """Model for proxy configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8083
    allowed_hosts: Optional[List[str]] = None
    excluded_hosts: Optional[List[str]] = None
    intercept_requests: bool = True
    intercept_responses: bool = True
    websocket_support: bool = False
    history_size: int = 1000
    keepalive_timeout: int = 5
    ca_cert_path: Optional[str] = None
    ca_key_path: Optional[str] = None
