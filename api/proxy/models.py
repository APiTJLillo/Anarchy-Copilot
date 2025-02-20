"""API models for proxy management."""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator, ConfigDict

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.proxy.database_models import ProxySession as DBProxySession, ProxyHistoryEntry

class CreateProxySession(BaseModel):
    """Model for creating a new proxy session."""
    name: str = Field(..., description="Session name")
    description: Optional[str] = Field(None, description="Session description")
    project_id: int = Field(..., description="ID of the project this session belongs to")
    created_by: int = Field(..., description="ID of the user creating this session")
    settings: Optional[Dict[str, Any]] = Field(None, description="Proxy settings for this session")

class ProxySessionResponse(BaseModel):
    """API response model for proxy sessions."""
    id: int
    name: str
    description: Optional[str]
    project_id: int
    start_time: datetime
    end_time: Optional[datetime]
    is_active: bool
    settings: Optional[Dict[str, Any]]
    created_by: int

    @classmethod
    def from_db(cls, session: 'DBProxySession') -> "ProxySessionResponse":
        """Create response model from database model."""
        return cls(
            id=session.id,
            name=session.name,
            description=session.description,
            project_id=session.project_id,
            start_time=session.start_time,
            end_time=session.end_time,
            is_active=session.is_active,
            settings=session.settings,
            created_by=session.created_by
        )

class Header(BaseModel):
    """HTTP header model."""
    name: str
    value: str

class InterceptedRequest(BaseModel):
    """Model for an intercepted HTTP request."""
    id: str
    method: str
    url: str
    headers: List[Header]
    body: Optional[str] = None

class InterceptedResponse(BaseModel):
    """Model for an intercepted HTTP response."""
    statusCode: int
    headers: List[Header]
    body: Optional[str] = None

class ProxySettings(BaseModel):
    """Proxy server configuration settings."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    host: str = Field(
        ..., 
        description="Proxy host address",
        examples=["127.0.0.1"]
    )
    port: int = Field(
        ...,
        gt=0,
        lt=65536,
        description="Proxy port",
        examples=[8080]
    )
    interceptRequests: bool = True
    interceptResponses: bool = True
    allowedHosts: List[str] = []
    excludedHosts: List[str] = []
    maxConnections: int = 100
    maxKeepaliveConnections: int = 20
    keepaliveTimeout: int = 30

    @validator("host")
    def validate_host(cls, v: str) -> str:
        """Validate host address."""
        v = v.strip()
        if len(v) < 1:
            raise ValueError("min_length validation failed")
        return v

class TagData(BaseModel):
    """Tag request data."""
    tag: str = Field(..., description="Tag to add")

class NoteData(BaseModel):
    """Note request data."""
    note: str = Field(..., description="Note to set")

class HistoryEntryResponse(BaseModel):
    """API response model for history entries."""
    id: int
    timestamp: datetime
    method: str
    url: str
    request_headers: Dict[str, Any]
    request_body: Optional[str]
    response_status: Optional[int]
    response_headers: Optional[Dict[str, Any]]
    response_body: Optional[str]
    duration: Optional[float]
    tags: List[str]
    notes: Optional[str]
    is_intercepted: bool
    session_id: int

    @classmethod
    def from_entry(cls, entry: 'ProxyHistoryEntry') -> 'HistoryEntryResponse':
        """Create response model from database history entry."""
        return cls(
            id=entry.id,
            timestamp=entry.timestamp,
            method=entry.method,
            url=entry.url,
            request_headers=entry.request_headers,
            request_body=entry.request_body,
            response_status=entry.response_status,
            response_headers=entry.response_headers,
            response_body=entry.response_body,
            duration=entry.duration,
            tags=entry.tags or [],
            notes=entry.notes,
            is_intercepted=entry.is_intercepted,
            session_id=entry.session_id
        )

__all__ = [
    'CreateProxySession',
    'ProxySessionResponse',
    'Header',
    'InterceptedRequest', 
    'InterceptedResponse',
    'ProxySettings',
    'TagData',
    'NoteData',
    'HistoryEntryResponse'
]
