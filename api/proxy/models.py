"""API models for proxy management."""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator, ConfigDict

from proxy.session import HistoryEntry

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
    id: str
    timestamp: str
    request: Dict[str, Any]
    response: Optional[Dict[str, Any]]
    duration: Optional[float]
    tags: List[str]
    notes: Optional[str]

    @classmethod
    def from_entry(cls, entry: HistoryEntry) -> "HistoryEntryResponse":
        """Create response model from history entry."""
        return cls(
            id=entry.id,
            timestamp=entry.timestamp.isoformat(),
            request=entry.request.to_dict(),
            response=entry.response.to_dict() if entry.response else None,
            duration=entry.duration,
            tags=list(entry.tags),
            notes=entry.notes
        )

__all__ = [
    'Header',
    'InterceptedRequest', 
    'InterceptedResponse',
    'ProxySettings',
    'TagData',
    'NoteData',
    'HistoryEntryResponse'
]
