"""Analysis-related API models."""
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from api.proxy.database_models import ProxyAnalysisResult as DBProxyAnalysisResult

class CreateAnalysisResult(BaseModel):
    """Model for creating a new analysis result."""
    analysis_type: str = Field(..., description="Type of analysis performed")
    findings: Dict[str, Any] = Field(..., description="Analysis findings")
    severity: Optional[str] = Field(None, description="Severity level if applicable")
    analysis_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional analysis metadata")
    session_id: int = Field(..., description="ID of the proxy session")
    history_entry_id: Optional[int] = Field(None, description="ID of the related history entry")

class AnalysisResultResponse(BaseModel):
    """API response model for analysis results."""
    id: int
    timestamp: datetime
    analysis_type: str
    findings: Dict[str, Any]
    severity: Optional[str]
    analysis_metadata: Optional[Dict[str, Any]]
    session_id: int
    history_entry_id: Optional[int]

    @classmethod
    def from_db(cls, result: DBProxyAnalysisResult) -> "AnalysisResultResponse":
        """Create response model from database model."""
        return cls(
            id=result.id,
            timestamp=result.timestamp,
            analysis_type=result.analysis_type,
            findings=result.findings,
            severity=result.severity,
            analysis_metadata=result.analysis_metadata,
            session_id=result.session_id,
            history_entry_id=result.history_entry_id
        )

__all__ = [
    'CreateAnalysisResult',
    'AnalysisResultResponse'
]
