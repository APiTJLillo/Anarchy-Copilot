"""Type definitions for proxy functionality."""
from datetime import datetime
from typing import Optional, Dict, Any, TypedDict

class SessionSettings(TypedDict, total=False):
    """Settings for proxy session."""
    intercept_requests: bool
    intercept_responses: bool
    ca_cert_path: str
    ca_key_path: str

class ProxySessionData:
    """Data class for proxy session information."""
    def __init__(self, id: int, settings: Dict[str, Any], is_active: bool = True):
        self.id = id
        self.settings = settings
        self.is_active = is_active
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    @classmethod
    def from_db(cls, db_session: Any) -> 'ProxySessionData':
        """Create from database model."""
        return cls(
            id=db_session.id,
            settings=db_session.settings or {},
            is_active=db_session.is_active
        )
