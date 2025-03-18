"""Database model for storing modified and resent proxy requests."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text, text
from sqlalchemy.orm import relationship
from database.base import Base

class ModifiedRequest(Base):
    """Model for storing modified and resent proxy requests."""
    __tablename__ = "modified_requests"

    id = Column(Integer, primary_key=True)
    original_request_id = Column(Integer, ForeignKey("proxy_history.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("proxy_sessions.id"), nullable=False)
    
    # Modified request data
    method = Column(String, nullable=False)
    url = Column(String, nullable=False)
    request_headers = Column(JSON, nullable=True)
    request_body = Column(Text, nullable=True)
    
    # Response data from resent request
    response_status = Column(Integer, nullable=True)
    response_headers = Column(JSON, nullable=True)
    response_body = Column(Text, nullable=True)
    
    # Tracking data
    modified_fields = Column(JSON, nullable=False)  # List of fields that were modified
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=text('CURRENT_TIMESTAMP'), nullable=False)
    sent_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    original_request = relationship("ProxyHistoryEntry", foreign_keys=[original_request_id])
    session = relationship("ProxySession", foreign_keys=[session_id])
    creator = relationship("User", foreign_keys=[created_by])

    def __repr__(self):
        return f"<ModifiedRequest(id={self.id}, original_id={self.original_request_id})>"
