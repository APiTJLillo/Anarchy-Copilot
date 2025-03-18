"""Database models for proxy history entries and sessions."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text, Boolean, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import text
from database.base import Base

class ProxySession(Base):
    """Model for storing proxy sessions."""
    __tablename__ = "proxy_sessions"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    settings = Column(JSON, nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=text('CURRENT_TIMESTAMP'), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=text('CURRENT_TIMESTAMP'), nullable=False)

    project = relationship("Project", back_populates="proxy_sessions")
    creator = relationship("User", backref="proxy_sessions", foreign_keys=[created_by])
    history_entries = relationship("ProxyHistoryEntry", back_populates="session", cascade="all, delete-orphan")
    modified_requests = relationship("ModifiedRequest", back_populates="session", cascade="all, delete-orphan")
    interception_rules = relationship("InterceptionRule", back_populates="session", cascade="all, delete-orphan")
    tunnel_metrics = relationship("TunnelMetrics", back_populates="session", cascade="all, delete-orphan")
    analysis_results = relationship("ProxyAnalysisResult", back_populates="session", cascade="all, delete-orphan")

class ProxyHistoryEntry(Base):
    """Model for storing proxy request/response history."""
    __tablename__ = "proxy_history"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=text('CURRENT_TIMESTAMP'), nullable=False)
    session_id = Column(Integer, ForeignKey("proxy_sessions.id"), nullable=False)
    method = Column(String, nullable=False)
    url = Column(String, nullable=False)
    host = Column(String, nullable=True)
    path = Column(String, nullable=True)
    status_code = Column(Integer, nullable=True)
    tls_version = Column(String, nullable=True)
    cipher_suite = Column(String, nullable=True)
    certificate_info = Column(JSON, nullable=True)
    
    # Request data
    request_headers = Column(JSON, nullable=True)
    request_body = Column(Text, nullable=True)  # Raw request body (base64 if binary)
    decrypted_request = Column(Text, nullable=True)  # Decrypted request body
    
    # Response data
    response_headers = Column(JSON, nullable=True)
    response_body = Column(Text, nullable=True)  # Raw response body (base64 if binary)
    decrypted_response = Column(Text, nullable=True)  # Decrypted response body
    
    # Metadata
    tags = Column(JSON, nullable=True)  # List of tags
    is_intercepted = Column(Boolean, default=True)
    is_encrypted = Column(Boolean, default=False)
    duration = Column(Float, nullable=True)  # Request duration in seconds
    notes = Column(Text, nullable=True)

    session = relationship("ProxySession", back_populates="history_entries")
    modified_requests = relationship("ModifiedRequest", back_populates="original_request")
    analysis_results = relationship("ProxyAnalysisResult", back_populates="history_entry")
