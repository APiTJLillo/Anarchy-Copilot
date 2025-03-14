"""Database models for proxy functionality."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Boolean, Table, Text, Float, text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.base import Base

class ProxySession(Base):
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
    analysis_results = relationship("ProxyAnalysisResult", back_populates="session", cascade="all, delete-orphan")
    interception_rules = relationship("InterceptionRule", back_populates="session", cascade="all, delete-orphan")
    tunnel_metrics = relationship("TunnelMetrics", back_populates="session", cascade="all, delete-orphan")

class InterceptionRule(Base):
    """Model for storing interception rules."""
    __tablename__ = "interception_rules"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    enabled = Column(Boolean, default=True)
    session_id = Column(Integer, ForeignKey("proxy_sessions.id"), nullable=False)
    conditions = Column(JSON, nullable=False)  # List of {field, operator, value, use_regex}
    action = Column(String, nullable=False)  # FORWARD, BLOCK, MODIFY
    modification = Column(JSON, nullable=True)  # Headers/body modifications
    priority = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=False)

    session = relationship("ProxySession", back_populates="interception_rules")

class ProxyHistoryEntry(Base):
    """Model for storing proxy request/response history."""
    __tablename__ = "proxy_history"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
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
    tags = Column(JSON, nullable=True)  # List of tags (e.g., ["raw", "encrypted", "request", "response"])
    is_intercepted = Column(Boolean, default=True)
    is_encrypted = Column(Boolean, default=False)

    session = relationship("ProxySession", back_populates="history_entries")

class TunnelMetrics(Base):
    """Model for storing HTTPS tunnel metrics and data flow."""
    __tablename__ = "tunnel_metrics"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    session_id = Column(Integer, ForeignKey("proxy_sessions.id"), nullable=False)
    connection_id = Column(String, nullable=False)
    bytes_sent = Column(Integer, nullable=False, default=0)
    bytes_received = Column(Integer, nullable=False, default=0)
    latency = Column(Float, nullable=True)

    session = relationship("ProxySession", back_populates="tunnel_metrics")

class ProxyAnalysisResult(Base):
    """Model for storing proxy traffic analysis results."""
    __tablename__ = "proxy_analysis_results"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    session_id = Column(Integer, ForeignKey("proxy_sessions.id"), nullable=False)
    analysis_type = Column(String, nullable=False)
    findings = Column(JSON, nullable=True)

    session = relationship("ProxySession", back_populates="analysis_results")
