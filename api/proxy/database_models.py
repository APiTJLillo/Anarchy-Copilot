"""Database models for proxy functionality."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Boolean, Table, Text, Float
from sqlalchemy.orm import relationship
from models.base import Base, Project

class ProxySession(Base):
    """Model for tracking proxy sessions."""
    __tablename__ = "proxy_sessions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Foreign Keys
    project_id = Column(Integer, ForeignKey("projects.id"))
    created_by = Column(Integer, ForeignKey("users.id"))
    
    # Configuration
    settings = Column(JSON, default={})  # Stores ProxySettings as JSON
    
    # Relationships
    # Note: These are reversed relationships - we define them here since the base models can't see this module
    project = relationship(Project, backref="proxy_sessions")
    creator = relationship("models.base.User", backref="proxy_sessions", foreign_keys=[created_by])
    history_entries = relationship("ProxyHistoryEntry", back_populates="session")
    analysis_results = relationship("ProxyAnalysisResult", back_populates="session")
    interception_rules = relationship("InterceptionRule", back_populates="session", cascade="all, delete-orphan")

class InterceptionRule(Base):
    """Model for storing interception rules."""
    __tablename__ = "interception_rules"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    enabled = Column(Boolean, default=True)
    session_id = Column(Integer, ForeignKey("proxy_sessions.id"), nullable=False)
    conditions = Column(JSON, nullable=False)  # List of {field, operator, value, use_regex}
    action = Column(String, nullable=False)  # FORWARD, BLOCK, MODIFY
    modification = Column(JSON, nullable=True)  # Headers/body modifications
    priority = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default="CURRENT_TIMESTAMP")
    updated_at = Column(DateTime, server_default="CURRENT_TIMESTAMP")

    # Relationships
    session = relationship("ProxySession", back_populates="interception_rules")

class ProxyHistoryEntry(Base):
    """Model for storing proxy request/response history."""
    __tablename__ = "proxy_history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    method = Column(String)
    url = Column(String)
    request_headers = Column(JSON)
    request_body = Column(Text, nullable=True)
    response_status = Column(Integer, nullable=True)
    response_headers = Column(JSON, nullable=True)
    response_body = Column(Text, nullable=True)
    duration = Column(Float, nullable=True)  # Request duration in seconds
    tags = Column(JSON, default=list)  # List of tags
    notes = Column(Text, nullable=True)
    is_intercepted = Column(Boolean, default=False)
    applied_rules = Column(JSON, nullable=True)  # List of {rule_id, action, modifications}
    
    # TLS Information
    tls_version = Column(String, nullable=True)  # TLS version (e.g., "TLSv1.3")
    cipher_suite = Column(String, nullable=True)  # Cipher suite used
    certificate_info = Column(JSON, nullable=True)  # Certificate details
    
    # Foreign Keys
    session_id = Column(Integer, ForeignKey("proxy_sessions.id"))
    
    # Relationships
    session = relationship("ProxySession", back_populates="history_entries")
    analysis_results = relationship("ProxyAnalysisResult", back_populates="history_entry")

class ProxyAnalysisResult(Base):
    """Model for storing proxy traffic analysis results."""
    __tablename__ = "proxy_analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    analysis_type = Column(String, index=True)  # Type of analysis performed
    findings = Column(JSON)  # Analysis findings
    severity = Column(String, nullable=True)  # Severity level if applicable
    analysis_metadata = Column(JSON, nullable=True)  # Additional analysis metadata
    
    # Foreign Keys
    session_id = Column(Integer, ForeignKey("proxy_sessions.id"))
    history_entry_id = Column(Integer, ForeignKey("proxy_history.id", ondelete="SET NULL"), nullable=True)
    
    # Relationships
    session = relationship("ProxySession", back_populates="analysis_results")
    history_entry = relationship("ProxyHistoryEntry", back_populates="analysis_results")
