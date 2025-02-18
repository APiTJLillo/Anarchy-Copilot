"""Database models for proxy functionality."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Boolean, Table, Text, Float
from sqlalchemy.orm import relationship
from database import Base

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
    project = relationship("Project", back_populates="proxy_sessions")
    history_entries = relationship("ProxyHistoryEntry", back_populates="session")
    analysis_results = relationship("ProxyAnalysisResult", back_populates="session")

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
