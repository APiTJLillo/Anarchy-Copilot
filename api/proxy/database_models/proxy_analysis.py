"""Database model for proxy analysis results."""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import text
from database.base import Base

class ProxyAnalysisResult(Base):
    """Model for storing proxy traffic analysis results."""
    __tablename__ = "proxy_analysis_results"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))
    analysis_type = Column(String)  # Type of analysis performed
    findings = Column(JSON)  # Analysis results/findings
    severity = Column(String)  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    analysis_metadata = Column(JSON)  # Additional metadata about the analysis
    session_id = Column(Integer, ForeignKey("proxy_sessions.id"))
    history_entry_id = Column(Integer, ForeignKey("proxy_history.id", ondelete='SET NULL'))

    session = relationship("ProxySession", back_populates="analysis_results")
    history_entry = relationship("ProxyHistoryEntry", back_populates="analysis_results")

    def __repr__(self):
        return f"<ProxyAnalysisResult(id={self.id}, type='{self.analysis_type}', severity='{self.severity}')>"
