"""Recon-related models."""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from database import Base
from .base import Project

class ReconResult(Base):
    __tablename__ = "recon_results"

    id = Column(Integer, primary_key=True, index=True)
    tool = Column(String, index=True)  # Name of the tool used (amass, subfinder, etc)
    domain = Column(String, index=True)  # Target domain
    results = Column(JSON)  # Structured results from the tool
    status = Column(String)  # Status of the scan (completed, failed, etc)
    error_message = Column(String, nullable=True)  # Error message if scan failed
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    scan_type = Column(String, index=True)  # Type of scan (subdomain_scan, port_scan, etc)
    scan_metadata = Column(JSON, nullable=True)  # Additional metadata and change tracking
    
    # Relationships
    project = relationship("Project", back_populates="recon_results")
    vulnerabilities = relationship("Vulnerability", back_populates="recon_result")

    def to_dict(self):
        return {
            "id": self.id,
            "tool": self.tool,
            "domain": self.domain,
            "results": self.results,
            "status": self.status,
            "error_message": self.error_message,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "project_id": self.project_id,
            "scan_metadata": self.scan_metadata
        }
