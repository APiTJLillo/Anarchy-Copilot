"""Recon-related models."""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime

from database import Base
from .base import Project

class ReconModule(Base):
    __tablename__ = "recon_modules"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)  # Module name (e.g. amass, subfinder, etc)
    description = Column(String, nullable=True)  # Module description
    is_enabled = Column(Boolean, default=True)  # Whether this module is enabled
    project_id = Column(Integer, ForeignKey("projects.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_run = Column(DateTime, nullable=True)  # Last time this module was run
    run_frequency = Column(String, nullable=True)  # How often to run this module
    config = Column(JSON, nullable=True)  # Module-specific configuration
    
    # Relationships
    project = relationship("Project", back_populates="recon_modules")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "is_enabled": self.is_enabled,
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_frequency": self.run_frequency,
            "config": self.config
        }

class ReconTarget(Base):
    __tablename__ = "recon_targets"

    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String, index=True)  # Target domain/hostname
    description = Column(String, nullable=True)  # Optional description of the target
    is_active = Column(Boolean, default=True)  # Whether this target is currently being scanned
    project_id = Column(Integer, ForeignKey("projects.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_scanned = Column(DateTime, nullable=True)  # Last time this target was scanned
    scan_frequency = Column(String, nullable=True)  # How often to scan this target (e.g. "daily", "weekly")
    target_metadata = Column(JSON, nullable=True)  # Additional target metadata
    
    # Relationships
    project = relationship("Project", back_populates="recon_targets")
    scan_results = relationship("ReconResult", back_populates="target")

    def to_dict(self):
        return {
            "id": self.id,
            "domain": self.domain,
            "description": self.description,
            "is_active": self.is_active,
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_scanned": self.last_scanned.isoformat() if self.last_scanned else None,
            "scan_frequency": self.scan_frequency,
            "target_metadata": self.target_metadata
        }

class ReconResult(Base):
    __tablename__ = "recon_results"

    id = Column(Integer, primary_key=True, index=True)
    tool = Column(String, index=True)  # Name of the tool used (amass, subfinder, etc)
    target_id = Column(Integer, ForeignKey("recon_targets.id"))
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
    target = relationship("ReconTarget", back_populates="scan_results")
    vulnerabilities = relationship("Vulnerability", back_populates="recon_result")

    def to_dict(self):
        return {
            "id": self.id,
            "tool": self.tool,
            "target_id": self.target_id,
            "results": self.results,
            "status": self.status,
            "error_message": self.error_message,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "project_id": self.project_id,
            "scan_metadata": self.scan_metadata
        }
