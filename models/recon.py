"""Recon models for the application."""
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from models.base import Base


class ReconModule(Base):
    """Recon module model."""

    __tablename__ = "recon_modules"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    name = Column(String, nullable=False)
    description = Column(Text)
    settings = Column(JSON, default=dict)
    is_active = Column(String, default="true")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="recon_modules")
    targets = relationship("ReconTarget", back_populates="module")
    results = relationship("ReconResult", back_populates="module")


class ReconTarget(Base):
    """Recon target model."""

    __tablename__ = "recon_targets"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    module_id = Column(Integer, ForeignKey("recon_modules.id"))
    target = Column(String, nullable=False)
    result_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="recon_targets")
    module = relationship("ReconModule", back_populates="targets")
    results = relationship("ReconResult", back_populates="target")


class ReconResult(Base):
    """Recon result model."""

    __tablename__ = "recon_results"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    module_id = Column(Integer, ForeignKey("recon_modules.id"))
    target_id = Column(Integer, ForeignKey("recon_targets.id"))
    result_data = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="recon_results")
    module = relationship("ReconModule", back_populates="results")
    target = relationship("ReconTarget", back_populates="results")
