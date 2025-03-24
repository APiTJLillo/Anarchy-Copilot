"""Core models for the application."""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Boolean, Table, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database.base import Base
from models.base import project_collaborators
from models.recon import ReconModule, ReconTarget, ReconResult
from api.proxy.database_models import ProxySession
from models.vulnerability import Vulnerability, VulnerabilityResult

class User(Base):
    """User model."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    projects = relationship("Project", back_populates="owner")

class Project(Base):
    """Project model."""
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    scope = Column(JSON, default=dict)
    is_archived = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    owner = relationship("User", back_populates="projects")
    collaborators = relationship("User", secondary=project_collaborators, lazy="selectin")
    proxy_sessions = relationship("ProxySession", back_populates="project")
    recon_modules = relationship("ReconModule", back_populates="project")
    recon_targets = relationship("ReconTarget", back_populates="project")
    recon_results = relationship("ReconResult", back_populates="project")
    vulnerabilities = relationship("Vulnerability", back_populates="project")
    vulnerability_results = relationship("VulnerabilityResult", back_populates="project") 