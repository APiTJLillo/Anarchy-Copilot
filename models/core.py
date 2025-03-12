"""Core SQLAlchemy models."""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Boolean, Text, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database.base import Base
from models.base import project_collaborators
from models.recon import ReconModule, ReconTarget, ReconResult
from api.proxy.database_models import ProxySession
from models.vulnerability import Vulnerability, VulnerabilityResult

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=True)
    email = Column(String, unique=True, nullable=True)
    hashed_password = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    owned_projects = relationship("Project", back_populates="owner", foreign_keys="Project.owner_id")
    collaborating_projects = relationship(
        "Project",
        secondary=project_collaborators,
        back_populates="collaborators"
    )

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    scope = Column(JSON, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    is_archived = Column(Boolean, default=False)

    # Relationships
    owner = relationship("User", back_populates="owned_projects", foreign_keys=[owner_id])
    collaborators = relationship(
        "User",
        secondary=project_collaborators,
        back_populates="collaborating_projects"
    )
    proxy_sessions = relationship("ProxySession", back_populates="project")
    recon_modules = relationship("ReconModule", back_populates="project")
    recon_targets = relationship("ReconTarget", back_populates="project")
    recon_results = relationship("ReconResult", back_populates="project")
    vulnerabilities = relationship("Vulnerability", back_populates="project")
    vulnerability_results = relationship("VulnerabilityResult", back_populates="project") 