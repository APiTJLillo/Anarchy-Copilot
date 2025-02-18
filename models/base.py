"""Core database models."""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text, Boolean, Table
from sqlalchemy.orm import relationship
from datetime import datetime

from database import Base

# Association table for project collaborators
project_collaborators = Table(
    'project_collaborators',
    Base.metadata,
    Column('project_id', Integer, ForeignKey('projects.id')),
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('role', String)  # Role within the project (e.g., admin, member, viewer)
)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    owned_projects = relationship("Project", back_populates="owner")
    projects = relationship("Project", secondary=project_collaborators, back_populates="collaborators")
    assigned_vulnerabilities = relationship("Vulnerability", back_populates="assigned_user")
    authored_reports = relationship("Report", back_populates="author")

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    scope = Column(JSON)  # List of in-scope domains/IPs/etc
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_archived = Column(Boolean, default=False)
    
    # Relationships
    owner = relationship("User", back_populates="owned_projects")
    collaborators = relationship("User", secondary=project_collaborators, back_populates="projects")
    recon_results = relationship("ReconResult", back_populates="project")
    vulnerabilities = relationship("Vulnerability", back_populates="project")
    reports = relationship("Report", back_populates="project")
    proxy_sessions = relationship("ProxySession", back_populates="project")
