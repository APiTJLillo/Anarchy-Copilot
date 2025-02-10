from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    projects = relationship("Project", back_populates="owner")

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    owner = relationship("User", back_populates="projects")
    recon_results = relationship("ReconResult", back_populates="project")

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
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    project = relationship("Project", back_populates="recon_results")

    def to_dict(self):
        """Convert the ReconResult object to a dictionary."""
        return {
            "id": self.id,
            "tool": self.tool,
            "domain": self.domain,
            "results": self.results,
            "status": self.status,
            "error_message": self.error_message,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "project_id": self.project_id
        }
