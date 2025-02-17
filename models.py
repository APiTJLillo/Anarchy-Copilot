from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Enum, Text, Boolean, Table, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

# Association tables for many-to-many relationships
vulnerability_tags = Table('vulnerability_tags',
    Base.metadata,
    Column('vulnerability_id', Integer, ForeignKey('vulnerabilities.id')),
    Column('tag_id', Integer, ForeignKey('tags.id'))
)

project_collaborators = Table('project_collaborators',
    Base.metadata,
    Column('project_id', Integer, ForeignKey('projects.id')),
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('role', String)  # Role within the project (e.g., admin, member, viewer)
)

class SeverityLevel(enum.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class VulnerabilityStatus(enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    FIXED = "fixed"
    WONT_FIX = "wont_fix"
    FALSE_POSITIVE = "false_positive"

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
    assigned_vulnerabilities = relationship("Vulnerability", back_populates="assigned_user")
    authored_reports = relationship("Report", back_populates="author")
    projects = relationship("Project", secondary=project_collaborators, back_populates="collaborators")

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

class Vulnerability(Base):
    __tablename__ = "vulnerabilities"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    severity = Column(Enum(SeverityLevel))
    status = Column(Enum(VulnerabilityStatus), default=VulnerabilityStatus.OPEN)
    cvss_score = Column(Float, nullable=True)
    proof_of_concept = Column(Text)
    steps_to_reproduce = Column(Text)
    technical_details = Column(Text)
    recommendation = Column(Text)
    discovered_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign keys
    project_id = Column(Integer, ForeignKey("projects.id"))
    assigned_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    recon_result_id = Column(Integer, ForeignKey("recon_results.id"), nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="vulnerabilities")
    assigned_user = relationship("User", back_populates="assigned_vulnerabilities")
    recon_result = relationship("ReconResult", back_populates="vulnerabilities")
    tags = relationship("Tag", secondary=vulnerability_tags, back_populates="vulnerabilities")
    comments = relationship("VulnerabilityComment", back_populates="vulnerability")
    reports = relationship("Report", secondary="report_vulnerabilities", back_populates="vulnerabilities")

class VulnerabilityComment(Base):
    __tablename__ = "vulnerability_comments"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign keys
    vulnerability_id = Column(Integer, ForeignKey("vulnerabilities.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    vulnerability = relationship("Vulnerability", back_populates="comments")
    user = relationship("User")

class Tag(Base):
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)
    
    # Relationships
    vulnerabilities = relationship("Vulnerability", secondary=vulnerability_tags, back_populates="tags")

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    executive_summary = Column(Text)
    methodology = Column(Text)
    findings_summary = Column(Text)
    recommendations = Column(Text)
    conclusion = Column(Text)
    report_type = Column(String)  # e.g., "Technical", "Executive", "Full"
    status = Column(String)  # e.g., "Draft", "In Review", "Final"
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign keys
    project_id = Column(Integer, ForeignKey("projects.id"))
    author_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    project = relationship("Project", back_populates="reports")
    author = relationship("User", back_populates="authored_reports")
    vulnerabilities = relationship("Vulnerability", secondary="report_vulnerabilities", back_populates="reports")

# Association table for reports and vulnerabilities
report_vulnerabilities = Table('report_vulnerabilities',
    Base.metadata,
    Column('report_id', Integer, ForeignKey('reports.id')),
    Column('vulnerability_id', Integer, ForeignKey('vulnerabilities.id'))
)
