from typing import Any, Dict, List, Optional
from datetime import datetime
from datetime import datetime
from sqlalchemy.ext.declarative import DeclarativeMeta

Base: DeclarativeMeta

class User(Base):
    id: int
    username: str
    email: str
    hashed_password: str
    projects: List["Project"]

class Project(Base):
    id: int
    name: str
    owner_id: int
    created_at: datetime
    owner: User
    recon_results: List["ReconResult"]

class ReconResult(Base):
    id: int
    tool: str
    domain: str
    results: Dict[str, Any]
    status: str
    error_message: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    project_id: Optional[int]
    project: Project

    def to_dict(self) -> Dict[str, Any]: ...
