"""Project management API models."""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

class CreateProject(BaseModel):
    """Model for creating a new project."""
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    scope: Optional[Dict[str, Any]] = Field(None, description="Project scope configuration")
    owner_id: int = Field(..., description="ID of the project owner")

class UpdateProject(BaseModel):
    """Model for updating a project."""
    name: Optional[str] = Field(None, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    scope: Optional[Dict[str, Any]] = Field(None, description="Project scope configuration")
    is_archived: Optional[bool] = Field(None, description="Whether the project is archived")

class ProjectResponse(BaseModel):
    """API response model for projects."""
    id: int
    name: str
    description: Optional[str]
    scope: Dict[str, Any]
    owner_id: int
    created_at: datetime
    updated_at: datetime
    is_archived: bool

    class Config:
        """Pydantic config."""
        from_attributes = True

class ProjectCollaborator(BaseModel):
    """Model for project collaborator data."""
    user_id: int = Field(..., description="ID of the collaborator")
    role: str = Field(..., description="Role in the project (admin, member, viewer)")

class ProjectCollaboratorResponse(BaseModel):
    """API response model for project collaborators."""
    user_id: int
    username: str
    email: str
    role: str

__all__ = [
    'CreateProject',
    'UpdateProject',
    'ProjectResponse',
    'ProjectCollaborator',
    'ProjectCollaboratorResponse'
]
