"""Project management API endpoints."""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete
from sqlalchemy.orm import selectinload
from fastapi.responses import JSONResponse

from database.session import get_async_session
from . import models
from models.core import Project, User
from . import router
from .models import (
    CreateProject,
    UpdateProject,
    ProjectResponse,
    ProjectCollaborator,
    ProjectCollaboratorResponse
)

logger = logging.getLogger(__name__)

@router.post("", response_model=ProjectResponse)
async def create_project(
    data: CreateProject,
    db: AsyncSession = Depends(get_async_session)
) -> ProjectResponse:
    """Create a new project."""
    # Verify owner exists
    result = await db.execute(
        select(User).where(User.id == data.owner_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="Owner not found")

    # Create project
    project = Project(
        name=data.name,
        description=data.description,
        scope=data.scope or {},
        owner_id=data.owner_id
    )
    
    db.add(project)
    await db.commit()
    await db.refresh(project)
    
    return project

@router.get("", response_model=List[ProjectResponse])
async def list_projects(
    db: AsyncSession = Depends(get_async_session)
) -> List[ProjectResponse]:
    """List all projects."""
    try:
        result = await db.execute(
            select(Project)
            .order_by(Project.created_at.desc())
        )
        projects = result.scalars().all()
        return list(projects)
    except Exception as e:
        logger.error(f"Error fetching projects: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    db: AsyncSession = Depends(get_async_session)
) -> ProjectResponse:
    """Get a project by ID."""
    try:
        logger.debug(f"Fetching project with ID: {project_id}")
        result = await db.execute(
            select(Project)
            .where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()
        if not project:
            logger.debug(f"Project {project_id} not found")
            raise HTTPException(status_code=404, detail="Project not found")
        
        logger.debug(f"Successfully fetched project {project_id}")
        # Convert project to response model
        response = ProjectResponse(
            id=project.id,
            name=project.name,
            description=project.description,
            owner_id=project.owner_id,
            scope=project.scope or {},
            is_archived=project.is_archived,
            created_at=project.created_at,
            updated_at=project.updated_at
        )
        logger.debug(f"Converted project {project_id} to response model: {response}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching project {project_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch project: {str(e)}")

@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    data: UpdateProject,
    db: AsyncSession = Depends(get_async_session)
) -> ProjectResponse:
    """Update a project."""
    # Get project
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Update fields
    update_data = data.dict(exclude_unset=True)
    if update_data:
        for key, value in update_data.items():
            setattr(project, key, value)
        project.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(project)
    
    return project

@router.delete("/{project_id}")
async def delete_project(
    project_id: int,
    db: AsyncSession = Depends(get_async_session)
) -> Dict[str, str]:
    """Delete a project."""
    # Check project exists
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Delete project
    await db.execute(
        delete(Project).where(Project.id == project_id)
    )
    await db.commit()
    
    return {"message": f"Project {project_id} deleted successfully"}

@router.post("/{project_id}/collaborators", response_model=ProjectCollaboratorResponse)
async def add_collaborator(
    project_id: int,
    data: ProjectCollaborator,
    db: AsyncSession = Depends(get_async_session)
) -> ProjectCollaboratorResponse:
    """Add a collaborator to a project."""
    # Get project and user
    project_result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = project_result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    user_result = await db.execute(
        select(User).where(User.id == data.user_id)
    )
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Add collaborator
    project.collaborators.append(user)
    await db.commit()
    
    return ProjectCollaboratorResponse(
        user_id=user.id,
        username=user.username,
        email=user.email,
        role=data.role
    )

@router.delete("/{project_id}/collaborators/{user_id}")
async def remove_collaborator(
    project_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_async_session)
) -> Dict[str, str]:
    """Remove a collaborator from a project."""
    # Get project
    result = await db.execute(
        select(Project)
        .options(selectinload(Project.collaborators))
        .where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Remove collaborator
    project.collaborators = [c for c in project.collaborators if c.id != user_id]
    await db.commit()
    
    return {"message": f"User {user_id} removed from project {project_id}"}

@router.get("/{project_id}/collaborators", response_model=List[ProjectCollaboratorResponse])
async def list_collaborators(
    project_id: int,
    db: AsyncSession = Depends(get_async_session)
) -> List[ProjectCollaboratorResponse]:
    """List all collaborators in a project."""
    result = await db.execute(
        select(Project)
        .options(selectinload(Project.collaborators))
        .where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get role for each collaborator
    collaborators = []
    for user in project.collaborators:
        role = "member"  # Get actual role from project_collaborators table
        collaborators.append(
            ProjectCollaboratorResponse(
                user_id=user.id,
                username=user.username,
                email=user.email,
                role=role
            )
        )
    
    return collaborators
