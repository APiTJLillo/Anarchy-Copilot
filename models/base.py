"""Base SQLAlchemy models and common tables."""
from sqlalchemy import Column, Integer, ForeignKey, Table
from database.base import Base

# Association table for project collaborators
project_collaborators = Table(
    'project_collaborators',
    Base.metadata,
    Column('project_id', Integer, ForeignKey('projects.id')),
    Column('user_id', Integer, ForeignKey('users.id'))
)

__all__ = ["Base", "project_collaborators"]
