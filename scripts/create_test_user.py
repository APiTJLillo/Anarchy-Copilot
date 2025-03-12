"""Script to create a test user and project for development."""
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database.session import AsyncSessionLocal
from models.core import User, Project
from sqlalchemy.orm import selectinload
from sqlalchemy import text
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_test_data():
    """Create test user and project if they don't exist."""
    async with AsyncSessionLocal() as db:
        # Check if test user exists
        result = await db.execute(
            select(User).where(User.username == "testuser")
        )
        user = result.scalar_one_or_none()

        if not user:
            logger.info("Creating test user...")
            user = User(
                username="testuser",
                email="test@example.com",
                hashed_password="test123",  # In production, this should be properly hashed
                is_active=True
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            logger.info(f"Created test user with ID: {user.id}")
        else:
            logger.info(f"Test user already exists with ID: {user.id}")

        # Check if test project exists
        result = await db.execute(
            select(Project)
            .options(selectinload(Project.collaborators))
            .where(Project.name == "Test Project")
        )
        project = result.scalar_one_or_none()

        if not project:
            logger.info("Creating test project...")
            project = Project(
                name="Test Project",
                description="Project for testing",
                owner_id=user.id,
                scope={"domains": ["example.com"]},
                is_archived=False
            )
            db.add(project)
            await db.commit()
            await db.refresh(project)
            logger.info(f"Created test project with ID: {project.id}")

            # Add user as collaborator using explicit SQL
            await db.execute(
                text("""
                INSERT INTO project_collaborators (project_id, user_id)
                VALUES (:project_id, :user_id)
                ON CONFLICT DO NOTHING
                """),
                {"project_id": project.id, "user_id": user.id}
            )
            await db.commit()
            logger.info("Added test user as collaborator")
        else:
            logger.info(f"Test project already exists with ID: {project.id}")

        # Create test proxy session if none exists
        result = await db.execute(text("SELECT * FROM proxy_sessions WHERE is_active = true"))
        active_session = result.first()

        if not active_session:
            logger.info("Creating test proxy session...")
            # Create new proxy session
            await db.execute(
                text("""
                    INSERT INTO proxy_sessions 
                    (name, project_id, created_by, start_time, is_active, settings)
                    VALUES (:name, :project_id, :user_id, :start_time, :is_active, :settings)
                """),
                {
                    "name": "Test Session",
                    "project_id": project.id,
                    "user_id": user.id,
                    "start_time": datetime.utcnow(),
                    "is_active": True,
                    "settings": json.dumps({"intercept_requests": True, "intercept_responses": True})
                }
            )
            await db.commit()
            logger.info("Created test proxy session")
        else:
            logger.info("Active proxy session already exists")

if __name__ == "__main__":
    asyncio.run(create_test_data())
