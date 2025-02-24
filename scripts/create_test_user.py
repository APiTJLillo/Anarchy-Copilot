"""Script to create a test user and project for development."""
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import AsyncSessionLocal
from models.base import User, Project
from sqlalchemy.orm import selectinload
import logging

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

            # Add user as collaborator
            project.collaborators.append(user)
            await db.commit()
            logger.info("Added test user as collaborator")
        else:
            logger.info(f"Test project already exists with ID: {project.id}")

if __name__ == "__main__":
    asyncio.run(create_test_data())
