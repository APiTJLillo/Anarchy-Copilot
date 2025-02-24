"""User management API endpoints."""
import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from database import get_async_session
from models.base import User
from .models import UserResponse, CreateUser, UpdateUser

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("", response_model=List[UserResponse])
async def list_users(
    db: AsyncSession = Depends(get_async_session)
) -> List[UserResponse]:
    """List all users."""
    result = await db.execute(
        select(User)
        .order_by(User.username)
    )
    users = result.scalars().all()
    return list(users)

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_async_session)
) -> UserResponse:
    """Get a user by ID."""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.post("", response_model=UserResponse)
async def create_user(
    data: CreateUser,
    db: AsyncSession = Depends(get_async_session)
) -> UserResponse:
    """Create a new user."""
    # Check if username or email already exists
    result = await db.execute(
        select(User)
        .where(
            (User.username == data.username) |
            (User.email == data.email)
        )
    )
    existing_user = result.scalar_one_or_none()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already exists")

    user = User(
        username=data.username,
        email=data.email,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    data: UpdateUser,
    db: AsyncSession = Depends(get_async_session)
) -> UserResponse:
    """Update a user."""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update allowed fields
    update_data = data.dict(exclude_unset=True)
    if update_data:
        for key, value in update_data.items():
            setattr(user, key, value)
        await db.commit()
        await db.refresh(user)
    
    return user
