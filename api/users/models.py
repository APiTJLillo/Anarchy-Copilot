"""User models for API."""
from pydantic import BaseModel, EmailStr
from typing import Optional

class UserBase(BaseModel):
    """Base model for user data."""
    username: str
    email: EmailStr

class UserResponse(UserBase):
    """Response model for user data."""
    id: int
    is_active: bool = True

    class Config:
        from_attributes = True

class CreateUser(UserBase):
    """Model for creating a new user."""
    pass

class UpdateUser(BaseModel):
    """Model for updating a user."""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None
