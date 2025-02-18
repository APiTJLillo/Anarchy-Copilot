"""Create a test user in the database."""
import sys
import os
sys.path.append(os.getcwd())

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from models.base import User, Base

def create_test_user():
    """Create a test user for development."""
    # Create database engine
    engine = create_engine('sqlite:///anarchy_copilot.db')
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create session
    session = Session(engine)
    
    # Check if test user already exists
    existing_user = session.query(User).filter(User.email == "test@example.com").first()
    if existing_user:
        print("Test user already exists!")
        return existing_user.id
    
    # Create test user
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="dummy_hashed_password",  # In production, this would be properly hashed
        is_active=True
    )
    
    session.add(user)
    session.commit()
    
    print(f"Created test user with ID: {user.id}")
    return user.id

if __name__ == "__main__":
    create_test_user()
