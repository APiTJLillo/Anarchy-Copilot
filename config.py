import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./anarchy_copilot.db")
