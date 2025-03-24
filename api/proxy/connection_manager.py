"""Connection manager initialization."""
from .connection import ConnectionManager

# Initialize the global connection manager instance
connection_manager = ConnectionManager()

__all__ = ['connection_manager'] 