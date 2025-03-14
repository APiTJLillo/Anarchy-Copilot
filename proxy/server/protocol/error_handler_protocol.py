"""Protocol definitions for error handler."""
from typing import Protocol

class ErrorHandlerProtocol(Protocol):
    """Protocol defining required error handler methods."""

    async def handle_error(self, error: Exception) -> None:
        """Handle error.
        
        Args:
            error: The exception to handle
            
        Returns:
            None when error handling is complete
        """
        ...
