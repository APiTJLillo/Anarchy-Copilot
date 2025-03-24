"""Protocol definitions for state management."""
from typing import Protocol, Dict, Any

class StateHandlerProtocol(Protocol):
    """Protocol defining required state handler methods."""

    def update_stats(self, **kwargs: Any) -> None:
        """Update connection statistics.
        
        Args:
            **kwargs: Statistics to update
        """
        ...

    def get_connection_state(self, connection_id: str) -> Dict[str, Any]:
        """Get state for a connection.
        
        Args:
            connection_id: The connection ID to get state for
            
        Returns:
            Dictionary containing connection state
        """
        ...
