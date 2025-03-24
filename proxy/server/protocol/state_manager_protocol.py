"""Protocol definitions for state manager."""
from typing import Protocol, Any

class StateManagerProtocol(Protocol):
    """Protocol defining required state manager methods."""
    
    def update_stats(self, **kwargs: Any) -> None:
        """Update connection statistics."""
        ...
