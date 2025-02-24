"""Proxy middleware and response classes."""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class ProxyResponse:
    """Represents a response from the proxy middleware."""
    status_code: int
    headers: Dict[str, str]
    body: Optional[bytes] = None
    modified: bool = False
    intercept_enabled: bool = True
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

async def proxy_middleware(request: Dict[str, Any]) -> Optional[ProxyResponse]:
    """Default proxy middleware for handling requests.
    
    Args:
        request: Dictionary containing request details
    
    Returns:
        Optional[ProxyResponse]: Response if request should be modified,
                               None to pass through unmodified
    """
    # Default implementation passes through requests unmodified
    return None
