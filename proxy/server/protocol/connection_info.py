"""Connection information protocols and types."""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ConnectionInfo:
    """Connection information."""
    hostname: str
    port: int
    peer_cert: Optional[bytes] = None
    alpn_protocol: Optional[str] = None
