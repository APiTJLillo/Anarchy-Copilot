"""Helper functions for socket operations."""
import socket
import ssl
import logging
from typing import Optional, cast

logger = logging.getLogger("proxy.core")

def get_raw_socket(ssl_sock: ssl.SSLSocket) -> Optional[socket.socket]:
    """Extract raw socket from SSL socket safely."""
    try:
        # Try standard socket attribute first
        if hasattr(ssl_sock, '_socket'):
            return cast(socket.socket, getattr(ssl_sock, '_socket'))
            
        # Try alternate attribute name
        if hasattr(ssl_sock, '_sock'):
            return cast(socket.socket, getattr(ssl_sock, '_sock'))
            
        # Fall back to fileno method
        if hasattr(ssl_sock, 'fileno'):
            fileno = ssl_sock.fileno()
            if fileno >= 0:
                return socket.fromfd(fileno, socket.AF_INET, socket.SOCK_STREAM)
                
    except (AttributeError, socket.error) as e:
        logger.debug(f"Error getting raw socket: {e}")
        
    return None
