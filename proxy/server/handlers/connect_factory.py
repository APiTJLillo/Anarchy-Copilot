"""Factory for creating ConnectHandler instances."""
import logging
from typing import Any, Optional
from dataclasses import dataclass

from ..protocol.base_types import TlsHandlerBase

logger = logging.getLogger("proxy.core")

@dataclass
class ConnectConfig:
    """Configuration for ConnectHandler."""
    connection_id: str
    transport: Optional[Any] = None
    connect_timeout: int = 30
    read_timeout: int = 60

def create_connect_handler(config: ConnectConfig, state_manager: Any, error_handler: Any, tls_handler: Optional[TlsHandlerBase] = None):
    """Create a ConnectHandler instance.
    
    Args:
        config: Connection configuration.
        state_manager: State manager instance.
        error_handler: Error handler instance.
        tls_handler: Optional existing TLS handler instance.
        
    Returns:
        A new ConnectHandler instance.
    """
    # Import here to avoid circular dependency
    from .connect import ConnectHandler
    
    logger.debug(f"Creating ConnectHandler for {config.connection_id}")
    return ConnectHandler(
        connection_id=config.connection_id,
        state_manager=state_manager,
        error_handler=error_handler,
        tls_handler=tls_handler
    )
