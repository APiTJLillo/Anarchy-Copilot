"""Factory for creating TLS handlers."""
import logging
from typing import Optional
import asyncio
from dataclasses import dataclass

from .tls_handler import TlsHandler
from .state_manager import StateManager
from .error_handler import ErrorHandler

logger = logging.getLogger("proxy.core")

@dataclass
class TlsHandlerConfig:
    """Configuration for TLS handler."""
    connection_id: str
    state_manager: StateManager
    error_handler: ErrorHandler
    loop: Optional[asyncio.AbstractEventLoop] = None

def create_tls_handler(config: Optional[TlsHandlerConfig] = None) -> TlsHandler:
    """Create a new TLS handler instance.
    
    Args:
        config: Configuration for the TLS handler.
        
    Returns:
        A new TLS handler instance.
    """
    if config is None:
        raise ValueError("TlsHandlerConfig is required")
        
    logger.debug("Creating new TLS handler instance")
    handler = TlsHandler(
        connection_id=config.connection_id,
        state_manager=config.state_manager,
        error_handler=config.error_handler,
        loop=config.loop
    )
    return handler
