"""Wrapper module providing context creation functions."""
from typing import Optional
import ssl
import os
import logging
from pathlib import Path

from ..tls_helper import cert_manager
from .context import context_factory, TLSContext

logger = logging.getLogger("proxy.core")

def get_server_context(hostname: Optional[str] = None) -> TLSContext:
    """Get server-side SSL context.
    
    Args:
        hostname: Optional hostname for certificate configuration.
        
    Returns:
        Configured SSL context for server use.
    """
    if not cert_manager.ca:
        raise RuntimeError("Certificate Authority not initialized")
        
    # Get paths from cert manager
    ca_cert = str(cert_manager.ca.ca_cert_path)
    ca_key = str(cert_manager.ca.ca_key_path)
    
    # Create context with proper error handling
    try:
        ctx = context_factory.create_server_context(ca_cert, ca_key)
        return ctx  # Return the TLSContext wrapper
    except Exception as e:
        logger.error(f"Failed to create server context: {e}")
        raise

def get_client_context(hostname: str) -> TLSContext:
    """Get client-side SSL context.
    
    Args:
        hostname: The hostname to create context for.
        
    Returns:
        Configured SSL context for client use.
    """
    try:
        # Create client context using factory to ensure proper wrapping
        return context_factory.create_client_context(hostname)
    except Exception as e:
        logger.error(f"Failed to create client context: {e}")
        raise
