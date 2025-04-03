"""
Integration of the FilterInterceptor with the proxy system.

This module registers the FilterInterceptor with the proxy system and ensures
proper handling of bypass mode and traffic recording.
"""

import logging
import os
from typing import Dict, List, Optional, Set, Any

from proxy.interceptor import ProxyInterceptor
from proxy.filter import FilterInterceptor, FilterManager, DatabaseFilterStorage
from proxy.instances import register_interceptor

logger = logging.getLogger(__name__)

# Initialize filter manager and interceptor
filter_storage = DatabaseFilterStorage()
filter_manager = FilterManager(filter_storage)
filter_interceptor = None

def initialize_filter_system() -> None:
    """Initialize the filter system and register the interceptor."""
    global filter_interceptor
    
    try:
        # Create filter interceptor
        filter_interceptor = FilterInterceptor(filter_manager)
        
        # Register with proxy system
        register_interceptor(filter_interceptor)
        
        logger.info("Filter system initialized and interceptor registered")
    except Exception as e:
        logger.error(f"Failed to initialize filter system: {e}")

# Initialize when module is imported
initialize_filter_system()
