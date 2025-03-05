"""
This module has been deprecated and split into multiple modules:

- models/server_state.py - Server state management
- models/ssl_context.py - SSL context handling
- models/connection.py - Connection handling
- models/server.py - Main server implementation
- utils/constants.py - Configuration constants
- utils/logging.py - Logging configuration
- main.py - Server entrypoint

Please use the new modular structure instead.
"""

raise ImportError(
    "This module has been deprecated. Please use the new modular structure. "
    "Import from proxy.models or use proxy.create_proxy_server() instead."
)
