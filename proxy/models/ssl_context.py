import os
import ssl
import gc
from pathlib import Path
from typing import Dict, Optional

from ..utils.logging import logger
from ..utils.constants import MemoryConfig, SSLConfig

class SSLContextManager:
    """Manages SSL contexts for the proxy server with efficient resource handling."""
    
    def __init__(self, cert_path: str, key_path: str):
        self.cert_path = str(Path(cert_path).resolve())
        self.key_path = str(Path(key_path).resolve())
        self._ssl_contexts: Dict[str, ssl.SSLContext] = {}
        self._active_connections: Dict[str, int] = {}
        self._initialized = False
        self._init_error = None
        self._max_cached_contexts = MemoryConfig.CONTEXT_CACHE_SIZE
        self._operation_count = 0

        logger.debug(f"Initializing SSL context manager with:")
        logger.debug(f"  Certificate: {self.cert_path}")
        logger.debug(f"  Private key: {self.key_path}")

        self._initialize_certificates()

    def _initialize_certificates(self) -> None:
        """Initialize and validate SSL certificates."""
        try:
            # Check file permissions and existence
            for path in (self.cert_path, self.key_path):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")
                if not os.access(path, os.R_OK):
                    raise PermissionError(f"No read permission for: {path}")
                    
            # Check certificate and key validity
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            try:
                context.load_cert_chain(self.cert_path, self.key_path)
            except (ssl.SSLError, IOError) as e:
                raise RuntimeError(f"Invalid certificate or key: {e}")
                
            # Verify certificate attributes
            with open(self.cert_path, 'rb') as f:
                cert_data = f.read()
                try:
                    from cryptography import x509
                    from cryptography.hazmat.backends import default_backend
                    cert = x509.load_pem_x509_certificate(cert_data, default_backend())
                    logger.info(f"Certificate details:")
                    logger.info(f"  Subject: {cert.subject}")
                    logger.info(f"  Issuer: {cert.issuer}")
                    logger.info(f"  Valid from: {cert.not_valid_before}")
                    logger.info(f"  Valid until: {cert.not_valid_after}")
                except ImportError:
                    logger.warning("cryptography package not available for detailed cert inspection")
            
            logger.info("Successfully initialized SSL context manager")
            self._initialized = True
            
        except Exception as e:
            self._init_error = str(e)
            logger.error("Failed to initialize SSL context manager:")
            logger.error(f"  Error: {e}")
            if isinstance(e, ssl.SSLError):
                logger.error("  Hint: Check certificate format and key matching")
            elif isinstance(e, PermissionError):
                logger.error("  Hint: Check file permissions")
            elif isinstance(e, FileNotFoundError):
                logger.error("  Hint: Verify certificate paths")

    def _cleanup_unused_contexts(self) -> None:
        """Clean up unused SSL contexts to free memory."""
        if len(self._ssl_contexts) > self._max_cached_contexts:
            # Remove contexts with no active connections
            to_remove = [
                hostname for hostname, context in self._ssl_contexts.items()
                if self._active_connections.get(hostname, 0) == 0
            ]
            
            for hostname in to_remove:
                try:
                    del self._ssl_contexts[hostname]
                    del self._active_connections[hostname]
                    logger.debug(f"Removed unused SSL context for {hostname}")
                except KeyError:
                    pass
                    
            gc.collect()

    def cleanup_resources(self) -> None:
        """Release memory and cleanup resources."""
        self._ssl_contexts.clear()
        self._active_connections.clear()
        self.cert_path = None
        self.key_path = None
        gc.collect()

    def _load_cert_with_fallback(self) -> ssl.SSLContext:
        """Try to load certificate with different protocols."""
        errors = []
        
        # Try different SSL/TLS protocol versions, start with most secure
        for protocol in [ssl.PROTOCOL_TLS]:  # Only use TLS now for better compatibility
            try:
                context = ssl.SSLContext(protocol)
                context.load_cert_chain(self.cert_path, self.key_path)
                protocol_name = protocol.name if hasattr(protocol, 'name') else str(protocol)
                logger.debug(f"Successfully loaded certificate with protocol: {protocol_name}")
                return context
            except ssl.SSLError as e:
                errors.append(f"Protocol {protocol}: {str(e)}")
                continue
                
        error_msg = "\n".join(errors)
        raise ssl.SSLError(f"Failed to load certificate with any protocol:\n{error_msg}")

    def create_client_context(self, hostname: str) -> ssl.SSLContext:
        """Create SSL context for client connections (where we act as server)."""
        if not self._initialized:
            raise RuntimeError(f"SSL context manager not initialized: {self._init_error}")
        
        # Track connection
        self._active_connections[hostname] = self._active_connections.get(hostname, 0) + 1
            
        if hostname in self._ssl_contexts:
            logger.debug(f"Reusing cached SSL context for {hostname}")
            return self._ssl_contexts[hostname]
            
        # Clean up if we have too many contexts
        self._cleanup_unused_contexts()
        
        # Create new context
        try:
            context = self._load_cert_with_fallback()
            self._configure_context(context, server_side=True, hostname=hostname)
            self._ssl_contexts[hostname] = context
            return context
            
        except Exception as e:
            logger.error(f"Error creating client context for {hostname}: {e}")
            raise

    def create_server_context(self, hostname: str) -> ssl.SSLContext:
        """Create SSL context for connecting to target servers."""
        if not hostname:
            raise ValueError("Hostname is required for server context")
            
        context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
        self._configure_context(context, server_side=False, hostname=hostname)
        
        logger.debug(f"Created server context for {hostname} with SNI and verification enabled")
        return context

    def _configure_context(self, context: ssl.SSLContext, server_side: bool = True, hostname: Optional[str] = None) -> None:
        """Configure SSL context with appropriate security settings."""
        # Configure TLS versions
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3

        # Configure session tickets and reuse
        context.options |= ssl.OP_NO_TICKET  # Disable session tickets
        context.options |= ssl.OP_NO_RENEGOTIATION  # Prevent renegotiation attacks

        if server_side:
            # Server mode settings
            context.verify_mode = ssl.CERT_NONE
            context.check_hostname = False
            
            # Server optimizations and security
            context.options |= ssl.OP_NO_COMPRESSION  # Prevent CRIME attack
            context.options |= ssl.OP_CIPHER_SERVER_PREFERENCE
            context.options |= ssl.OP_SINGLE_DH_USE
            context.options |= ssl.OP_SINGLE_ECDH_USE
            context.options |= getattr(ssl, 'OP_IGNORE_UNEXPECTED_EOF', 0)  # Ignore unexpected EOFs
            context.options |= getattr(ssl, 'OP_NO_TLSv1_3_MIDDLEBOX_COMPAT', 0)  # Disable middlebox compat
            
            try:
                context.set_alpn_protocols(['h2', 'http/1.1'])
            except (NotImplementedError, ssl.SSLError) as e:
                logger.debug(f"ALPN not supported in server mode: {e}")
                
        else:
            # Client mode settings
            if not hostname:
                raise ValueError("Hostname is required for client SSL context")
                
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            context.hostname = hostname  # For SNI
            
            try:
                context.load_default_certs(purpose=ssl.Purpose.SERVER_AUTH)
                logger.debug(f"Loaded default certificates for {hostname}")
            except Exception as e:
                logger.warning(f"Could not load default certificates: {e}")
                context.verify_mode = ssl.CERT_NONE
                context.check_hostname = False
            
            try:
                context.set_alpn_protocols(['h2', 'http/1.1'])
                logger.debug(f"ALPN protocols set for {hostname}")
            except (NotImplementedError, ssl.SSLError) as e:
                logger.debug(f"ALPN not supported in client mode: {e}")

        # Set secure cipher list
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')
        
        # Enable Perfect Forward Secrecy
        context.options |= ssl.OP_SINGLE_ECDH_USE
        context.options |= ssl.OP_SINGLE_DH_USE

    def remove_context(self, hostname: str) -> None:
        """Remove a context and decrement its connection count."""
        if hostname in self._active_connections:
            self._active_connections[hostname] -= 1
            if self._active_connections[hostname] <= 0:
                del self._active_connections[hostname]
                if hostname in self._ssl_contexts:
                    del self._ssl_contexts[hostname]
                    logger.debug(f"Removed SSL context for {hostname}")
