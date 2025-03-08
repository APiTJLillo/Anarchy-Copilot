"""TLS certificate and connection management.

This module provides a comprehensive TLS certificate management system for MITM proxy
operations with features for certificate generation, caching, and monitoring.
"""

import asyncio
import logging
import os
import ssl
import stat
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, TYPE_CHECKING, Set, NamedTuple, Union, Literal, List
from typing_extensions import TypedDict
from dataclasses import dataclass
from functools import wraps
import inspect
import traceback
import time
import heapq

import OpenSSL
from cryptography import x509
import pwd
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509.oid import NameOID

if TYPE_CHECKING:
    from .certificates import CertificateAuthority

logger = logging.getLogger("proxy.core")

# Type definitions
PathLike = Union[str, Path]
HealthStatus = Literal["healthy", "warning", "error", "critical"]

class CertificateValidationError(Exception):
    """Raised when certificate validation fails."""
    pass

class RateLimitExceededError(Exception):
    """Raised when certificate generation rate limit is exceeded."""
    pass

class CANotInitializedError(Exception):
    """Raised when CA is not properly initialized."""
    pass

@dataclass
class CertificateStats:
    """Statistics for certificate operations."""
    total_generated: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    validation_failures: int = 0
    generation_failures: int = 0
    average_generation_time: float = 0.0
    total_certificates: int = 0
    expired_certificates: int = 0
    cached_contexts: int = 0

class CertificateHealth(NamedTuple):
    """Health status information for the certificate manager."""
    status: HealthStatus
    details: Dict[str, Any]
    last_error: Optional[str]
    last_check: datetime

class BaseCertificateManager:
    """Base class for certificate management functionality."""
    
    CERT_AGE_LIMIT = 86400 * 30  # 30 days
    LOCK_EXPIRE_TIME = 3600  # 1 hour
    CLEANUP_INTERVAL = 3600  # 1 hour
    CERT_AGE_WARN = 86400 * 25  # Warn when cert is 25 days old
    STATS_ROLLOVER = 86400  # Reset stats daily
    
    def __init__(self, cert_cache_dir: Optional[Path] = None):
        """Initialize base certificate manager."""
        self.cert_cache_dir = cert_cache_dir or Path("/tmp/proxy_certs")
        self._cert_cache: Dict[str, Tuple[Path, Path, datetime]] = {}
        self._contexts: Dict[str, ssl.SSLContext] = {}
        self._temp_files: Set[Path] = set()
        self._locks: Dict[str, Tuple[asyncio.Lock, datetime]] = {}
        self._stats = CertificateStats()
        self._last_error = None
        self._last_error_time = None
        self._last_health_check = None
        self._cleanup_task = None
        self._monitoring_task = None
        self._loop = None
        
        # Ensure directory exists with proper permissions
        self._setup_cert_dir()

    def _setup_cert_dir(self) -> None:
        """Set up certificate directory with secure permissions."""
        if not self.cert_cache_dir.exists():
            self.cert_cache_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self.cert_cache_dir, stat.S_IRWXU)  # 700 permissions
        else:
            current_mode = os.stat(self.cert_cache_dir).st_mode
            if current_mode & 0o777 != 0o700:
                os.chmod(self.cert_cache_dir, stat.S_IRWXU)

class CertificateManager(BaseCertificateManager):
    """Manage TLS certificates for interception with caching and monitoring."""
    
    def __init__(
        self,
        ca: Optional['CertificateAuthority'] = None,
        cert_cache_dir: Optional[Path] = None
    ):
        """Initialize certificate manager.
        
        Args:
            ca: Certificate Authority for signing certificates
            cert_cache_dir: Directory for certificate storage
        """
        super().__init__(cert_cache_dir)
        
        # Initialize CA
        self.ca = ca
        self._ca_cert = None
        self._ca_key = None
        
        if ca:
            try:
                self._load_ca()
            except Exception as e:
                logger.error(f"Failed to load CA during initialization: {e}")
                raise

        # Start background tasks
        self._init_tasks()

    def _init_tasks(self) -> None:
        """Initialize background tasks."""
        # Tasks will be initialized when start() is called
        pass

    def get_context(self, hostname: str, is_server: bool = True) -> ssl.SSLContext:
        """Get SSL context for the given hostname.

        Args:
            hostname: The hostname to create context for
            is_server: Whether to create a server or client context

        Returns:
            ssl.SSLContext: Configured SSL context with loaded certificate

        Raises:
            CANotInitializedError: If CA is not properly initialized
            RuntimeError: If context creation fails
        """
        cache_key = f"{'server' if is_server else 'client'}_{hostname}"
        if cache_key in self._contexts:
            return self._contexts[cache_key]

        try:
            cert_path, key_path = self.get_cert_pair(hostname)
            
            # Create appropriate context type
            if is_server:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(certfile=cert_path, keyfile=key_path)
            else:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            
            # Load CA cert separately for verification if needed
            if self._ca_cert and hasattr(self.ca, 'ca_cert_path'):
                context.load_verify_locations(cafile=str(self.ca.ca_cert_path))
            
            # Configure modern TLS settings
            context.options |= (
                ssl.OP_NO_SSLv2 | 
                ssl.OP_NO_SSLv3 |
                ssl.OP_CIPHER_SERVER_PREFERENCE |
                ssl.OP_SINGLE_DH_USE |
                ssl.OP_NO_COMPRESSION
            )

            self._contexts[cache_key] = context
            return context

        except Exception as e:
            logger.error(f"Failed to create SSL context for {hostname}: {e}")
            raise RuntimeError(f"Failed to create SSL context: {e}")

    def set_ca(self, ca: 'CertificateAuthority') -> None:
        """Set or update the CA instance."""
        try:
            self.ca = ca
            self._load_ca()
            
            # Clear caches since we have a new CA
            self._cert_cache.clear()
            self._contexts.clear()
            self._stats = CertificateStats()
            
            logger.info("CA updated successfully")
            
        except Exception as e:
            self.ca = None
            self._ca_cert = None
            self._ca_key = None
            logger.error(f"Failed to set CA: {e}")
            raise RuntimeError(f"Failed to set CA: {e}")

    def get_cert_pair(self, hostname: str) -> Tuple[str, str]:
        """Get certificate path for a given hostname.

        Args:
            hostname: The hostname to get certificate for

        Returns:
            str: Path to the certificate file

        Raises:
            CANotInitializedError: If CA is not properly initialized
            RuntimeError: If certificate generation fails
        """
        if not self.ca:
            raise CANotInitializedError("No CA provided")

        try:
            cert_path, key_path = self.ca.get_certificate(hostname)
            if not os.path.exists(cert_path) or not os.path.exists(key_path):
                raise FileNotFoundError(f"Certificate or key file missing for {hostname}")
            return cert_path, key_path
        except Exception as e:
            logger.error(f"Failed to get certificate path for {hostname}: {e}")
            raise RuntimeError(f"Failed to get certificate path: {e}")

    def _load_ca(self) -> None:
        """Load CA certificate and private key."""
        if not self.ca:
            raise CANotInitializedError("No CA provided")

        try:
            if not self.ca.ca_cert_path.exists():
                raise FileNotFoundError(f"CA certificate not found: {self.ca.ca_cert_path}")
            if not self.ca.ca_key_path.exists():
                raise FileNotFoundError(f"CA private key not found: {self.ca.ca_key_path}")
            
            # Load and validate CA certificate using cryptography
            with open(self.ca.ca_cert_path, 'rb') as f:
                cert_data = f.read()
                self._ca_cert = x509.load_pem_x509_certificate(cert_data)
            
            # Load and validate CA private key using cryptography
            with open(self.ca.ca_key_path, 'rb') as f:
                key_data = f.read()
                self._ca_key = serialization.load_pem_private_key(key_data, password=None)
            
            # Verify certificate/key pair
            self._verify_ca_pair()
            
            # Set secure permissions
            self._set_ca_permissions()
            
            logger.info("CA loaded successfully")
            
        except Exception as e:
            self._ca_cert = None
            self._ca_key = None
            logger.error(f"Failed to load CA: {e}")
            raise

    def _verify_ca_pair(self) -> None:
        """Verify CA certificate and private key match."""
        try:
            # Create test data and sign it
            test_data = b"test"
            signature = self._ca_key.sign(
                test_data,
                padding.PKCS1v15(),
                hashes.SHA256()
            )

            # Verify the signature
            self._ca_cert.public_key().verify(
                signature,
                test_data,
                padding.PKCS1v15(),
                hashes.SHA256()
            )

            # Check expiration - convert to UTC for comparison
            now = datetime.now(self._ca_cert.not_valid_after_utc.tzinfo)
            if self._ca_cert.not_valid_after_utc < now:
                raise ValueError("CA certificate has expired")

            logger.debug("CA certificate and private key validated successfully")

        except Exception as e:
            logger.error(f"CA certificate and private key validation failed: {e}")
            raise ValueError(f"CA certificate and private key validation failed: {e}")

    def _set_ca_permissions(self) -> None:
        """Set secure permissions for CA files."""
        try:
            os.chmod(self.ca.ca_cert_path, 0o644)  # rw-r--r--
            os.chmod(self.ca.ca_key_path, 0o600)   # rw-------
        except Exception as e:
            logger.error(f"Failed to set CA file permissions: {e}")
            raise

    async def cleanup_old_certs(self, max_age: Optional[int] = None) -> None:
        """Clean up expired certificates.
        
        Args:
            max_age: Optional maximum age in seconds. If provided, remove all certs older than this.
                    If not provided, uses CERT_AGE_LIMIT.
        """
        try:
            age_limit = max_age if max_age is not None else self.CERT_AGE_LIMIT
            now = datetime.utcnow()
            expired = []

            # Check cached certificates
            for hostname, (cert_path, key_path, created) in list(self._cert_cache.items()):
                if (now - created).total_seconds() > age_limit:
                    expired.append((hostname, cert_path, key_path))

            # Remove expired certificates
            for hostname, cert_path, key_path in expired:
                try:
                    if os.path.exists(cert_path):
                        os.unlink(cert_path)
                    if os.path.exists(key_path):
                        os.unlink(key_path)
                    del self._cert_cache[hostname]
                    if hostname in self._contexts:
                        del self._contexts[hostname]
                except Exception as e:
                    logger.warning(f"Failed to remove expired cert {cert_path}: {e}")

            # Clean up context cache entries without certs
            for hostname in list(self._contexts.keys()):
                if hostname not in self._cert_cache:
                    del self._contexts[hostname]

            # Update stats
            self._stats.expired_certificates += len(expired)

        except Exception as e:
            logger.error(f"Error during certificate cleanup: {e}")
            self._last_error = e
            self._last_error_time = datetime.utcnow()
            raise

    async def _periodic_cleanup(self) -> None:
        """Run periodic cleanup tasks."""
        while True:
            try:
                await asyncio.sleep(self.CLEANUP_INTERVAL)
                await self.cleanup_old_certs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    async def _periodic_monitoring(self) -> None:
        """Run periodic monitoring tasks."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                health = self.get_health()
                if health.status != "healthy":
                    logger.warning(f"Health check failed: {health.status}")
                    logger.warning(f"Health details: {health.details}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring task: {e}")

    def get_health(self) -> CertificateHealth:
        """Get current health status of certificate manager.

        Returns:
            CertificateHealth: Current health status and details
        """
        now = datetime.utcnow()
        details = {
            "total_certificates": len(self._cert_cache),
            "cached_contexts": len(self._contexts),
            "ca_initialized": bool(self.ca),
            "stats": {
                "total_generated": self._stats.total_generated,
                "cache_hits": self._stats.cache_hits,
                "cache_misses": self._stats.cache_misses,
                "validation_failures": self._stats.validation_failures,
                "generation_failures": self._stats.generation_failures
            }
        }

        # Determine health status
        status: HealthStatus = "healthy"
        if not self.ca:
            status = "critical"
        elif self._last_error and (now - self._last_error_time < timedelta(minutes=5)):
            status = "error"
        elif self._stats.generation_failures > 0:
            status = "warning"

        self._last_health_check = now
        return CertificateHealth(
            status=status,
            details=details,
            last_error=str(self._last_error) if self._last_error else None,
            last_check=now
        )

    async def stop(self) -> None:
        """Stop the certificate manager and clean up resources."""
        logger.info("Stopping certificate manager...")
        
        # Cancel background tasks
        for task in [self._cleanup_task, self._monitoring_task]:
            if task and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        # Final cleanup
        try:
            await self.cleanup_old_certs(max_age=0)  # Clean all certificates
            logger.info("Certificate manager stopped")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise

    def start(self):
        """Start the certificate manager with a running event loop."""
        if self._cleanup_task is None or self._monitoring_task is None:
            self._loop = asyncio.get_event_loop()
            self._cleanup_task = self._loop.create_task(self._periodic_cleanup())
            self._monitoring_task = self._loop.create_task(self._periodic_monitoring())

# Create global instance
cert_manager = CertificateManager()

__all__ = ['CertificateManager', 'cert_manager', 'CertificateValidationError',
           'RateLimitExceededError', 'CANotInitializedError']
