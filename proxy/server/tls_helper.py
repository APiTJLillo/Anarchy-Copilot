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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, TYPE_CHECKING, Set, NamedTuple, Union, Literal, List, TypedDict
from dataclasses import dataclass
from functools import wraps
import inspect
import traceback
import time
import heapq

import OpenSSL
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509.oid import NameOID
from cryptography.hazmat.backends import default_backend

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
        self._is_running = False
        
        if ca:
            try:
                self._load_ca()
                logger.info("Certificate Authority loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CA during initialization: {e}")
                self.ca = None
                logger.warning("HTTPS interception will be disabled")

        # Initialize tasks as None, they'll be created when start() is called
        self._cleanup_task = None
        self._monitoring_task = None
        self._loop = None

    def _init_tasks(self) -> None:
        """Initialize background tasks.
        
        This sets up the task references but doesn't start them.
        The actual tasks are created and started in the start() method.
        """
        if self._cleanup_task is None and self._monitoring_task is None:
            logger.debug("Task references initialized")

    def is_running(self) -> bool:
        """Check if the certificate manager is running.
        
        Returns:
            bool: True if running, False otherwise
        """
        return self._is_running

    async def get_context(self, hostname: str, is_server: bool = True) -> ssl.SSLContext:
        """Get SSL context for the given hostname.
        
        Args:
            hostname: The hostname to get SSL context for
            is_server: Whether this is a server-side context
            
        Returns:
            ssl.SSLContext: Configured SSL context
            
        Raises:
            RuntimeError: If context creation fails
        """
        cache_key = f"{'server' if is_server else 'client'}_{hostname}"
        if cache_key in self._contexts:
            return self._contexts[cache_key]

        try:
            cert_path, key_path = await self.get_cert_pair(hostname)
            
            # Create appropriate context type
            if is_server:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(certfile=cert_path, keyfile=key_path)
            else:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            
            # Load CA cert separately for verification if needed
            if self._ca_cert and self.ca and self.ca.ca_cert_path:
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

    async def get_cert_pair(self, hostname: str) -> Tuple[str, str]:
        """Get certificate path for a given hostname.

        Args:
            hostname: The hostname to get certificate for

        Returns:
            Tuple[str, str]: Tuple of (cert_path, key_path)

        Raises:
            CANotInitializedError: If CA is not properly initialized
            RuntimeError: If certificate generation fails
        """
        if not self.ca:
            raise CANotInitializedError("No CA provided")

        try:
            cert_path, key_path = await self.ca.get_certificate(hostname)
            if not os.path.exists(cert_path) or not os.path.exists(key_path):
                raise FileNotFoundError(f"Certificate or key file missing for {hostname}")
            return cert_path, key_path
        except Exception as e:
            logger.error(f"Failed to get certificate path for {hostname}: {e}")
            raise RuntimeError(f"Failed to get certificate path: {e}")

    def _load_ca(self) -> None:
        """Load CA certificate and private key."""
        if not self.ca:
            logger.error("No CA instance provided")
            raise CANotInitializedError("No CA provided")

        logger.debug(f"Loading CA with cert_path={self.ca.ca_cert_path}, key_path={self.ca.ca_key_path}")
        try:
            if not os.path.exists(self.ca.ca_cert_path):
                logger.error(f"CA certificate file not found at: {self.ca.ca_cert_path}")
                raise FileNotFoundError(f"CA certificate not found: {self.ca.ca_cert_path}")
            else:
                logger.debug(f"Found CA certificate at {self.ca.ca_cert_path}")
                
            if not os.path.exists(self.ca.ca_key_path):
                logger.error(f"CA key file not found at: {self.ca.ca_key_path}")
                raise FileNotFoundError(f"CA private key not found: {self.ca.ca_key_path}")
            else:
                logger.debug(f"Found CA key at {self.ca.ca_key_path}")
            
            # Load and validate CA certificate using cryptography
            logger.debug("Loading CA certificate...")
            with open(self.ca.ca_cert_path, 'rb') as f:
                cert_data = f.read()
                logger.debug(f"Read {len(cert_data)} bytes from certificate file")
                self._ca_cert = x509.load_pem_x509_certificate(cert_data)
                logger.debug("Successfully loaded CA certificate")
            
            # Load and validate CA private key using cryptography
            logger.debug("Loading CA private key...")
            with open(self.ca.ca_key_path, 'rb') as f:
                key_data = f.read()
                logger.debug(f"Read {len(key_data)} bytes from key file")
                self._ca_key = serialization.load_pem_private_key(key_data, password=None)
                logger.debug("Successfully loaded CA private key")
            
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
        logger.debug("Starting CA certificate/key pair verification...")
        if not self._ca_key or not self._ca_cert:
            logger.error("CA certificate or key not loaded before verification")
            raise ValueError("CA certificate or key not loaded")

        try:
            logger.debug("Testing key pair by signing test data...")
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
            # Safely get expiration time
            expiry = getattr(self._ca_cert, 'not_valid_after_utc', None)
            if not expiry:
                expiry = self._ca_cert.not_valid_after
            if not expiry:
                raise ValueError("Could not determine certificate expiration")

            now = datetime.now(timezone.utc)
            if expiry < now:
                raise ValueError("CA certificate has expired")

            logger.debug("CA certificate and private key validated successfully")

        except Exception as e:
            logger.error(f"CA certificate and private key validation failed: {e}")
            raise ValueError(f"CA certificate and private key validation failed: {e}")

    def _set_ca_permissions(self) -> None:
        """Set secure permissions for CA files."""
        if not self.ca:
            raise CANotInitializedError("No CA provided")

        try:
            if hasattr(self.ca, 'ca_cert_path'):
                os.chmod(self.ca.ca_cert_path, 0o644)  # rw-r--r--
            if hasattr(self.ca, 'ca_key_path'):
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
        
        # Check if CA is fully initialized
        ca_initialized = bool(self.ca and self._ca_cert and self._ca_key)
        
        # Count actual certificates in cache directory
        total_certs = 0
        try:
            if self.cert_cache_dir.exists():
                total_certs = len([f for f in self.cert_cache_dir.glob("*.crt")])
        except Exception as e:
            logger.warning(f"Error counting certificates: {e}")

        details = {
            "total_certificates": total_certs,
            "cached_contexts": len(self._contexts),
            "ca_initialized": ca_initialized,
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
        if not ca_initialized:
            status = "critical"
            self._last_error = "CA not fully initialized"
            self._last_error_time = now
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

        self._is_running = False

        # Final cleanup
        try:
            await self.cleanup_old_certs(max_age=0)  # Clean all certificates
            logger.info("Certificate manager stopped")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise

    async def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Start the certificate manager with the provided or current event loop."""
        if self._is_running:
            logger.debug("Certificate manager already running, skipping start")
            return

        logger.debug("Starting certificate manager...")
        try:
            # Get existing loop or current loop - don't create a new one
            if loop:
                self._loop = loop
                logger.debug(f"Using provided event loop: {loop}")
            else:
                self._loop = asyncio.get_running_loop()
                logger.debug("Using current event loop")
            
            # Initialize CA if needed
            if self.ca:
                logger.debug(f"Initializing CA with cert_path={self.ca.ca_cert_path}, key_path={self.ca.ca_key_path}")
                try:
                    # Load CA first - this initializes self._ca_cert and self._ca_key
                    logger.info("Starting CA initialization...")
                    try:
                        self._load_ca()
                        logger.info("CA certificates loaded successfully")

                        # Verify certificates are valid
                        if not self._ca_cert or not self._ca_key:
                            raise CANotInitializedError("CA certificate or key not loaded")
                        if not isinstance(self._ca_cert, x509.Certificate):
                            raise CANotInitializedError("Invalid CA certificate type")
                        if not isinstance(self._ca_key, rsa.RSAPrivateKey):
                            raise CANotInitializedError("Invalid CA key type")

                        # Ensure CA is started
                        if hasattr(self.ca, 'start'):
                            logger.info("Starting CA service...")
                            await self.ca.start()
                            logger.info("CA service started successfully")
                            
                        # Verify CA is working by generating and verifying a test certificate
                        logger.info("Testing certificate generation...")
                        test_hostname = "test.local"
                        test_start = time.time()
                        cert_path, key_path = await self.ca.get_certificate(test_hostname)
                        test_duration = time.time() - test_start
                        logger.info(f"Test certificate generated in {test_duration:.2f} seconds")

                        # Verify test certificate was created
                        if not os.path.exists(cert_path):
                            raise RuntimeError(f"Test certificate not found at {cert_path}")
                        if not os.path.exists(key_path):
                            raise RuntimeError(f"Test key not found at {key_path}")

                        # Load and verify test certificate
                        logger.info("Verifying test certificate...")
                        test_cert_data = None
                        with open(cert_path, 'rb') as f:
                            test_cert_data = f.read()
                        test_cert = x509.load_pem_x509_certificate(test_cert_data)

                        # Verify test certificate properties
                        test_cn = test_cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
                        if test_cn != test_hostname:
                            raise RuntimeError(f"Test certificate CN mismatch: {test_cn} != {test_hostname}")
                        
                        # Verify certificate signature using CA public key
                        try:
                            # Get CA public key for verification
                            ca_public_key = self._ca_cert.public_key()
                            
                            # Get the signature algorithm and hash algorithm from the certificate
                            signature_algorithm = test_cert.signature_algorithm_oid
                            if signature_algorithm == x509.SignatureAlgorithmOID.RSA_WITH_SHA256:
                                hash_algorithm = hashes.SHA256()
                            elif signature_algorithm == x509.SignatureAlgorithmOID.RSA_WITH_SHA384:
                                hash_algorithm = hashes.SHA384()
                            elif signature_algorithm == x509.SignatureAlgorithmOID.RSA_WITH_SHA512:
                                hash_algorithm = hashes.SHA512()
                            else:
                                raise RuntimeError(f"Unsupported signature algorithm: {signature_algorithm}")
                            
                            # Verify the test certificate was signed by our CA
                            ca_public_key.verify(
                                test_cert.signature,
                                test_cert.tbs_certificate_bytes,
                                padding.PKCS1v15(),
                                hash_algorithm
                            )
                            
                            # Additional verification steps
                            now = datetime.now(timezone.utc)
                            if test_cert.not_valid_before_utc > now or test_cert.not_valid_after_utc < now:
                                raise RuntimeError("Test certificate has invalid validity period")
                            
                            if test_cert.issuer != self._ca_cert.subject:
                                raise RuntimeError("Test certificate issuer does not match CA subject")
                                
                            logger.info("Test certificate signature and validity verified successfully")
                        except Exception as e:
                            logger.error(f"Test certificate verification failed: {str(e)}")
                            logger.debug("Test certificate details:", extra={
                                'subject': str(test_cert.subject),
                                'issuer': str(test_cert.issuer),
                                'not_before': str(test_cert.not_valid_before),
                                'not_after': str(test_cert.not_valid_after)
                            })
                            raise RuntimeError(f"Certificate verification failed: {e}")

                        # Clean up test certificate
                        try:
                            os.unlink(cert_path)
                            os.unlink(key_path)
                            logger.info("Test certificate cleanup successful")
                        except Exception as e:
                            logger.warning(f"Failed to clean up test certificate: {e}")

                        logger.info("CA initialization and verification completed successfully")

                    except Exception as e:
                        logger.error(f"CA initialization failed: {str(e)}\n{traceback.format_exc()}")
                        self.ca = None
                        self._ca_cert = None
                        self._ca_key = None
                        raise RuntimeError(f"CA initialization failed: {e}")
                        
                except Exception as e:
                    logger.error(f"Failed to initialize CA during start: {e}")
                    self.ca = None
                    self._ca_cert = None
                    self._ca_key = None
                    raise

            # Only create tasks if they don't exist
            if not self._cleanup_task or self._cleanup_task.done():
                self._cleanup_task = self._loop.create_task(self._periodic_cleanup())
            if not self._monitoring_task or self._monitoring_task.done():
                self._monitoring_task = self._loop.create_task(self._periodic_monitoring())
            
            logger.debug("Started certificate manager background tasks")
            self._is_running = True
            logger.info("Certificate manager started")
            
        except Exception as e:
            logger.error(f"Failed to start certificate manager: {e}")
            raise

    def verify_with_backend(self, test_cert: x509.Certificate) -> bool:
        """Verify a test certificate using the CA's public key."""
        try:
            if not self._ca_cert:
                raise ValueError("CA certificate not loaded")
                
            ca_public_key = self._ca_cert.public_key()
            
            # Get the signature algorithm OID
            signature_algorithm_oid = test_cert.signature_algorithm_oid._name
            logger.debug(f"Verifying certificate with signature algorithm: {signature_algorithm_oid}")
            
            # Map signature algorithm to hash algorithm
            hash_algorithm = None
            if "sha256" in signature_algorithm_oid.lower():
                hash_algorithm = hashes.SHA256()
            elif "sha384" in signature_algorithm_oid.lower():
                hash_algorithm = hashes.SHA384()
            elif "sha512" in signature_algorithm_oid.lower():
                hash_algorithm = hashes.SHA512()
            else:
                raise ValueError(f"Unsupported signature algorithm: {signature_algorithm_oid}")
            
            # Verify the certificate
            try:
                ca_public_key.verify(
                    test_cert.signature,
                    test_cert.tbs_certificate_bytes,
                    padding.PKCS1v15(),
                    hash_algorithm
                )
                logger.debug("Certificate verification successful")
                return True
            except Exception as e:
                logger.error(f"Certificate verification failed: {e}", exc_info=True)
                return False
                
        except Exception as e:
            logger.error(f"Error during certificate verification: {e}", exc_info=True)
            return False

# Create global instance
cert_manager = CertificateManager()

__all__ = ['CertificateManager', 'cert_manager', 'CertificateValidationError',
           'RateLimitExceededError', 'CANotInitializedError']
