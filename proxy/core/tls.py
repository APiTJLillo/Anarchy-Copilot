"""Core TLS functionality for the proxy."""
import ssl
import os
import logging
from typing import Optional, Dict, Any, Union, Tuple, Type
from dataclasses import dataclass
from datetime import datetime
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

logger = logging.getLogger(__name__)

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
    """Statistics about certificate operations."""
    total_generated: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    validation_errors: int = 0
    last_generated: Optional[datetime] = None

@dataclass
class CertificateHealth:
    """Health status of certificate operations."""
    ca_initialized: bool = False
    cache_size: int = 0
    error_rate: float = 0.0
    last_error: Optional[str] = None

class CertificateManager:
    """Manages TLS certificates for MITM proxy."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'CertificateManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize certificate manager."""
        self.cert_dir = os.path.expanduser("~/.proxy/certs")
        os.makedirs(self.cert_dir, exist_ok=True)
        self._ca_cert = None
        self._ca_key = None
        self._cert_cache = {}
        self._stats = CertificateStats()
        self._initialize_ca()

    def _initialize_ca(self) -> None:
        """Initialize the Certificate Authority."""
        ca_key_path = os.path.join(self.cert_dir, "ca.key")
        ca_cert_path = os.path.join(self.cert_dir, "ca.crt")
        
        if not os.path.exists(ca_key_path) or not os.path.exists(ca_cert_path):
            self._generate_ca()
        else:
            self._load_ca(ca_key_path, ca_cert_path)

    def _generate_ca(self) -> None:
        """Generate new CA certificate and private key."""
        # Generate key
        self._ca_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u"Anarchy Copilot CA"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Anarchy Copilot"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, u"Security Testing")
        ])
        
        self._ca_cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            self._ca_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow().replace(year=datetime.utcnow().year + 10)
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True
        ).sign(self._ca_key, hashes.SHA256())
        
        # Save to files
        with open(os.path.join(self.cert_dir, "ca.key"), "wb") as f:
            f.write(self._ca_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
            
        with open(os.path.join(self.cert_dir, "ca.crt"), "wb") as f:
            f.write(self._ca_cert.public_bytes(serialization.Encoding.PEM))

    def _load_ca(self, key_path: str, cert_path: str) -> None:
        """Load CA certificate and private key from files."""
        with open(key_path, "rb") as f:
            self._ca_key = serialization.load_pem_private_key(
                f.read(),
                password=None
            )
            
        with open(cert_path, "rb") as f:
            self._ca_cert = x509.load_pem_x509_certificate(f.read())

    def get_context(self, hostname: str, is_server: bool = False) -> ssl.SSLContext:
        """Get SSL context for the given hostname.
        
        Args:
            hostname: Target hostname
            is_server: Whether this is a server-side context
            
        Returns:
            Configured SSL context
        """
        if not self._ca_cert or not self._ca_key:
            raise CANotInitializedError("CA not initialized")
            
        context = ssl.create_default_context(
            purpose=ssl.Purpose.CLIENT_AUTH if is_server else ssl.Purpose.SERVER_AUTH
        )
        
        if is_server:
            # Generate or get cached cert for hostname
            cert_path = self._get_cert_path(hostname)
            context.load_cert_chain(cert_path)
        else:
            # Load CA cert for client verification
            context.load_verify_locations(
                cafile=os.path.join(self.cert_dir, "ca.crt")
            )
            
        return context

    def _get_cert_path(self, hostname: str) -> str:
        """Get path to certificate for hostname, generating if needed."""
        cert_path = os.path.join(self.cert_dir, f"{hostname}.crt")
        
        if not os.path.exists(cert_path):
            self._generate_cert(hostname, cert_path)
            
        return cert_path

    def _generate_cert(self, hostname: str, cert_path: str) -> None:
        """Generate certificate for hostname."""
        # Generate key
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Generate certificate
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, hostname)
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            self._ca_cert.subject
        ).public_key(
            key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow().replace(year=datetime.utcnow().year + 1)
        ).add_extension(
            x509.SubjectAlternativeName([x509.DNSName(hostname)]),
            critical=False
        ).sign(self._ca_key, hashes.SHA256())
        
        # Save to file
        with open(cert_path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
            f.write(cert.public_bytes(serialization.Encoding.PEM))

def get_tls_context(hostname: str, is_server: bool = False) -> ssl.SSLContext:
    """Get SSL context for the given hostname.
    
    Args:
        hostname: Target hostname to get context for
        is_server: Whether this is a server-side context
        
    Returns:
        Configured SSL context
    """
    cert_mgr = CertificateManager.get_instance()
    return cert_mgr.get_context(hostname, is_server)

__all__ = [
    'get_tls_context',
    'CertificateManager',
    'CertificateValidationError',
    'RateLimitExceededError',
    'CANotInitializedError',
    'CertificateStats',
    'CertificateHealth'
] 