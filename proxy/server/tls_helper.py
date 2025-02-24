"""TLS certificate and connection management."""
import asyncio
import logging
import os
import ssl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, TYPE_CHECKING
import OpenSSL
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

if TYPE_CHECKING:
    from .certificates import CertificateAuthority

logger = logging.getLogger("proxy.core")

class CertificateManager:
    """Manage TLS certificates for interception."""
    
    def __init__(self, ca: Optional['CertificateAuthority'] = None, cert_cache_dir: Optional[Path] = None):
        """Initialize certificate manager."""
        self.cert_cache_dir = cert_cache_dir or Path("/tmp/proxy_certs")
        self.cert_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._cert_cache: Dict[str, Tuple[Path, Path, datetime]] = {}
        self._contexts: Dict[str, ssl.SSLContext] = {}
        self.ca = ca
        self._ca_cert = None
        self._ca_key = None
        
        if ca:
            self._load_ca()

    def _load_ca(self) -> None:
        """Load CA certificate and private key."""
        try:
            with open(self.ca.cert_path, 'rb') as f:
                self._ca_cert = OpenSSL.crypto.load_certificate(
                    OpenSSL.crypto.FILETYPE_PEM, f.read()
                )
            with open(self.ca.key_path, 'rb') as f:
                self._ca_key = OpenSSL.crypto.load_privatekey(
                    OpenSSL.crypto.FILETYPE_PEM, f.read()
                )
        except Exception as e:
            logger.error(f"Failed to load CA certificate/key: {e}")
            raise

    def set_ca(self, ca: 'CertificateAuthority') -> None:
        """Set or update the CA instance."""
        self.ca = ca
        self._load_ca()
        # Clear caches since we have a new CA
        self._cert_cache.clear()
        self._contexts.clear()

    def _generate_cert(self, hostname: str) -> Tuple[Path, Path]:
        """Generate a certificate for a hostname."""
        if not self.ca:
            raise RuntimeError("CA not initialized")

        try:
            # Generate key
            key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Create certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, hostname)
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=365)
            ).add_extension(
                x509.SubjectAlternativeName([x509.DNSName(hostname)]),
                critical=False,
            ).sign(key, hashes.SHA256())
            
            # Save to files
            cert_path = self.cert_cache_dir / f"{hostname}.crt"
            key_path = self.cert_cache_dir / f"{hostname}.key"
            
            with open(cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            with open(key_path, "wb") as f:
                f.write(key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            return cert_path, key_path
            
        except Exception as e:
            logger.error(f"Failed to generate certificate for {hostname}: {e}")
            raise

    def get_context(self, hostname: str) -> ssl.SSLContext:
        """Get or create an SSL context for a hostname."""
        if not self.ca:
            raise RuntimeError("CA not initialized")

        try:
            if hostname in self._contexts:
                return self._contexts[hostname]
            
            # Check cache
            if hostname in self._cert_cache:
                cert_path, key_path, _ = self._cert_cache[hostname]
                if cert_path.exists() and key_path.exists():
                    ctx = self._create_context(cert_path, key_path)
                    self._contexts[hostname] = ctx
                    return ctx
            
            # Generate new certificate
            cert_path, key_path = self._generate_cert(hostname)
            ctx = self._create_context(cert_path, key_path)
            
            # Cache results
            self._cert_cache[hostname] = (cert_path, key_path, datetime.now())
            self._contexts[hostname] = ctx
            
            return ctx
            
        except Exception as e:
            logger.error(f"Failed to get SSL context for {hostname}: {e}")
            raise

    def _create_context(self, cert_path: Path, key_path: Path) -> ssl.SSLContext:
        """Create an SSL context with the given certificate and key."""
        try:
            ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ctx.load_cert_chain(cert_path, key_path)
            # Disable hostname verification for MITM
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            # Configure TLS settings for better compatibility and performance
            ctx.options |= (
                ssl.OP_NO_COMPRESSION |  # Disable compression to prevent CRIME attack
                ssl.OP_CIPHER_SERVER_PREFERENCE  # Use server's cipher preferences
            )
            # Allow larger TLS record size
            if hasattr(ssl, 'OP_NO_RENEGOTIATION'):
                ctx.options |= ssl.OP_NO_RENEGOTIATION
            # Set buffer sizes
            ctx.set_ciphers('ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256')
            return ctx
        except Exception as e:
            logger.error(f"Failed to create SSL context: {e}")
            raise

    async def cleanup_old_certs(self, max_age: int = 86400) -> None:
        """Clean up old certificates from the cache directory."""
        try:
            now = datetime.now()
            for hostname, (cert_path, key_path, timestamp) in list(self._cert_cache.items()):
                if (now - timestamp).total_seconds() > max_age:
                    try:
                        cert_path.unlink(missing_ok=True)
                        key_path.unlink(missing_ok=True)
                        del self._cert_cache[hostname]
                        if hostname in self._contexts:
                            del self._contexts[hostname]
                    except Exception as e:
                        logger.error(f"Error cleaning up certs for {hostname}: {e}")
                        
        except Exception as e:
            logger.error(f"Error during certificate cleanup: {e}")

# Global instance (initialized without CA)
cert_manager = CertificateManager()
