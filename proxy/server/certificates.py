"""Certificate handling for HTTPS interception."""
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path
import tempfile
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

logger = logging.getLogger("proxy.core")

class CertificateAuthority:
    """Manages certificates for HTTPS interception."""

    def __init__(self, ca_cert_path: Path, ca_key_path: Path):
        """Initialize the Certificate Authority.
        
        Args:
            ca_cert_path: Path to CA certificate file
            ca_key_path: Path to CA private key file
        """
        self.ca_cert_path = ca_cert_path
        self.ca_key_path = ca_key_path
        self._cert_cache = {}
        
        # Load CA cert and key
        try:
            with open(ca_cert_path, 'rb') as f:
                self.ca_cert = x509.load_pem_x509_certificate(f.read())
            
            with open(ca_key_path, 'rb') as f:
                self.ca_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
                
        except Exception as e:
            logger.error(f"Failed to load CA certificate/key: {e}")
            raise

    def get_certificate(self, hostname: str) -> tuple[str, str]:
        """Get or generate certificate for hostname.
        
        Args:
            hostname: The hostname to generate certificate for
            
        Returns:
            Tuple of (cert_path, key_path)
        """
        if hostname in self._cert_cache:
            return self._cert_cache[hostname]

        # Generate new certificate
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )

        # Create certificate
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, hostname)
        ])

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            self.ca_cert.subject
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
            critical=False
        ).sign(self.ca_key, hashes.SHA256())

        # Save to temp files
        cert_path = os.path.join(tempfile.gettempdir(), f"{hostname}.crt")
        key_path = os.path.join(tempfile.gettempdir(), f"{hostname}.key")

        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        with open(key_path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        self._cert_cache[hostname] = (cert_path, key_path)
        return cert_path, key_path
