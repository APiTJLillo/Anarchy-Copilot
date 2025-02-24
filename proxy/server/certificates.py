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
        self._cert_cache_dir = Path(tempfile.gettempdir()) / "proxy_certs"
        self._cert_cache_dir.mkdir(exist_ok=True)
        
        # Load CA cert and key
        try:
            with open(ca_cert_path, 'rb') as f:
                self.ca_cert = x509.load_pem_x509_certificate(f.read())
            
            with open(ca_key_path, 'rb') as f:
                self.ca_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
            
            # Validate CA certificate
            if not isinstance(self.ca_key, rsa.RSAPrivateKey):
                raise ValueError("CA key must be an RSA private key")
            
            if self.ca_cert.not_valid_after < datetime.utcnow():
                raise ValueError("CA certificate has expired")
                
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
            cert_path, key_path = self._cert_cache[hostname]
            # Validate cached cert still exists and is valid
            try:
                with open(cert_path, 'rb') as f:
                    cert = x509.load_pem_x509_certificate(f.read())
                    if cert.not_valid_after > datetime.utcnow():
                        return cert_path, key_path
            except:
                pass  # Generate new cert if validation fails
            
        # Generate new private key with modern defaults
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,  # Minimum recommended size
        )

        # Create certificate with modern defaults
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "MITM Proxy"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "HTTPS Intercept")
        ])

        # Build certificate with recommended extensions
        builder = x509.CertificateBuilder()
        builder = builder.subject_name(subject)
        builder = builder.issuer_name(self.ca_cert.subject)
        builder = builder.public_key(key.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        
        # Set validity period (shorter than CA cert)
        builder = builder.not_valid_before(datetime.utcnow())
        builder = builder.not_valid_after(
            datetime.utcnow() + timedelta(days=90)  # 90 days is a common choice for short-lived certs
        )
        
        # Add recommended extensions
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(hostname)]),
            critical=False
        )
        
        # Modern extensions for security
        builder = builder.add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True
        )
        
        builder = builder.add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True
        )
        
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH
            ]),
            critical=False
        )
        
        # Sign certificate
        cert = builder.sign(
            private_key=self.ca_key,
            algorithm=hashes.SHA256()
        )

        # Save to cache directory with hostname-based unique names
        cert_path = str(self._cert_cache_dir / f"{hostname}_{datetime.utcnow().strftime('%Y%m%d')}.crt")
        key_path = str(self._cert_cache_dir / f"{hostname}_{datetime.utcnow().strftime('%Y%m%d')}.key")

        # Save certificate with proper permissions
        with open(cert_path, "wb") as f:
            os.chmod(cert_path, 0o644)  # Read by owner, read by others
            f.write(cert.public_bytes(
                encoding=serialization.Encoding.PEM
            ))

        # Save private key with restricted permissions
        with open(key_path, "wb") as f:
            os.chmod(key_path, 0o600)  # Read/write by owner only
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        self._cert_cache[hostname] = (cert_path, key_path)
        return cert_path, key_path

    def cleanup_old_certs(self, max_age_days: int = 7) -> None:
        """Clean up expired certificates from the cache directory."""
        try:
            cutoff = datetime.utcnow() - timedelta(days=max_age_days)
            for cert_file in self._cert_cache_dir.glob("*.crt"):
                if cert_file.stat().st_mtime < cutoff.timestamp():
                    try:
                        cert_file.unlink()
                        key_file = self._cert_cache_dir / (cert_file.stem + ".key")
                        if key_file.exists():
                            key_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove old cert {cert_file}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning up old certificates: {e}")

    def __del__(self):
        """Cleanup when the CA is destroyed."""
        try:
            self.cleanup_old_certs(max_age_days=0)  # Remove all cached certs
            if self._cert_cache_dir.exists():
                self._cert_cache_dir.rmdir()  # Remove directory if empty
        except:
            pass  # Ignore cleanup errors on shutdown
