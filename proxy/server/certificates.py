"""Certificate handling for HTTPS interception."""

import logging
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import tempfile
from base64 import b64decode
import warnings
import asyncio
from typing import Optional
warnings.filterwarnings('ignore', category=DeprecationWarning)

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.x509.oid import SignatureAlgorithmOID
from cryptography.hazmat.primitives.hashes import HashAlgorithm
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.backends.openssl import backend as openssl_backend
from cryptography.x509.oid import NameOID

def sign_certificate(builder, private_key, legacy_mode=False):
    """Sign certificate using SHA256 or SHA1 with RSA based on legacy mode."""
    try:
        # Initialize key variables
        backend = default_backend()
        algorithm = hashes.SHA1() if legacy_mode else hashes.SHA256()
        expected_algo = (
            x509.SignatureAlgorithmOID.RSA_WITH_SHA1 if legacy_mode 
            else x509.SignatureAlgorithmOID.RSA_WITH_SHA256
        )
        
        # Sign the certificate with selected algorithm
        logger.debug("Signing certificate with %s", 
                    "SHA1" if legacy_mode else "SHA256")
        cert = builder.sign(
            private_key=private_key,
            algorithm=algorithm,
            backend=backend
        )
        
        # Log the actual signature algorithm used
        sig_algo = cert.signature_algorithm_oid
        logger.debug("Certificate signed with algorithm: %s", sig_algo._name)
        
        # Build the certificate chain
        logger.debug("Building certificate chain")
        cert_chain = [cert]
        issuer_pubkey = private_key.public_key()
        
        # Validate certificate structure
        for c in cert_chain:
            # Basic structure validation
            if not c.subject or not c.issuer:
                raise ValueError("Certificate missing subject or issuer")
            
            # Validate required extensions are present
            required_extensions = {
                x509.oid.ExtensionOID.SUBJECT_KEY_IDENTIFIER: "Subject Key Identifier",
                x509.oid.ExtensionOID.AUTHORITY_KEY_IDENTIFIER: "Authority Key Identifier",
                x509.oid.ExtensionOID.KEY_USAGE: "Key Usage",
                x509.oid.ExtensionOID.EXTENDED_KEY_USAGE: "Extended Key Usage",
                x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME: "Subject Alternative Name"
            }
            
            for ext_oid, ext_name in required_extensions.items():
                try:
                    c.extensions.get_extension_for_oid(ext_oid)
                except x509.ExtensionNotFound:
                    raise ValueError(f"Missing required {ext_name} extension")
        
        # Verify key usage
        key_usage = cert.extensions.get_extension_for_oid(
            x509.oid.ExtensionOID.KEY_USAGE
        ).value
        if not (key_usage.digital_signature and key_usage.key_encipherment):
            raise ValueError("Certificate missing required key usage flags")
        
        # Verify certificate signature with retry
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                issuer_pubkey.verify(
                    cert.signature,
                    cert.tbs_certificate_bytes,
                    padding.PKCS1v15(),
                    cert.signature_hash_algorithm
                )
                logger.debug("Certificate signature verified successfully")
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning("Signature verification attempt %d failed: %s", 
                                 attempt + 1, str(e))
                    continue
                
        if last_error:
            raise ValueError(f"Certificate signature verification failed after "
                           f"{max_retries} attempts: {str(last_error)}")
        
        # Verify signature algorithm matches expected
        if sig_algo != expected_algo:
            raise ValueError(
                f"Expected {expected_algo._name} signature, got {sig_algo._name}"
            )
        
        logger.debug("Certificate validation completed with %s signature", 
                    "SHA1" if legacy_mode else "SHA256")
        return cert
        
    except Exception as e:
        logger.error("Certificate signing/validation failed: %s", str(e))
        raise ValueError(f"Certificate signing/validation failed: {str(e)}")

logger = logging.getLogger("proxy.core")

# Define allowed PEM types and their aliases
PEM_TYPE_MAPPINGS = {
    b'CERTIFICATE': {b'CERTIFICATE'},
    b'PRIVATE KEY': {b'PRIVATE KEY', b'RSA PRIVATE KEY'},
}

def normalize_pem_type(marker_type: bytes) -> bytes:
    """Normalize PEM marker type to standard format."""
    for standard_type, aliases in PEM_TYPE_MAPPINGS.items():
        if marker_type in aliases:
            return standard_type
    return marker_type

def is_compatible_type(marker_type: bytes, required_type: bytes) -> bool:
    """Check if a marker type is compatible with required type."""
    if marker_type == required_type:
        return True
    # Look up in mappings
    for standard_type, aliases in PEM_TYPE_MAPPINGS.items():
        if required_type == standard_type and marker_type in aliases:
            return True
    return False

def validate_base64_content(content: bytes) -> bool:
    """Validate base64 content for PEM certificates."""
    try:
        # Check for valid base64 characters
        if any(c > 127 for c in content):
            return False
            
        # Try decoding base64 with appropriate padding
        pad_len = (4 - (len(content) % 4)) % 4
        if pad_len:
            content = content.rstrip(b'=') + b'=' * pad_len
            
        b64decode(content)
        return True
    except Exception:
        return False

def debug_pem_content(pem_data: bytes, description: str = "PEM data") -> None:
    """Debug helper to analyze PEM content structure."""
    logger.debug(f"Analyzing {description}")
    logger.debug("Total length: %d bytes", len(pem_data))
    
    lines = pem_data.replace(b'\r\n', b'\n').split(b'\n')
    logger.debug("Line count: %d", len(lines))
    
    begin_count = sum(1 for line in lines if b'BEGIN' in line and b'-----' in line)
    end_count = sum(1 for line in lines if b'END' in line and b'-----' in line)
    logger.debug("Markers found: %d BEGIN, %d END", begin_count, end_count)
    
    types_found = set()
    current_type = None
    
    for i, line in enumerate(lines, 1):
        if b'-----BEGIN' in line:
            try:
                parts = line.split(b'-----')
                marker_type = parts[1].split(b'BEGIN ', 1)[1].strip()
                types_found.add(marker_type.decode('ascii'))
                current_type = marker_type
                logger.debug("Line %d: BEGIN %s", i, marker_type.decode('ascii'))
            except Exception as e:
                logger.error("Line %d: Invalid BEGIN marker: %s", i, e)
        elif b'-----END' in line:
            logger.debug("Line %d: END marker", i)
            current_type = None
        elif current_type:
            # Log base64 content stats
            try:
                stripped = line.strip()
                if stripped:
                    logger.debug("Line %d: %d bytes base64 content", i, len(stripped))
            except Exception as e:
                logger.error("Line %d: Invalid content: %s", i, e)

    logger.debug("PEM types found: %s", ", ".join(sorted(types_found)))

def format_pem_block(base64_data: bytes, marker_type: bytes) -> bytes:
    """Format a single PEM block with proper headers, line wrapping and validation.
    
    Args:
        base64_data: The base64 encoded certificate data
        marker_type: The PEM marker type (e.g. CERTIFICATE)
    """
    if not validate_base64_content(base64_data):
        raise ValueError("Invalid base64 content")
    
    # Normalize and clean content
    content = bytes(base64_data).strip()
    content = content.replace(b'\n', b'').replace(b'\r', b'')
    
    # Ensure proper base64 padding
    pad_len = (4 - (len(content) % 4)) % 4
    if pad_len:
        content = content.rstrip(b'=') + b'=' * pad_len
    
    # Validate padded content
    try:
        if not validate_base64_content(content):
            raise ValueError("Invalid base64 content after padding")
    except Exception as e:
        logger.error(f"Base64 validation error: {e}")
        raise ValueError(f"Invalid base64 content: {e}")
    
    result = bytearray()
    
    # Add BEGIN marker with CRLF
    marker = marker_type.decode('ascii').strip().upper()
    result.extend(f"-----BEGIN {marker}-----\r\n".encode('ascii'))
    
    # Add content with proper line wrapping and CRLF endings
    for i in range(0, len(content), 64):
        chunk = content[i:i+64]
        if chunk:
            if not validate_base64_content(chunk):
                raise ValueError(f"Invalid base64 chunk at offset {i}: {chunk}")
            result.extend(chunk + b'\r\n')
            
    # Add END marker with CRLF
    result.extend(f"-----END {marker}-----\r\n".encode('ascii'))
    
    # Verify proper line endings exist
    try:
        lines = result.splitlines(keepends=True)
        if len(lines) < 3:
            raise ValueError("Invalid PEM block line count")
            
        # Verify content lines
        for i, line in enumerate(lines[:-1]):  # All lines except last should have CRLF
            if i > 0 and i < len(lines) - 2:  # Content lines (skip headers)
                stripped = line.rstrip(b'\r\n')
                if not stripped:
                    raise ValueError(f"Empty content line found at position {i+1}")
                if len(stripped) > 64:
                    raise ValueError(f"Line too long at position {i+1}: {len(stripped)} chars")
                    
            # Every line except the last should end with CRLF
            if not line.endswith(b'\r\n'):
                logger.error(f"Line {i+1} missing CRLF ending: {line!r}")
                raise ValueError(f"Line {i+1} missing CRLF ending")
            
    except Exception as e:
        logger.error(f"PEM block validation error: {e}")
        raise ValueError(f"Invalid PEM block structure: {str(e)}")
        
    return bytes(result)

def format_pem(pem_data: bytes, force_type: bytes = None) -> bytes:
    """Format PEM data with proper markers and line endings.
    
    Args:
        pem_data: Raw PEM data to format
        force_type: Optional PEM type to enforce (e.g. CERTIFICATE)
    """
    logger.debug("Formatting PEM data with %s lines", len(pem_data.split(b'\n')))
    if force_type:
        logger.debug("Enforcing PEM type: %s", force_type.decode('ascii'))
        
    blocks = []
    lines = pem_data.replace(b'\r\n', b'\n').split(b'\n')
    
    current_type = None
    base64_data = bytearray()
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        logger.debug("Processing line %d: %s...", line_num, 
                    line[:20].decode('ascii', errors='replace'))
            
        if line.startswith(b'-----'):
            if b'BEGIN' in line:
                # Extract marker type
                try:
                    parts = line.split(b'-----')
                    if len(parts) != 3:
                        raise ValueError("Invalid PEM marker format")
                        
                    current_type = parts[1].split(b'BEGIN ', 1)[1].strip()
                    if not current_type:
                        raise ValueError("Empty PEM marker type")
                        
                    if force_type:
                        if not is_compatible_type(current_type, force_type):
                            raise ValueError(
                                f"Incompatible PEM type: {current_type.decode('ascii')} "
                                f"(expected {force_type.decode('ascii')})"
                            )
                        current_type = force_type  # Normalize to required type
                        
                    logger.debug("Found BEGIN marker type: %s", current_type.decode('ascii'))
                        
                    # Process any previous base64 data
                    if base64_data:
                        blocks.append(format_pem_block(bytes(base64_data), current_type))
                        base64_data = bytearray()
                        
                except Exception as e:
                    logger.error("Failed to parse BEGIN marker on line %d: %s", line_num, e)
                    raise ValueError(f"Failed to parse BEGIN marker: {str(e)}")
                    
            elif b'END' in line:
                if not current_type:
                    logger.error("Found END marker without type: %s", line)
                    raise ValueError("Found END marker without type")
                    
                # Format accumulated base64 data
                if base64_data:
                    blocks.append(format_pem_block(bytes(base64_data), current_type))
                base64_data = bytearray()
                current_type = None
                
        elif current_type:
            # Accumulate base64 content
            if any(c > 127 for c in line):
                raise ValueError("Invalid base64 character in input")
            base64_data.extend(line)
            
    # Handle any remaining data
    if base64_data:
        if not current_type:
            logger.error("Found base64 content without PEM type")
            raise ValueError("Found base64 content without PEM type")
        blocks.append(format_pem_block(bytes(base64_data), current_type))
        
    if not blocks:
        raise ValueError("No valid PEM blocks found")

    # Join blocks with double CRLF between them
    return b'\r\n\r\n'.join(blocks)

class CertificateAuthority:
    """Certificate Authority for generating and managing certificates."""

    def __init__(self, ca_cert_path: Path, ca_key_path: Path):
        """Initialize the Certificate Authority.
        
        Args:
            ca_cert_path: Path to the CA certificate file
            ca_key_path: Path to the CA private key file
            
        Raises:
            ValueError: If either path is None or the files don't exist/aren't readable
            IOError: If there are issues reading the certificate files
            Exception: For other initialization errors
        """
        if not ca_cert_path or not ca_key_path:
            raise ValueError("Both CA certificate and key paths must be provided")
            
        if not isinstance(ca_cert_path, Path):
            ca_cert_path = Path(ca_cert_path)
        if not isinstance(ca_key_path, Path):
            ca_key_path = Path(ca_key_path)
            
        # Store paths as instance variables
        self.ca_cert_path = ca_cert_path
        self.ca_key_path = ca_key_path
            
        # Validate certificate path
        if not ca_cert_path.exists():
            raise ValueError(f"CA certificate file does not exist: {ca_cert_path}")
        if not os.access(ca_cert_path, os.R_OK):
            raise ValueError(f"CA certificate file is not readable: {ca_cert_path}")
            
        # Validate key path
        if not ca_key_path.exists():
            raise ValueError(f"CA key file does not exist: {ca_key_path}")
        if not os.access(ca_key_path, os.R_OK):
            raise ValueError(f"CA key file is not readable: {ca_key_path}")
            
        # Initialize instance variables
        self.ca_cert = None
        self.ca_key = None
        self._lock = asyncio.Lock()
        self._cert_cache = {}
        self._cert_cache_dir = Path("/tmp/proxy_certs")
        self._cert_cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Certificate Authority instance created successfully")

    async def start(self) -> None:
        """Initialize the CA asynchronously."""
        try:
            # Get the current event loop - don't create a new one
            loop = asyncio.get_running_loop()

            # Load and validate CA certificate using cryptography
            with open(self.ca_cert_path, 'rb') as f:
                cert_data = f.read()
                self.ca_cert = x509.load_pem_x509_certificate(cert_data)
                
                # Verify certificate uses allowed signature algorithms
                allowed_algorithms = [
                    x509.SignatureAlgorithmOID.RSA_WITH_SHA256,
                    x509.SignatureAlgorithmOID.RSA_WITH_SHA1
                ]
                if self.ca_cert.signature_algorithm_oid not in allowed_algorithms:
                    logger.warning("CA certificate must use RSA with SHA256 or SHA1")
                    raise ValueError("CA certificate must use RSA with either SHA256 or SHA1")
            
            # Load and validate CA private key
            with open(self.ca_key_path, 'rb') as f:
                key_data = f.read()
                # Try multiple formats for loading CA key
                load_errors = []
                for fmt in [serialization.PrivateFormat.TraditionalOpenSSL, 
                           serialization.PrivateFormat.PKCS8]:
                    try:
                        self.ca_key = serialization.load_pem_private_key(
                            key_data,
                            password=None
                        )
                        break
                    except Exception as e:
                        load_errors.append(f"{fmt}: {str(e)}")
                else:
                    error_msg = "Failed to load CA key in any format: " + "; ".join(load_errors)
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            # Validate CA key is RSA and compatible with TLS 1.0
            if not isinstance(self.ca_key, rsa.RSAPrivateKey):
                raise ValueError("CA key must be an RSA private key")
            
            # Verify key size and parameters 
            key_size = self.ca_key.key_size
            if key_size < 2048:
                raise ValueError(f"CA key size {key_size} is too small, minimum 2048 bits required")
            
            # Use timezone-aware datetime comparison
            now = datetime.now(timezone.utc)
            if now > self.ca_cert.not_valid_after_utc:
                raise ValueError("CA certificate has expired")
                
            # Verify the key pair matches
            await self._verify_key_pair()
            logger.info("Successfully verified CA certificate and key pair")
                
            logger.info("Certificate Authority initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CA: {e}")
            self.ca_cert = None
            self.ca_key = None
            raise

    async def _verify_key_pair(self):
        """Verify that the CA certificate and private key form a valid pair."""
        try:
            # Get the public key from the certificate
            cert_public_key = self.ca_cert.public_key()
            
            # Get the public key from the private key
            private_key_public = self.ca_key.public_key()
            
            # Compare the public key numbers
            if cert_public_key.public_numbers() != private_key_public.public_numbers():
                raise ValueError("CA certificate and private key do not form a valid pair")
                
        except Exception as e:
            logger.error(f"Failed to verify CA certificate and key pair: {str(e)}")
            raise

    async def get_certificate(self, hostname: str) -> tuple[str, str]:
        """Get or generate certificate for hostname.
        
        This function is thread-safe and handles concurrent requests.
        
        Args:
            hostname: The hostname to generate a certificate for

        Returns:
            tuple[str, str]: Paths to the certificate and key files

        Raises:
            RuntimeError: If certificate generation fails
        """
        if not self.ca_cert or not self.ca_key:
            raise RuntimeError("CA not initialized")

        async with self._lock:  # Ensure thread safety for concurrent requests
            # Check cache first
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

            try:
                # Generate key
                loop = asyncio.get_running_loop()
                backend = default_backend()
                key = await loop.run_in_executor(None, lambda: rsa.generate_private_key(
                    public_exponent=65537,  # Standard RSA public exponent
                    key_size=2048,  # Minimum size required by modern browsers
                    backend=backend
                ))

                # Generate certificate
                subject = issuer = x509.Name([
                    x509.NameAttribute(NameOID.COMMON_NAME, hostname),
                ])

                # Create certificate builder
                cert_builder = x509.CertificateBuilder().subject_name(
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
                    datetime.utcnow() + timedelta(days=365)  # 1 year validity
                ).add_extension(
                    x509.SubjectAlternativeName([x509.DNSName(hostname)]),
                    critical=False,
                )

                # Sign the certificate with the CA key
                certificate = cert_builder.sign(
                    private_key=self.ca_key,
                    algorithm=hashes.SHA256()
                )

                # Save the certificate and key
                cert_path = self._cert_cache_dir / f"{hostname}.crt"
                key_path = self._cert_cache_dir / f"{hostname}.key"

                # Write certificate
                with open(cert_path, "wb") as f:
                    f.write(certificate.public_bytes(serialization.Encoding.PEM))

                # Write private key
                with open(key_path, "wb") as f:
                    f.write(key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    ))

                # Set secure permissions
                os.chmod(cert_path, 0o644)  # rw-r--r--
                os.chmod(key_path, 0o600)   # rw-------

                # Update cache
                self._cert_cache[hostname] = (str(cert_path), str(key_path))

                logger.info(f"Generated new certificate for {hostname}")
                return str(cert_path), str(key_path)

            except Exception as e:
                logger.error(f"Failed to generate certificate for {hostname}: {e}")
                raise RuntimeError(f"Failed to generate certificate: {e}")

    async def cleanup_old_certs(self, max_age_days: int = 7) -> None:
        """Clean up expired certificates from the cache directory."""
        async with self._lock:
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
            asyncio.run(self.cleanup_old_certs(max_age_days=0))
            if self._cert_cache_dir.exists():
                self._cert_cache_dir.rmdir()
        except:
            pass
