"""Certificate handling for HTTPS interception."""

import logging
from datetime import datetime, timedelta
import os
from pathlib import Path
import tempfile
from base64 import b64decode
import warnings
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
from cryptography.x509.oid import NameOID

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
                cert_data = f.read()
                try:
                    self.ca_cert = x509.load_pem_x509_certificate(cert_data)
                    # Verify certificate uses allowed signature algorithms
                    allowed_algorithms = [
                        x509.SignatureAlgorithmOID.RSA_WITH_SHA256,
                        x509.SignatureAlgorithmOID.RSA_WITH_SHA1
                    ]
                    if self.ca_cert.signature_algorithm_oid not in allowed_algorithms:
                        logger.warning("CA certificate must use RSA with SHA256 or SHA1")
                        raise ValueError("CA certificate must use RSA with either SHA256 or SHA1")
                except Exception as e:
                    logger.error(f"Failed to load CA certificate: {e}")
                    raise ValueError(f"Invalid CA certificate format: {e}")
            
            with open(ca_key_path, 'rb') as f:
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
            
            if self.ca_cert.not_valid_after < datetime.utcnow():
                raise ValueError("CA certificate has expired")
                
        except Exception as e:
            logger.error(f"Failed to load CA certificate/key: {e}")
            raise

    def get_certificate(self, hostname: str) -> tuple[str, str]:
        """Get or generate certificate for hostname."""
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
            
        # Generate RSA key with legacy-compatible parameters
        # Get backend that supports SHA1
        backend = default_backend()
        key = rsa.generate_private_key(
            public_exponent=65537,  # Standard RSA public exponent
            key_size=2048,  # Minimum size required by modern browsers
            backend=backend
        )
        logger.debug("Generated 2048-bit RSA key with e=65537")

        # Create certificate with TLS 1.0 compatible settings
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "MITM Proxy"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "HTTPS Intercept")
        ])

        # Build certificate
        builder = x509.CertificateBuilder()
        builder = builder.subject_name(subject)
        builder = builder.issuer_name(self.ca_cert.subject)
        builder = builder.public_key(key.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        # Set validity period with skew tolerance
        now = datetime.utcnow()
        builder = builder.not_valid_before(
            now - timedelta(minutes=5)  # Allow for clock skew
        )
        builder = builder.not_valid_after(
            now + timedelta(days=90)
        )
        logger.debug(f"Certificate validity period: {now - timedelta(minutes=5)} to {now + timedelta(days=90)}")
        
        # Configure certificate for maximum TLS compatibility
        try:
            # Pre-build extensions in TLS 1.0 preferred order
            extensions = [
                (x509.BasicConstraints(ca=False, path_length=None), True),
                (x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=True,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False
                ), True),
                (x509.SubjectKeyIdentifier.from_public_key(key.public_key()), False),
                (x509.AuthorityKeyIdentifier.from_issuer_public_key(self.ca_key.public_key()), False),
                (x509.SubjectAlternativeName([x509.DNSName(hostname)]), False),
                (x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), False),
            ]

            # Add extensions in order
            for ext, critical in extensions:
                builder = builder.add_extension(ext, critical=critical)
                logger.debug(f"Added {ext.__class__.__name__} extension (critical={critical})")

        except ValueError as e:
            logger.error("Failed to add certificate extensions: %s", e)
            raise ValueError(f"Certificate extension error: {e}")

        # Sign and verify certificate
        try:
            # Determine if legacy mode is needed based on CA cert
            legacy_mode = self.ca_cert.signature_algorithm_oid == x509.SignatureAlgorithmOID.RSA_WITH_SHA1
            
            cert = sign_certificate(builder, self.ca_key, legacy_mode=legacy_mode)
            logger.debug("Successfully signed certificate")

            # Verify certificate with proper validation based on mode
            def verify_cert(cert_to_verify, issuer_cert, legacy=False):
                """Verify a certificate's validity and signature."""
                try:
                    # Time validation with clock skew allowance
                    now = datetime.utcnow()
                    skew = timedelta(minutes=5 if legacy else 1)  # More lenient for legacy mode
                    
                    # Handle timezone naive comparison with extra buffer for clock skew
                    not_valid_before = cert_to_verify.not_valid_before
                    not_valid_after = cert_to_verify.not_valid_after
                    
                    # Add extra buffer for TLS1.0 compatibility
                    if legacy:
                        now = now + timedelta(minutes=2)  # Future bias for old clients
                    
                    # Validate with skew allowance on both sides
                    effective_start = not_valid_before - skew
                    effective_end = not_valid_after + skew
                    
                    if now < effective_start:
                        raise ValueError(f"Certificate not yet valid (becomes valid at {not_valid_before}, with {skew} allowance)")
                    if now > effective_end:
                        raise ValueError(f"Certificate has expired (expired at {not_valid_after}, with {skew} allowance)")
                    
                    logger.debug(f"Certificate time validation passed: valid from {effective_start} to {effective_end}")
                        
                    # Signature verification
                    issuer_cert.public_key().verify(
                        cert_to_verify.signature,
                        cert_to_verify.tbs_certificate_bytes,
                        padding.PKCS1v15(),
                        cert_to_verify.signature_hash_algorithm
                    )

                    # Check certificate chain integrity
                    if cert_to_verify.issuer != issuer_cert.subject:
                        raise ValueError("Certificate issuer does not match CA subject")

                    # Verify key usage flags
                    key_usage = cert_to_verify.extensions.get_extension_for_oid(
                        x509.oid.ExtensionOID.KEY_USAGE
                    ).value
                    if not (key_usage.digital_signature and key_usage.key_encipherment):
                        raise ValueError("Certificate missing required key usage flags")

                    # Verify algorithm matches legacy mode
                    if legacy:
                        expected_algo = x509.SignatureAlgorithmOID.RSA_WITH_SHA1
                    else:
                        expected_algo = x509.SignatureAlgorithmOID.RSA_WITH_SHA256
                        
                    if cert_to_verify.signature_algorithm_oid != expected_algo:
                        raise ValueError(
                            f"Incorrect signature algorithm for {'legacy' if legacy else 'modern'} mode: "
                            f"got {cert_to_verify.signature_algorithm_oid._name}, "
                            f"expected {expected_algo._name}"
                        )

                    logger.debug("Certificate verification successful")
                    return True
                    
                except ValueError as e:
                    logger.error(f"Certificate validation error: {e}")
                    return False
                except Exception as e:
                    logger.error(f"Unexpected error during certificate verification: {e}")
                    return False
            
            # Perform verification with retries
            max_retries = 3
            success = False
            for attempt in range(max_retries):
                if verify_cert(cert, self.ca_cert, legacy_mode):
                    success = True
                    break
                logger.warning(f"Verification attempt {attempt + 1} failed, retrying...")
                
            if not success:
                raise ValueError("Certificate verification failed after multiple attempts")
                
            logger.debug("Successfully verified certificate signature and chain")
        except Exception as e:
            logger.error("Failed to sign/verify certificate: %s", str(e))
            raise ValueError(f"Certificate signing/verification failed: {str(e)}")


        # Save certificate chain with proper permissions
        cert_path = str(self._cert_cache_dir / f"{hostname}_{datetime.utcnow().strftime('%Y%m%d')}.crt")
        key_path = str(self._cert_cache_dir / f"{hostname}_{datetime.utcnow().strftime('%Y%m%d')}.key")
        
        try:
            # Ensure proper directory permissions
            os.chmod(self._cert_cache_dir, 0o755)

            # Generate certificates with enhanced formatting
            try:
                # Get raw certificate data with explicit encoding
                host_cert_raw = cert.public_bytes(
                    encoding=serialization.Encoding.PEM
                )
                ca_cert_raw = self.ca_cert.public_bytes(
                    encoding=serialization.Encoding.PEM
                )
                    
                logger.debug("Generated raw certificates - host: %d bytes, CA: %d bytes", 
                           len(host_cert_raw), len(ca_cert_raw))
                
                # Enhanced certificate cleaning and validation
                def clean_cert(raw_data: bytes, cert_type: str) -> tuple[bytes, x509.Certificate]:
                    """Clean and validate certificate PEM data with enhanced validation."""
                    try:
                        # Parse and verify certificate first
                        test_cert = x509.load_pem_x509_certificate(raw_data)
                        
                        # Extract and validate base64 content
                        content = b''
                        in_content = False
                        for line in raw_data.split(b'\n'):
                            line = line.strip()
                            if b'-----BEGIN ' in line:
                                in_content = True
                                continue
                            elif b'-----END ' in line:
                                break
                            elif in_content and line:
                                # Validate each base64 line
                                if not validate_base64_content(line):
                                    raise ValueError(f"Invalid base64 line in {cert_type} certificate")
                                content += line
                        
                        if not content:
                            raise ValueError(f"No valid content found in {cert_type} certificate")
                        
                        # Verify content can be decoded
                        try:
                            from base64 import b64decode
                            b64decode(content)
                        except Exception as e:
                            raise ValueError(f"Base64 decoding failed for {cert_type} certificate: {e}")
                        
                        logger.debug(f"Cleaned {cert_type} certificate: {len(content)} bytes")
                        return content, test_cert
                        
                    except Exception as e:
                        logger.error(f"Failed to clean {cert_type} certificate: {e}")
                        raise ValueError(f"Invalid {cert_type} certificate format: {str(e)}")
                
                # Clean and validate both certificates
                host_content, host_cert = clean_cert(host_cert_raw, "host")
                ca_content, ca_cert = clean_cert(ca_cert_raw, "CA")
                    
                # Format certificates with strict PEM encoding
                def format_cert_pem(content: bytes, marker_type: bytes) -> bytes:
                    """Format certificate content as PEM with strict validation."""
                    return format_pem_block(content, marker_type)
                
                # Assemble certificate chain with explicit line endings
                chain_data = bytearray()
                
                # Add host certificate
                chain_data.extend(format_cert_pem(host_content, b"CERTIFICATE"))
                # Add double newline separator
                chain_data.extend(b"\r\n\r\n")
                # Add CA certificate
                chain_data.extend(format_cert_pem(ca_content, b"CERTIFICATE"))
                # Add final newline
                chain_data.extend(b"\r\n")
                
                chain_data = bytes(chain_data)
                
                # Extra validation
                begin_markers = chain_data.count(b'-----BEGIN CERTIFICATE-----')
                end_markers = chain_data.count(b'-----END CERTIFICATE-----')
                logger.debug(f"Chain markers: {begin_markers} BEGIN, {end_markers} END")
                
                if begin_markers != 2 or end_markers != 2:
                    raise ValueError(f"Invalid chain format: found {begin_markers} BEGIN and {end_markers} END markers")
                
                # Verify the complete chain can be parsed
                try:
                    test_certs = []
                    remaining = chain_data
                    while b'-----BEGIN CERTIFICATE-----' in remaining:
                        cert_der, remaining = remaining.split(b'-----END CERTIFICATE-----', 1)
                        if not cert_der:
                            continue
                        cert_der += b'-----END CERTIFICATE-----'
                        test_cert = x509.load_pem_x509_certificate(cert_der)
                        test_certs.append(test_cert)
                        logger.debug(f"Successfully parsed certificate from chain: {test_cert.subject}")
                except Exception as e:
                    logger.error("Failed to parse complete chain:", exc_info=True)
                    raise ValueError(f"Chain assembly validation failed: {str(e)}")
                
                if len(test_certs) != 2:
                    raise ValueError(f"Expected 2 certificates in chain, found {len(test_certs)}")
                    
                logger.debug("Certificate chain assembly completed successfully")
                debug_pem_content(chain_data, "Final assembled chain")
                    
            except Exception as e:
                logger.error("Failed to format certificate chain: %s", e)
                # Log raw certificate details for debugging
                logger.debug("Raw host certificate: %d bytes", len(host_cert_raw))
                logger.debug("Raw CA certificate: %d bytes", len(ca_cert_raw))
                logger.debug("Cleaned host content: %d bytes", len(host_content) if 'host_content' in locals() else 0)
                logger.debug("Cleaned CA content: %d bytes", len(ca_content) if 'ca_content' in locals() else 0)
                if 'chain_data' in locals():
                    debug_pem_content(chain_data, "Failed chain assembly")
                raise ValueError(f"Invalid certificate chain format: {str(e)}")
            
            # Write chain with secure permissions
            # Write chain with explicit fsync to ensure persistence
            cert_fd = os.open(cert_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.fchmod(cert_fd, 0o644)
            with os.fdopen(cert_fd, 'wb') as f:
                f.write(chain_data)
                f.flush()
                os.fsync(f.fileno())

            # Verify the chain was written correctly
            with open(cert_path, 'rb') as f:
                written_data = f.read()
                if written_data != chain_data:
                    raise ValueError("Certificate chain verification failed after write")
                logger.debug("Certificate chain write verified successfully")
            
        except Exception as e:
            logger.error("Failed to format certificates: %s", e)
            if os.path.exists(cert_path):
                os.unlink(cert_path)
            raise ValueError(f"Failed to format certificates: {str(e)}")

        # Generate and save private key
        try:
            # Force RSA PRIVATE KEY format for best compatibility
            key_pem = key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,  # This gives us RSA PRIVATE KEY format
                encryption_algorithm=serialization.NoEncryption()
            ).replace(b'\r\n', b'\n')  # Normalize line endings
            
            debug_pem_content(key_pem, "Raw RSA private key")
            
            key_fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.fchmod(key_fd, 0o600)
            with os.fdopen(key_fd, 'wb') as f:
                f.write(key_pem)
                f.flush()
                os.fsync(f.fileno())

            # Verify the key was written correctly
            with open(key_path, 'rb') as f:
                written_key = f.read()
                if written_key != key_pem:
                    raise ValueError("Private key verification failed after write")
                logger.debug("Private key write verified successfully")
                
        except Exception as e:
            logger.error("Failed to save private key: %s", e)
            os.unlink(cert_path)
            if os.path.exists(key_path):
                os.unlink(key_path)
            raise ValueError(f"Failed to save private key: {str(e)}")

        # Verify the generated files
        try:
            # Verify certificate chain
            with open(cert_path, 'rb') as f:
                cert_data = f.read()
                debug_pem_content(cert_data, "Raw certificate chain")
                
                # Split and parse certificates
                formatted = format_pem(cert_data)
                cert_parts = formatted.split(b'-----END CERTIFICATE-----')
                cert_blocks = []
                for part in cert_parts:
                    if b'-----BEGIN CERTIFICATE-----' in part:
                        cert_blocks.append(part.strip() + b'-----END CERTIFICATE-----')
                
                logger.debug("Found %d certificates in chain", len(cert_blocks))
                
                # Parse each certificate with retries
                certs = []
                for i, block in enumerate(cert_blocks):
                    tries = 3
                    last_error = None
                    while tries > 0:
                        try:
                            cert = x509.load_pem_x509_certificate(block)
                            cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
                            algo = cert.signature_algorithm_oid._name
                            logger.debug("Certificate %d: %s, signed with %s", i + 1, cn, algo)
                            certs.append(cert)
                            break
                        except Exception as e:
                            last_error = e
                            tries -= 1
                            if tries > 0:
                                logger.warning("Retrying certificate %d parse, error: %s", i + 1, e)
                    
                    if tries == 0:
                        logger.error("Failed to load certificate %d after retries: %s", i + 1, last_error)
                        raise ValueError(f"Invalid certificate format at position {i+1}: {last_error}")
                        
                if len(certs) != 2:
                    raise ValueError(f"Expected 2 certificates in chain, found {len(certs)}")
                
                # Verify chain order and validity
                if certs[0].issuer != certs[1].subject:
                    raise ValueError("Certificate chain is in wrong order")
                    
                # Verify certificate validity with proper timezone handling
                now = datetime.utcnow()
                # Add buffer for TLS 1.0 compatibility
                for cert_idx, cert in enumerate(certs):
                    try:
                        # Use UTC timestamps for comparison
                        not_valid_before = cert.not_valid_before.replace(tzinfo=None)
                        not_valid_after = cert.not_valid_after.replace(tzinfo=None)
                        
                        # Add 2-minute buffer for legacy TLS
                        if legacy_mode:
                            not_valid_before = not_valid_before - timedelta(minutes=2)
                            not_valid_after = not_valid_after + timedelta(minutes=2)
                            
                        if now < not_valid_before or now > not_valid_after:
                            raise ValueError(
                                f"Certificate {cert_idx + 1} validity error: "
                                f"Current time: {now}, "
                                f"Valid from {not_valid_before} to {not_valid_after}"
                            )
                        logger.debug(
                            f"Certificate {cert_idx + 1} validity verified: "
                            f"current={now}, valid={not_valid_before} to {not_valid_after}"
                        )
                    except Exception as e:
                        logger.error(f"Certificate {cert_idx + 1} validation failed: {e}")
                        raise ValueError(f"Invalid certificate timing: {str(e)}")

            # Verify private key and certificate compatibility
            with open(key_path, 'rb') as f:
                key_content = f.read()
                debug_pem_content(key_content, "Raw private key")
                
                pkey = serialization.load_pem_private_key(key_content, password=None)
                if not isinstance(pkey, rsa.RSAPrivateKey):
                    raise ValueError("Private key must be RSA")
                    
                # Verify key matches certificate
                cert_key_der = certs[0].public_key().public_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                key_der = pkey.public_key().public_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                if cert_key_der != key_der:
                    raise ValueError("Private key does not match certificate")
                    
            logger.debug("Successfully verified certificate chain and key for %s", hostname)
            
        except Exception as e:
            logger.error("Certificate verification failed: %s", e)
            if os.path.exists(cert_path):
                os.unlink(cert_path)
            if os.path.exists(key_path):
                os.unlink(key_path)
            raise ValueError(f"Certificate verification failed: {str(e)}")

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
            self.cleanup_old_certs(max_age_days=0)
            if self._cert_cache_dir.exists():
                self._cert_cache_dir.rmdir()
        except:
            pass
