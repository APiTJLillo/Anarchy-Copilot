"""TLS context creation and configuration."""
import sys
import ssl
import socket
import os
from typing import Tuple, Optional, List, Dict, Union, Any
import logging
from dataclasses import dataclass, field
from pathlib import Path
from cryptography import x509
from cryptography.x509 import (
    DNSName, ExtensionOID, Certificate, CertificateRevocationList,
    NameAttribute, Name
)
from cryptography.x509.extensions import Extension
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.x509.oid import NameOID, ExtensionOID
from datetime import datetime, timezone
import re

logger = logging.getLogger("proxy.core")

# Add missing constants in Python 3.10
if not hasattr(ssl, 'SESS_CACHE_OFF'):
    ssl.SESS_CACHE_OFF = 0
    ssl.SESS_CACHE_CLIENT = 1
    ssl.SESS_CACHE_SERVER = 2
    ssl.SESS_CACHE_BOTH = 3

class SessionCache:
    """Session cache modes."""
    OFF = ssl.SESS_CACHE_OFF
    CLIENT = ssl.SESS_CACHE_CLIENT
    SERVER = ssl.SESS_CACHE_SERVER
    BOTH = ssl.SESS_CACHE_BOTH

@dataclass
class TlsConfig:
    """TLS configuration settings."""
    minimum_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_2
    cipher_string: str = (
        # TLS 1.3 ciphers
        "TLS_AES_256_GCM_SHA384:"
        "TLS_CHACHA20_POLY1305_SHA256:"
        "TLS_AES_128_GCM_SHA256:"
        # TLS 1.2 ciphers
        "ECDHE-ECDSA-AES256-GCM-SHA384:"
        "ECDHE-RSA-AES256-GCM-SHA384:"
        "ECDHE-ECDSA-CHACHA20-POLY1305:"
        "ECDHE-RSA-CHACHA20-POLY1305:"
        "ECDHE-ECDSA-AES128-GCM-SHA256:"
        "ECDHE-RSA-AES128-GCM-SHA256:"
        # Fallback ciphers
        "DHE-RSA-AES256-GCM-SHA384:"
        "DHE-RSA-AES128-GCM-SHA256"
    )
    verify_mode: ssl.VerifyMode = ssl.CERT_NONE
    check_hostname: bool = False  # Changed to False for intercepting proxy
    session_tickets: bool = True
    session_cache_mode: int = SessionCache.BOTH
    alpn_protocols: list[str] = field(default_factory=lambda: ['h2', 'http/1.1'])

class CertificateError(ssl.SSLError):
    """Custom certificate error."""
    pass

class TestModeContext:
    """Wrapper for test mode SSL context."""
    def __init__(self, real_context: ssl.SSLContext):
        self._ctx = real_context
        self._test_mode = True
        self._session_reused = False
        self._verify_mode = ssl.CERT_NONE
        self._check_hostname = True
        self.session_cache_mode = SessionCache.BOTH
        self._session_cache: Dict[bytes, bytes] = {}
        self._certs: List[Certificate] = []

    def __getattr__(self, name):
        if name == 'session_reused':
            return lambda: self._session_reused
        if name == 'get_certs':
            return lambda: self._certs
        return getattr(self._ctx, name)

class CertificateHelper:
    """Helper class for certificate operations."""

    @staticmethod 
    def get_subject_info(cert: Certificate) -> Dict[str, Any]:
        """Get certificate subject information."""
        try:
            info = []
            subject = cert.subject
            for attr in subject:
                oid = attr.oid
                if hasattr(oid, '_name'):
                    info.append([(oid._name, attr.value)])
            return {"subject": info} if info else {}
        except Exception as e:
            logger.error(f"Error getting subject info: {e}")
            return {}

    @staticmethod
    def get_subject_cn(cert: Certificate) -> Optional[str]:
        """Get certificate subject Common Name."""
        try:
            if isinstance(cert, x509.Certificate):
                attrs = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
                if attrs:
                    return attrs[0].value
            return None
        except Exception as e:
            logger.error(f"Error getting subject CN: {e}")
            return None

    @staticmethod
    def get_subject_alt_names(cert: Certificate) -> List[Tuple[str, str]]:
        """Get certificate Subject Alternative Names."""
        names = []
        try:
            ext = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            if ext and ext.value:
                for name in ext.value:
                    if isinstance(name, DNSName):
                        names.append((name.value, "DNSName"))
        except x509.extensions.ExtensionNotFound:
            pass  # SANs are optional
        except Exception as e:
            logger.error(f"Error getting subject alt names: {e}")
        return names

    @staticmethod
    def is_valid(cert: Certificate) -> bool:
        """Check if certificate is currently valid."""
        try:
            now = datetime.now(timezone.utc)
            not_before = cert.not_valid_before_utc if hasattr(cert, 'not_valid_before_utc') else cert.not_valid_before
            not_after = cert.not_valid_after_utc if hasattr(cert, 'not_valid_after_utc') else cert.not_valid_after
            return not_before <= now <= not_after
        except Exception as e:
            logger.error(f"Error checking certificate validity: {e}")
            return False

    @staticmethod 
    def has_expired(crl: CertificateRevocationList) -> bool:
        """Check if CRL has expired."""
        try:
            now = datetime.now(timezone.utc)
            return not crl.next_update or now >= crl.next_update
        except Exception as e:
            logger.error(f"Error checking CRL expiry: {e}")
            return True

    @staticmethod
    def is_revoked(cert: Certificate, crl: CertificateRevocationList) -> bool:
        """Check if certificate is revoked."""
        try:
            if CertificateHelper.has_expired(crl):
                raise ssl.SSLError("CRL has expired")

            if hasattr(crl, 'get_revoked_certificate_by_serial_number'):
                revoked = crl.get_revoked_certificate_by_serial_number(cert.serial_number)
                if revoked:
                    raise ssl.SSLError(f"Certificate {cert.serial_number} has been revoked")
            else:
                for rev_cert in crl:
                    if rev_cert.serial_number == cert.serial_number:
                        raise ssl.SSLError(f"Certificate {cert.serial_number} has been revoked")
            return False
        except ssl.SSLError:
            raise
        except Exception as e:
            logger.error(f"Error checking certificate revocation: {e}")
            raise ssl.SSLError(str(e))

    @staticmethod
    def extract_cert_info(cert: Certificate) -> Dict[str, Any]:
        """Extract key information from a certificate."""
        return {
            "subject": CertificateHelper.get_subject_info(cert),
            "san": CertificateHelper.get_subject_alt_names(cert),
            "not_before": cert.not_valid_before_utc if hasattr(cert, 'not_valid_before_utc') else cert.not_valid_before,
            "not_after": cert.not_valid_after_utc if hasattr(cert, 'not_valid_after_utc') else cert.not_valid_after,
            "serial_number": str(cert.serial_number),
            "common_name": CertificateHelper.get_subject_cn(cert)
        }

class TLSContext:
    """Enhanced SSLContext with additional functionality."""
    
    def __init__(self, ssl_context: ssl.SSLContext):
        self._ctx = ssl_context
        self._certs: List[Certificate] = []
        self._session_cache: Dict[bytes, bytes] = {}
        self._session_cache_mode = SessionCache.OFF
        self._server_hostname = None
        self._is_client = isinstance(ssl_context, ssl.SSLContext) and ssl_context.verify_mode != ssl.CERT_NONE
        self._crls: List[CertificateRevocationList] = []
        self._test_mode = False
        self._session_reused = False

    def enable_test_mode(self) -> None:
        """Enable test mode."""
        self._test_mode = True
        self._ctx = TestModeContext(self._ctx)

    @property
    def session_reused(self) -> bool:
        """Check if session was reused."""
        if self._test_mode:
            return bool(self._session_reused)
        try:
            reused = getattr(self._ctx, 'session_reused', None)
            if reused is None:
                return False
            return bool(reused() if callable(reused) else reused)
        except (AttributeError, TypeError):
            return self._session_reused

    def get_server_certificate(self) -> Optional[str]:
        """Get server certificate in PEM format."""
        if not self._certs:
            return None
        try:
            cert = self._certs[0]
            return cert.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode('utf-8')
        except Exception as e:
            logger.error(f"Error getting server certificate: {e}")
            return None

    def get_certs(self) -> List[Certificate]:
        """Get all certificates."""
        return self._certs.copy()  # Return a copy to prevent external modification

    def add_cert(self, cert_path: str) -> None:
        """Add a certificate from file."""
        try:
            with open(cert_path, 'rb') as f:
                cert_data = f.read()
                cert = x509.load_pem_x509_certificate(cert_data, default_backend())
                if not CertificateHelper.is_valid(cert):
                    raise CertificateError("Certificate is not valid")
                self._certs.append(cert)
        except Exception as e:
            logger.error(f"Failed to load certificate: {e}")
            raise ssl.SSLError(str(e))

    def __getattr__(self, name):
        """Forward unknown attributes to the underlying SSLContext."""
        return getattr(self._ctx, name)

    def set_session_cache_mode(self, mode: int) -> None:
        """Set the session cache mode.
        
        Args:
            mode: One of the SessionCache mode constants
        """
        self._session_cache_mode = mode
        # Store mode in our wrapper
        if hasattr(self._ctx, 'set_session_cache_mode'):
            self._ctx.set_session_cache_mode(mode)
        # If the underlying context doesn't support it, we'll handle it ourselves
        
    def store_session(self, session_id: bytes, session_data: bytes) -> None:
        """Store a session in the cache."""
        if self._session_cache_mode in (SessionCache.SERVER, SessionCache.BOTH):
            self._session_cache[session_id] = session_data
            
    def get_session(self, session_id: bytes) -> Optional[bytes]:
        """Get a session from the cache."""
        if self._session_cache_mode in (SessionCache.CLIENT, SessionCache.BOTH):
            return self._session_cache.get(session_id)

    def verify_certificate(self, cert: Certificate) -> bool:
        """Verify certificate validity and revocation status."""
        try:
            if not CertificateHelper.is_valid(cert):
                raise ssl.SSLError("Certificate is not valid")
                
            for crl in self._crls:
                if CertificateHelper.is_revoked(cert, crl):
                    return False
                    
            return True
        except ssl.SSLError:
            raise
        except Exception as e:
            logger.error(f"Certificate verification error: {e}")
            raise ssl.SSLError(str(e))

    def close(self) -> None:
        """Close context and cleanup resources."""
        if hasattr(self._ctx, 'close'):
            self._ctx.close()
        self._session_cache.clear()
        self._certs.clear()
        self._crls.clear()

class TlsContextFactory:
    """Factory for creating TLS contexts."""

    def __init__(self, config: Optional[TlsConfig] = None):
        self.config = config or TlsConfig()

    def create_server_context(self, cert_path: Union[str, Path], key_path: Union[str, Path]) -> TLSContext:
        """Create a server-side SSL context."""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        
        # Convert paths to strings and resolve them
        cert_path = str(Path(cert_path).resolve())
        key_path = str(Path(key_path).resolve())
        
        # Verify files exist
        if not os.path.exists(cert_path):
            raise CertificateError(f"Certificate file not found: {cert_path}")
        if not os.path.exists(key_path):
            raise CertificateError(f"Private key file not found: {key_path}")
        
        # Configure TLS version and ciphers
        context.minimum_version = self.config.minimum_version
        context.set_ciphers(self.config.cipher_string)
        
        # Load certificate and private key with detailed error handling
        try:
            context.load_cert_chain(cert_path, key_path)
            logger.info(f"Successfully loaded certificate from {cert_path}")
        except (ssl.SSLError, IOError) as e:
            logger.error(f"Failed to load certificate: {e}")
            raise CertificateError(f"Failed to load certificate: {e}")
            
        # Configure verification
        context.verify_mode = self.config.verify_mode
        context.check_hostname = self.config.check_hostname
        
        # Enable session management
        context.options |= ssl.OP_NO_TICKET
        
        # Set ALPN protocols
        if self.config.alpn_protocols:
            context.set_alpn_protocols(self.config.alpn_protocols)
            
        # Create TLSContext wrapper and configure session cache
        tls_context = TLSContext(context)
        tls_context.set_session_cache_mode(self.config.session_cache_mode)
        return tls_context

    def create_client_context(self, server_hostname: str) -> TLSContext:
        """Create a client-side SSL context."""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        
        # Configure TLS version and ciphers
        context.minimum_version = self.config.minimum_version
        context.set_ciphers(self.config.cipher_string)
        
        # Configure verification settings for client
        # Must set check_hostname to False before setting verify_mode to CERT_NONE
        context.check_hostname = False  # For intercepting proxy
        context.verify_mode = ssl.CERT_NONE  # For intercepting proxy
        
        # Enable session management
        if self.config.session_tickets:
            context.options &= ~ssl.OP_NO_TICKET
        
        # Set ALPN protocols
        if self.config.alpn_protocols:
            context.set_alpn_protocols(self.config.alpn_protocols)
            
        # Create TLSContext wrapper and configure session cache
        tls_context = TLSContext(context)
        tls_context.set_session_cache_mode(self.config.session_cache_mode)
        return tls_context

    def verify_hostname(self, cert: Certificate, hostname: str) -> bool:
        """Verify certificate hostname against SAN or CN."""
        try:
            if not cert:
                return False

            # First check Subject Alternative Names
            san_found = False
            for name, name_type in CertificateHelper.get_subject_alt_names(cert):
                if name_type == "DNSName" and self._hostname_matches(name, hostname):
                    san_found = True
                    break

            # Fall back to Common Name if no SAN matched
            if not san_found:
                cn = CertificateHelper.get_subject_cn(cert)
                if cn and self._hostname_matches(cn, hostname):
                    return True

            return san_found
        except Exception as e:
            logger.error(f"Hostname verification error: {e}")
            return False

    def _hostname_matches(self, pattern: str, hostname: str) -> bool:
        """Match hostname against pattern with wildcard support."""
        if not pattern or not hostname:
            return False

        hostname = hostname.lower()
        pattern = pattern.lower()
        
        if pattern.startswith("*."):
            # Wildcard matching
            suffix = pattern[2:]
            if not suffix or '*' in suffix:
                return False
            if '*' in hostname:
                return False
            if '.' not in hostname:
                return False
            return hostname.endswith(suffix) and hostname.count('.') == suffix.count('.') + 1
        
        return pattern == hostname

# Global instance with default configuration
context_factory = TlsContextFactory()
