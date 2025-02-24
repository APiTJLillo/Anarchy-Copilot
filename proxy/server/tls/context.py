"""TLS context creation and configuration."""
import sys
import ssl
import socket
from typing import Tuple, Optional, List, Dict, Union, Any
import logging
from dataclasses import dataclass
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
        "ECDHE-ECDSA-AES256-GCM-SHA384:"
        "ECDHE-RSA-AES256-GCM-SHA384:"
        "ECDHE-ECDSA-CHACHA20-POLY1305:"
        "ECDHE-RSA-CHACHA20-POLY1305"
    )
    verify_mode: ssl.VerifyMode = ssl.CERT_NONE
    check_hostname: bool = True
    session_tickets: bool = True
    session_cache_mode: int = SessionCache.BOTH
    alpn_protocols: list[str] = None

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
        return getattr(self._ctx, name)

    def set_session_cache_mode(self, mode: int) -> None:
        """Set session cache mode."""
        if mode not in (SessionCache.OFF, SessionCache.CLIENT, SessionCache.SERVER, SessionCache.BOTH):
            raise ValueError("Invalid session cache mode")
        self._session_cache_mode = mode
        if mode == SessionCache.OFF:
            self._session_cache.clear()
            self._session_reused = False

    def store_session(self, session_id: bytes, session_data: bytes) -> None:
        """Store session data."""
        if self._session_cache_mode == SessionCache.OFF:
            return
        self._session_cache[session_id] = session_data
        self._session_reused = True

    def get_session(self, session_id: bytes) -> Optional[bytes]:
        """Retrieve session data."""
        if self._session_cache_mode == SessionCache.OFF:
            return None
        data = self._session_cache.get(session_id)
        if data:
            self._session_reused = True
        return data

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
    """Factory for creating TLS contexts with enhanced functionality."""
    
    def __init__(self, config: Optional[TlsConfig] = None):
        self.config = config or TlsConfig()
        self.config.alpn_protocols = self.config.alpn_protocols or ['h2', 'http/1.1']

    @staticmethod
    def get_connection_info(ssl_object: ssl.SSLSocket) -> Dict[str, Any]:
        """Extract TLS connection information."""
        try:
            info = {
                "version": None,
                "cipher": {"name": None, "version": None, "bits": None},
                "compression": None,
                "alpn_protocol": None,
                "server_hostname": None,
                "peer_certificate": None,
                "peer_cert": None,
                "session_reused": False
            }

            # Get version
            try:
                ver = ssl_object.version()
                info["version"] = ver() if callable(ver) else ver
            except (AttributeError, TypeError):
                pass

            # Get cipher info
            try:
                cipher = ssl_object.cipher()
                if cipher and len(cipher) >= 3:
                    info["cipher"].update({
                        "name": cipher[0],
                        "version": cipher[1],
                        "bits": cipher[2]
                    })
            except (AttributeError, TypeError):
                pass

            # Get compression and ALPN
            try:
                info["compression"] = ssl_object.compression()
            except AttributeError:
                pass

            try:
                if hasattr(ssl_object, 'selected_alpn_protocol'):
                    info["alpn_protocol"] = ssl_object.selected_alpn_protocol()
            except AttributeError:
                pass

            # Get server hostname
            info["server_hostname"] = getattr(ssl_object, 'server_hostname', None)

            # Get peer certificate
            try:
                cert = None
                if hasattr(ssl_object, 'getpeercert'):
                    der_cert = ssl_object.getpeercert(binary_form=True)
                    if der_cert:
                        cert = x509.load_der_x509_certificate(der_cert, default_backend())
                elif hasattr(ssl_object, 'get_peer_certificate'):
                    cert_data = ssl_object.get_peer_certificate()
                    if cert_data:
                        cert = x509.load_der_x509_certificate(cert_data, default_backend())
                
                if cert:
                    cert_info = CertificateHelper.extract_cert_info(cert)
                    info["peer_certificate"] = cert_info
                    info["peer_cert"] = cert_info  # Keep both for compatibility
            except Exception as e:
                logger.error(f"Error getting peer certificate: {e}")
                info["peer_certificate"] = None
                info["peer_cert"] = None

            # Get session reuse info
            try:
                reused = getattr(ssl_object, 'session_reused', None)
                info["session_reused"] = bool(reused() if callable(reused) else reused if reused is not None else False)
            except (AttributeError, TypeError):
                pass

            return info
        except Exception as e:
            logger.error(f"Error getting connection info: {e}")
            return {}

    def create_server_context(self, cert_path: str, key_path: str) -> TLSContext:
        """Create server-side SSL context with modern security settings."""
        try:
            if not Path(cert_path).exists() or not Path(key_path).exists():
                raise FileNotFoundError(f"Certificate or key file not found: {cert_path}, {key_path}")
            
            ctx = TLSContext(ssl.create_default_context(ssl.Purpose.CLIENT_AUTH))
            ctx._ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
            ctx.add_cert(cert_path)
            
            # Version handling
            ctx._ctx.minimum_version = self.config.minimum_version
            if ctx._ctx.minimum_version < ssl.TLSVersion.TLSv1_2:
                raise ValueError("TLS version must be 1.2 or higher")
            ctx._ctx.maximum_version = ssl.TLSVersion.TLSv1_3
            ctx._ctx.set_ciphers(self.config.cipher_string)
            
            # Security options
            ctx._ctx.options |= (
                ssl.OP_NO_COMPRESSION |
                ssl.OP_SINGLE_DH_USE |
                ssl.OP_SINGLE_ECDH_USE |
                ssl.OP_NO_RENEGOTIATION |
                ssl.OP_NO_TLSv1 |
                ssl.OP_NO_TLSv1_1
            )
            
            # Configure ALPN
            if hasattr(ssl, 'HAS_ALPN') and ssl.HAS_ALPN:
                ctx._ctx.set_alpn_protocols(self.config.alpn_protocols)
            
            # Configure session handling
            ctx._ctx.options |= ssl.OP_NO_TICKET if not self.config.session_tickets else 0
            ctx.set_session_cache_mode(self.config.session_cache_mode)
            
            return ctx
        except Exception as e:
            logger.error(f"Failed to create server TLS context: {e}")
            raise

    def create_client_context(self, server_hostname: str) -> TLSContext:
        """Create client-side SSL context."""
        try:
            ctx = TLSContext(ssl.create_default_context())
            ctx._server_hostname = server_hostname
            
            # Version handling
            ctx._ctx.minimum_version = self.config.minimum_version
            if ctx._ctx.minimum_version < ssl.TLSVersion.TLSv1_2:
                raise ValueError("TLS version must be 1.2 or higher")
            ctx._ctx.maximum_version = ssl.TLSVersion.TLSv1_3
            ctx._ctx.set_ciphers(self.config.cipher_string)
            
            # Configure hostname verification and cert validation
            ctx._ctx.check_hostname = self.config.check_hostname
            if self.config.check_hostname:
                ctx._ctx.verify_mode = ssl.CERT_REQUIRED
                ctx._ctx.verify_flags = ssl.VERIFY_X509_STRICT
            else:
                ctx._ctx.verify_mode = self.config.verify_mode
            
            # Configure session handling
            ctx._ctx.options |= ssl.OP_NO_TICKET if not self.config.session_tickets else 0
            ctx.set_session_cache_mode(self.config.session_cache_mode)
            
            # Configure ALPN
            if hasattr(ssl, 'HAS_ALPN') and ssl.HAS_ALPN:
                ctx._ctx.set_alpn_protocols(self.config.alpn_protocols)

            return ctx
        except Exception as e:
            logger.error(f"Failed to create client TLS context: {e}")
            raise

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
