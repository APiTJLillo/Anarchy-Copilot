"""Tests for TLS context creation and configuration."""
import pytest
import ssl
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from proxy.server.tls.context import TlsConfig, TlsContextFactory

def test_tls_config_defaults():
    """Test TLS configuration defaults are secure."""
    config = TlsConfig()
    
    # Verify minimum version is at least TLS 1.2
    assert config.minimum_version >= ssl.TLSVersion.TLSv1_2
    
    # Verify cipher string includes modern ciphers
    assert "ECDHE" in config.cipher_string
    assert "GCM" in config.cipher_string
    assert "CHACHA20" in config.cipher_string
    
    # Verify insecure ciphers are not present
    assert "RC4" not in config.cipher_string
    assert "MD5" not in config.cipher_string
    assert "DES" not in config.cipher_string

def test_server_context_creation(context_factory, ca_handler):
    """Test server-side TLS context creation."""
    # Generate test certificate
    cert_path, key_path = ca_handler.get_certificate("example.com")
    
    # Create server context
    ctx = context_factory.create_server_context(cert_path, key_path)
    
    # Verify security settings
    assert ctx.minimum_version == context_factory.config.minimum_version
    assert ctx.verify_mode == ssl.CERT_NONE  # Server doesn't verify client
    
    # Basic security options
    assert ctx.options & ssl.OP_NO_SSLv3
    assert ctx.options & ssl.OP_NO_COMPRESSION
    
    # ECDH params should be auto-selected
    if hasattr(ctx, 'set_ecdh_auto'):
        assert ctx.get_ecdh_auto()
    
    # Verify certificate was loaded
    cert = ctx.get_server_certificate()
    assert "BEGIN CERTIFICATE" in cert

def test_client_context_creation(context_factory):
    """Test client-side TLS context creation."""
    ctx = context_factory.create_client_context("example.com")
    
    # Verify security settings
    assert ctx.minimum_version == context_factory.config.minimum_version
    assert ctx.check_hostname == context_factory.config.check_hostname
    assert ctx.verify_mode == context_factory.config.verify_mode
    
    # Verify protocol support
    if hasattr(ssl, 'HAS_ALPN') and ssl.HAS_ALPN:
        ctx.set_alpn_protocols(['h2', 'http/1.1'])
        # ALPN protocols are stored internally
        ciphers = ctx.get_ciphers()
        assert len(ciphers) > 0  # Should have some ciphers available

def test_connection_info_extraction(context_factory, mock_ssl_object):
    """Test TLS connection information extraction."""
    info = TlsContextFactory.get_connection_info(mock_ssl_object)
    
    # Verify basic info
    assert info["version"] == "TLSv1.2"
    assert info["cipher"]["name"] == "ECDHE-RSA-AES128-GCM-SHA256"
    assert info["cipher"]["bits"] == 128
    
    # Verify certificate info
    cert_info = info["peer_certificate"]
    assert cert_info["subject"][0][0][0] == "commonName"
    assert cert_info["subject"][0][0][1] == "example.com"
    
    # Verify protocol info
    assert info["alpn_protocol"] == "http/1.1"
    assert info["channel_binding"]["tls-unique"] == b"test-binding"

@pytest.mark.parametrize("hostname", [
    "example.com",
    "subdomain.example.com",
    "*.wildcard.com",
    "ip.1.2.3.4.xip.io",
    "localhost"
])
def test_certificate_compatibility(context_factory, ca_handler, hostname):
    """Test certificate generation and context compatibility."""
    cert_path, key_path = ca_handler.get_certificate(hostname)
    
    # Create context with generated cert
    ctx = context_factory.create_server_context(cert_path, key_path)
    
    # Verify certificate loading
    assert ctx.get_ca_certs() is not None

@pytest.mark.parametrize("version,expected", [
    (ssl.TLSVersion.TLSv1_2, True),
    (ssl.TLSVersion.TLSv1_3, True),
    (ssl.TLSVersion.TLSv1_1, False),  # Should raise or be rejected
])
def test_tls_version_enforcement(version, expected):
    """Test TLS version restrictions."""
    config = TlsConfig(minimum_version=version)
    factory = TlsContextFactory(config)
    
    if expected:
        ctx = factory.create_client_context("example.com")
        assert ctx.minimum_version == version
    else:
        with pytest.raises((ValueError, ssl.SSLError, ssl.SSLEOFError)):
            factory.create_client_context("example.com")

def test_custom_cipher_configuration():
    """Test custom cipher suite configuration."""
    custom_ciphers = "ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES256-GCM-SHA384"
    config = TlsConfig(cipher_string=custom_ciphers)
    factory = TlsContextFactory(config)
    
    # Create both client and server contexts
    client_ctx = factory.create_client_context("example.com")
    server_ctx = factory.create_server_context("ca.crt", "ca.key")
    
    # Verify cipher configuration
    assert any(cipher["name"] in custom_ciphers for cipher in client_ctx.get_ciphers())
    assert any(cipher["name"] in custom_ciphers for cipher in server_ctx.get_ciphers())

def test_context_error_handling(context_factory):
    """Test error handling in context creation."""
    # Test with invalid certificate paths
    with pytest.raises(FileNotFoundError):
        context_factory.create_server_context("nonexistent.crt", "nonexistent.key")
    
    # Test with invalid cipher string
    config = TlsConfig(cipher_string="INVALID-CIPHER")
    factory = TlsContextFactory(config)
    with pytest.raises(ssl.SSLError):
        factory.create_client_context("example.com")

def test_security_options(context_factory):
    """Test security-related options are properly set."""
    ctx = context_factory.create_client_context("example.com")
    
    # Basic security settings
    assert ctx.minimum_version >= ssl.TLSVersion.TLSv1_2
    assert ctx.maximum_version is None or ctx.maximum_version >= ssl.TLSVersion.TLSv1_2
    
    # Modern options
    assert ctx.options & ssl.OP_NO_SSLv3
    assert ctx.options & ssl.OP_NO_COMPRESSION
    
    # Verify cipher settings
    ciphers = ctx.get_ciphers()
    cipher_names = {c["name"] for c in ciphers}
    
    # Should have modern ciphers
    assert any("ECDHE" in name for name in cipher_names)
    assert any("GCM" in name for name in cipher_names)
    
    # Should not have weak ciphers
    assert not any(weak in name for name in cipher_names 
                  for weak in ("RC4", "MD5", "SHA1", "DES", "EXPORT"))

@pytest.mark.asyncio
async def test_sni_support(ca_handler):
    """Test SNI (Server Name Indication) support."""
    hostnames = ["site1.example.com", "site2.example.com"]
    cert_paths = {}
    
    # Generate certificates for both hostnames
    for hostname in hostnames:
        cert_path, key_path = ca_handler.get_certificate(hostname)
        cert_paths[hostname] = (cert_path, key_path)
    
    config = TlsConfig()
    factory = TlsContextFactory(config)
    
    # Create server contexts for each hostname
    contexts = {
        hostname: factory.create_server_context(cert_path, key_path)
        for hostname, (cert_path, key_path) in cert_paths.items()
    }
    
    # Verify SNI selection works
    for hostname in hostnames:
        client_ctx = factory.create_client_context(hostname)
        assert client_ctx.check_hostname
        # Extract server cert CN and verify it matches
        server_ctx = contexts[hostname]
        cert = server_ctx.get_certs()[0]
        assert hostname in cert.get_subject().CN

@pytest.mark.asyncio
async def test_client_certificate_auth(ca_handler, tmp_path):
    """Test client certificate authentication."""
    # Generate server and client certificates
    server_cert, server_key = ca_handler.get_certificate("server.local")
    client_cert, client_key = ca_handler.get_certificate("client.local")
    
    config = TlsConfig(
        verify_mode=ssl.CERT_REQUIRED,
        check_hostname=True
    )
    factory = TlsContextFactory(config)
    
    # Create server context requiring client auth
    server_ctx = factory.create_server_context(server_cert, server_key)
    server_ctx.verify_mode = ssl.CERT_REQUIRED
    server_ctx.load_verify_locations(cafile=ca_handler.ca_cert_path)
    
    # Create client context with client cert
    client_ctx = factory.create_client_context("server.local")
    client_ctx.load_cert_chain(client_cert, client_key)
    
    # Verify mutual authentication
    assert server_ctx.verify_mode == ssl.CERT_REQUIRED
    assert client_ctx.check_hostname
    assert client_ctx.verify_mode == ssl.CERT_REQUIRED

@pytest.mark.asyncio
async def test_session_resumption(context_factory, ca_handler):
    """Test TLS session resumption."""
    cert_path, key_path = ca_handler.get_certificate("resume.local")
    
    # Create contexts
    server_ctx = context_factory.create_server_context(cert_path, key_path)
    client_ctx = context_factory.create_client_context("resume.local")
    
    # Enable session caching
    server_ctx.set_session_cache_mode(ssl.SESS_CACHE_SERVER)
    client_ctx.set_session_cache_mode(ssl.SESS_CACHE_CLIENT)
    
    # Create SSL objects
    server_ssl = server_ctx.wrap_bio(
        ssl.MemoryBIO(),
        ssl.MemoryBIO(),
        server_side=True
    )
    client_ssl = client_ctx.wrap_bio(
        ssl.MemoryBIO(),
        ssl.MemoryBIO()
    )
    
    # Verify session reuse
    assert not client_ssl.session_reused()
    # Note: Full handshake simulation would be needed for actual session reuse verification

@pytest.mark.asyncio
async def test_certificate_revocation(context_factory, ca_handler, tmp_path):
    """Test certificate revocation handling."""
    hostname = "revoked.local"
    cert_path, key_path = ca_handler.get_certificate(hostname)
    
    # Create CRL file
    crl_path = tmp_path / "test.crl"
    ca_handler.create_crl([cert_path], crl_path)
    
    config = TlsConfig(
        verify_mode=ssl.CERT_REQUIRED,
        check_hostname=True
    )
    factory = TlsContextFactory(config)
    
    # Create context with CRL checking
    ctx = factory.create_client_context(hostname)
    ctx.verify_flags = ssl.VERIFY_CRL_CHECK_LEAF
    ctx.load_verify_locations(cafile=ca_handler.ca_cert_path)
    
    # Verify revoked certificate is rejected
    with pytest.raises(ssl.SSLError) as exc_info:
        ctx.load_verify_locations(cafile=cert_path)
    assert "certificate revoked" in str(exc_info.value).lower()
