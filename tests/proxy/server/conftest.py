"""Test fixtures for proxy server components."""
import pytest
import asyncio
import ssl
from typing import AsyncGenerator, Tuple
from uuid import uuid4

from proxy.server.tls.connection_manager import ConnectionManager
from proxy.server.tls.context import TlsConfig, TlsContextFactory
from proxy.server.tls.transport import BufferedTransport, BufferConfig
from proxy.server.certificates import CertificateAuthority

@pytest.fixture
def connection_id() -> str:
    """Generate a unique connection ID."""
    return str(uuid4())

@pytest.fixture
def connection_manager() -> ConnectionManager:
    """Create a fresh connection manager instance."""
    return ConnectionManager()

@pytest.fixture
def buffer_config() -> BufferConfig:
    """Create test buffer configuration."""
    return BufferConfig(
        chunk_size=1024,  # 1KB chunks for testing
        max_buffer_size=4096,  # 4KB total buffer
        write_buffer_size=2048,  # 2KB write buffer
        high_water_mark=0.8,
        low_water_mark=0.2
    )

@pytest.fixture
def tls_config() -> TlsConfig:
    """Create test TLS configuration."""
    return TlsConfig(
        minimum_version=ssl.TLSVersion.TLSv1_2,
        cipher_string="ECDHE-RSA-AES128-GCM-SHA256",
        verify_mode=ssl.CERT_NONE,
        check_hostname=False
    )

@pytest.fixture
async def echo_server(unused_tcp_port: int) -> AsyncGenerator[Tuple[str, int], None]:
    """Create an echo server for testing connections."""
    connections = []
    
    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            while True:
                data = await reader.read(1024)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(
        handle_client,
        '127.0.0.1',
        unused_tcp_port
    )
    
    async with server:
        yield ('127.0.0.1', unused_tcp_port)

class TestCertificateAuthority(CertificateAuthority):
    """Extended CA class with additional test functionality."""
    
    def create_crl(self, revoked_certs, crl_path):
        """Create a Certificate Revocation List."""
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives import serialization
        from datetime import datetime, timedelta

        builder = x509.CertificateRevocationListBuilder()
        builder = builder.issuer_name(self.ca_cert.subject)
        builder = builder.last_update(datetime.utcnow())
        builder = builder.next_update(datetime.utcnow() + timedelta(days=1))

        # Add revoked certificates
        for cert_path in revoked_certs:
            with open(cert_path, 'rb') as f:
                cert_data = f.read()
            cert = x509.load_pem_x509_certificate(cert_data)
            revoked_cert = x509.RevokedCertificateBuilder(
            ).serial_number(
                cert.serial_number
            ).revocation_date(
                datetime.utcnow()
            ).build()
            builder = builder.add_revoked_certificate(revoked_cert)

        crl = builder.sign(
            private_key=self.ca_key,
            algorithm=hashes.SHA256()
        )

        with open(crl_path, 'wb') as f:
            f.write(crl.public_bytes(serialization.Encoding.PEM))

@pytest.fixture
def ca_handler(tmp_path) -> TestCertificateAuthority:
    """Create a test CA for certificate generation."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    from datetime import datetime, timedelta

    # Generate key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Generate cert
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, u"Test CA"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.utcnow()
    ).not_valid_after(
        datetime.utcnow() + timedelta(days=365)
    ).add_extension(
        x509.BasicConstraints(ca=True, path_length=None),
        critical=True,
    ).sign(private_key, hashes.SHA256())

    # Write cert and key
    cert_path = tmp_path / "ca.crt"
    key_path = tmp_path / "ca.key"

    with open(str(cert_path), "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    with open(str(key_path), "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    return TestCertificateAuthority(str(cert_path), str(key_path))

@pytest.fixture
def mock_transport(mocker):
    """Create a mock transport."""
    transport = mocker.MagicMock()
    transport.is_closing.return_value = False
    transport.get_extra_info.return_value = ('127.0.0.1', 12345)
    return transport

@pytest.fixture
def buffered_transport(connection_id: str, buffer_config: BufferConfig, 
                      mock_transport) -> BufferedTransport:
    """Create a buffered transport instance."""
    transport = BufferedTransport(
        connection_id=connection_id,
        target_transport=mock_transport,
        config=buffer_config
    )
    transport.connection_made(mock_transport)
    return transport

@pytest.fixture
def context_factory(tls_config: TlsConfig) -> TlsContextFactory:
    """Create a TLS context factory."""
    return TlsContextFactory(config=tls_config)

@pytest.fixture
def test_data() -> bytes:
    """Generate test data of various sizes."""
    return b"".join([
        b"Small data",
        b"Medium " * 100,  # ~600 bytes
        b"Large " * 1000,  # ~5000 bytes
    ])

@pytest.fixture
def mock_ssl_object(mocker):
    """Create a mock SSL object with required methods."""
    ssl_obj = mocker.MagicMock()
    ssl_obj.version.return_value = "TLSv1.2"
    ssl_obj.cipher.return_value = ("ECDHE-RSA-AES128-GCM-SHA256", "TLSv1.2", 128)
    ssl_obj.compression.return_value = None
    ssl_obj.getpeercert.return_value = {
        'subject': ((('commonName', 'example.com'),),),
        'issuer': ((('commonName', 'Test CA'),),),
        'version': 3,
        'serialNumber': '1234',
        'notBefore': 'Jan 1 00:00:00 2020 GMT',
        'notAfter': 'Jan 1 00:00:00 2025 GMT',
    }
    ssl_obj.selected_alpn_protocol.return_value = "http/1.1"
    ssl_obj.selected_npn_protocol.return_value = None
    ssl_obj.get_channel_binding.return_value = b"test-binding"
    ssl_obj.shared_ciphers.return_value = [("ECDHE-RSA-AES128-GCM-SHA256", "TLSv1.2", 128)]
    ssl_obj.get_ciphers.return_value = [{"name": "ECDHE-RSA-AES128-GCM-SHA256", "protocol": "TLSv1.2", "bits": 128}]
    return ssl_obj

@pytest.fixture
def initial_cert_config(tmp_path):
    """Create initial certificate configuration."""
    from cryptography import x509
    from cryptography.x509 import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    from datetime import datetime, timedelta

    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )

    cert = x509.CertificateBuilder().subject_name(
        x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u'Test CA'),
        ])
    ).issuer_name(
        x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u'Test CA'),
        ])
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.utcnow()
    ).not_valid_after(
        datetime.utcnow() + timedelta(days=365)
    ).add_extension(
        x509.BasicConstraints(ca=True, path_length=None), critical=True
    ).sign(key, hashes.SHA256())

    ca_key_path = tmp_path / "ca.key"
    ca_cert_path = tmp_path / "ca.crt"

    with open(ca_key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    with open(ca_cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    return ca_cert_path, ca_key_path
