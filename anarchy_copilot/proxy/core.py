"""
Core proxy server implementation for Anarchy Copilot.

This module implements the main proxy server functionality, handling HTTP/HTTPS connections
and integrating with the interceptor and session management components.
"""
import asyncio
import logging
import ssl
from typing import List, Optional, Set
import aiohttp
from aiohttp import web
import certifi
import OpenSSL.crypto as crypto
from pathlib import Path

from .config import ProxyConfig
from .interceptor import (
    InterceptedRequest,
    InterceptedResponse,
    RequestInterceptor,
    ResponseInterceptor
)
from .session import ProxySession

logger = logging.getLogger(__name__)

class CertificateAuthority:
    """Manages SSL/TLS certificates for HTTPS interception."""
    
    def __init__(self, ca_cert_path: Path, ca_key_path: Path):
        """Initialize the certificate authority.
        
        Args:
            ca_cert_path: Path to the CA certificate
            ca_key_path: Path to the CA private key
        """
        self.ca_cert_path = ca_cert_path
        self.ca_key_path = ca_key_path
        self._load_or_create_ca()
    
    def _load_or_create_ca(self) -> None:
        """Load existing CA certificate and key or create new ones."""
        try:
            with open(self.ca_cert_path, 'rb') as cert_file:
                self.ca_cert = crypto.load_certificate(
                    crypto.FILETYPE_PEM, cert_file.read())
            with open(self.ca_key_path, 'rb') as key_file:
                self.ca_key = crypto.load_privatekey(
                    crypto.FILETYPE_PEM, key_file.read())
        except FileNotFoundError:
            self._create_ca()
    
    def _create_ca(self) -> None:
        """Create a new CA certificate and private key."""
        # Generate key
        self.ca_key = crypto.PKey()
        self.ca_key.generate_key(crypto.TYPE_RSA, 2048)
        
        # Generate certificate
        cert = crypto.X509()
        cert.get_subject().CN = "Anarchy Copilot CA"
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(10*365*24*60*60)  # 10 years
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(self.ca_key)
        cert.add_extensions([
            crypto.X509Extension(
                b"basicConstraints",
                True,
                b"CA:TRUE, pathlen:0"
            ),
            crypto.X509Extension(
                b"keyUsage",
                True,
                b"keyCertSign, cRLSign"
            ),
            crypto.X509Extension(
                b"subjectKeyIdentifier",
                False,
                b"hash",
                subject=cert
            ),
        ])
        cert.sign(self.ca_key, 'sha256')
        self.ca_cert = cert
        
        # Save to files
        with open(self.ca_cert_path, 'wb') as cert_file:
            cert_file.write(
                crypto.dump_certificate(crypto.FILETYPE_PEM, self.ca_cert))
        with open(self.ca_key_path, 'wb') as key_file:
            key_file.write(
                crypto.dump_privatekey(crypto.FILETYPE_PEM, self.ca_key))
    
    def generate_certificate(self, hostname: str) -> tuple[bytes, bytes]:
        """Generate a certificate for a specific hostname.
        
        Args:
            hostname: The hostname to generate a certificate for
            
        Returns:
            Tuple of (certificate bytes, private key bytes)
        """
        # Generate key
        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 2048)
        
        # Generate certificate
        cert = crypto.X509()
        cert.get_subject().CN = hostname
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(365*24*60*60)  # 1 year
        cert.set_issuer(self.ca_cert.get_subject())
        cert.set_pubkey(key)
        cert.add_extensions([
            crypto.X509Extension(
                b"basicConstraints",
                True,
                b"CA:FALSE"
            ),
            crypto.X509Extension(
                b"subjectAltName",
                False,
                f"DNS:{hostname}".encode()
            ),
        ])
        cert.sign(self.ca_key, 'sha256')
        
        return (
            crypto.dump_certificate(crypto.FILETYPE_PEM, cert),
            crypto.dump_privatekey(crypto.FILETYPE_PEM, key)
        )

class ProxyServer:
    """Main proxy server implementation."""
    
    def __init__(self, config: ProxyConfig):
        """Initialize the proxy server.
        
        Args:
            config: Proxy configuration
        """
        self.config = config
        self.session = ProxySession(max_history=config.history_size)
        self._request_interceptors: List[RequestInterceptor] = []
        self._response_interceptors: List[ResponseInterceptor] = []
        self._ssl_contexts: dict[str, ssl.SSLContext] = {}
        
        if config.ca_cert_path and config.ca_key_path:
            self._ca = CertificateAuthority(
                config.ca_cert_path,
                config.ca_key_path
            )
        else:
            self._ca = None
    
    def add_request_interceptor(self, interceptor: RequestInterceptor) -> None:
        """Add a request interceptor to the chain."""
        self._request_interceptors.append(interceptor)
    
    def add_response_interceptor(self, interceptor: ResponseInterceptor) -> None:
        """Add a response interceptor to the chain."""
        self._response_interceptors.append(interceptor)
    
    async def _get_ssl_context(self, hostname: str) -> Optional[ssl.SSLContext]:
        """Get or create an SSL context for a hostname."""
        if not self._ca:
            return None
            
        if hostname not in self._ssl_contexts:
            cert_bytes, key_bytes = self._ca.generate_certificate(hostname)
            ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ctx.load_cert_chain(
                certfile=certifi.where(),
                keyfile=None,
                password=None
            )
            self._ssl_contexts[hostname] = ctx
            
        return self._ssl_contexts[hostname]
    
    async def _handle_request(self, request: web.Request) -> web.Response:
        """Handle an incoming proxy request."""
        if not self.config.is_in_scope(request.host):
            return web.Response(
                status=403,
                text="Host not in scope"
            )
        
        # Create intercepted request
        intercepted = InterceptedRequest(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            body=await request.read()
        )
        
        # Apply request interceptors
        for interceptor in self._request_interceptors:
            try:
                intercepted = await interceptor.intercept(intercepted)
            except Exception as e:
                logger.error(f"Request interceptor error: {e}")
        
        # Create history entry
        history_entry = self.session.create_history_entry(intercepted)
        
        # Forward request
        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(
                    method=intercepted.method,
                    url=intercepted.url,
                    headers=intercepted.headers,
                    data=intercepted.body,
                    verify_ssl=self.config.verify_ssl,
                    timeout=aiohttp.ClientTimeout(
                        total=self.config.connection_timeout
                    )
                ) as response:
                    body = await response.read()
                    
                    # Create intercepted response
                    intercepted_response = InterceptedResponse(
                        status_code=response.status,
                        headers=dict(response.headers),
                        body=body
                    )
                    
                    # Apply response interceptors
                    for interceptor in self._response_interceptors:
                        try:
                            intercepted_response = await interceptor.intercept(
                                intercepted_response,
                                intercepted
                            )
                        except Exception as e:
                            logger.error(f"Response interceptor error: {e}")
                    
                    # Complete history entry
                    self.session.complete_history_entry(
                        history_entry.id,
                        intercepted_response
                    )
                    
                    return web.Response(
                        status=intercepted_response.status_code,
                        headers=intercepted_response.headers,
                        body=intercepted_response.body
                    )
                    
            except Exception as e:
                logger.error(f"Proxy error: {e}")
                return web.Response(
                    status=502,
                    text=f"Proxy error: {str(e)}"
                )
    
    async def start(self) -> None:
        """Start the proxy server."""
        app = web.Application()
        app.router.add_route("*", "/{path:.*}", self._handle_request)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(
            runner,
            self.config.host,
            self.config.port
        )
        
        await site.start()
        logger.info(
            f"Proxy server running on {self.config.host}:{self.config.port}")
