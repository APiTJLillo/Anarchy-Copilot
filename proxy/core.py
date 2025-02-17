"""
Core proxy server implementation for Anarchy Copilot.

This module implements the main proxy server functionality, handling HTTP/HTTPS/WebSocket
connections and integrating with the interceptor and session management components.
"""
import asyncio
import logging
import socket
import ssl
from typing import Dict, List, Optional, Set, Any
import aiohttp
from aiohttp import web
import certifi
import OpenSSL.crypto as crypto
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlparse, urlunparse

from .config import ProxyConfig
from .interceptor import (
    InterceptedRequest,
    InterceptedResponse,
    RequestInterceptor,
    ResponseInterceptor
)
from .encoding import ContentEncodingInterceptor
from .session import ProxySession
from .websocket import WebSocketManager, DebugInterceptor
from .analysis.analyzer import TrafficAnalyzer  # Update import path

logger = logging.getLogger(__name__)

class PendingRequest:
    """Represents a request waiting for user interaction."""
    def __init__(self, request: InterceptedRequest):
        self.request = request
        self.future = asyncio.Future()

class PendingResponse:
    """Represents a response waiting for user interaction."""
    def __init__(self, request_id: str, response: InterceptedResponse):
        self.request_id = request_id
        self.response = response
        self.future = asyncio.Future()

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
    
    def generate_certificate(self, hostname: str) -> "tuple[bytes, bytes]":
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
    
    def __init__(self, config: ProxyConfig, add_default_interceptors: bool = False):
        """Initialize the proxy server.
        
        Args:
            config: Proxy configuration
            add_default_interceptors: Whether to add default interceptors
        """
        self.config = config
        self.session = ProxySession(max_history=config.history_size)
        self._request_interceptors: List[RequestInterceptor] = []
        self._response_interceptors: List[ResponseInterceptor] = []

        # Only add default interceptors if specified
        if add_default_interceptors:
            self._request_interceptors.append(ContentEncodingInterceptor())

        self._ssl_contexts: dict[str, ssl.SSLContext] = {}
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._is_running: bool = False
        
        # Initialize WebSocket support
        self._ws_manager = WebSocketManager()
        if config.websocket_support:
            self._ws_manager.add_interceptor(DebugInterceptor())

        # Initialize traffic analyzer
        self._analyzer = TrafficAnalyzer()
        
        # Track pending requests/responses
        self._pending_requests: Dict[str, PendingRequest] = {}
        self._pending_responses: Dict[str, PendingResponse] = {}
        
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

    async def intercept_request(self, request_id: str, headers: Dict[str, str], body: Optional[bytes] = None) -> Dict[str, Any]:
        """Handle user interception of a request.
        
        Args:
            request_id: ID of the request to modify
            headers: Modified headers
            body: Modified request body
            
        Returns:
            Modified request data
        """
        if request_id not in self._pending_requests:
            raise ValueError(f"No pending request with ID {request_id}")

        pending = self._pending_requests[request_id]
        
        # Update request with modified data
        pending.request.headers = headers
        pending.request.body = body
        
        # Mark as handled
        pending.future.set_result(pending.request)
        return {"status": "success"}

    async def intercept_response(self, request_id: str, status_code: int, headers: Dict[str, str], body: Optional[bytes] = None) -> Dict[str, Any]:
        """Handle user interception of a response.
        
        Args:
            request_id: ID of the original request
            status_code: Modified status code
            headers: Modified headers
            body: Modified response body
            
        Returns:
            Modified response data
        """
        if request_id not in self._pending_responses:
            raise ValueError(f"No pending response with ID {request_id}")

        pending = self._pending_responses[request_id]
        
        # Update response with modified data
        pending.response.status_code = status_code
        pending.response.headers = headers
        pending.response.body = body
        
        # Mark as handled
        pending.future.set_result(pending.response)
        return {"status": "success"}

    async def forward_request(self, request_id: str, modified: Dict[str, Any]) -> None:
        """Forward a modified request.
        
        Args:
            request_id: ID of the request to forward
            modified: Modified request data
        """
        if request_id not in self._pending_requests:
            raise ValueError(f"No pending request with ID {request_id}")

        pending = self._pending_requests[request_id]
        
        # Update request with modified data
        pending.request.method = modified["method"]
        pending.request.url = modified["url"]
        pending.request.headers = {h["name"]: h["value"] for h in modified["headers"]}
        pending.request.body = modified["body"].encode() if modified.get("body") else None
        
        # Mark as handled
        pending.future.set_result(pending.request)
        del self._pending_requests[request_id]

    async def forward_response(self, request_id: str, modified: Dict[str, Any]) -> None:
        """Forward a modified response.
        
        Args:
            request_id: ID of the original request
            modified: Modified response data
        """
        if request_id not in self._pending_responses:
            raise ValueError(f"No pending response with ID {request_id}")

        pending = self._pending_responses[request_id]
        
        # Update response with modified data
        pending.response.status_code = modified["statusCode"]
        pending.response.headers = {h["name"]: h["value"] for h in modified["headers"]}
        pending.response.body = modified["body"].encode() if modified.get("body") else None
        
        # Mark as handled
        pending.future.set_result(pending.response)
        del self._pending_responses[request_id]

    async def drop_request(self, request_id: str) -> None:
        """Drop a pending request.
        
        Args:
            request_id: ID of the request to drop
        """
        if request_id not in self._pending_requests:
            raise ValueError(f"No pending request with ID {request_id}")

        pending = self._pending_requests[request_id]
        pending.future.set_exception(Exception("Request dropped"))
        del self._pending_requests[request_id]

    async def drop_response(self, request_id: str) -> None:
        """Drop a pending response.
        
        Args:
            request_id: ID of the original request
        """
        if request_id not in self._pending_responses:
            raise ValueError(f"No pending response with ID {request_id}")

        pending = self._pending_responses[request_id]
        pending.future.set_exception(Exception("Response dropped"))
        del self._pending_responses[request_id]
    
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

    def _rewrite_url_for_docker(self, url: str, target_host: str) -> str:
        """Rewrite URL for Docker networking.
        
        Args:
            url: Original URL
            target_host: Target hostname
            
        Returns:
            Rewritten URL for Docker networking
        """
        parsed = urlparse(url)
        if parsed.netloc == 'localhost':
            # Use Docker network hostname
            netloc = 'httpbin'
        elif parsed.netloc == 'httpbin':
            # Keep Docker network hostname
            netloc = 'httpbin'
        elif ':' not in parsed.netloc:
            # Add port 80 if not specified
            netloc = f'{target_host}:80'
        else:
            netloc = parsed.netloc

        logger.debug(f"URL rewrite: {parsed.netloc} -> {netloc}")
        return urlunparse(parsed._replace(netloc=netloc))
    
    async def _handle_request(self, request: web.Request) -> web.Response:
        """Handle an incoming proxy request."""
        if not self.config.is_in_scope(request.host):
            return web.Response(
                status=403,
                text="Host not in scope"
            )
        
        # Check for WebSocket upgrade
        if request.headers.get('Upgrade', '').lower() == 'websocket':
            target_url = f"ws{'s' if request.scheme == 'https' else ''}://{request.host}{request.path_qs}"
            return await self._ws_manager.handle_websocket((request, target_url, dict(request.headers)))
        
        # Create intercepted request object
        request_id = str(uuid4())
        
        # Construct proper URL with port if needed
        url = str(request.url)
        if ':' not in request.host:
            # Add port 80 for HTTP
            url = url.replace(request.host, f"{request.host}:80")
        logger.debug(f"Forwarding request to: {url}")
        
        intercepted = InterceptedRequest(
            id=request_id,
            method=request.method,
            url=url,
            headers=dict(request.headers),
            body=await request.read()
        )
        
        # Apply request interceptors
        for interceptor in self._request_interceptors:
            try:
                intercepted = await interceptor.intercept(intercepted)
            except ValueError as e:
                # Return 400 Bad Request for validation errors
                return web.Response(status=400, text=str(e))
            except Exception as e:
                logger.error(f"Request interceptor error: {e}")
                return web.Response(status=500, text=f"Internal proxy error: {str(e)}")
        
        # Handle user interception if enabled
        if self.config.intercept_requests:
            pending = PendingRequest(intercepted)
            self._pending_requests[request_id] = pending
            try:
                intercepted = await pending.future
            except Exception as e:
                return web.Response(
                    status=500,
                    text=str(e)
                )
            finally:
                del self._pending_requests[request_id]
        
        # Create history entry and analyze request
        history_entry = self.session.create_history_entry(intercepted)
        self._analyzer.analyze_request(intercepted)
        
        # Forward request
        connector = aiohttp.TCPConnector(
            verify_ssl=False,  # Disable SSL verification for internal traffic
            force_close=True,  # Don't reuse connections
            enable_cleanup_closed=True,  # Clean up closed connections
            ttl_dns_cache=0,  # Disable DNS caching
            use_dns_cache=False,  # Disable DNS caching
            ssl=None,  # Disable SSL verification for internal traffic
            keepalive_timeout=30.0,  # Increase keepalive timeout
            limit=100,  # Increase max connections
            limit_per_host=10  # Increase per-host connections
        )
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(
                total=120,  # Match client timeout
                connect=30,  # Increase connect timeout
                sock_read=120,  # Match client read timeout
                sock_connect=30  # Increase connect timeout
            ),
            trust_env=False,  # Don't use environment variables
            version="1.1",  # Force HTTP/1.1
            skip_auto_headers=["Accept-Encoding"],  # Don't add compression headers
            auto_decompress=False  # Disable automatic decompression
        ) as session:
            try:
                logger.debug(f"Sending request: {intercepted.method} {intercepted.url}")
                logger.debug(f"Request headers: {intercepted.headers}")
                
                # Prepare headers for Docker network DNS
                headers = dict(intercepted.headers)
                target_host = request.host.split(':')[0]  # Remove port from Host header
                headers['Host'] = target_host
                
                # Rewrite URL for Docker networking
                target_url = self._rewrite_url_for_docker(intercepted.url, target_host)
                
                logger.debug(f"Original URL: {intercepted.url}")
                logger.debug(f"Forwarding to URL: {target_url}")
                logger.debug(f"Using headers: {headers}")

                # Retry mechanism for connecting to target
                max_retries = 5
                retry_delay = 2.0  # seconds
                last_error = None

                # Wait longer for Docker DNS to propagate and network stability
                await asyncio.sleep(2.0)

                # Attempt request with retries
                for attempt in range(max_retries):
                    # Add delay between attempts
                    if attempt > 0:
                        await asyncio.sleep(retry_delay)
                    try:
                        async with session.request(
                            method=intercepted.method,
                            url=target_url,
                            headers=headers,
                            data=intercepted.body,
                            allow_redirects=True,
                            proxy=None,  # Disable any proxy settings
                            verify_ssl=False,  # Disable SSL verification for internal traffic
                            timeout=aiohttp.ClientTimeout(
                                total=60,  # Longer timeout for reliability
                                connect=20,
                                sock_read=60,
                                sock_connect=20
                            )
                        ) as response:
                            try:
                                body = await asyncio.wait_for(
                                    response.read(),
                                    timeout=30.0  # Shorter timeout for retries
                                )
                                break  # Success, exit retry loop
                            except asyncio.TimeoutError as e:
                                last_error = e
                                logger.warning(f"Timeout reading response body from {target_url} (attempt {attempt + 1}/{max_retries})")
                                if attempt == max_retries - 1:  # Last attempt
                                    raise
                                await asyncio.sleep(retry_delay)
                                continue
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Error connecting to {target_url} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        if attempt == max_retries - 1:  # Last attempt
                            raise
                        await asyncio.sleep(retry_delay)
                        continue

                if last_error:
                    logger.error(f"All retries failed for {target_url}: {str(last_error)}")
                    return web.Response(
                        status=504,
                        text=f"Gateway Timeout: Failed to connect to {target_url} after {max_retries} attempts"
                    )
                
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
                        
                # Handle user interception if enabled
                if self.config.intercept_responses:
                    pending = PendingResponse(request_id, intercepted_response)
                    self._pending_responses[request_id] = pending
                    try:
                        intercepted_response = await pending.future
                    except Exception as e:
                        return web.Response(
                            status=500,
                            text=str(e)
                        )
                    finally:
                        del self._pending_responses[request_id]
                
                # Complete history entry and analyze response
                self.session.complete_history_entry(
                    history_entry.id,
                    intercepted_response
                )
                self._analyzer.analyze_response(intercepted_response, intercepted)
                
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
    
    @property
    def is_running(self) -> bool:
        """Check if the proxy server is running."""
        return self._is_running

    async def start(self) -> None:
        """Start the proxy server."""
        if self._is_running:
            raise RuntimeError("Proxy server is already running")

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", self._handle_request)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        # Wait for port to be fully released
        await asyncio.sleep(2.0)  # Give more time for OS to release port
        
        # Create site with retries
        max_retries = 3
        retry_delay = 2.0
        last_error = None
        site = None
        
        for attempt in range(max_retries):
            try:
                # Create TCPSite with reuse options
                site = web.TCPSite(
                    runner,
                    self.config.host,
                    self.config.port,
                    reuse_address=True,
                    reuse_port=True,
                    backlog=128
                )
                # Wait for potential socket cleanup
                await asyncio.sleep(1.0)
                await site.start()
                logger.info(f"Successfully started site on attempt {attempt + 1}")
                break
            except OSError as e:
                last_error = e
                logger.warning(f"Failed to start site (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("All attempts to start site failed")
                    raise
                # Cleanup failed site
                if site:
                    try:
                        await site.stop()
                    except Exception as stop_error:
                        logger.warning(f"Error stopping failed site: {stop_error}")
                await asyncio.sleep(retry_delay)

        self._runner = runner
        self._site = site
        self._is_running = True
        logger.info(
            f"Proxy server running on {self.config.host}:{self.config.port}")

    def get_analyzer(self) -> Optional[TrafficAnalyzer]:
        """Get the traffic analyzer instance.
        
        Returns:
            The traffic analyzer if the server is running, None otherwise
        """
        return self._analyzer if self._is_running else None

    async def stop(self) -> bool:
        """Stop the proxy server and cleanup all resources.
        
        Returns:
            bool: True if stopped successfully, False if already stopped
        """
        if not self._is_running:
            return False

        try:
            logger.info("Stopping proxy server...")
            self._is_running = False  # Set this first to prevent new connections

            # Cancel pending operations first
            logger.debug("Canceling pending operations...")
            for pending in list(self._pending_requests.values()):
                try:
                    if not pending.future.done():
                        pending.future.set_exception(Exception("Server stopped"))
                except Exception:
                    pass
            self._pending_requests.clear()

            for pending in list(self._pending_responses.values()):
                try:
                    if not pending.future.done():
                        pending.future.set_exception(Exception("Server stopped"))
                except Exception:
                    pass
            self._pending_responses.clear()

            # Cancel active connections first
            try:
                # Only cancel our own tasks, not system ones
                tasks = [t for t in asyncio.all_tasks() 
                        if t is not asyncio.current_task() and 
                        not t.get_name().startswith('Task-')]
                if tasks:
                    logger.debug(f"Canceling {len(tasks)} tasks...")
                    for task in tasks:
                        task.cancel()
                    # Use short timeout for cancellation
                    await asyncio.wait(tasks, timeout=2.0)
            except Exception as e:
                logger.warning(f"Error canceling tasks: {e}")
                return False

            # Stop the site with timeout
            if self._site:
                logger.debug("Stopping site...")
                try:
                    await asyncio.wait_for(self._site.stop(), timeout=3.0)
                except asyncio.TimeoutError:
                    logger.warning("Site stop timed out")
                except Exception as e:
                    logger.warning(f"Error stopping site: {e}")

            # Clean up runner
            if self._runner:
                logger.debug("Cleaning up runner...")
                try:
                    await self._runner.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up runner: {e}")

            # Clean up WebSocket connections
            if self._ws_manager:
                logger.debug("Cleaning up WebSocket connections...")
                try:
                    await self._ws_manager.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up WebSocket manager: {e}")

            # Clear all contexts and state
            logger.debug("Clearing contexts and state...")
            self._ssl_contexts.clear()
            self.session = ProxySession(max_history=self.config.history_size)
            self._site = None
            self._runner = None

            # Final cleanup
            await asyncio.sleep(1.0)

            # Successful cleanup
            return True

        except Exception as e:
            logger.error(f"Error during proxy server cleanup: {e}")
            # Attempt to reset state even on error
            self._is_running = False
            self._site = None
            self._runner = None
            self._ssl_contexts.clear()
            return False
