"""
Example usage of the Anarchy Copilot proxy module.

This script demonstrates how to set up and use the proxy server with custom interceptors
for security testing and traffic analysis.
"""
import asyncio
import logging
import sys
from pathlib import Path

from anarchy_copilot.proxy import (
    ProxyConfig,
    ProxyServer,
    RequestInterceptor,
    ResponseInterceptor,
    InterceptedRequest,
    InterceptedResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VulnScanInterceptor(RequestInterceptor):
    """Example interceptor that looks for potential vulnerabilities in requests."""
    
    SUSPICIOUS_PATTERNS = [
        ("XSS", r"[<>]script"),
        ("SQLi", r"'\s*OR\s*'1'='1"),
        ("Path Traversal", r"\.\.\/"),
        ("Command Injection", r"[;&|`]"),
    ]
    
    async def intercept(self, request: InterceptedRequest) -> InterceptedRequest:
        """Check request for suspicious patterns."""
        url = request.url.lower()
        body = request.body.decode('utf-8', errors='ignore').lower() if request.body else ''
        
        for vuln_type, pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, url) or re.search(pattern, body):
                logger.warning(
                    f"Potential {vuln_type} detected in request to {request.url}"
                )
        
        return request

class SecurityHeadersInterceptor(ResponseInterceptor):
    """Example interceptor that analyzes security headers."""
    
    EXPECTED_HEADERS = {
        'X-Frame-Options',
        'X-Content-Type-Options',
        'X-XSS-Protection',
        'Content-Security-Policy',
        'Strict-Transport-Security',
    }
    
    async def intercept(
        self,
        response: InterceptedResponse,
        request: InterceptedRequest
    ) -> InterceptedResponse:
        """Check for missing security headers."""
        missing_headers = self.EXPECTED_HEADERS - set(response.headers.keys())
        if missing_headers:
            logger.warning(
                f"Missing security headers for {request.url}: {missing_headers}"
            )
        return response

async def main():
    """Run the example proxy server."""
    # Create proxy configuration
    config = ProxyConfig(
        host="127.0.0.1",
        port=8080,
        # Generate CA certificate in the current directory
        ca_cert_path=Path("./ca.crt"),
        ca_key_path=Path("./ca.key"),
        # Allow all hosts for this example
        allowed_hosts=set(),
        # Store up to 1000 requests in history
        history_size=1000,
        # Enable all features
        intercept_requests=True,
        intercept_responses=True,
        websocket_support=True,
        http2_support=True
    )
    
    # Create and configure proxy server
    server = ProxyServer(config)
    
    # Add custom interceptors
    server.add_request_interceptor(VulnScanInterceptor())
    server.add_response_interceptor(SecurityHeadersInterceptor())
    
    try:
        # Start the proxy server
        await server.start()
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down proxy server...")
        sys.exit(0)

if __name__ == "__main__":
    # Run the proxy server
    asyncio.run(main())
