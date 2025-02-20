"""Test proxy server connection handling."""
import asyncio
import aiohttp
import logging
import pytest
import ssl
import websockets
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ProxyTestClient:
    """Test client for proxy connections."""

    def __init__(self, proxy_url: str = "http://localhost:8080"):
        self.proxy_url = proxy_url
        self.session: Optional[aiohttp.ClientSession] = None
        self._trust_env = True  # Use system certificates

    async def __aenter__(self):
        """Set up HTTP session."""
        # Configure SSL context to trust our CA
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Create session with proxy configuration
        self.session = aiohttp.ClientSession(
            trust_env=self._trust_env,
            connector=aiohttp.TCPConnector(ssl=ssl_context)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up session."""
        if self.session:
            await self.session.close()

    async def test_http_connection(self, url: str) -> bool:
        """Test HTTP connection through proxy."""
        try:
            async with self.session.get(url, proxy=self.proxy_url) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"HTTP connection error: {e}")
            return False

    async def test_https_connection(self, url: str) -> bool:
        """Test HTTPS connection through proxy."""
        try:
            async with self.session.get(url, proxy=self.proxy_url) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"HTTPS connection error: {e}")
            return False

    async def test_websocket_connection(self, url: str) -> bool:
        """Test WebSocket connection through proxy."""
        try:
            async with websockets.connect(
                url,
                proxy=self.proxy_url,
                ssl=ssl.create_default_context()
            ) as ws:
                await ws.send("Hello")
                response = await ws.recv()
                return bool(response)
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            return False

@pytest.mark.asyncio
async def test_http_proxy():
    """Test HTTP proxy functionality."""
    async with ProxyTestClient() as client:
        # Test HTTP connection
        result = await client.test_http_connection("http://httpbin.org/get")
        assert result, "HTTP connection failed"

@pytest.mark.asyncio
async def test_https_proxy():
    """Test HTTPS proxy functionality."""
    async with ProxyTestClient() as client:
        # Test HTTPS connection
        result = await client.test_https_connection("https://httpbin.org/get")
        assert result, "HTTPS connection failed"

@pytest.mark.asyncio
async def test_websocket_proxy():
    """Test WebSocket proxy functionality."""
    async with ProxyTestClient() as client:
        # Test WebSocket connection
        result = await client.test_websocket_connection("wss://echo.websocket.org")
        assert result, "WebSocket connection failed"

@pytest.mark.asyncio
async def test_multiple_connections():
    """Test multiple simultaneous connections."""
    async with ProxyTestClient() as client:
        # Create multiple concurrent connections
        tasks = [
            client.test_https_connection("https://httpbin.org/get"),
            client.test_https_connection("https://api.github.com"),
            client.test_websocket_connection("wss://echo.websocket.org")
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Connection {i} failed with error"
            assert result, f"Connection {i} failed"

@pytest.mark.asyncio
async def test_error_handling():
    """Test proxy error handling."""
    async with ProxyTestClient() as client:
        # Test invalid URL
        result = await client.test_https_connection("https://invalid.example.com")
        assert not result, "Invalid URL should fail"

        # Test connection timeout
        result = await client.test_https_connection("https://10.255.255.1")
        assert not result, "Timeout should fail"

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    async def run_tests():
        """Run all tests."""
        # Create test client
        async with ProxyTestClient() as client:
            # Test HTTP
            logger.info("Testing HTTP connection...")
            result = await client.test_http_connection("http://httpbin.org/get")
            logger.info(f"HTTP result: {result}")

            # Test HTTPS
            logger.info("Testing HTTPS connection...")
            result = await client.test_https_connection("https://httpbin.org/get")
            logger.info(f"HTTPS result: {result}")

            # Test WebSocket
            logger.info("Testing WebSocket connection...")
            result = await client.test_websocket_connection("wss://echo.websocket.org")
            logger.info(f"WebSocket result: {result}")

            # Test multiple connections
            logger.info("Testing multiple connections...")
            tasks = [
                client.test_https_connection("https://httpbin.org/get"),
                client.test_https_connection("https://api.github.com"),
                client.test_websocket_connection("wss://echo.websocket.org")
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                logger.info(f"Multiple connection {i} result: {result}")

    # Run tests
    asyncio.run(run_tests())
