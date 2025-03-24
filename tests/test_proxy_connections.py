"""Test proxy server connection handling."""
import asyncio
import aiohttp
import logging
import pytest
import ssl
from typing import Optional
from aiohttp import web
from proxy.core import ProxyServer

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ProxyTestClient:
    """Test client for proxy connections."""

    def __init__(self, proxy_url: str = "http://localhost:8083"):
        self.proxy_url = proxy_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Set up HTTP session."""
        # Configure SSL context to trust our CA
        ssl_context = ssl.create_default_context(cafile='certs/ca.crt')
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Create session with proxy configuration
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        self.session = aiohttp.ClientSession(connector=connector)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up session."""
        if self.session:
            await self.session.close()

    async def test_http_connection(self, url: str) -> bool:
        """Test HTTP connection through proxy."""
        try:
            async with self.session.get(
                url,
                proxy=self.proxy_url,
                allow_redirects=True,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                await response.text()
                return response.status == 200
        except Exception as e:
            logger.error(f"HTTP connection error: {e}")
            return False

    async def test_https_connection(self, url: str) -> bool:
        """Test HTTPS connection through proxy."""
        try:
            async with self.session.get(
                url,
                proxy=self.proxy_url,
                allow_redirects=True,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                await response.text()
                return response.status == 200
        except Exception as e:
            logger.error(f"HTTPS connection error: {e}")
            return False

class TestServer:
    def __init__(self):
        self.app = web.Application()
        self.runner = None
        self.site = None
        self.ssl_context = None

    async def setup(self, use_ssl=False):
        try:
            # Define routes
            async def hello(request):
                return web.Response(text="Hello, World!")

            async def delay(request):
                delay_time = float(request.query.get('time', '1'))
                await asyncio.sleep(delay_time)
                return web.Response(text=f"Delayed {delay_time}s")

            self.app.router.add_get('/', hello)
            self.app.router.add_get('/delay', delay)

            # Set up runner
            self.runner = web.AppRunner(self.app)
            await asyncio.wait_for(self.runner.setup(), timeout=5.0)

            # Configure SSL if needed
            if use_ssl:
                self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                self.ssl_context.load_cert_chain('certs/ca.crt', 'certs/ca.key')
                self.site = web.TCPSite(self.runner, 'localhost', 8443, ssl_context=self.ssl_context)
            else:
                self.site = web.TCPSite(self.runner, 'localhost', 8081)

            # Start the site with timeout
            await asyncio.wait_for(self.site.start(), timeout=5.0)
            return f"http{'s' if use_ssl else ''}://localhost:{8443 if use_ssl else 8081}"
            
        except asyncio.TimeoutError:
            logger.error("Timeout while setting up test server")
            await self.cleanup()
            raise
        except Exception as e:
            logger.error(f"Error setting up test server: {str(e)}")
            await self.cleanup()
            raise

    async def cleanup(self):
        """Clean up server resources."""
        if self.runner:
            try:
                await asyncio.wait_for(self.runner.cleanup(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.error("Timeout while cleaning up test server")
            except Exception as e:
                logger.error(f"Error cleaning up test server: {str(e)}")


@pytest.mark.asyncio
async def test_timeout_handling(test_https_server, proxy_server):
    """Test timeout handling."""
    async with ProxyTestClient() as client:
        # Test delay endpoint with timeout
        url = test_https_server.replace('/', '/delay?time=2')  # 2 second delay
        start_time = asyncio.get_event_loop().time()
        result = await client.test_https_connection(url)
        end_time = asyncio.get_event_loop().time()
        
        # Verify request completed and took appropriate time
        assert result, "Request to delay endpoint failed"
        assert end_time - start_time >= 2.0, "Request completed too quickly"

@pytest.fixture
async def test_http_server():
    """Create a test HTTP server."""
    server = TestServer()
    url = await server.setup(use_ssl=False)
    yield url
    await server.cleanup()

@pytest.fixture
async def test_https_server():
    """Create a test HTTPS server."""
    server = TestServer()
    url = await server.setup(use_ssl=True)
    yield url
    await server.cleanup()

@pytest.fixture
async def proxy_server():
    """Create and run a proxy server instance."""
    server = ProxyServer(
        host='localhost',
        port=8083,
        cert_path='certs/ca.crt',
        key_path='certs/ca.key'
    )
    task = asyncio.create_task(server.start())
    await asyncio.sleep(0.1)  # Give the server time to start
    yield server
    server.close()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_http_proxy(test_http_server, proxy_server):
    """Test HTTP proxy functionality."""
    async with ProxyTestClient() as client:
        result = await client.test_http_connection(test_http_server)
        assert result, "HTTP connection failed"

@pytest.mark.asyncio
async def test_https_proxy(test_https_server, proxy_server):
    """Test HTTPS proxy functionality."""
    async with ProxyTestClient() as client:
        result = await client.test_https_connection(test_https_server)
        assert result, "HTTPS connection failed"

@pytest.mark.asyncio
async def test_multiple_connections(proxy_server):
    """Test multiple simultaneous connections."""
    urls = [
        "https://httpbin.org/get",
        "https://api.github.com",
        "https://example.com"
    ]
    
    async def test_connection(url: str):
        async with ProxyTestClient() as client:
            return await client.test_https_connection(url)

    # Create tasks for each connection
    tasks = [test_connection(url) for url in urls]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Connection to {urls[i]} failed with error: {result}")
            continue
        assert result, f"Connection to {urls[i]} failed"

@pytest.mark.asyncio
async def test_error_handling(proxy_server):
    """Test proxy error handling."""
    test_cases = [
        {
            "url": "https://invalid.example.com",
            "name": "invalid domain",
            "expected_error": True
        },
        {
            "url": "https://10.255.255.1",
            "name": "timeout",
            "expected_error": True
        },
        {  
            "url": "https://localhost:1234",
            "name": "connection refused",
            "expected_error": True
        }
    ]
    
    async with ProxyTestClient() as client:
        for case in test_cases:
            result = await client.test_https_connection(case["url"])
            assert result is False, f"Connection to {case['name']} should fail but succeeded"

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    async def run_tests():
        """Run all tests."""
        async with ProxyTestClient() as client:
            # Test HTTP with local test server
            app = web.Application()
            async def hello(request):
                return web.Response(text="Hello, World!")
            app.router.add_get('/', hello)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', 8081)
            await site.start()
            
            logger.info("Testing HTTP connection...")
            result = await client.test_http_connection("http://localhost:8081")
            logger.info(f"HTTP result: {result}")
            
            # Test HTTPS
            logger.info("Testing HTTPS connection...")
            result = await client.test_https_connection("https://httpbin.org/get")
            logger.info(f"HTTPS result: {result}")

            # Test multiple connections
            logger.info("Testing multiple connections...")
            tasks = [
                client.test_https_connection("https://httpbin.org/get"),
                client.test_https_connection("https://api.github.com"),
                client.test_https_connection("https://example.com")
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Connection {i} error: {result}")
                else:
                    logger.info(f"Connection {i} result: {result}")
            
            await runner.cleanup()

    # Run tests
    asyncio.run(run_tests())
