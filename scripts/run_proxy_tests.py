"""Run proxy server tests in the test container environment."""
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_proxy_connections import (
    test_http_proxy,
    test_https_proxy,
    test_websocket_proxy,
    test_multiple_connections,
    test_error_handling
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("proxy.test")

async def run_tests():
    """Run all proxy tests in sequence."""
    logger.info("Starting proxy tests in container environment...")

    try:
        # Wait for proxy to be ready
        logger.info("Waiting for proxy server to be ready...")
        await asyncio.sleep(2)

        # Environment setup verification
        httpbin_url = os.getenv("HTTPBIN_URL", "http://httpbin")
        logger.info(f"Using httpbin URL: {httpbin_url}")
        
        # Run tests
        tests = [
            ("HTTP Proxy", test_http_proxy),
            ("HTTPS Proxy", test_https_proxy),
            ("WebSocket Proxy", test_websocket_proxy),
            ("Multiple Connections", test_multiple_connections),
            ("Error Handling", test_error_handling)
        ]

        for test_name, test_func in tests:
            try:
                logger.info(f"Running test: {test_name}")
                await test_func()
                logger.info(f"Test passed: {test_name}")
            except Exception as e:
                logger.error(f"Test failed: {test_name}")
                logger.error(f"Error: {e}", exc_info=True)
                raise

        logger.info("All tests completed successfully!")

    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        # Run tests
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test runner failed: {e}", exc_info=True)
        sys.exit(1)
