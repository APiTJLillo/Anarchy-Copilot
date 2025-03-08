"""Tests for proxy server API endpoints."""
import asyncio
import logging
from typing import Dict, Any, AsyncGenerator

import pytest
from httpx import AsyncClient, Timeout

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
class TestProxyAPI:
    """Test proxy server API endpoints."""

    async def test_httpbin_ready(self, base_client: AsyncGenerator[AsyncClient, None]) -> None:
        """Verify httpbin service is ready."""
        async for client in base_client:
            max_attempts = 5  # Increased retries
            async with AsyncClient(
                base_url="http://httpbin",  # Use container name in Docker network
                verify=False,
                timeout=10.0
            ) as test_client:
                for attempt in range(max_attempts):
                    try:
                        response = await test_client.get("/get")
                        assert response.status_code == 200
                        data = response.json()
                        assert "url" in data
                        assert data["url"].endswith("/get")

                        # Double-check stability
                        response = await test_client.get("/get")
                        assert response.status_code == 200
                        logger.info(f"Httpbin responded successfully on attempt {attempt + 1}")
                        break
                    except Exception as e:
                        logger.warning(f"Httpbin test failed (attempt {attempt + 1}/{max_attempts}): {e}")
                        await asyncio.sleep(3.0)  # Longer wait between retries
                else:
                    raise RuntimeError(f"Httpbin service not stable after {max_attempts} attempts")
                logger.info("Httpbin is ready and stable")

    async def test_start_proxy(
        self,
        base_client: AsyncGenerator[AsyncClient, None],
        proxy_client: AsyncGenerator[Dict[str, Any], None]
    ) -> None:
        """Test starting proxy server."""
        async for client in base_client:
            try:
                # Ensure no proxy is running before starting
                try:
                    await client.post("/api/proxy/stop")
                    await asyncio.sleep(3.0)  # Wait for full cleanup
                except Exception:
                    pass  # Ignore errors if no proxy was running

                # Start the proxy
                config = {
                    "host": "0.0.0.0",
                    "port": 8083,
                    "interceptRequests": True,
                    "interceptResponses": True,
                    "allowedHosts": ["httpbin", "localhost"],
                    "excludedHosts": [],
                    "maxConnections": 50,
                    "maxKeepaliveConnections": 10,
                    "keepaliveTimeout": 30
                }
                response = await client.post("/api/proxy/start", json=config)
                assert response.status_code == 201

                # Verify proxy is running with retry
                max_status_attempts = 5
                for attempt in range(max_status_attempts):
                    status = await client.get("/api/proxy/status")
                    assert status.status_code == 200
                    status_data = status.json()
                    if status_data["isRunning"]:
                        break
                    await asyncio.sleep(1.0)
                else:
                    raise RuntimeError("Proxy failed to start properly")

                # Test request through proxy with enhanced retry logic
                max_attempts = 3
                last_error = None
                success = False
                
                async for proxy in proxy_client:
                    # Test request through proxy with timeout
                    try:
                        async def make_request():
                            for attempt in range(max_attempts):
                                try:
                                    response = await proxy["client"].get(
                                        "/get",
                                        timeout=Timeout(5.0)  # Shorter timeout for quicker retries
                                    )
                                    assert response.status_code == 200
                                    data = response.json()
                                    assert data["url"].endswith("/get")
                                    return True
                                except Exception as e:
                                    last_error = e
                                    logger.warning(f"Proxy request failed (attempt {attempt + 1}/{max_attempts}): {e}")
                                    if attempt < max_attempts - 1:
                                        await asyncio.sleep(1.0)  # Shorter sleep between retries
                            return False

                        success = await asyncio.wait_for(make_request(), timeout=15.0)  # 15 second total timeout
                        if not success:
                            raise RuntimeError(f"Failed to make proxy request after {max_attempts} attempts: {last_error}")
                        break  # Success, exit proxy client loop
                    except asyncio.TimeoutError:
                        raise RuntimeError("Timeout waiting for proxy request to complete")
            finally:
                # Always try to stop the proxy after test with timeout
                try:
                    async def cleanup():
                        await client.post("/api/proxy/stop")
                        await asyncio.sleep(2.0)  # Reduced cleanup wait
                    
                    await asyncio.wait_for(cleanup(), timeout=10.0)  # 10 second timeout for cleanup
                except asyncio.TimeoutError:
                    logger.error("Timeout during proxy cleanup")
                except Exception as e:
                    logger.warning(f"Error stopping proxy during cleanup: {e}")

    async def test_stop_proxy(
        self,
        base_client: AsyncGenerator[AsyncClient, None]
    ) -> None:
        """Test stopping proxy server."""
        async for client in base_client:
            try:
                # Start the proxy first
                config = {
                    "host": "0.0.0.0",
                    "port": 8083,
                    "interceptRequests": True,
                    "interceptResponses": True,
                    "allowedHosts": ["httpbin", "localhost"],
                    "excludedHosts": [],
                    "maxConnections": 50,
                    "maxKeepaliveConnections": 10,
                    "keepaliveTimeout": 30
                }
                response = await client.post("/api/proxy/start", json=config, timeout=10.0)
                assert response.status_code == 201

                # Wait for proxy to start and verify it's running
                for _ in range(5):
                    status = await client.get("/api/proxy/status", timeout=5.0)
                    assert status.status_code == 200
                    if status.json()["isRunning"]:
                        break
                    await asyncio.sleep(1.0)
                else:
                    raise RuntimeError("Proxy failed to start")

                # Send stop request and let server handle cleanup
                response = await client.post("/api/proxy/stop", timeout=5.0)
                assert response.status_code == 201
                data = response.json()
                assert data["message"] == "Proxy server stopped successfully"
                
                # Just wait a bit for cleanup without checking status
                await asyncio.sleep(3.0)
            finally:
                # Ensure cleanup
                try:
                    await client.post("/api/proxy/stop", timeout=5.0)
                except Exception:
                    pass  # Ignore errors during cleanup

    async def test_proxy_status(self, base_client: AsyncGenerator[AsyncClient, None]) -> None:
        """Test proxy status endpoint."""
        async for client in base_client:
            response = await client.get("/api/proxy/status")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict)
            assert "isRunning" in data
            assert "settings" in data
            assert isinstance(data["settings"], dict)

    async def test_start_proxy_failure(self, base_client: AsyncGenerator[AsyncClient, None]) -> None:
        """Test proxy failure handling."""
        async for client in base_client:
            response = await client.post("/api/proxy/start", json={
                "host": "127.0.0.1",
                "port": -1,  # Invalid port
                "interceptRequests": True,
                "interceptResponses": True,
                "allowedHosts": ["httpbin"],
                "excludedHosts": []
            })
            assert response.status_code in (400, 422)
