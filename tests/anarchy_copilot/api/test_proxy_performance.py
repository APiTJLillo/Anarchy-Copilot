"""Test proxy server performance and load handling."""
import pytest
import asyncio
import time
import statistics
import psutil
import os
from typing import List, Tuple, Dict, Any, AsyncGenerator
from httpx import AsyncClient, Response

pytestmark = pytest.mark.asyncio

async def make_concurrent_requests(
    client: AsyncClient,
    num_requests: int,
    endpoint: str,
    **kwargs: Any
) -> List[Tuple[float, int]]:
    """Make concurrent requests and measure response times."""
    async def make_request() -> Tuple[float, int]:
        start_time = time.time()
        try:
            response = await client.request(
                method=kwargs.get("method", "GET"),
                url=endpoint,
                **{k: v for k, v in kwargs.items() if k != "method"}
            )
            end_time = time.time()
            return (end_time - start_time, response.status_code)
        except Exception as e:
            end_time = time.time()
            return (end_time - start_time, 500)

    tasks = [make_request() for _ in range(num_requests)]
    return await asyncio.gather(*tasks)

async def test_proxy_concurrent_requests(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test proxy handling concurrent requests."""
    async for client_data in proxy_client:
        client = client_data["client"]

        # Make concurrent requests through proxy
        results = await make_concurrent_requests(
            client,
            num_requests=50,
            endpoint="http://httpbin.org/get",
            method="GET"
        )

        # Analyze results
        response_times = [time for time, _ in results]
        status_codes = [code for _, code in results]

        # Calculate statistics
        avg_time = statistics.mean(response_times)
        p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        success_rate = sum(1 for code in status_codes if code < 500) / len(status_codes)

        # Assert performance requirements
        assert avg_time < 2.0, f"Average response time too high: {avg_time:.2f}s"
        assert p95_time < 4.0, f"95th percentile response time too high: {p95_time:.2f}s"
        assert success_rate > 0.90, f"Success rate too low: {success_rate:.2%}"

async def test_proxy_websocket_load(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test WebSocket connection handling under load."""
    async for client_data in proxy_client:
        base_client = client_data["base_client"]

        async def websocket_session() -> Tuple[bool, float]:
            start_time = time.time()
            try:
                async with base_client.stream(
                    "GET",
                    "/api/proxy/ws/intercept",
                    headers={
                        "Connection": "Upgrade",
                        "Upgrade": "websocket",
                        "Sec-WebSocket-Version": "13",
                        "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="
                    }
                ) as response:
                    assert response.status_code == 101  # Switching Protocols
                    await asyncio.sleep(0.1)  # Keep connection open briefly
                    return True, time.time() - start_time
            except Exception:
                return False, time.time() - start_time

        try:
            # Create multiple concurrent WebSocket connections
            num_connections = 20
            tasks = [websocket_session() for _ in range(num_connections)]
            results = await asyncio.gather(*tasks)

            # Analyze results
            successes = [success for success, _ in results]
            connection_times = [time for _, time in results]

            success_rate = sum(successes) / len(successes)
            avg_time = statistics.mean(connection_times)

            assert success_rate > 0.90, f"WebSocket connection success rate too low: {success_rate:.2%}"
            assert avg_time < 1.0, f"Average WebSocket connection time too high: {avg_time:.2f}s"
        finally:
            await asyncio.sleep(0.1)  # Allow connections to close

async def test_proxy_memory_usage(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test proxy memory usage under load."""
    async for client_data in proxy_client:
        client = client_data["client"]

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        try:
            # Generate load with large payloads
            tasks = []
            for _ in range(50):  # Reduced to 50 requests to prevent overload
                tasks.append(
                    client.post(
                        "http://httpbin.org/post",
                        json={"data": "x" * 500000}  # 500KB payload
                    )
                )
            await asyncio.gather(*tasks)

            # Check memory usage
            final_memory = process.memory_info().rss
            memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB

            assert memory_increase < 200, f"Memory usage increased too much: {memory_increase:.1f}MB"
        finally:
            # Force garbage collection
            import gc
            gc.collect()
            await asyncio.sleep(0.1)

async def test_proxy_connection_limits(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test proxy connection limit handling."""
    async for client_data in proxy_client:
        client = client_data["client"]
        base_client = client_data["base_client"]

        # Update proxy settings with limits
        await base_client.post("/api/proxy/settings", json={
            "maxConnections": 20,
            "maxKeepaliveConnections": 10,
            "keepaliveTimeout": 5
        })

        try:
            # Test with different batch sizes
            for batch_size in [5, 10, 20]:
                tasks = []
                for _ in range(batch_size * 2):  # Try twice the batch size
                    tasks.append(client.get("http://httpbin.org/get"))
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                successful = sum(
                    1 for r in responses 
                    if isinstance(r, Response) and 200 <= r.status_code < 300
                )
                assert successful >= batch_size, f"Too few successful connections with batch size {batch_size}"
                await asyncio.sleep(0.1)  # Let connections close
        finally:
            # Reset settings
            await base_client.post("/api/proxy/settings", json={
                "maxConnections": 100,
                "maxKeepaliveConnections": 50,
                "keepaliveTimeout": 30
            })

async def test_proxy_large_response_handling(proxy_client: AsyncGenerator[Dict[str, Any], None]):
    """Test handling of large response bodies."""
    async for client_data in proxy_client:
        client = client_data["client"]

        try:
            start_time = time.time()
            async with client.stream('GET', 'http://httpbin.org/stream/50') as response:
                # Read response in chunks
                chunks = []
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    chunks.append(chunk)
            
            end_time = time.time()
            total_size = sum(len(chunk) for chunk in chunks)
            processing_time = end_time - start_time

            # Assert reasonable performance with large responses
            assert processing_time < 10.0, f"Large response processing too slow: {processing_time:.2f}s"
            assert total_size > 0, "No data received"
        finally:
            await asyncio.sleep(0.1)  # Let cleanup complete
