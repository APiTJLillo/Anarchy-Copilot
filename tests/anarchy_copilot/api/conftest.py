"""Test fixtures for API tests."""
import asyncio
import logging
import socket
import subprocess
from typing import Dict, Any, Generator, AsyncGenerator, List, Union
import signal
import os
import psutil
import time
import sys
import re

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, Timeout, Limits
from httpx._transports.asgi import ASGITransport

from api import create_app
from api.proxy import proxy_server as global_proxy_server, reset_state

logger = logging.getLogger("proxy_tests")
logger.setLevel(logging.DEBUG)

def is_valid_pid_to_kill(pid: int) -> bool:
    """Check if PID is valid to kill."""
    if pid <= 1:  # Never kill init process
        return False
    
    if sys.platform == "win32":
        return True  # Windows needs different handling
        
    try:
        proc = psutil.Process(pid)
        # In container, be more aggressive about killing processes
        cmdline = " ".join(proc.cmdline()).lower()
        return any(term in cmdline for term in ['python', 'proxy', 'aiohttp'])
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

async def find_process_by_port(port: int) -> List[int]:
    """Find process IDs using the specified port."""
    pids = set()
    
    for attempt in range(3):  # Multiple attempts for reliability
        # First check if it's a container service using docker port binding
        try:
            output = subprocess.check_output(
                f"docker ps --format '{{{{.ID}}}}\t{{{{.Ports}}}}' | grep -w {port}",
                shell=True,
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()
            if output:
                # If it's a docker container, we'll handle it differently
                return [0]  # Special marker for docker container
        except Exception:
            pass

        # Look for host processes
        try:
            output = subprocess.check_output(
                f"fuser {port}/tcp 2>/dev/null",
                shell=True,
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()
            if output:
                pids.update(int(pid) for pid in output.split() if pid.isdigit() and int(pid) > 1)
                if pids:
                    return list(pids)
        except Exception:
            pass

        await asyncio.sleep(0.5)

    return list(set(pids))  # Remove duplicates

def kill_process(pid: int, force: bool = True) -> bool:
    """Kill a process with specified PID."""
    if not is_valid_pid_to_kill(pid):
        return False

    try:
        proc = psutil.Process(pid)
        
        # First try graceful shutdown for our own processes
        if not force:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
                if not proc.is_running():
                    return True
            except (psutil.TimeoutExpired, psutil.NoSuchProcess):
                pass

        # Force kill if still running
        proc.kill()
        try:
            proc.wait(timeout=3.0)
            return True
        except (psutil.TimeoutExpired, psutil.NoSuchProcess):
            if not proc.is_running():
                return True
            # Last resort: system kill
            subprocess.run(
                f"kill -9 {pid}",
                shell=True,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return not proc.is_running()
    except Exception as e:
        logger.warning(f"Error killing process {pid}: {e}")
        return False

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for entire session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def app() -> FastAPI:
    """Create test FastAPI application."""
    return create_app()

async def is_port_in_use(port: int) -> bool:
    """Check if port is in use."""
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result == 0
    except Exception:
        return True
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass

async def force_kill_port(port: int, check_interval: float = 0.5) -> bool:
    """Force kill any process using specified port."""
    # First try graceful shutdown of our proxy server
    if global_proxy_server and global_proxy_server.is_running:
        try:
            logger.debug("Attempting graceful proxy shutdown...")
            await global_proxy_server.stop()
            await asyncio.sleep(1.0)
            if not await is_port_in_use(port):
                return True
        except Exception as e:
            logger.debug(f"Graceful shutdown failed: {e}")

    reset_state()  # Reset internal state

    # Check for processes first
    killed = False
    pids = await find_process_by_port(port)

    if pids:
        logger.debug(f"Found PIDs using port {port}: {pids}")

        # Special case for docker container
        if pids == [0]:
            # Try to aggressively find and kill any process using the port
            try:
                # Use netstat to find processes
                netstat_output = subprocess.check_output(
                    f"netstat -tuln | grep {port}",
                    shell=True,
                    text=True,
                    stderr=subprocess.DEVNULL
                )
                if netstat_output:
                    # Try to kill using fuser
                    subprocess.run(
                        f"fuser -k {port}/tcp",
                        shell=True,
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    await asyncio.sleep(1.0)
                    if not await is_port_in_use(port):
                        return True

                # Try killing Python processes that might be holding the port
                subprocess.run(
                    "pkill -f 'python.*port.*8080'",
                    shell=True,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                await asyncio.sleep(1.0)
                if not await is_port_in_use(port):
                    return True
            except Exception as e:
                logger.debug(f"Port cleanup failed: {e}")
        else:
            # Regular process cleanup
            for pid in pids:
                if kill_process(pid, force=False):
                    killed = True
                    await asyncio.sleep(check_interval)
                    if not await is_port_in_use(port):
                        return True
                elif kill_process(pid, force=True):
                    killed = True
                    await asyncio.sleep(check_interval)
                    if not await is_port_in_use(port):
                        return True

    # Final cleanup attempt using all available methods
    if await is_port_in_use(port):
        try:
            # Kill all Python processes that might be using the port
            subprocess.run(
                "pkill -9 -f 'python.*proxy'",
                shell=True,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            await asyncio.sleep(1.0)

            # Last resort: Force kill anything using the port
            subprocess.run(
                f"fuser -k -9 {port}/tcp",
                shell=True,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            await asyncio.sleep(1.0)
        except Exception as e:
            logger.debug(f"Final cleanup failed: {e}")

    # Final check
    return not await is_port_in_use(port)

async def ensure_proxy_stopped(client: Union[AsyncClient, AsyncGenerator[AsyncClient, None]], max_attempts: int = 3) -> None:
    """Ensure proxy is stopped via API."""
    async def _stop_proxy(c: AsyncClient) -> None:
        for attempt in range(max_attempts):
            try:
                # Check current status
                status = await c.get("/api/proxy/status")
                if status.status_code == 200 and not status.json()["isRunning"]:
                    return

                # Try to stop the proxy
                logger.debug(f"Stopping proxy attempt {attempt + 1}/{max_attempts}")
                await c.post("/api/proxy/stop")
                await asyncio.sleep(1.0)

                # Verify it stopped
                status = await c.get("/api/proxy/status")
                if status.status_code == 200 and not status.json()["isRunning"]:
                    return
            except Exception as e:
                logger.warning(f"Error stopping proxy (attempt {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1.0)

        logger.error("Failed to stop proxy server")

    if isinstance(client, AsyncClient):
        await _stop_proxy(client)
    else:
        async for c in client:
            await _stop_proxy(c)
            break  # Only try with first client

async def wait_for_httpbin(timeout: float = 30.0) -> bool:
    """Wait for httpbin to be ready."""
    async with AsyncClient(
        base_url="http://httpbin",
        verify=False,
        timeout=Timeout(5.0)
    ) as client:
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                response = await client.get("/get")
                if response.status_code == 200:
                    await asyncio.sleep(0.5)  # Add small delay between checks
                    response = await client.get("/get")
                    if response.status_code == 200:
                        logger.info("Httpbin is ready and stable")
                        return True
            except Exception as e:
                logger.warning(f"Waiting for httpbin: {e}")
            await asyncio.sleep(2.0)
        return False

@pytest.fixture(autouse=True)
async def cleanup_state(base_client: AsyncClient) -> AsyncGenerator[None, None]:
    """Ensure clean state before and after each test."""
    async def do_cleanup():
        """Perform cleanup with timeout."""
        try:
            reset_state()
            await asyncio.wait_for(ensure_proxy_stopped(base_client), timeout=5.0)  # 5 second timeout for cleanup
        except asyncio.TimeoutError:
            logger.warning("Cleanup timed out")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

    # Pre-test cleanup
    await do_cleanup()
    yield
    # Post-test cleanup
    await do_cleanup()

@pytest.fixture
async def base_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create base test client."""
    timeout = Timeout(30.0, connect=10.0, read=30.0, write=30.0)
    transport = ASGITransport(app=app)
    
    async with AsyncClient(
        base_url="http://testserver",
        transport=transport,
        timeout=timeout,
        follow_redirects=True
    ) as client:
        if not await wait_for_httpbin():
            raise RuntimeError("Httpbin service not ready after 30 seconds")
        yield client

@pytest.fixture
def security_headers() -> Dict[str, str]:
    """Get secure default headers."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }

async def wait_for_proxy(client: AsyncClient, timeout: float = 10.0) -> bool:
    """Wait for proxy to be ready."""
    start_time = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - start_time) < timeout:
        try:
            response = await client.get("/get", timeout=5.0)
            if response.status_code == 200:
                logger.info("Proxy connection successful")
                return True
        except Exception as e:
            logger.warning(f"Waiting for proxy: {e}")
        await asyncio.sleep(1.0)
    return False

@pytest.fixture
async def proxy_client(
    base_client: AsyncGenerator[AsyncClient, None],
    security_headers: Dict[str, str]
) -> AsyncGenerator[Dict[str, Any], None]:
    """Create proxy test client."""
    async for api_client in base_client:
        # Initial cleanup
        await ensure_proxy_stopped(api_client)
        await asyncio.sleep(1.0)

        config = {
            "host": "0.0.0.0",
            "port": 8080,
            "interceptRequests": True,
            "interceptResponses": True,
            "allowedHosts": ["httpbin", "localhost"],
            "excludedHosts": [],
            "maxConnections": 50,
            "maxKeepaliveConnections": 10,
            "keepaliveTimeout": 30
        }
        
        response = await api_client.post("/api/proxy/start", json=config, timeout=10.0)
        if response.status_code != 201:
            raise RuntimeError(f"Failed to start proxy: {response.text}")
    
        await asyncio.sleep(2.0)

        timeout = Timeout(10.0, connect=5.0, read=5.0, write=5.0)
        limits = Limits(max_connections=50, max_keepalive_connections=10)
        client = None
        try:
            client = AsyncClient(
                base_url="http://httpbin",
                proxy="http://0.0.0.0:8080",
                verify=False,
                follow_redirects=True,
                timeout=timeout,
                limits=limits,
                headers=security_headers
            )

            if not await wait_for_proxy(client):
                raise RuntimeError("Proxy server did not become ready")

            yield {
                "client": client,
                "base_client": api_client,
                "config": config
            }
        except Exception as e:
            if client:
                await client.aclose()
            await ensure_proxy_stopped(api_client)
            raise RuntimeError(f"Failed to establish proxy connection: {e}")
        finally:
            if client:
                await client.aclose()
            await ensure_proxy_stopped(api_client)
            reset_state()
