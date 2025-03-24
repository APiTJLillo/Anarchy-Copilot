"""Utility functions for proxy management."""
import asyncio
import logging
import socket
import subprocess
import psutil
from typing import Optional, List

async def try_close_sockets(port: int, logger: logging.Logger) -> bool:
    """Try to close any sockets on the given port."""
    sock = None
    try:
        # Try to bind to check if port is already free
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('0.0.0.0', port))
            return True
        except Exception:
            pass

        # Try to find and terminate processes using the port
        for attempt in range(3):
            try:
                for proc in psutil.process_iter(['pid', 'connections']):
                    try:
                        for conn in proc.info['connections']:
                            if hasattr(conn, 'laddr') and isinstance(conn.laddr, tuple):
                                if len(conn.laddr) >= 2 and conn.laddr[1] == port:
                                    proc.terminate()  # Try graceful shutdown first
                                    break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                await asyncio.sleep(1.0)
                # Check if port is now free
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('0.0.0.0', port))
                return True
            except Exception:
                await asyncio.sleep(1.0)
                continue

        return False
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass

async def find_processes_using_port(port: int) -> List[int]:
    """Find all processes using a specific port."""
    pids = set()
    try:
        # Check using lsof first (most reliable)
        output = subprocess.check_output(
            f"lsof -t -i:{port}",
            shell=True,
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        if output:
            pids.update(int(pid) for pid in output.split() if pid.isdigit())
    except Exception:
        pass

    # Also check using psutil
    try:
        for proc in psutil.process_iter(['pid', 'connections']):
            try:
                for conn in proc.info['connections']:
                    if hasattr(conn, 'laddr') and isinstance(conn.laddr, tuple):
                        if len(conn.laddr) >= 2 and conn.laddr[1] == port:
                            pids.add(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass

    return list(pids)

async def cleanup_port(port: int, logger: logging.Logger) -> bool:
    """Attempt to cleanup port usage."""
    try:
        # First try to close sockets gracefully
        if await try_close_sockets(port, logger):
            return True

        # Find all processes using the port
        pids = await find_processes_using_port(port)
        if not pids:
            return True

        # Try SIGTERM first
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                proc.terminate()
            except Exception:
                pass

        # Wait and check if port is free
        await asyncio.sleep(2.0)
        if await try_close_sockets(port, logger):
            return True

        # Try SIGKILL if needed
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                if proc.is_running():
                    proc.kill()
            except Exception:
                pass

        # Final check
        await asyncio.sleep(1.0)
        return await try_close_sockets(port, logger)

    except Exception as e:
        logger.warning(f"Port cleanup failed: {e}")
        return False

def is_valid_port(port: int) -> bool:
    """Check if a port number is valid."""
    return isinstance(port, int) and 0 < port < 65536

def get_free_port(start_port: int = 8083, max_attempts: int = 100) -> Optional[int]:
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.bind(('0.0.0.0', port))
            return port
        except Exception:
            continue
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
    return None
