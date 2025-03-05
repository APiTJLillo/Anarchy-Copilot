from typing import Optional, List, Set
import asyncio
import socket
from .server_state import ServerState
from .ssl_context import SSLContextManager
from .connection import ProxyConnection

class ProxyServer:
    state: ServerState
    _server: Optional[asyncio.AbstractServer]
    _socket: Optional[socket.socket]
    host: str
    port: int
    _cleanup_lock: asyncio.Lock
    ssl_manager: SSLContextManager

    def __init__(self, host: Optional[str] = None, 
                 port: Optional[int] = None,
                 cert_path: Optional[str] = None, 
                 key_path: Optional[str] = None) -> None: ...

    async def cleanup_resources(self) -> None: ...
    def _configure_socket(self, sock: socket.socket) -> socket.socket: ...
    def close(self) -> None: ...
    async def start(self) -> None: ...
    async def handle_client(self, reader: asyncio.StreamReader,
                          writer: asyncio.StreamWriter) -> None: ...
