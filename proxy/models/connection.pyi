from typing import Optional, Tuple, Any
import asyncio
import socket
import ssl
from .server_state import ServerState
from .ssl_context import SSLContextManager

class ProxyConnection:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    ssl_manager: SSLContextManager
    state: ServerState
    client_addr: Tuple[str, int]
    hostname: Optional[str]
    _closing: bool
    _cleanup_lock: asyncio.Lock
    _server_transport: Optional[asyncio.BaseTransport]
    _client_transport: Optional[asyncio.BaseTransport]

    def __init__(self, 
                 reader: asyncio.StreamReader,
                 writer: asyncio.StreamWriter,
                 ssl_manager: SSLContextManager,
                 server_state: ServerState) -> None: ...

    async def cleanup(self) -> None: ...
    
    async def __aenter__(self) -> 'ProxyConnection': ...
    async def __aexit__(self, 
                       exc_type: Optional[type],
                       exc_val: Optional[Exception],
                       exc_tb: Optional[object]) -> None: ...

    async def handle_connect(self, hostname: str, port: int) -> None: ...

    async def _setup_tls_tunnel(self,
                              sock: socket.socket,
                              ssl_context: ssl.SSLContext,
                              server_reader: asyncio.StreamReader,
                              server_writer: asyncio.StreamWriter) -> None: ...

    @staticmethod
    def _is_fatal_ssl_error(error_msg: str) -> bool: ...

    async def _proxy_data(self,
                       client_reader: asyncio.StreamReader,
                       client_writer: asyncio.StreamWriter,
                       server_reader: asyncio.StreamReader,
                       server_writer: asyncio.StreamWriter) -> None: ...

    @property
    def is_active(self) -> bool: ...
    
    @property
    def connection_info(self) -> dict[str, Any]: ...
    
    def get_transfer_stats(self) -> dict[str, int]: ...
    
    async def send_keepalive(self) -> bool: ...
    
    def get_cipher_info(self) -> Optional[dict[str, Any]]: ...
    
    async def graceful_shutdown(self, timeout: float = 5.0) -> bool: ...
