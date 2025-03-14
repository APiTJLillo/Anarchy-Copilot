"""Protocol definitions for TLS handlers."""
import asyncio
from typing import Protocol, runtime_checkable

@runtime_checkable
class TlsHandlerProtocol(Protocol):
    """Protocol defining required TLS handler methods."""
    async def process_decrypted_data(self, data: bytes, protocol: asyncio.Protocol) -> None: ...
