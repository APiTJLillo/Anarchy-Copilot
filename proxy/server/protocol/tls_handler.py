"""TLS tunnel establishment and certificate management."""
import asyncio
import socket
import ssl
import logging
from typing import Optional, Tuple, Dict, Any, Union, List, cast, TYPE_CHECKING
from dataclasses import dataclass, field

from .socket_helpers import get_raw_socket
from .ssl_transport import SslTransport
from .tls_handler_protocol import TlsHandlerProtocol
from .tls_handler_base_http import BaseHandlerProtocol, BaseHttpHandler
from .state_handler_protocol import StateHandlerProtocol
from .error_handler_protocol import ErrorHandlerProtocol
from .connection_info import ConnectionInfo
from ..tls.context_wrapper import get_server_context, get_client_context
from .base_types import TlsCapableProtocol, TlsContextProvider, TlsHandlerBase
from .base_h2 import H2EventHandlerMixin
from h2.connection import H2Connection
from h2.events import (
    RequestReceived, ResponseReceived,
    DataReceived, StreamEnded,
    StreamReset, SettingsAcknowledged,
    RemoteSettingsChanged, WindowUpdated,
    PriorityUpdated
)
from h2.config import H2Configuration
from h2.settings import SettingCodes
from h2.exceptions import ProtocolError

if TYPE_CHECKING:
    from proxy.interceptor import InterceptedRequest

logger = logging.getLogger("proxy.core")

def get_default_h2_settings() -> Dict[int, int]:
    """Get default HTTP/2 settings."""
    return {
        SettingCodes.HEADER_TABLE_SIZE: 4096,
        SettingCodes.INITIAL_WINDOW_SIZE: 65535,
        SettingCodes.MAX_CONCURRENT_STREAMS: 100,
        SettingCodes.MAX_HEADER_LIST_SIZE: 16384,
    }

@dataclass
class TlsHandlerState:
    """State container for TLS handler."""
    connection_id: str
    state_handler: Optional[StateHandlerProtocol] = None
    error_handler: Optional[ErrorHandlerProtocol] = None
    http_buffer: bytearray = field(default_factory=bytearray)
    header_block: Optional[bytes] = None
    current_request: Optional['InterceptedRequest'] = None
    is_request: bool = True
    database_interceptor: Any = None
    closing: bool = False
    http2_preface_seen: bool = False
    h2_conn: Optional[H2Connection] = None
    h2_settings: Dict[int, int] = field(default_factory=get_default_h2_settings)

class TlsHandler(TlsHandlerBase, H2EventHandlerMixin, TlsHandlerProtocol, BaseHttpHandler):
    """Handles TLS connections and certificate management."""

    def __init__(self, connection_id: str,
                state_handler: Optional[StateHandlerProtocol] = None,
                error_handler: Optional[ErrorHandlerProtocol] = None):
        """Initialize TLS handler."""
        TlsHandlerBase.__init__(self)
        H2EventHandlerMixin.__init__(self)
        self._streams: Dict[int, Dict[str, Any]] = {}

        # Initialize state
        self._state = TlsHandlerState(
            connection_id=connection_id,
            state_handler=state_handler,
            error_handler=error_handler
        )
        logger.debug(f"[{connection_id}] TLS handler initialized")

    @property
    def _connection_id(self) -> str:
        return self._state.connection_id

    @property
    def _database_interceptor(self) -> Any:
        return self._state.database_interceptor

    @property
    def _is_request(self) -> bool:
        return self._state.is_request

    @_is_request.setter
    def _is_request(self, value: bool) -> None:
        self._state.is_request = value

    @property
    def _http2_preface_seen(self) -> bool:
        return self._state.http2_preface_seen

    @property
    def _header_block(self) -> Optional[bytes]:
        return self._state.header_block

    @property
    def _current_request(self) -> Optional['InterceptedRequest']:
        return self._state.current_request

    @_current_request.setter 
    def _current_request(self, value: Optional['InterceptedRequest']) -> None:
        self._state.current_request = value

    async def _ensure_interceptor(self) -> None:
        """Initialize database interceptor."""
        if not self._state.database_interceptor:
            from proxy.interceptors.database import DatabaseInterceptor
            self._state.database_interceptor = DatabaseInterceptor(self._connection_id)
            logger.debug(f"[{self._connection_id}] Database interceptor initialized")

    async def process_decrypted_data(self, data: Union[bytes, bytearray], protocol: asyncio.Protocol) -> None:
        """Process decrypted data from the SSL socket."""
        try:
            # Convert to bytes if needed
            data_bytes = bytes(data) if isinstance(data, bytearray) else data
            
            # Add data to buffer
            self._state.http_buffer.extend(data_bytes)
            
            # Check for HTTP/2 preface
            if not self._state.http2_preface_seen and self._state.http_buffer.startswith(b'PRI * HTTP/2.0\r\n'):
                self._state.http2_preface_seen = True
                logger.debug(f"[{self._connection_id}] HTTP/2 connection preface detected")
                # Initialize HTTP/2 connection
                config = H2Configuration(
                    client_side=False,
                    header_encoding='utf-8'
                )
                self._state.h2_conn = H2Connection(config=config)
                if self._state.h2_conn:
                    self._state.h2_conn.initiate_connection()
                    # Apply settings
                    self._state.h2_conn.update_settings(self._state.h2_settings)
                    # Send initial settings
                    initial_data = self._state.h2_conn.data_to_send()
                    if initial_data:
                        protocol.data_received(bytes(initial_data))
                    # Process HTTP/2 data
                    if data_bytes:
                        await self._handle_http2_data(data_bytes, protocol)
                return
                
            # If we've seen HTTP/2 preface, handle as HTTP/2
            if self._state.http2_preface_seen and self._state.h2_conn:
                await self._handle_http2_data(bytes(self._state.http_buffer), protocol)
                return
                
            # Process HTTP/1.x messages
            await self._process_http1_data(protocol)
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error in process_decrypted_data: {e}")
            if self._state.error_handler:
                await self._state.error_handler.handle_error(e)
            protocol.data_received(data_bytes)
            self._state.http_buffer.clear()
            self._state.header_block = None

    async def _process_http1_data(self, protocol: asyncio.Protocol) -> None:
        """Process HTTP/1.x data."""
        while self._state.http_buffer:
            if b'\r\n\r\n' not in self._state.http_buffer:
                break
            
            # Split headers and body
            if not self._state.header_block:
                try:
                    header_block, rest = self._state.http_buffer.split(b'\r\n\r\n', 1)
                    self._state.header_block = bytes(header_block)
                    self._state.http_buffer = bytearray(rest)
                except ValueError:
                    # Incomplete message
                    break
            else:
                rest = self._state.http_buffer

            try:
                # Parse headers
                header_lines = self._state.header_block.split(b'\r\n')
                first_line = header_lines[0].decode('utf-8', errors='ignore')
                
                # Build headers dict
                headers: Dict[str, str] = {}
                content_length = 0
                
                for line in header_lines[1:]:
                    line = line.decode('utf-8', errors='ignore')
                    if ': ' in line:
                        name, value = line.split(': ', 1)
                        headers[name.lower()] = value
                        if name.lower() == 'content-length':
                            content_length = int(value)

                # Check complete message
                if len(rest) < content_length:
                    break
                
                # Extract body
                body = bytes(rest[:content_length]) if content_length > 0 else None
                
                # Process message
                if first_line.startswith(('GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH', 'CONNECT')):
                    method, target, *_ = first_line.split(' ')
                    await self._handle_request(method, target, headers, body, protocol)
                elif first_line.startswith('HTTP/'):
                    version, status_code, *_ = first_line.split(' ')
                    await self._handle_response(int(status_code), headers, body, protocol)
                else:
                    if body is not None:
                        await self._handle_raw_data(bytes(rest[:content_length]), protocol)
                    else:
                        await self._handle_raw_data(b'', protocol)

                # Update buffer state
                self._state.http_buffer = bytearray(rest[content_length:])
                self._state.header_block = None

            except Exception as e:
                logger.error(f"[{self._connection_id}] Error processing HTTP message: {e}")
                if self._state.error_handler:
                    await self._state.error_handler.handle_error(e)
                if self._state.http_buffer:
                    await self._handle_raw_data(bytes(self._state.http_buffer), protocol)
                self._state.http_buffer.clear()
                self._state.header_block = None
                break

    async def _handle_http2_data(self, data: bytes, protocol: asyncio.Protocol) -> None:
        """Handle HTTP/2 data."""
        if not self._state.h2_conn:
            return

        try:
            events = self._state.h2_conn.receive_data(data)
            processed = False

            for event in events:
                processed = True
                if isinstance(event, RequestReceived):
                    await super()._handle_h2_request(event)  # Call mixin method
                elif isinstance(event, ResponseReceived):
                    await super()._handle_h2_response(event)  # Call mixin method
                elif isinstance(event, DataReceived):
                    await super()._handle_h2_data_frame(event)  # Call mixin method
                elif isinstance(event, StreamEnded):
                    await self._handle_http2_stream(event.stream_id, protocol)
                elif isinstance(event, (SettingsAcknowledged, RemoteSettingsChanged)):
                    if (event.changed_settings.get(SettingCodes.INITIAL_WINDOW_SIZE) and 
                        self._state.h2_conn is not None):
                        value = event.changed_settings[SettingCodes.INITIAL_WINDOW_SIZE].new_value
                        self._state.h2_conn.increment_flow_control_window(value)

            # Send any pending data
            if self._state.h2_conn and (data_to_send := self._state.h2_conn.data_to_send()):
                protocol.data_received(bytes(data_to_send))

            # Forward original data if not processed
            if not processed:
                protocol.data_received(data)

            # Clear buffer
            self._state.http_buffer.clear()

        except ProtocolError as e:
            logger.error(f"[{self._connection_id}] HTTP/2 protocol error: {e}")
            if self._state.error_handler:
                await self._state.error_handler.handle_error(e)
            if self._state.h2_conn:
                try:
                    self._state.h2_conn.close_connection()
                    if data_to_send := self._state.h2_conn.data_to_send():
                        protocol.data_received(bytes(data_to_send))
                except Exception as err:
                    if self._state.error_handler:
                        await self._state.error_handler.handle_error(err)
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error handling HTTP/2 data: {e}")
            if self._state.error_handler:
                await self._state.error_handler.handle_error(e)
            protocol.data_received(data)
            self._state.http_buffer.clear()

    async def _handle_http2_stream(self, stream_id: int, protocol: asyncio.Protocol) -> None:
        """Handle a complete HTTP/2 stream."""
        try:
            stream_data = self._streams[stream_id]
            await self._ensure_interceptor()
            
            if not stream_data.get('request_handled'):
                from proxy.interceptor import InterceptedRequest
                request = InterceptedRequest(
                    method=stream_data['method'],
                    url=stream_data['url'],
                    headers=stream_data['headers'],
                    body=bytes(stream_data['body']),
                    connection_id=self._connection_id
                )
                if self._state.database_interceptor:
                    await self._state.database_interceptor.intercept(request)
                    stream_data['request_handled'] = True
                    stream_data['intercepted_request'] = request
            
            if 'status' in stream_data and not stream_data.get('response_handled'):
                from proxy.interceptor import InterceptedResponse
                response = InterceptedResponse(
                    status_code=stream_data['status'],
                    headers=stream_data['response_headers'],
                    body=bytes(stream_data['response_body']),
                    connection_id=self._connection_id
                )
                
                if self._state.database_interceptor:
                    await self._state.database_interceptor.intercept(
                        response,
                        stream_data.get('intercepted_request')
                    )
                    stream_data['response_handled'] = True
            
            # Clean up stream after handling
            if stream_data.get('request_handled') and stream_data.get('response_handled'):
                del self._streams[stream_id]
                
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error handling HTTP/2 stream {stream_id}: {e}")
            if self._state.error_handler:
                await self._state.error_handler.handle_error(e)

    def get_context(self, hostname: str, is_server: bool = True) -> ssl.SSLContext:
        """Get SSL context for the given hostname."""
        if not hostname:
            raise ValueError("Hostname is required")

        if is_server:
            context = get_server_context(hostname)
            if not isinstance(context, ssl.SSLContext):
                raise TypeError("Expected SSLContext from get_server_context")
            return context
        context = get_client_context(hostname)  # Pass hostname here
        if not isinstance(context, ssl.SSLContext):
            raise TypeError("Expected SSLContext from get_client_context")  
        return context

    async def wrap_client(self,
                       protocol: TlsCapableProtocol,
                       server_hostname: str,
                       alpn_protocols: Optional[List[str]] = None) -> Tuple[asyncio.Transport, TlsCapableProtocol]:
        """Wrap client connection with TLS."""
        try:
            if not server_hostname:
                raise ValueError("server_hostname is required")

            context = self.get_context(hostname=server_hostname, is_server=False)  # Use get_context method with hostname
            if alpn_protocols:
                context.set_alpn_protocols(alpn_protocols)
            
            transport = protocol.transport
            if not transport:
                raise RuntimeError("No transport available")
            
            ssl_transport = await SslTransport.create(
                self,
                transport,
                context,  # Use the context from get_context
                server_side=False,
                server_hostname=server_hostname
            )
            
            # Get connection info for state tracking
            peer_info = transport.get_extra_info('peername')
            port = peer_info[1] if peer_info else 0
            
            conn_info = ConnectionInfo(
                hostname=server_hostname,
                port=port,
                alpn_protocol=ssl_transport.get_extra_info('alpn_protocol'),
                peer_cert=ssl_transport.get_extra_info('peercert')
            )

            if self._state.state_handler:
                self._state.state_handler.update_stats(
                    connection_id=self._connection_id,
                    info=conn_info
                )
            
            return ssl_transport, protocol
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to wrap client connection: {e}")
            if self._state.error_handler:
                await self._state.error_handler.handle_error(e)
            raise

    async def _handle_response(self, status_code: int, headers: Dict[str, str], 
                             body: Optional[bytes], protocol: asyncio.Protocol) -> None:
        """Handle intercepted response."""
        try:
            from proxy.interceptor import InterceptedResponse

            response = InterceptedResponse(
                status_code=status_code,
                headers=headers,
                body=body if body else b'',
                connection_id=self._connection_id
            )

            if self._state.database_interceptor and self._current_request:
                await self._state.database_interceptor.intercept(
                    response,
                    self._current_request
                )

            # Track state
            if self._state.state_handler:
                self._state.state_handler.update_stats(
                    connection_id=self._connection_id,
                    status_code=status_code,
                    event='response'
                )

        except Exception as e:
            logger.error(f"[{self._connection_id}] Error handling response: {e}")
            if self._state.error_handler:
                await self._state.error_handler.handle_error(e)
            raise

    def update_connection_stats(self, connection_id: str, **kwargs: Any) -> None:
        """Update connection statistics."""
        if (connection_id == self._connection_id and 
            self._state.state_handler):
            try:
                self._state.state_handler.update_stats(**kwargs)
            except Exception as e:
                logger.error(f"[{self._connection_id}] Failed to update stats: {e}")

    def close_connection(self, connection_id: str) -> None:
        """Close connection and clean up."""
        if connection_id != self._connection_id:
            return
            
        self._state.closing = True
        if self._state.database_interceptor:
            asyncio.create_task(self._state.database_interceptor.close())

    @property
    def closing(self) -> bool:
        """Check if handler is closing."""
        return self._state.closing
