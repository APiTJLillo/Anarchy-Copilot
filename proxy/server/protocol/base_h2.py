"""Base definitions for HTTP/2 handling."""
from typing import Dict, Any, Protocol

from h2.events import RequestReceived, ResponseReceived, DataReceived

class H2StreamData(Protocol):
    """Protocol defining expected stream data structure."""
    method: str
    url: str
    headers: Dict[str, str]
    body: bytearray
    complete: bool
    status: int
    response_headers: Dict[str, str]
    response_body: bytearray

class H2EventHandlerMixin:
    """Mixin providing HTTP/2 event handling methods."""
    
    def __init__(self) -> None:
        """Initialize H2 event handler."""
        self._streams: Dict[int, Dict[str, Any]] = {}

    async def _handle_h2_request(self, event: RequestReceived) -> None:
        """Handle HTTP/2 request event."""
        headers = dict(event.headers)
        method = headers.get(b':method', b'GET').decode('utf-8')
        scheme = headers.get(b':scheme', b'https').decode('utf-8')
        authority = headers.get(b':authority', b'').decode('utf-8')
        path = headers.get(b':path', b'/').decode('utf-8')
        
        # Create URL
        url = f"{scheme}://{authority}{path}"
        
        # Convert headers
        standard_headers = {
            k.decode('utf-8').strip(':').lower(): v.decode('utf-8')
            for k, v in headers.items()
            if not k.startswith(b':')
        }
        
        # Store stream data
        self._streams[event.stream_id] = {
            'method': method,
            'url': url,
            'headers': standard_headers,
            'body': bytearray(),
            'complete': False,
            'response_body': bytearray(),
            'response_headers': {}
        }

    async def _handle_h2_response(self, event: ResponseReceived) -> None:
        """Handle HTTP/2 response event."""
        headers = dict(event.headers)
        status = int(headers.get(b':status', b'200').decode('utf-8'))
        
        # Convert headers
        standard_headers = {
            k.decode('utf-8').strip(':').lower(): v.decode('utf-8')
            for k, v in headers.items()
            if not k.startswith(b':')
        }
        
        # Update stream data if exists
        if event.stream_id in self._streams:
            self._streams[event.stream_id].update({
                'status': status,
                'response_headers': standard_headers,
            })

    async def _handle_h2_data_frame(self, event: DataReceived) -> None:
        """Handle HTTP/2 data frame event."""
        if event.stream_id in self._streams:
            stream_data = self._streams[event.stream_id]
            if 'status' not in stream_data:
                # Request body
                if isinstance(event.data, (bytes, bytearray)):
                    stream_data['body'].extend(event.data)
            else:
                # Response body
                if isinstance(event.data, (bytes, bytearray)):
                    stream_data['response_body'].extend(event.data)
