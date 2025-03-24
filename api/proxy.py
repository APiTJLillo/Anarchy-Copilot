"""Main proxy API module re-exports."""
from .proxy import router  # Export FastAPI router
from .proxy import proxy_server  # Export global proxy server instance
from .proxy import reset_state  # Export reset function

# Export all endpoint handlers
from .proxy.endpoints import (
    get_proxy_status,
    start_proxy,
    stop_proxy,
    get_proxy_history,
    get_history_entry,
    intercept_request,
    intercept_response,
    websocket_endpoint
)

# Export models
from .proxy.models import (
    ProxySettings,
    InterceptedRequest,
    InterceptedResponse,
    HistoryEntryResponse,
    Header,
    TagData,
    NoteData
)

__all__ = [
    'router',
    'proxy_server',
    'reset_state',
    'get_proxy_status',
    'start_proxy',
    'stop_proxy',
    'get_proxy_history',
    'get_history_entry',
    'intercept_request',
    'intercept_response',
    'websocket_endpoint',
    'ProxySettings',
    'InterceptedRequest',
    'InterceptedResponse',
    'HistoryEntryResponse',
    'Header',
    'TagData',
    'NoteData'
]
