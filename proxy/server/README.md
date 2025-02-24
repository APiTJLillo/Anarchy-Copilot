# HTTPS Interception Proxy Architecture

This directory contains the core HTTPS interception proxy implementation, split into modular components for maintainability and testability.

## Architecture Overview

```
proxy/server/
├── tls/                    # TLS/SSL handling components
│   ├── connection_manager  # Connection tracking and metrics
│   ├── context            # TLS context creation and configuration
│   └── transport          # Buffered transport with flow control
├── handlers/              # Protocol handlers
│   ├── http              # HTTP request/response processing
│   └── connect           # CONNECT method and tunnel setup
└── https_intercept_protocol.py  # Main protocol orchestration
```

## Component Responsibilities

### TLS Components

- **connection_manager.py**
  - Tracks active connections and their states
  - Maintains connection metrics
  - Handles WebSocket updates for UI
  - Provides global connection monitoring

- **context.py**
  - Creates and configures TLS contexts
  - Manages cipher suites and security settings
  - Handles certificate operations
  - Provides TLS connection information

- **transport.py**
  - Implements buffered data transport
  - Handles flow control
  - Manages write buffering and backpressure
  - Tracks transport metrics

### Protocol Handlers

- **http.py**
  - Processes HTTP requests and responses
  - Manages request/response state
  - Handles HTTP protocol (h11) events
  - Records transaction history

- **connect.py**
  - Handles CONNECT method requests
  - Sets up TLS tunneling
  - Manages connection establishment
  - Coordinates client/server transports

### Main Protocol

- **https_intercept_protocol.py**
  - Orchestrates components
  - Manages connection lifecycle
  - Routes data between handlers
  - Handles high-level errors

## Data Flow

1. Connection established → `HttpsInterceptProtocol` created
2. CONNECT request → `ConnectHandler` establishes tunnel
3. TLS setup → `TlsContextFactory` creates contexts
4. Data flow begins → `BufferedTransport` handles transmission
5. HTTP processing → `HttpRequestHandler` processes messages
6. Metrics/state → `ConnectionManager` tracks everything

## Key Features

- Modular design for maintainability
- Comprehensive error handling
- Flow control and backpressure
- Connection monitoring and metrics
- Clean separation of concerns
- Buffered data handling
- Protocol state management

## Usage Example

```python
from proxy.server.https_intercept_protocol import HttpsInterceptProtocol
from proxy.server.certificates import CertificateAuthority

# Create CA for certificate generation
ca = CertificateAuthority("ca.crt", "ca.key")

# Create protocol factory
factory = HttpsInterceptProtocol.create_protocol_factory(ca=ca)

# Use with asyncio server
server = await loop.create_server(
    factory,
    host="127.0.0.1",
    port=8443
)
```

## Testing

Each component has its own test suite:

```bash
# Run specific component tests
pytest tests/proxy/server/tls/test_connection_manager.py
pytest tests/proxy/server/tls/test_context.py
pytest tests/proxy/server/tls/test_transport.py
pytest tests/proxy/server/handlers/test_http.py
pytest tests/proxy/server/handlers/test_connect.py

# Run all proxy tests
pytest tests/proxy/
```

## Future Improvements

- HTTP/2 support
- WebSocket tunneling
- Protocol switching
- Enhanced metrics
- Traffic replay capabilities
- Custom protocol handlers
