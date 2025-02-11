# Proxy Module Implementation

This PR adds a comprehensive proxy module to Anarchy Copilot, providing features similar to Burp Suite while integrating with our existing reconnaissance and vulnerability scanning capabilities.

## Features

### Core Proxy Features
- HTTP/HTTPS traffic interception
- Request/response modification
- Session management and history tracking
- Certificate authority for HTTPS inspection
- Configurable scope and filtering
- Plugin system for custom interceptors

### Frontend Integration
- Real-time proxy status monitoring
- Request/response history viewer
- Request modification interface
- Tag and note management
- Proxy settings configuration

### API Endpoints
- `GET /api/proxy/status` - Get proxy status and history
- `POST /api/proxy/start` - Start proxy server
- `POST /api/proxy/stop` - Stop proxy server
- `POST /api/proxy/settings` - Update proxy settings
- `GET /api/proxy/history/{entry_id}` - Get detailed history entry
- `POST /api/proxy/history/{entry_id}/tags` - Add tags to history entry
- `POST /api/proxy/history/{entry_id}/notes` - Update entry notes
- `POST /api/proxy/history/clear` - Clear proxy history

## Testing

### Manual Testing Steps

1. Start the development environment:
```bash
docker-compose up -d
```

2. Install the CA certificate:
```bash
# The certificate will be generated at ./certs/ca.crt
# Import this into your browser/system trust store
```

3. Configure your browser to use the proxy:
- Proxy address: `localhost`
- Port: `8080`

4. Visit the web interface:
- Open `http://localhost:3000`
- Navigate to the Proxy Dashboard
- Click "Start Proxy"

5. Test basic functionality:
- Browse some websites and verify traffic appears in history
- Try adding tags and notes to requests
- Test request modification
- Verify HTTPS interception works

### Running Tests
```bash
# Install dependencies
pip install -e ".[dev]"
pip install -r tests/requirements-proxy-test.txt

# Run proxy-specific tests
make test-proxy

# Run all tests including proxy
make test
```

## Configuration

### Environment Variables
- `PROXY_HOST`: Proxy server host (default: 127.0.0.1)
- `PROXY_PORT`: Proxy server port (default: 8080)
- `CA_CERT_PATH`: Path to CA certificate
- `CA_KEY_PATH`: Path to CA private key

### Docker Configuration
The proxy service is configured in `docker-compose.yml` with:
- Port 8080 exposed for proxy traffic
- Volume mounts for certificates and history
- Required environment variables

## Known Limitations
- WebSocket support is basic and may need enhancement
- No authentication yet for proxy management endpoints
- Performance testing needed for high traffic scenarios

## Future Improvements
- Add authentication for proxy management
- Enhance WebSocket support
- Add more built-in security checks
- Implement request/response pattern matching
- Add export/import functionality for history
- Integrate with other scanning tools

## Breaking Changes
None. This is a new module that doesn't affect existing functionality.

## Related Issues
Closes #XXX - Add proxy functionality similar to Burp Suite
