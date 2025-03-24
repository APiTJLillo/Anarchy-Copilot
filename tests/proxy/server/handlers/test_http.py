"""Tests for HTTP request/response handling."""
import pytest
import h11
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from proxy.server.handlers.http import HttpRequestHandler

@pytest.fixture
def http_handler(connection_id):
    """Create a fresh HTTP handler for testing."""
    return HttpRequestHandler(connection_id)

@pytest.fixture
def sample_request():
    """Create a sample HTTP request."""
    return {
        "method": "GET",
        "url": "https://example.com/test",
        "headers": [
            (b"host", b"example.com"),
            (b"user-agent", b"test-client"),
            (b"content-length", b"0")
        ]
    }

@pytest.fixture
def sample_response():
    """Create a sample HTTP response."""
    return {
        "status_code": 200,
        "headers": [
            (b"content-type", b"text/plain"),
            (b"content-length", b"11"),
            (b"server", b"test-server")
        ],
        "body": b"Hello World"
    }

@pytest.mark.asyncio
async def test_request_handling(http_handler, sample_request):
    """Test basic request handling."""
    # Create h11 request event
    request = h11.Request(
        method=sample_request["method"].encode(),
        target=sample_request["url"].encode(),
        headers=sample_request["headers"]
    )
    
    # Send request through handler
    data = http_handler.client_conn.send(request)
    response_data = await http_handler.handle_client_data(data)
    
    # Verify request state
    assert http_handler._current_request is not None
    assert http_handler._current_request["method"] == sample_request["method"]
    assert http_handler._current_request["url"] == sample_request["url"]
    assert http_handler._current_request_start_time is not None

@pytest.mark.asyncio
async def test_response_handling(http_handler, sample_response):
    """Test basic response handling."""
    # Create h11 response event
    response = h11.Response(
        status_code=sample_response["status_code"],
        headers=sample_response["headers"],
        reason=b"OK"
    )
    
    # Send response through handler
    data = http_handler.server_conn.send(response)
    client_data = await http_handler.handle_server_data(data)
    
    # Verify response state
    assert http_handler._current_response is not None
    assert http_handler._current_response["status_code"] == sample_response["status_code"]
    assert len(http_handler._current_response["headers"]) == len(sample_response["headers"])

@pytest.mark.asyncio
async def test_complete_transaction(http_handler, sample_request, sample_response):
    """Test complete HTTP transaction flow."""
    # Send request
    request = h11.Request(
        method=sample_request["method"].encode(),
        target=sample_request["url"].encode(),
        headers=sample_request["headers"]
    )
    end = h11.EndOfMessage()
    
    request_data = http_handler.client_conn.send(request)
    await http_handler.handle_client_data(request_data)
    end_data = http_handler.client_conn.send(end)
    await http_handler.handle_client_data(end_data)
    
    # Send response
    response = h11.Response(
        status_code=sample_response["status_code"],
        headers=sample_response["headers"],
        reason=b"OK"
    )
    data = h11.Data(data=sample_response["body"])
    
    response_data = http_handler.server_conn.send(response)
    await http_handler.handle_server_data(response_data)
    body_data = http_handler.server_conn.send(data)
    await http_handler.handle_server_data(body_data)
    end_data = http_handler.server_conn.send(end)
    await http_handler.handle_server_data(end_data)
    
    # Verify transaction was completed and cleared
    assert http_handler._current_request is None
    assert http_handler._current_response is None
    assert len(http_handler._current_request_body) == 0
    assert len(http_handler._current_response_body) == 0

@pytest.mark.asyncio
async def test_request_body_handling(http_handler):
    """Test handling of requests with bodies."""
    # Create request with body
    headers = [
        (b"host", b"example.com"),
        (b"content-type", b"application/json"),
        (b"content-length", b"18")
    ]
    body = b'{"test": "value"}'
    
    request = h11.Request(method=b"POST", target=b"/api", headers=headers)
    data = h11.Data(data=body)
    end = h11.EndOfMessage()
    
    # Send request parts
    for event in [request, data, end]:
        event_data = http_handler.client_conn.send(event)
        await http_handler.handle_client_data(event_data)
    
    # Verify body was accumulated
    assert http_handler._current_request is not None
    assert http_handler._current_request["request_body"] == body

@pytest.mark.asyncio
async def test_response_body_handling(http_handler):
    """Test handling of responses with bodies."""
    # Create response with body
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", b"18")
    ]
    body = b'{"status":"ok"}'
    
    response = h11.Response(status_code=200, headers=headers, reason=b"OK")
    data = h11.Data(data=body)
    end = h11.EndOfMessage()
    
    # Send response parts
    for event in [response, data, end]:
        event_data = http_handler.server_conn.send(event)
        await http_handler.handle_server_data(event_data)
    
    # Verify body was accumulated
    assert http_handler._current_response is not None
    assert http_handler._current_response["body"] == body

@pytest.mark.asyncio
async def test_error_handling(http_handler):
    """Test error handling in request/response processing."""
    # Test invalid HTTP data
    invalid_data = b"NOT HTTP DATA"
    result = await http_handler.handle_client_data(invalid_data)
    assert result is None  # Should handle error gracefully
    
    # Test protocol error
    bad_request = h11.Request(
        method=b"GET",
        target=b"/test",
        headers=[(b"content-length", b"-1")]  # Invalid content length
    )
    data = http_handler.client_conn.send(bad_request)
    result = await http_handler.handle_client_data(data)
    assert result is None

def test_cleanup(http_handler):
    """Test cleanup of handler resources."""
    # Set some state
    http_handler._current_request = {"test": "data"}
    http_handler._current_response = {"test": "data"}
    http_handler._current_request_body.extend(b"test")
    http_handler._current_response_body.extend(b"test")
    
    # Cleanup
    http_handler.close()
    
    # Verify state was cleared
    assert http_handler._current_request is None
    assert http_handler._current_response is None
    assert len(http_handler._current_request_body) == 0
    assert len(http_handler._current_response_body) == 0

@pytest.mark.asyncio
async def test_connection_state_handling(http_handler):
    """Test connection state transitions."""
    # Send connection close
    close = h11.ConnectionClosed()
    
    # From client
    client_data = http_handler.client_conn.send(close)
    await http_handler.handle_client_data(client_data)
    assert http_handler.client_conn.states == {"START": h11.DONE}
    
    # From server
    server_data = http_handler.server_conn.send(close)
    await http_handler.handle_server_data(server_data)
    assert http_handler.server_conn.states == {"START": h11.DONE}

@pytest.mark.asyncio
async def test_history_entry_creation(http_handler, sample_request):
    """Test creation of history entries."""
    # Send request
    request = h11.Request(
        method=sample_request["method"].encode(),
        target=sample_request["url"].encode(),
        headers=sample_request["headers"]
    )
    end = h11.EndOfMessage()
    
    # Process request
    request_data = http_handler.client_conn.send(request)
    await http_handler.handle_client_data(request_data)
    end_data = http_handler.client_conn.send(end)
    await http_handler.handle_client_data(end_data)
    
    # Verify history entry was created
    assert http_handler._history_entry_id is not None
