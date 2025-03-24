"""Tests for buffered transport implementation."""
import pytest
import asyncio
import ssl
from unittest.mock import Mock, call

from proxy.server.tls.transport import BufferedTransport, BufferConfig, TransportMetrics

@pytest.mark.asyncio
async def test_basic_data_flow(buffered_transport, test_data):
    """Test basic data flow through transport."""
    # Mock target transport
    target = Mock()
    target.write.return_value = len(test_data)  # Simulate successful write
    buffered_transport.set_target(target)
    
    # Send data
    buffered_transport.data_received(test_data)
    
    # Allow event loop to process
    await asyncio.sleep(0.1)  # Give more time for processing
    
    # Process any pending writes
    buffered_transport._process_pending_writes()
    
    # Verify data was forwarded
    target.write.assert_called_once_with(test_data)
    
    # Verify metrics
    metrics = buffered_transport.get_metrics()
    assert metrics.bytes_received == len(test_data)
    assert metrics.bytes_sent == len(test_data)  # Should be updated after write

@pytest.mark.asyncio
async def test_write_operation(buffered_transport):
    """Test write operation basics."""
    data = b"test data"
    target = Mock()
    target.write.return_value = len(data)
    buffered_transport.set_target(target)

    # Direct write
    buffered_transport.write(data)
    await asyncio.sleep(0.1)

    # Verify write occurred
    target.write.assert_called_once_with(data)
    assert buffered_transport.get_metrics().bytes_sent == len(data)

@pytest.mark.asyncio
async def test_flow_control(buffered_transport):
    """Test flow control with large data chunks."""
    # Create data larger than chunk size
    chunk_size = buffered_transport.config.chunk_size
    large_data = b"X" * (chunk_size * 3)
    
    # Mock target transport
    target = Mock()
    target.write.side_effect = lambda data: len(data)  # Return bytes written
    buffered_transport.set_target(target)
    
    # Send large data
    buffered_transport.data_received(large_data)
    await asyncio.sleep(0.1)  # Allow time for processing
    
    # Verify data was split into chunks
    expected_chunks = [
        large_data[i:i + buffered_transport.config.chunk_size]
        for i in range(0, len(large_data), buffered_transport.config.chunk_size)
    ]
    
    assert target.write.call_count == len(expected_chunks)
    target.write.assert_has_calls([call(chunk) for chunk in expected_chunks])

@pytest.mark.asyncio
async def test_buffer_overflow_handling(buffered_transport):
    """Test handling of buffer overflow conditions."""
    # Create data larger than max buffer size
    overflow_data = b"X" * (buffered_transport.config.max_buffer_size * 2)
    
    # Mock target transport and make it appear busy
    target = Mock()
    buffered_transport.set_target(target)
    buffered_transport.pause_writing()  # Simulate backpressure
    
    # Send large data
    buffered_transport.data_received(overflow_data)
    
    # Verify metrics show overflow
    metrics = buffered_transport.get_metrics()
    assert metrics.buffer_overflows > 0
    assert metrics.current_buffer_size <= buffered_transport.config.max_buffer_size

@pytest.mark.asyncio
async def test_write_pausing(buffered_transport):
    """Test write pausing when buffer fills up."""
    # Create data chunk
    chunk = b"X" * buffered_transport.config.write_buffer_size
    
    # Mock target transport
    target = Mock()
    buffered_transport.set_target(target)
    
    # Setup pause callback
    pause_called = False
    def on_pause():
        nonlocal pause_called
        pause_called = True
    
    buffered_transport.register_flow_control_callbacks(
        pause_cb=on_pause,
        resume_cb=lambda: None
    )
    
    # Fill the buffer
    buffered_transport.pause_writing()
    for _ in range(3):  # Send enough to exceed high water mark
        buffered_transport.data_received(chunk)
    
    # Verify pause was triggered
    assert pause_called
    assert buffered_transport._write_paused

@pytest.mark.asyncio
async def test_write_resuming(buffered_transport):
    """Test write resuming when buffer drains."""
    # Setup resume callback
    resume_called = False
    def on_resume():
        nonlocal resume_called
        resume_called = True
    
    buffered_transport.register_flow_control_callbacks(
        pause_cb=lambda: None,
        resume_cb=on_resume
    )
    
    # Pause and then resume writing
    buffered_transport.pause_writing()
    buffered_transport.resume_writing()
    
    # Verify resume was triggered
    assert resume_called
    assert not buffered_transport._write_paused

@pytest.mark.asyncio
async def test_pending_writes_processing(buffered_transport):
    """Test processing of pending writes."""
    # Create test data
    chunks = [b"chunk1", b"chunk2", b"chunk3"]
    
    # Mock target transport
    target = Mock()
    buffered_transport.set_target(target)
    
    # Queue up some writes while paused
    buffered_transport.pause_writing()
    for chunk in chunks:
        buffered_transport._buffer_data(chunk)
    
    # Resume writing
    buffered_transport.resume_writing()
    
    # Verify all chunks were written
    target.write.assert_has_calls([call(chunk) for chunk in chunks])

def test_connection_cleanup(buffered_transport):
    """Test cleanup on connection loss."""
    # Setup mocks
    target = Mock()
    buffered_transport.set_target(target)
    
    # Add some pending data
    buffered_transport._buffer_data(b"test data")
    
    # Simulate connection loss
    buffered_transport.connection_lost(Exception("test error"))
    
    # Verify cleanup
    assert len(buffered_transport._pending_writes) == 0
    assert buffered_transport._transport is None
    assert buffered_transport._target_transport is None

@pytest.mark.asyncio
async def test_metrics_accuracy(buffered_transport):
    """Test accuracy of transport metrics."""
    # Create test data
    data_chunks = [b"chunk1", b"chunk2", b"chunk3"]
    total_bytes = sum(len(chunk) for chunk in data_chunks)
    
    # Mock target transport that tracks writes
    target = Mock()
    target.write.side_effect = lambda data: len(data)  # Return bytes written
    buffered_transport.set_target(target)
    
    # Send data with processing time
    for chunk in data_chunks:
        buffered_transport.data_received(chunk)
        await asyncio.sleep(0.1)  # Allow write processing
        buffered_transport._process_pending_writes()
    
    # Verify metrics
    metrics = buffered_transport.get_metrics()
    assert metrics.bytes_received == total_bytes
    assert metrics.bytes_sent == total_bytes
    assert metrics.peak_buffer_size >= metrics.current_buffer_size
    
    # Verify all data was written
    assert sum(len(c.args[0]) for c in target.write.call_args_list) == total_bytes

@pytest.mark.asyncio
async def test_error_handling(buffered_transport):
    """Test error handling during data transmission."""
    # Mock target transport that raises error
    target = Mock()
    target.write.side_effect = ConnectionError("Test error")
    buffered_transport.set_target(target)
    
    # Send data and verify error doesn't propagate
    try:
        buffered_transport.data_received(b"test data")
    except ConnectionError:
        pytest.fail("Error should have been handled")
    
    # Verify transport was closed
    assert buffered_transport._closed

def test_custom_buffer_config():
    """Test custom buffer configuration."""
    config = BufferConfig(
        chunk_size=1000,
        max_buffer_size=5000,
        write_buffer_size=2000,
        high_water_mark=0.9,
        low_water_mark=0.1
    )
    
    transport = BufferedTransport(
        connection_id="test",
        config=config
    )
    
    assert transport.config.chunk_size == 1000
    assert transport.config.max_buffer_size == 5000
    assert transport.config.write_buffer_size == 2000
    assert transport.config.high_water_mark == 0.9
    assert transport.config.low_water_mark == 0.1

@pytest.mark.asyncio
async def test_connection_interruption_recovery(buffered_transport):
    """Test recovery from connection interruptions."""
    # Mock target transport
    target = Mock()
    buffered_transport.set_target(target)
    
    # Setup data chunks
    chunks = [b"chunk1", b"chunk2", b"chunk3"]
    total_data = b"".join(chunks)
    
    # Initial successful write
    target.write.side_effect = lambda data: len(data)
    buffered_transport.data_received(chunks[0])
    await asyncio.sleep(0.1)
    
    # Simulate connection interruption during second chunk
    buffered_transport.pause_writing()
    target.write.side_effect = ConnectionError("Simulated interruption")
    buffered_transport.data_received(chunks[1])
    await asyncio.sleep(0.1)  # Let error be processed
    
    # Send final chunk while still interrupted
    buffered_transport.data_received(chunks[2])
    await asyncio.sleep(0.1)
    
    # Verify data is buffered
    assert len(buffered_transport._pending_writes) > 0
    
    # Restore connection and resume
    target.write.side_effect = lambda data: len(data)
    buffered_transport.resume_writing()
    await asyncio.sleep(0.1)  # Allow writes to process
    
    # Verify metrics and data
    metrics = buffered_transport.get_metrics()
    assert metrics.bytes_received == len(total_data)
    
    # Verify all data was eventually sent in order
    received_data = b"".join(call.args[0] for call in target.write.call_args_list)
    assert received_data == total_data

@pytest.mark.asyncio
async def test_partial_write_recovery(buffered_transport):
    """Test recovery from partial writes."""
    # Mock target that only writes partial data
    target = Mock()
    written_size = 0
    
    def partial_write(data):
        nonlocal written_size
        # Only write half the data each time
        write_size = len(data) // 2
        written_size += write_size
        return write_size
    
    target.write.side_effect = partial_write
    buffered_transport.set_target(target)
    
    # Send large data
    test_data = b"X" * 1000
    buffered_transport.data_received(test_data)
    
    # Verify data was split into multiple writes
    assert target.write.call_count > 1
    assert written_size > 0

@pytest.mark.asyncio
async def test_tls_renegotiation_handling(buffered_transport):
    """Test handling of TLS renegotiation."""
    # Mock target transport with renegotiation simulation
    target = Mock()
    renegotiation_state = {'triggered': False, 'completed': False, 'attempts': 0}
    
    def write_with_renegotiation(data):
        nonlocal renegotiation_state
        if not renegotiation_state['triggered'] and len(data) > 100:
            # Initial renegotiation trigger
            renegotiation_state['triggered'] = True
            renegotiation_state['attempts'] += 1
            raise ssl.SSLWantWriteError()
        
        if renegotiation_state['triggered'] and not renegotiation_state['completed']:
            # Still in renegotiation
            renegotiation_state['attempts'] += 1
            if renegotiation_state['attempts'] >= 3:  # Simulate completion after attempts
                renegotiation_state['completed'] = True
                return len(data)
            raise ssl.SSLWantWriteError()
        
        return len(data)  # Normal write after renegotiation
    
    target.write.side_effect = write_with_renegotiation
    buffered_transport.set_target(target)
    
    # Send data that will trigger renegotiation
    test_data = b"X" * 200
    buffered_transport.data_received(test_data)
    await asyncio.sleep(0.1)  # Allow initial write attempt
    
    # Should be buffering during renegotiation
    metrics = buffered_transport.get_metrics()
    assert metrics.bytes_received == len(test_data)
    assert len(buffered_transport._pending_writes) > 0
    
    # Simulate completion of renegotiation
    for _ in range(5):  # Multiple event loop iterations to process writes
        await asyncio.sleep(0.1)
        buffered_transport._process_pending_writes()
    
    # Verify final state
    assert renegotiation_state['completed']
    assert buffered_transport.get_metrics().bytes_sent == len(test_data)
    total_written = sum(len(call.args[0]) for call in target.write.call_args_list)
    assert total_written == len(test_data)

@pytest.mark.asyncio
async def test_buffer_memory_management(buffered_transport):
    """Test memory management during buffering."""
    initial_size = buffered_transport.get_metrics().current_buffer_size
    
    # Fill buffer
    large_data = b"X" * buffered_transport.config.max_buffer_size
    buffered_transport.pause_writing()
    buffered_transport.data_received(large_data)
    
    # Verify buffer size is capped
    metrics = buffered_transport.get_metrics()
    assert metrics.current_buffer_size <= buffered_transport.config.max_buffer_size
    assert metrics.peak_buffer_size <= buffered_transport.config.max_buffer_size
    
    # Clear buffer
    buffered_transport.resume_writing()
    buffered_transport._process_pending_writes()
    
    # Verify memory was released
    final_metrics = buffered_transport.get_metrics()
    assert final_metrics.current_buffer_size == initial_size
