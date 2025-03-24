"""Test WebSocket connectivity."""
import asyncio
import websockets
import json
import logging
import os
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logging.getLogger("websockets").setLevel(logging.DEBUG)

# Use environment-aware WebSocket URL
WS_HOST = os.environ.get("WS_HOST", "dev")  # Default to 'dev' for Docker environment
WS_PORT = os.environ.get("WS_PORT", "8000")
WS_URL = f"ws://{WS_HOST}:{WS_PORT}/api/proxy/ws"  # Note: No trailing slash

RESPONSE_TIMEOUT = 10  # seconds

async def wait_for_message(websocket, timeout):
    """Wait for a message with timeout."""
    try:
        return await asyncio.wait_for(websocket.recv(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"Timeout waiting for response after {timeout} seconds")
        raise
    except Exception as e:
        logger.error(f"Error receiving message: {str(e)}")
        raise

async def test_ws_connection():
    """Test WebSocket connection and heartbeat."""
    logger.info(f"Attempting to connect to: {WS_URL}")
    logger.info("Connection details:")
    logger.info(f"  Host: {WS_HOST}")
    logger.info(f"  Port: {WS_PORT}")
    logger.info(f"  Full URL: {WS_URL}")
    
    try:
        async with websockets.connect(
            WS_URL,
            ping_interval=20,
            max_size=None,
            compression=None,  # Disable compression for clearer debugging
        ) as websocket:
            logger.info("Connected to WebSocket server")
            
            # Wait for initial state message
            logger.info("Waiting for initial state...")
            initial_msg = await wait_for_message(websocket, RESPONSE_TIMEOUT)
            logger.info(f"Received initial state: {initial_msg}")
            
            # Send a heartbeat
            heartbeat = {
                "type": "heartbeat",
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.info("Sending heartbeat...")
            await websocket.send(json.dumps(heartbeat))
            logger.info("Heartbeat sent")
            
            # Wait for heartbeat response
            logger.info("Waiting for heartbeat response...")
            response = await wait_for_message(websocket, RESPONSE_TIMEOUT)
            logger.info(f"Received heartbeat response: {response}")
            
            # Send status request
            status_req = {
                "type": "get_status",
                "channel": "proxy"  # Added channel
            }
            logger.info("Sending status request...")
            await websocket.send(json.dumps(status_req))
            logger.info("Status request sent")
            
            # Wait for status response
            logger.info("Waiting for status response...")
            status_resp = await wait_for_message(websocket, RESPONSE_TIMEOUT)
            logger.info(f"Received status response: {status_resp}")
            
            # Parse and validate the status response
            try:
                status_data = json.loads(status_resp)
                if status_data.get("type") != "status_update":
                    raise ValueError(f"Expected status_update message, got: {status_data.get('type')}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse status response: {e}")
                raise
            except KeyError as e:
                logger.error(f"Missing required field in status response: {e}")
                raise

    except websockets.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        raise
    except Exception as e:
        logger.error(f"WebSocket test failed: {e}")
        raise

async def main():
    """Run the test."""
    retry_count = 0
    max_retries = 5
    retry_delay = 2

    while retry_count < max_retries:
        try:
            await test_ws_connection()
            logger.info("WebSocket test completed successfully")
            return
        except Exception as e:
            retry_count += 1
            logger.error(f"Test attempt {retry_count} failed: {str(e)}")
            if retry_count < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error("Max retries reached. Test failed.")
                sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(0)
