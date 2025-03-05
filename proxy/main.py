"""Main entry point for the proxy server."""

import asyncio
import signal
import sys
from typing import Optional
from contextlib import suppress

from .utils.logging import logger
from .utils.constants import NetworkConfig
from .models.server import ProxyServer

async def shutdown(sig: signal.Signals, loop: asyncio.AbstractEventLoop, server: Optional[ProxyServer]) -> None:
    """Handle graceful shutdown."""
    logger.info(f'Received exit signal {sig.name}...')

    if server:
        logger.info('Closing server')
        server.close()

        try:
            # Wait for shutdown with timeout
            logger.info('Waiting for active connections to close...')
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    server.state.wait_for_shutdown(),
                    timeout=NetworkConfig.SHUTDOWN_TIMEOUT
                )
        except Exception as e:
            logger.error(f'Error waiting for shutdown: {e}')

        logger.info('Cleaning up resources')
        try:
            await asyncio.wait_for(
                server.cleanup_resources(),
                timeout=NetworkConfig.CLEANUP_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning('Resource cleanup timed out')
        except Exception as e:
            logger.error(f'Error during resource cleanup: {e}')

    # Cancel pending tasks
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    if tasks:
        logger.info(f'Cancelling {len(tasks)} outstanding tasks')
        for task in tasks:
            task.cancel()
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=NetworkConfig.CLEANUP_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning('Some tasks did not complete in time')
        except Exception as e:
            logger.error(f'Error during task cleanup: {e}')

    logger.info('Stopping event loop')
    try:
        # Stop accepting new tasks
        loop.stop()
        
        # Run final iteration to clean up
        with suppress(Exception):
            loop.run_until_complete(asyncio.sleep(0))
    except Exception as e:
        logger.error(f'Error stopping event loop: {e}')
    
    # Force final garbage collection
    import gc
    gc.collect()
    logger.info('Shutdown complete')

def handle_exception(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """Handle exceptions that escape the event loop."""
    exception = context.get('exception')
    message = context.get('message')
    
    if exception:
        if isinstance(exception, (KeyboardInterrupt, SystemExit)):
            logger.info('Received interrupt signal')
            asyncio.create_task(shutdown(signal.SIGINT, loop, None))
            return

    logger.error(
        'Unhandled error in event loop: %s',
        message or str(exception),
        exc_info=bool(exception)
    )
    
    # Log additional context
    for key in ('future', 'handle', 'protocol', 'transport'):
        if key in context:
            logger.error('%s: %r', key.title(), context[key])
    
    logger.info('Initiating shutdown due to error')
    try:
        asyncio.create_task(shutdown(signal.SIGTERM, loop, None))
    except Exception as e:
        logger.error(f'Error initiating shutdown: {e}')
        sys.exit(1)

async def run_server(host: Optional[str] = None, 
                    port: Optional[int] = None,
                    cert_path: Optional[str] = None,
                    key_path: Optional[str] = None) -> None:
    """Run the proxy server with the given configuration."""
    # Create and configure event loop
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    
    # Set up exception handler
    loop.set_exception_handler(handle_exception)
    
    # Create proxy server
    logger.info('Initializing proxy server')
    server = ProxyServer(
        host=host,
        port=port,
        cert_path=cert_path,
        key_path=key_path
    )
    
    # Set up signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(
                    shutdown(s, loop, server)
                )
            )
            logger.info(f'Signal handler set up for {sig.name}')
        except NotImplementedError:
            # Windows doesn't support SIGHUP
            if sig != signal.SIGHUP:
                raise
    
    # Start the server
    try:
        logger.info('Starting proxy server...')
        await server.start()
    except KeyboardInterrupt:
        logger.info('Received keyboard interrupt')
    except Exception as e:
        logger.error(f'Error running server: {e}', exc_info=True)
        raise
    finally:
        # Run cleanup
        logger.info('Running final cleanup...')
        await shutdown(signal.SIGTERM, loop, server)

def main():
    """Command-line entry point."""
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info('Exiting due to keyboard interrupt')
    except Exception as e:
        logger.error(f'Fatal error: {e}', exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
