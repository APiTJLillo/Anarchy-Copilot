import uuid
import logging
import asyncio
import ssl

logger = logging.getLogger(__name__)

class RemoteConnection:
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer
        
    async def send(self, data):
        self.writer.write(data)
        await self.writer.drain()
        
    async def receive(self):
        try:
            data = await self.reader.read(8192)  # Standard buffer size
            return data
        except Exception as e:
            logger.error(f"Error receiving data: {str(e)}")
            return None
            
    async def close(self):
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception as e:
            logger.error(f"Error closing connection: {str(e)}")

class ProxyServer:
    async def handle_connect(self, scope, receive, send):
        client_addr = scope.get('client', ('unknown', 0))[0]
        request_id = str(uuid.uuid4())
        
        try:
            target = self._parse_connect_target(scope, client_addr, request_id)
            host, port = target
            
            logger.info(f"[{client_addr}] [{request_id}] Establishing MITM tunnel to {host}:{port}")
            
            # Generate certificate for the domain
            cert = await self.cert_manager.get_cert_for_host(host)
            
            # Create connection to remote server with TLS
            remote_conn = await self._create_remote_connection(host, port, tls=True)
            
            # Send 200 Connection Established
            await send({
                'type': 'http.response.start',
                'status': 200,
                'headers': []
            })
            await send({
                'type': 'http.response.body',
                'body': b'',
                'more_body': False
            })

            # Create TLS context with our generated cert
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(cert.cert_path, cert.key_path)

            # Wrap client connection in TLS
            message = await receive()
            if message['type'] != 'http.request':
                raise Exception("Expected http.request message after CONNECT")

            client_stream = message.get('stream')
            if not client_stream:
                raise Exception("No stream available in ASGI scope")

            client_tls = await self._wrap_socket(client_stream, ssl_context, server_side=True)

            # Now we can intercept the decrypted traffic
            async def forward_client_to_remote():
                try:
                    while True:
                        data = await client_tls.read(8192)
                        if not data:
                            break
                        # Here we can inspect/modify the decrypted request
                        await self._log_request(request_id, data)
                        await remote_conn.send(data)
                except Exception as e:
                    logger.error(f"Error in client->remote: {str(e)}")

            async def forward_remote_to_client():
                try:
                    while True:
                        data = await remote_conn.receive()
                        if not data:
                            break
                        # Here we can inspect/modify the decrypted response
                        await self._log_response(request_id, data)
                        await client_tls.write(data)
                except Exception as e:
                    logger.error(f"Error in remote->client: {str(e)}")

            # Run both forwarding tasks concurrently
            forward_tasks = [
                asyncio.create_task(forward_client_to_remote()),
                asyncio.create_task(forward_remote_to_client())
            ]
            
            await asyncio.wait(forward_tasks, return_when=asyncio.FIRST_COMPLETED)
            
        except Exception as e:
            logger.error(f"[{client_addr}] [{request_id}] Error establishing tunnel: {str(e)}", exc_info=True)
            raise
        finally:
            if 'remote_conn' in locals():
                await remote_conn.close()

    async def _create_remote_connection(self, host, port, tls=False):
        try:
            if tls:
                reader, writer = await asyncio.open_connection(host, port, ssl=ssl.create_default_context())
            else:
                reader, writer = await asyncio.open_connection(host, port)
            return RemoteConnection(reader, writer)
        except Exception as e:
            logger.error(f"Failed to create remote connection to {host}:{port}: {str(e)}")
            raise 