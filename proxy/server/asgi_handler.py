async def handle_connect(self, scope, receive, send):
    client_addr = scope.get('client', ('unknown', 0))[0]
    try:
        # First send 200 OK to establish tunnel
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

        # Handle the tunnel through the proxy server
        await self.proxy_server.handle_connect(scope, receive, send)
        
    except Exception as e:
        logger.error(f"[{client_addr}] Error handling CONNECT request: {str(e)}", exc_info=True)
        await send({
            'type': 'http.response.start',
            'status': 502,
            'headers': [(b'content-type', b'text/plain')]
        })
        await send({
            'type': 'http.response.body',
            'body': b'Bad Gateway',
        }) 