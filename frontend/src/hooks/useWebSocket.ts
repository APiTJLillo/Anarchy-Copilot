import { useEffect, useRef, useState, useCallback } from 'react';

interface WebSocketHookOptions {
    onMessage?: (data: any) => void;
    onOpen?: () => void;
    onClose?: () => void;
    onError?: (error: Event) => void;
    reconnectAttempts?: number;
    reconnectInterval?: number;
    keepAlive?: boolean;
}

// Singleton WebSocket instance for persistent connections
const globalWebSockets = new Map<string, WebSocket>();

export const useWebSocket = (url: string, options: WebSocketHookOptions = {}) => {
    const {
        onMessage,
        onOpen,
        onClose,
        onError,
        reconnectAttempts = 5,
        reconnectInterval = 3000,
        keepAlive = true
    } = options;

    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<Event | null>(null);
    const ws = useRef<WebSocket | null>(null);
    const reconnectCount = useRef(0);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
    const mountedRef = useRef(true);

    const connect = useCallback(() => {
        try {
            // Check if we already have a connection for this URL
            let socket = keepAlive ? globalWebSockets.get(url) : null;

            // If no existing connection or it's closed, create a new one
            if (!socket || socket.readyState === WebSocket.CLOSED) {
                socket = new WebSocket(url);
                if (keepAlive) {
                    globalWebSockets.set(url, socket);
                }
            }

            socket.onopen = () => {
                console.log('WebSocket connected');
                if (mountedRef.current) {
                    setIsConnected(true);
                    setError(null);
                }
                reconnectCount.current = 0;
                if (onOpen) onOpen();
            };

            socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (onMessage) onMessage(data);
            };

            socket.onclose = () => {
                console.log('WebSocket closed');
                if (mountedRef.current) {
                    setIsConnected(false);
                }
                if (onClose) onClose();

                // Remove from global map if connection is closed
                if (keepAlive) {
                    globalWebSockets.delete(url);
                }

                // Attempt to reconnect
                if (reconnectCount.current < reconnectAttempts) {
                    reconnectCount.current += 1;
                    reconnectTimeoutRef.current = setTimeout(() => {
                        if (mountedRef.current) {
                            console.log(`Reconnecting... Attempt ${reconnectCount.current}`);
                            connect();
                        }
                    }, reconnectInterval);
                }
            };

            socket.onerror = (err) => {
                console.error('WebSocket error:', err);
                if (mountedRef.current) {
                    setError(err);
                }
                if (onError) onError(err);
            };

            ws.current = socket;
        } catch (err) {
            console.error('Failed to create WebSocket:', err);
            if (mountedRef.current) {
                setError(err as Event);
            }
        }
    }, [url, onMessage, onOpen, onClose, onError, reconnectAttempts, reconnectInterval, keepAlive]);

    useEffect(() => {
        mountedRef.current = true;

        // If we have an existing connection, use it
        const existingSocket = keepAlive ? globalWebSockets.get(url) : null;
        if (existingSocket && existingSocket.readyState === WebSocket.OPEN) {
            ws.current = existingSocket;
            setIsConnected(true);
            if (onOpen) onOpen();
        } else {
            connect();
        }

        return () => {
            mountedRef.current = false;
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            // Only close the socket if we're not keeping it alive
            if (!keepAlive && ws.current) {
                ws.current.close();
            }
        };
    }, [connect, url, keepAlive, onOpen]);

    const send = useCallback((data: any) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify(data));
        }
    }, []);

    return {
        isConnected,
        error,
        send
    };
};
