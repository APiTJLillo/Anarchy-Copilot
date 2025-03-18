import { useEffect, useRef, useState, useCallback } from 'react';

interface WebSocketHookOptions<T = any> {
    onMessage?: (data: T) => void;
    onOpen?: () => void;
    onClose?: () => void;
    onError?: (error: Event) => void;
    reconnectAttempts?: number;
    reconnectInterval?: number;
    keepAlive?: boolean;
}

interface WebSocketConnection<T = any> {
    socket: WebSocket;
    subscribers: Set<(data: T) => void>;
}

// Singleton WebSocket instance and subscriber management
const globalWebSockets = new Map<string, WebSocketConnection>();

const addSubscriber = <T>(url: string, onMessage: (data: T) => void) => {
    const connection = globalWebSockets.get(url);
    if (connection) {
        connection.subscribers.add(onMessage);
    }
};

const removeSubscriber = <T>(url: string, onMessage: (data: T) => void) => {
    const connection = globalWebSockets.get(url);
    if (connection) {
        connection.subscribers.delete(onMessage);
        // If no more subscribers and socket is closed, clean up
        if (connection.subscribers.size === 0 && connection.socket.readyState === WebSocket.CLOSED) {
            globalWebSockets.delete(url);
        }
    }
};

export const useWebSocket = <T = any>(url: string, options: WebSocketHookOptions<T> = {}) => {
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
            let connection = keepAlive ? globalWebSockets.get(url) : null;

            // If no existing connection or it's closed, create a new one
            if (!connection || connection.socket.readyState === WebSocket.CLOSED) {
                const socket = new WebSocket(url);
                connection = { socket, subscribers: new Set() };
                if (keepAlive) {
                    globalWebSockets.set(url, connection);
                }
            }

            if (onMessage) {
                addSubscriber(url, onMessage);
            }

            connection.socket.onopen = () => {
                console.log('WebSocket connected');
                if (mountedRef.current) {
                    setIsConnected(true);
                    setError(null);
                }
                reconnectCount.current = 0;
                if (onOpen) onOpen();
            };

            connection.socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                // Broadcast to all subscribers
                connection?.subscribers.forEach(subscriber => subscriber(data));
            };

            connection.socket.onclose = () => {
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

            connection.socket.onerror = (err) => {
                console.error('WebSocket error:', err);
                if (mountedRef.current) {
                    setError(err);
                }
                if (onError) onError(err);
            };

            ws.current = connection.socket;
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
        const existingConnection = keepAlive ? globalWebSockets.get(url) : null;
        if (existingConnection && existingConnection.socket.readyState === WebSocket.OPEN) {
            ws.current = existingConnection.socket;
            if (onMessage) {
                addSubscriber(url, onMessage);
            }
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
            // Remove the message subscriber
            if (onMessage) {
                removeSubscriber(url, onMessage);
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
