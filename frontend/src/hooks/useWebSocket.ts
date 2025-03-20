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
    isConnecting: boolean;
    reconnectCount: number;
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
    const mountedRef = useRef(true);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

    const connect = useCallback(() => {
        try {
            // Check if we already have a connection for this URL
            let connection = keepAlive ? globalWebSockets.get(url) : null;

            // Don't create a new connection if one is already being established
            if (connection?.isConnecting) {
                return;
            }

            // If no existing connection or it's closed, create a new one
            if (!connection || connection.socket.readyState === WebSocket.CLOSED) {
                const socket = new WebSocket(url);
                connection = {
                    socket,
                    subscribers: new Set(),
                    isConnecting: true,
                    reconnectCount: 0
                };
                if (keepAlive) {
                    globalWebSockets.set(url, connection);
                }
            }

            if (onMessage) {
                addSubscriber(url, onMessage);
            }

            connection.socket.onopen = () => {
                if (!mountedRef.current) return;

                connection!.isConnecting = false;
                connection!.reconnectCount = 0;
                setIsConnected(true);
                setError(null);
                if (onOpen) onOpen();
            };

            connection.socket.onmessage = (event) => {
                if (!mountedRef.current) return;

                try {
                    const data = JSON.parse(event.data);
                    // Broadcast to all subscribers
                    connection?.subscribers.forEach(subscriber => subscriber(data));
                } catch (err) {
                    console.error('Failed to parse WebSocket message:', err);
                }
            };

            connection.socket.onclose = () => {
                if (!mountedRef.current) return;

                setIsConnected(false);
                if (onClose) onClose();

                // Only attempt reconnect if we're keeping the connection alive
                if (keepAlive && connection!.reconnectCount < reconnectAttempts) {
                    connection!.reconnectCount += 1;
                    connection!.isConnecting = false;
                    globalWebSockets.delete(url);

                    reconnectTimeoutRef.current = setTimeout(() => {
                        if (mountedRef.current) {
                            connect();
                        }
                    }, reconnectInterval);
                } else {
                    // Clean up the connection
                    globalWebSockets.delete(url);
                }
            };

            connection.socket.onerror = (err) => {
                if (!mountedRef.current) return;

                setError(err);
                if (onError) onError(err);
            };

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
            if (!keepAlive) {
                const connection = globalWebSockets.get(url);
                if (connection) {
                    connection.socket.close();
                    globalWebSockets.delete(url);
                }
            }
        };
    }, [connect, url, keepAlive, onMessage, onOpen]);

    const send = useCallback((data: any) => {
        const connection = globalWebSockets.get(url);
        if (connection?.socket.readyState === WebSocket.OPEN) {
            connection.socket.send(JSON.stringify(data));
        }
    }, [url]);

    return {
        isConnected,
        error,
        send
    };
};
