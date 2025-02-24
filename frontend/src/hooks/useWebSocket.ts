import { useEffect, useRef, useState, useCallback } from 'react';

interface WebSocketHookOptions {
    onMessage?: (data: any) => void;
    onOpen?: () => void;
    onClose?: () => void;
    onError?: (error: Event) => void;
    reconnectAttempts?: number;
    reconnectInterval?: number;
}

export const useWebSocket = (url: string, options: WebSocketHookOptions = {}) => {
    const {
        onMessage,
        onOpen,
        onClose,
        onError,
        reconnectAttempts = 5,
        reconnectInterval = 3000,
    } = options;

    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<Event | null>(null);
    const ws = useRef<WebSocket | null>(null);
    const reconnectCount = useRef(0);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

    const connect = useCallback(() => {
        try {
            const socket = new WebSocket(url);

            socket.onopen = () => {
                console.log('WebSocket connected');
                setIsConnected(true);
                setError(null);
                reconnectCount.current = 0;
                if (onOpen) onOpen();
            };

            socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (onMessage) onMessage(data);
            };

            socket.onclose = () => {
                console.log('WebSocket closed');
                setIsConnected(false);
                if (onClose) onClose();

                // Attempt to reconnect
                if (reconnectCount.current < reconnectAttempts) {
                    reconnectCount.current += 1;
                    reconnectTimeoutRef.current = setTimeout(() => {
                        console.log(`Reconnecting... Attempt ${reconnectCount.current}`);
                        connect();
                    }, reconnectInterval);
                }
            };

            socket.onerror = (err) => {
                console.error('WebSocket error:', err);
                setError(err);
                if (onError) onError(err);
            };

            ws.current = socket;
        } catch (err) {
            console.error('Failed to create WebSocket:', err);
            setError(err as Event);
        }
    }, [url, onMessage, onOpen, onClose, onError, reconnectAttempts, reconnectInterval]);

    useEffect(() => {
        connect();

        return () => {
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            if (ws.current) {
                ws.current.close();
            }
        };
    }, [connect]);

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
