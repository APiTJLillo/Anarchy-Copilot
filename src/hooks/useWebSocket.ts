import { useState, useCallback, useRef, useEffect } from 'react';
import { ReadyState } from './types';

export interface WebSocketOptions {
    onMessage?: (data: any) => void;
    onOpen?: () => void;
    onClose?: (event: CloseEvent) => void;
    onError?: (event: Event) => void;
    reconnectAttempts?: number;
    reconnectInterval?: number;
    reconnectOnError?: boolean;
    heartbeatInterval?: number;
    debug?: boolean;
}

export function useWebSocket<T = any>(url: string, options: WebSocketOptions = {}) {
    const [readyState, setReadyState] = useState<ReadyState>(ReadyState.UNINSTANTIATED);
    const [lastMessage, setLastMessage] = useState<T | null>(null);
    const [error, setError] = useState<Error | null>(null);
    const wsRef = useRef<WebSocket | null>(null);

    const send = useCallback((data: any) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            try {
                wsRef.current.send(JSON.stringify(data));
                return true;
            } catch (err) {
                console.error('[WebSocket] Error sending message:', err);
                return false;
            }
        }
        return false;
    }, []);

    const reconnect = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = new WebSocket(url);
            setReadyState(ReadyState.CONNECTING);
        }
    }, [url]);

    useEffect(() => {
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
            setReadyState(ReadyState.OPEN);
            setError(null);
            if (options.onOpen) options.onOpen();
        };

        ws.onclose = (event) => {
            setReadyState(ReadyState.CLOSED);
            if (options.onClose) options.onClose(event);
        };

        ws.onerror = (event) => {
            setError(event as any);
            if (options.onError) options.onError(event);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                setLastMessage(data);
                if (options.onMessage) options.onMessage(data);
            } catch (err) {
                console.error('[WebSocket] Error parsing message:', err);
            }
        };

        setReadyState(ReadyState.CONNECTING);

        return () => {
            ws.close();
        };
    }, [url, options]);

    return {
        readyState,
        isConnected: readyState === ReadyState.OPEN,
        lastMessage,
        error,
        send,
        reconnect
    };
} 