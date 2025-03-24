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
    protocols?: string | string[];
    isInternal?: boolean;
}

interface ConnectionState {
    isConnecting: boolean;
    lastActivity: number;
    hasError: boolean;
    errorMessage?: string;
}

const HEARTBEAT_INTERVAL = 15000;
const DEFAULT_RECONNECT_ATTEMPTS = 5;
const DEFAULT_RECONNECT_INTERVAL = 3000;

export function useWebSocket<T = any>(url: string, options: WebSocketOptions = {}) {
    const [readyState, setReadyState] = useState<ReadyState>(ReadyState.UNINSTANTIATED);
    const [lastMessage, setLastMessage] = useState<T | null>(null);
    const [error, setError] = useState<Error | null>(null);
    const [connectionState, setConnectionState] = useState<ConnectionState>({
        isConnecting: false,
        lastActivity: Date.now(),
        hasError: false,
        errorMessage: undefined
    });

    const socketRef = useRef<WebSocket | null>(null);
    const timeoutRef = useRef<number>();
    const heartbeatIntervalRef = useRef<number>();
    const heartbeatTimeoutRef = useRef<number>();
    const reconnectAttemptsRef = useRef(0);
    const mountedRef = useRef(true);
    const cleanupInProgressRef = useRef(false);

    const debugLog = useCallback((message: string, data?: any) => {
        if (options.debug) {
            if (data) {
                console.log(`[WebSocket] ${message}`, data);
            } else {
                console.log(`[WebSocket] ${message}`);
            }
        }
    }, [options.debug]);

    const cleanup = useCallback(async () => {
        if (cleanupInProgressRef.current) {
            debugLog('Cleanup already in progress');
            return;
        }

        const cleanupPromise = new Promise<void>((resolve) => {
            cleanupInProgressRef.current = true;
            debugLog('Starting cleanup');

            const clearTimers = () => {
                if (timeoutRef.current) {
                    clearTimeout(timeoutRef.current);
                    timeoutRef.current = undefined;
                }
                if (heartbeatIntervalRef.current) {
                    clearInterval(heartbeatIntervalRef.current);
                    heartbeatIntervalRef.current = undefined;
                }
                if (heartbeatTimeoutRef.current) {
                    clearTimeout(heartbeatTimeoutRef.current);
                    heartbeatTimeoutRef.current = undefined;
                }
            };

            clearTimers();

            if (socketRef.current) {
                const socket = socketRef.current;
                debugLog('Cleaning up socket', { readyState: socket.readyState });

                // Remove all event listeners
                socket.onopen = null;
                socket.onclose = null;
                socket.onerror = null;
                socket.onmessage = null;

                if (socket.readyState !== WebSocket.CLOSING && socket.readyState !== WebSocket.CLOSED) {
                    // Set up a one-time close listener to resolve the promise
                    socket.onclose = () => {
                        socketRef.current = null;
                        cleanupInProgressRef.current = false;
                        debugLog('Socket closed and cleanup completed');
                        resolve();
                    };

                    // Close the socket
                    socket.close(1000, 'Client cleanup');

                    // Set a timeout in case the close event doesn't fire
                    setTimeout(() => {
                        if (cleanupInProgressRef.current) {
                            socketRef.current = null;
                            cleanupInProgressRef.current = false;
                            debugLog('Socket close timed out, forcing cleanup completion');
                            resolve();
                        }
                    }, 1000);
                } else {
                    socketRef.current = null;
                    cleanupInProgressRef.current = false;
                    debugLog('Socket was already closed, cleanup completed');
                    resolve();
                }
            } else {
                cleanupInProgressRef.current = false;
                debugLog('No socket to clean up');
                resolve();
            }
        });

        return cleanupPromise;
    }, [debugLog]);

    const connect = useCallback(async () => {
        await cleanup();

        // Add delay between reconnection attempts
        if (reconnectAttemptsRef.current > 0) {
            const delay = (options.reconnectInterval || DEFAULT_RECONNECT_INTERVAL) * 
                Math.pow(1.5, reconnectAttemptsRef.current - 1);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
        debugLog('Connecting to WebSocket', { url, protocols: options.protocols });

        try {
            setConnectionState(prev => ({ ...prev, isConnecting: true, hasError: false }));
            const protocols = options.isInternal ? ['proxy-internal'] : options.protocols;
            const socket = new WebSocket(url, protocols);
            socketRef.current = socket;

            socket.onopen = () => {
                debugLog('WebSocket connection opened');
                setReadyState(ReadyState.OPEN);
                setError(null);
                setConnectionState(prev => ({
                    ...prev,
                    isConnecting: false,
                    lastActivity: Date.now(),
                    hasError: false,
                    errorMessage: undefined
                }));
                reconnectAttemptsRef.current = 0;

                if (mountedRef.current) {
                    if (options.onOpen) options.onOpen();
                }

                // Start heartbeat
                heartbeatIntervalRef.current = window.setInterval(() => {
                    if (socket.readyState === WebSocket.OPEN) {
                        try {
                            socket.send(JSON.stringify({
                                type: 'heartbeat',
                                timestamp: Date.now()
                            }));

                            // Set up heartbeat timeout
                            if (heartbeatTimeoutRef.current) {
                                clearTimeout(heartbeatTimeoutRef.current);
                            }
                            heartbeatTimeoutRef.current = window.setTimeout(() => {
                                debugLog('Heartbeat timeout, reconnecting');
                                setConnectionState(prev => ({
                                    ...prev,
                                    hasError: true,
                                    errorMessage: 'Heartbeat timeout'
                                }));
                                cleanup();
                                connect();
                            }, 10000); // 10 second timeout
                        } catch (err) {
                            debugLog('Error sending heartbeat', err);
                            setConnectionState(prev => ({
                                ...prev,
                                hasError: true,
                                errorMessage: 'Failed to send heartbeat'
                            }));
                        }
                    }
                }, options.heartbeatInterval || HEARTBEAT_INTERVAL);
            };

            socket.onclose = (event) => {
                debugLog('WebSocket connection closed', event);
                setReadyState(ReadyState.CLOSED);
                setConnectionState(prev => ({
                    ...prev,
                    isConnecting: false,
                    hasError: event.code !== 1000,
                    errorMessage: event.code !== 1000 ? `Connection closed: ${event.reason || 'Unknown reason'}` : undefined
                }));

                if (mountedRef.current) {
                    if (options.onClose) options.onClose(event);
                }

                    // Don't reconnect on normal closure or if unmounted
                    if (event.code !== 1000 && mountedRef.current) {
                        const shouldReconnect = reconnectAttemptsRef.current < 
                            (options.reconnectAttempts || DEFAULT_RECONNECT_ATTEMPTS);
                        if (shouldReconnect) {
                            debugLog(`Scheduling reconnection (attempt ${reconnectAttemptsRef.current + 1})`);
                            reconnectAttemptsRef.current++;
                            connect();
                        } else {
                            debugLog('Max reconnection attempts reached');
                            setError(new Error('Max reconnection attempts reached'));
                        }
                    }
            };

            socket.onerror = (event) => {
                const errorMessage = event instanceof ErrorEvent ? event.message : 'Unknown WebSocket error';
                debugLog('WebSocket error occurred', { error: errorMessage, event });
                setError(new Error(errorMessage));
                setConnectionState(prev => ({
                    ...prev,
                    hasError: true,
                    errorMessage: errorMessage
                }));

                if (options.onError) options.onError(event);

                if (options.reconnectOnError) {
                    cleanup();
                    connect();
                }
            };

            socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    debugLog('Received message', data);
                    setConnectionState(prev => ({
                        ...prev,
                        lastActivity: Date.now()
                    }));

                    // Handle heartbeat responses
                    if (data.type === 'heartbeat_response') {
                        if (heartbeatTimeoutRef.current) {
                            clearTimeout(heartbeatTimeoutRef.current);
                            heartbeatTimeoutRef.current = undefined;
                        }
                        return;
                    }

                    setLastMessage(data);
                    if (options.onMessage) options.onMessage(data);
                } catch (err) {
                    debugLog('Error processing message', err);
                    setConnectionState(prev => ({
                        ...prev,
                        hasError: true,
                        errorMessage: 'Failed to process message'
                    }));
                }
            };

            setReadyState(ReadyState.CONNECTING);
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to create WebSocket connection';
            debugLog('Error creating WebSocket', err);
            setError(new Error(errorMessage));
            setConnectionState(prev => ({
                ...prev,
                isConnecting: false,
                hasError: true,
                errorMessage: errorMessage
            }));
            if (options.onError) options.onError(err as any);
        }
    }, [url, options, cleanup, debugLog]);

    useEffect(() => {
        connect();
        return () => {
            mountedRef.current = false;
            cleanup();
        };
    }, [connect, cleanup]);

    const send = useCallback((data: any) => {
        if (socketRef.current?.readyState === WebSocket.OPEN) {
            try {
                socketRef.current.send(JSON.stringify(data));
                return true;
            } catch (err) {
                debugLog('Error sending message', err);
                return false;
            }
        }
        debugLog('Cannot send message - socket not connected');
        return false;
    }, [debugLog]);

    const reconnect = useCallback(() => {
        debugLog('Manual reconnection requested');
        cleanup();
        reconnectAttemptsRef.current = 0;
        connect();
    }, [cleanup, connect, debugLog]);

    return {
        readyState,
        isConnected: readyState === ReadyState.OPEN,
        lastMessage,
        error,
        connectionState,
        send,
        reconnect
    };
}
