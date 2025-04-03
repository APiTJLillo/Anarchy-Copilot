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
    heartbeatTimeout?: number; // Added explicit heartbeat timeout option
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
const DEFAULT_HEARTBEAT_TIMEOUT = 10000; // Default heartbeat timeout
const DEFAULT_RECONNECT_ATTEMPTS = 5;
const DEFAULT_RECONNECT_INTERVAL = 3000;
const MAX_RECONNECT_DELAY = 30000; // Maximum reconnect delay to prevent excessive backoff

// Global reconnection tracking to prevent multiple simultaneous reconnection attempts
const globalReconnectState = {
    isReconnecting: false,
    lastReconnectAttempt: 0,
    reconnectCount: 0
};

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
    const instanceIdRef = useRef<string>(Math.random().toString(36).substring(2, 9)); // Unique instance ID
    
    // Store message handlers in refs to persist them across reconnections
    const onMessageHandlerRef = useRef<((data: any) => void) | undefined>(options.onMessage);
    const onOpenHandlerRef = useRef<(() => void) | undefined>(options.onOpen);
    const onCloseHandlerRef = useRef<((event: CloseEvent) => void) | undefined>(options.onClose);
    const onErrorHandlerRef = useRef<((event: Event) => void) | undefined>(options.onError);

    // Update handler refs when options change
    useEffect(() => {
        onMessageHandlerRef.current = options.onMessage;
        onOpenHandlerRef.current = options.onOpen;
        onCloseHandlerRef.current = options.onClose;
        onErrorHandlerRef.current = options.onError;
    }, [options.onMessage, options.onOpen, options.onClose, options.onError]);

    const debugLog = useCallback((message: string, data?: any) => {
        if (options.debug) {
            if (data) {
                console.log(`[WebSocket:${instanceIdRef.current}] ${message}`, data);
            } else {
                console.log(`[WebSocket:${instanceIdRef.current}] ${message}`);
            }
        }
    }, [options.debug]);

    const cleanup = useCallback(async () => {
        if (cleanupInProgressRef.current) {
            debugLog('Cleanup already in progress, waiting...');
            // Wait for existing cleanup to complete
            let attempts = 0;
            while (cleanupInProgressRef.current && attempts < 10) {
                await new Promise(resolve => setTimeout(resolve, 100));
                attempts++;
            }
            if (cleanupInProgressRef.current) {
                debugLog('Cleanup still in progress after waiting, forcing continuation');
            }
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
                    try {
                        socket.close(1000, 'Client cleanup');
                    } catch (err) {
                        debugLog('Error closing socket', err);
                    }

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
        // Check if global reconnection is in progress
        const now = Date.now();
        if (globalReconnectState.isReconnecting) {
            const timeSinceLastAttempt = now - globalReconnectState.lastReconnectAttempt;
            if (timeSinceLastAttempt < 1000) {
                debugLog('Another reconnection is in progress, delaying attempt');
                await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 1000));
            }
        }

        await cleanup();

        // Add delay between reconnection attempts with jitter to prevent thundering herd
        if (reconnectAttemptsRef.current > 0) {
            const baseDelay = options.reconnectInterval || DEFAULT_RECONNECT_INTERVAL;
            const exponentialDelay = baseDelay * Math.pow(1.5, reconnectAttemptsRef.current - 1);
            const jitter = Math.random() * 0.3 * exponentialDelay; // Add up to 30% jitter
            const delay = Math.min(exponentialDelay + jitter, MAX_RECONNECT_DELAY);
            
            debugLog(`Delaying reconnection attempt ${reconnectAttemptsRef.current} by ${delay}ms`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }

        // Mark global reconnection state
        globalReconnectState.isReconnecting = true;
        globalReconnectState.lastReconnectAttempt = Date.now();
        globalReconnectState.reconnectCount++;

        debugLog('Connecting to WebSocket', { url, protocols: options.protocols, reconnectAttempt: reconnectAttemptsRef.current });

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
                globalReconnectState.isReconnecting = false;

                if (mountedRef.current) {
                    // Use the handler from ref to ensure we have the latest
                    if (onOpenHandlerRef.current) onOpenHandlerRef.current();
                }

                // Start heartbeat with a more robust implementation
                if (heartbeatIntervalRef.current) {
                    clearInterval(heartbeatIntervalRef.current);
                }
                
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
                            
                            // Use configurable timeout with default
                            const heartbeatTimeoutDuration = options.heartbeatTimeout || DEFAULT_HEARTBEAT_TIMEOUT;
                            
                            heartbeatTimeoutRef.current = window.setTimeout(() => {
                                debugLog('Heartbeat timeout, reconnecting');
                                setConnectionState(prev => ({
                                    ...prev,
                                    hasError: true,
                                    errorMessage: 'Heartbeat timeout'
                                }));
                                
                                // Prevent multiple reconnection attempts from the same heartbeat timeout
                                if (heartbeatTimeoutRef.current) {
                                    clearTimeout(heartbeatTimeoutRef.current);
                                    heartbeatTimeoutRef.current = undefined;
                                }
                                
                                // Only attempt reconnect if not already reconnecting
                                if (!globalReconnectState.isReconnecting) {
                                    cleanup().then(() => {
                                        // Check if still mounted before reconnecting
                                        if (mountedRef.current) {
                                            connect();
                                        }
                                    });
                                }
                            }, heartbeatTimeoutDuration);
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

                // Clear any pending heartbeat timers
                if (heartbeatIntervalRef.current) {
                    clearInterval(heartbeatIntervalRef.current);
                    heartbeatIntervalRef.current = undefined;
                }
                
                if (heartbeatTimeoutRef.current) {
                    clearTimeout(heartbeatTimeoutRef.current);
                    heartbeatTimeoutRef.current = undefined;
                }

                if (mountedRef.current) {
                    // Use the handler from ref to ensure we have the latest
                    if (onCloseHandlerRef.current) onCloseHandlerRef.current(event);
                }

                // Don't reconnect on normal closure or if unmounted
                if (event.code !== 1000 && mountedRef.current) {
                    const maxReconnectAttempts = options.reconnectAttempts || DEFAULT_RECONNECT_ATTEMPTS;
                    const shouldReconnect = reconnectAttemptsRef.current < maxReconnectAttempts;
                    
                    if (shouldReconnect) {
                        debugLog(`Scheduling reconnection (attempt ${reconnectAttemptsRef.current + 1})`);
                        reconnectAttemptsRef.current++;
                        
                        // Use setTimeout to prevent immediate reconnection
                        setTimeout(() => {
                            if (mountedRef.current) {
                                connect();
                            }
                        }, 500);
                    } else {
                        debugLog('Max reconnection attempts reached');
                        setError(new Error('Max reconnection attempts reached'));
                        globalReconnectState.isReconnecting = false;
                    }
                } else {
                    globalReconnectState.isReconnecting = false;
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

                // Use the handler from ref to ensure we have the latest
                if (onErrorHandlerRef.current) onErrorHandlerRef.current(event);

                // Only reconnect on error if explicitly enabled and not already reconnecting
                if (options.reconnectOnError && !globalReconnectState.isReconnecting) {
                    cleanup().then(() => {
                        if (mountedRef.current) {
                            connect();
                        }
                    });
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
                    if (data.type === 'heartbeat' || data.type === 'heartbeat_response') {
                        if (heartbeatTimeoutRef.current) {
                            clearTimeout(heartbeatTimeoutRef.current);
                            heartbeatTimeoutRef.current = undefined;
                        }
                        
                        // If it's a heartbeat request (not response), send a response back
                        if (data.type === 'heartbeat' && !data.response) {
                            try {
                                socket.send(JSON.stringify({
                                    type: 'heartbeat_response',
                                    timestamp: Date.now()
                                }));
                            } catch (err) {
                                debugLog('Error sending heartbeat response', err);
                            }
                        }
                        return;
                    }

                    setLastMessage(data);
                    // Use the handler from ref to ensure we have the latest
                    if (onMessageHandlerRef.current) onMessageHandlerRef.current(data);
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
            // Use the handler from ref to ensure we have the latest
            if (onErrorHandlerRef.current) onErrorHandlerRef.current(err as any);
            globalReconnectState.isReconnecting = false;
        }
    }, [url, options.reconnectAttempts, options.reconnectInterval, options.reconnectOnError, 
        options.heartbeatInterval, options.heartbeatTimeout, options.isInternal, options.protocols, 
        cleanup, debugLog]);

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
        // Only proceed if not already reconnecting
        if (!globalReconnectState.isReconnecting) {
            cleanup().then(() => {
                reconnectAttemptsRef.current = 0;
                connect();
            });
        } else {
            debugLog('Reconnection already in progress, request ignored');
        }
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
