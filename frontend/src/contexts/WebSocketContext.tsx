import React, { createContext, useContext, useEffect, useMemo, ReactNode } from 'react';
import WebSocketManager, { WebSocketState } from '../services/websocket/WebSocketManager';
import { getWebSocketUrl } from '../config';

interface WebSocketContextValue {
    isConnected: boolean;
    error: Error | null;
    subscribe: (channel: string, handler: (data: any) => void) => () => void;
    send: (data: any) => void;
}

const WebSocketContext = createContext<WebSocketContextValue | null>(null);

interface WebSocketProviderProps {
    children: ReactNode;
    options?: {
        reconnectAttempts?: number;
        reconnectInterval?: number;
        heartbeatInterval?: number;
        debug?: boolean;
    };
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children, options = {} }) => {
    const wsUrl = useMemo(() => {
        const url = getWebSocketUrl();
        console.debug('[WebSocketContext] Using WebSocket URL:', url);
        return url;
    }, []);

    const wsManager = useMemo(() => {
        console.debug('[WebSocketContext] Creating WebSocket manager with URL:', wsUrl);
        return WebSocketManager.getInstance(wsUrl, {
            ...options,
            debug: true // Force debug mode for now
        });
    }, [wsUrl, options]);

    const [state, setState] = React.useState<WebSocketState>(wsManager.getState());

    useEffect(() => {
        console.debug('[WebSocketContext] Setting up WebSocket state change listener');
        const unsubscribe = wsManager.onStateChange((newState) => {
            console.debug('[WebSocketContext] WebSocket state changed:', newState);
            setState(newState);
        });

        // Initiate connection
        console.debug('[WebSocketContext] Initiating WebSocket connection');
        wsManager.connect();

        return () => {
            console.debug('[WebSocketContext] Cleaning up WebSocket state change listener');
            unsubscribe();
        };
    }, [wsManager]);

    const contextValue = useMemo(() => ({
        isConnected: state.isConnected,
        error: state.error,
        subscribe: wsManager.subscribe.bind(wsManager),
        send: wsManager.send.bind(wsManager)
    }), [state.isConnected, state.error, wsManager]);

    return (
        <WebSocketContext.Provider value={contextValue}>
            {children}
        </WebSocketContext.Provider>
    );
};

export const useWebSocket = () => {
    const context = useContext(WebSocketContext);
    if (!context) {
        throw new Error('useWebSocket must be used within a WebSocketProvider');
    }
    return context;
};

export default WebSocketContext; 