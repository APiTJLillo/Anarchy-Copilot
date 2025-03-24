import { useEffect, useCallback } from 'react';
import { useWebSocket } from '../contexts/WebSocketContext';

interface UseWebSocketChannelOptions<T> {
    onMessage?: (data: T) => void;
    transform?: (data: any) => T;
}

function useWebSocketChannel<T = any>(channel: string, options: UseWebSocketChannelOptions<T> = {}) {
    const { subscribe, send, isConnected, error } = useWebSocket();
    const { onMessage, transform } = options;

    const handleMessage = useCallback((data: any) => {
        if (onMessage) {
            const transformedData = transform ? transform(data) : data;
            onMessage(transformedData);
        }
    }, [onMessage, transform]);

    useEffect(() => {
        const unsubscribe = subscribe(channel, handleMessage);
        return () => unsubscribe();
    }, [channel, subscribe, handleMessage]);

    const sendToChannel = useCallback((data: any) => {
        send({
            channel,
            ...data
        });
    }, [channel, send]);

    return {
        isConnected,
        error,
        send: sendToChannel
    };
}

export default useWebSocketChannel; 