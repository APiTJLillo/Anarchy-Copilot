import { useState } from 'react';
import { useApi } from './useApi';

interface ResendRequestParams {
    requestId: number;
    method?: string;
    url?: string;
    headers?: Record<string, string>;
    body?: string;
}

export const useResendRequest = () => {
    const [isResending, setIsResending] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const api = useApi();

    const resendRequest = async (params: ResendRequestParams) => {
        setIsResending(true);
        setError(null);
        
        try {
            const response = await api.post('/proxy/resend', params);
            return response.data;
        } catch (err: any) {
            setError(err.message || 'Failed to resend request');
            throw err;
        } finally {
            setIsResending(false);
        }
    };

    return {
        resendRequest,
        isResending,
        error
    };
};
