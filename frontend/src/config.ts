/**
 * Global configuration for the frontend application.
 */

// Helper function to determine if we're running in Docker
const isDocker = process.env.REACT_APP_ENVIRONMENT === 'docker';

/**
 * Base URL for API endpoints.
 * When running in Docker, requests are made to 'http://dev:8000'
 * When accessing through browser, requests go to 'http://localhost:8000'
 */
export const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

/**
 * WebSocket endpoint for proxy monitoring.
 */
export const WS_ENDPOINT = process.env.REACT_APP_WS_URL || (isDocker ? 'ws://dev:8000/api/proxy/ws' : 'ws://localhost:8000/api/proxy/ws');

/**
 * WebSocket endpoint for proxy interception.
 * Using environment variable or falling back to relative URL that matches backend route
 */
export const WS_INTERCEPT_ENDPOINT = process.env.REACT_APP_WS_INTERCEPT_ENDPOINT || (isDocker ? 'ws://dev:8000/api/proxy/ws/intercept' : 'ws://localhost:8000/api/proxy/ws/intercept');

/**
 * Proxy URL for direct proxy connections.
 */
export const PROXY_URL = process.env.REACT_APP_PROXY_URL || (isDocker ? 'http://proxy:8083' : 'http://localhost:8083');

/**
 * Get the WebSocket URL based on the current environment
 */
export const getWebSocketUrl = (isInternal: boolean = false) => {
    // First try the environment variable
    if (process.env.REACT_APP_WS_URL) {
        console.debug('[Config] Using WebSocket URL from environment:', process.env.REACT_APP_WS_URL);
        return process.env.REACT_APP_WS_URL + (isInternal ? '/internal' : '');
    }

    // If we're in the browser, construct the URL based on the current window location
    if (typeof window !== 'undefined') {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const path = isInternal ? '/api/proxy/ws/internal' : '/api/proxy/ws';
        const url = `${protocol}//${host}${path}`;
        console.debug('[Config] Using WebSocket URL from window location:', url);
        return url;
    }

    // Fallback to default WebSocket endpoint
    const baseUrl = WS_ENDPOINT || 'ws://localhost:8000/api/proxy/ws';
    const url = isInternal ? `${baseUrl}/internal` : baseUrl;
    console.debug('[Config] Using fallback WebSocket URL:', url);
    return url;
};

/**
 * Configuration options.
 */
export const CONFIG = {
    /**
     * Polling interval for proxy status updates (in milliseconds).
     */
    POLLING_INTERVAL: 5000,

    /**
     * Default proxy settings.
     */
    DEFAULT_PROXY_CONFIG: {
        host: isDocker ? 'proxy' : '127.0.0.1',
        port: 8083,
        intercept_requests: true,
        intercept_responses: true,
        allowed_hosts: [],
        excluded_hosts: []
    },

    /**
     * Analysis severity levels and their color mappings.
     */
    SEVERITY_LEVELS: {
        high: 'error',
        medium: 'warning',
        low: 'info'
    } as const
};
