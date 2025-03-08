/**
 * Global configuration for the frontend application.
 */

/**
 * Base URL for API endpoints.
 * When running in Docker, requests are made to 'http://dev:8000'
 * When accessing through browser, requests go to 'http://localhost:8000'
 */
export const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

/**
 * WebSocket endpoint for proxy monitoring.
 * Browser always connects through localhost since Docker DNS resolution
 * isn't available in the browser context
 */
export const WS_ENDPOINT = `${API_BASE_URL.replace('http', 'ws')}/api/proxy/ws`;

/**
 * WebSocket endpoint for proxy interception.
 * Browser always connects through localhost since Docker DNS resolution
 * isn't available in the browser context
 */
export const WS_INTERCEPT_ENDPOINT = `${API_BASE_URL.replace('http', 'ws')}/api/proxy/ws/intercept`;

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
        host: '127.0.0.1',
        port: 8083,
        interceptRequests: true,
        interceptResponses: true,
        allowedHosts: [],
        excludedHosts: []
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
