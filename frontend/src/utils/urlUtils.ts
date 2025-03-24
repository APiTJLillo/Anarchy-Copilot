export const reconstructUrl = (request: {
    url: string;
    host?: string;
    path?: string;
    request_headers?: Record<string, string> | string;
}): string => {
    // If it's a raw data entry, try to extract host from headers
    if (request.url === 'raw://data' || request.url === '/') {
        if (request.request_headers) {
            try {
                const headers = typeof request.request_headers === 'string' ?
                    JSON.parse(request.request_headers) :
                    request.request_headers;

                // Try to get the host from headers
                const host = headers['Host'] || headers['host'];
                if (host) {
                    // Check if host already includes protocol
                    if (host.startsWith('http://') || host.startsWith('https://')) {
                        return `${host}${request.path || '/'}`;
                    }

                    // Check if we have a Referer header to determine protocol
                    const referer = headers['Referer'] || headers['referer'];
                    if (referer && (referer.startsWith('http://') || referer.startsWith('https://'))) {
                        const protocol = referer.split('://')[0];
                        return `${protocol}://${host}${request.path || '/'}`;
                    }

                    // Default to https if no protocol specified
                    return `https://${host}${request.path || '/'}`;
                }
            } catch (e) {
                console.error('Failed to parse request headers:', e);
            }
        }
    }

    // If URL doesn't have a protocol, try to add one
    if (request.url && !request.url.includes('://')) {
        // Check if we have request headers with protocol info
        if (request.request_headers) {
            try {
                const headers = typeof request.request_headers === 'string' ?
                    JSON.parse(request.request_headers) :
                    request.request_headers;

                const referer = headers['Referer'] || headers['referer'];
                if (referer && (referer.startsWith('http://') || referer.startsWith('https://'))) {
                    const protocol = referer.split('://')[0];
                    return request.url.startsWith('/') ?
                        `${protocol}://${request.host || ''}${request.url}` :
                        `${protocol}://${request.url}`;
                }
            } catch (e) {
                console.error('Failed to parse request headers for protocol detection:', e);
            }
        }

        // Default to https if we can't determine protocol
        return request.url.startsWith('/') ?
            `https://${request.host || ''}${request.url}` :
            `https://${request.url}`;
    }

    return request.url;
}; 