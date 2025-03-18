/**
 * Determines the editor language based on content type or data structure
 */
export const getEditorLanguage = (content: string | object | undefined): string => {
    if (!content) return 'plaintext';

    // If it's already an object, it's likely JSON
    if (typeof content === 'object') return 'json';

    // Try parsing as JSON
    try {
        JSON.parse(content);
        return 'json';
    } catch {
        // Check if it looks like HTML
        if (content.trim().startsWith('<') && content.trim().endsWith('>')) {
            return 'html';
        }

        // Default to plaintext if we can't determine the type
        return 'plaintext';
    }
};

/**
 * Formats headers object into a pretty-printed JSON string
 */
export const formatHeaders = (headers: Record<string, string> | string | undefined): string => {
    if (!headers) return '';

    // If headers is already a string, try to parse it as JSON
    if (typeof headers === 'string') {
        try {
            const parsed = JSON.parse(headers);
            return JSON.stringify(parsed, null, 2);
        } catch {
            // If parsing fails, return the original string
            return headers;
        }
    }

    // If headers is an object, stringify it
    return JSON.stringify(headers, null, 2);
}; 