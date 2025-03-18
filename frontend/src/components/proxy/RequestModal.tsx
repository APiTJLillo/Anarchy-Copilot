import React, { useState } from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    Typography,
    Box,
    Tab,
    Tabs,
    IconButton,
    Tooltip,
    CircularProgress,
} from '@mui/material';
import Editor from "@monaco-editor/react";
import { getEditorLanguage, formatHeaders } from '../../utils/editorUtils';
import { reconstructUrl } from '../../utils/urlUtils';
import ReplayIcon from '@mui/icons-material/Replay';
import { useResendRequest } from '../../hooks/useResendRequest';
import KeyboardArrowLeftIcon from '@mui/icons-material/KeyboardArrowLeft';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import CloseIcon from '@mui/icons-material/Close';

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

const TabPanel = (props: TabPanelProps) => {
    const { children, value, index, ...other } = props;

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`request-modal-tabpanel-${index}`}
            aria-labelledby={`request-modal-tab-${index}`}
            {...other}
            style={{ padding: '16px 0' }}
        >
            {value === index && children}
        </div>
    );
};

interface RequestData {
    id: number;
    method: string;
    url: string;
    host?: string;
    path?: string;
    status_code?: number;
    response_status: number | null;
    request_headers: Record<string, string> | string;
    request_body: string | null;
    response_headers: Record<string, string> | string;
    response_body: string | null;
    raw_request: string;
    raw_response: string;
    decrypted_request?: string;
    decrypted_response?: string;
    duration: number | null;
    is_intercepted: boolean;
    is_encrypted?: boolean;
    applied_rules: any | null;
    tags: string[] | string;
    notes: string | null;
    session_id: number;
    timestamp: string;
}

interface RequestModalProps {
    open: boolean;
    onClose: () => void;
    request: RequestData;
    history: RequestData[];
    currentIndex: number;
    onNavigate: (index: number) => void;
    onRequestResent?: () => void;
}

export const RequestModal: React.FC<RequestModalProps> = ({
    open,
    onClose,
    request,
    history,
    currentIndex,
    onNavigate,
    onRequestResent
}) => {
    const [tabValue, setTabValue] = React.useState(0);
    const { resendRequest, isResending, error: resendError } = useResendRequest();

    const decodeBase64Data = (data: string | null | undefined): string => {
        if (!data) return '';

        console.log('Decoding data:', {
            isBase64Url: data.startsWith('base64://'),
            dataLength: data.length,
            firstFewChars: data.substring(0, 50)
        });

        // First try to decode base64:// prefixed data
        if (data.startsWith('base64://')) {
            try {
                const decoded = atob(data.substring('base64://'.length));
                console.log('Successfully decoded base64:// data');
                return decoded;
            } catch (e) {
                console.error('Failed to decode base64:// data:', e);
            }
        }

        // Then try to decode potential raw base64 data
        try {
            // Simple heuristic: if the string contains only valid base64 characters
            if (/^[A-Za-z0-9+/=]+$/.test(data)) {
                const decoded = atob(data);
                // Check if it looks like an HTTP message or contains printable characters
                if (decoded.startsWith('HTTP/') ||
                    decoded.includes('\r\n') ||
                    decoded.includes('\n') ||
                    /^[\x20-\x7E\t\n\r]*$/.test(decoded)) {
                    console.log('Successfully decoded raw base64 data');
                    return decoded;
                }
            }
        } catch (e) {
            console.error('Failed to decode potential base64 data:', e);
        }

        // Return original data if we couldn't decode it
        return data;
    };

    const parseTags = (tags: string | string[] | undefined | null): string[] => {
        if (!tags) return [];

        // Handle case where tags is already an array
        if (Array.isArray(tags)) return tags;

        // Handle string case
        if (typeof tags === 'string') {
            try {
                // Try parsing as JSON array
                return JSON.parse(tags.replace(/'/g, '"'));
            } catch {
                // Fall back to splitting comma-separated string
                return tags.replace(/[\[\]'"\s]/g, '').split(',');
            }
        }

        return [];
    };

    const getDataVersion = (tags: string | string[] | undefined | null): 'decrypted' | 'raw' => {
        if (Array.isArray(tags)) {
            return tags.includes('decrypted') ? 'decrypted' : 'raw';
        }
        const parsedTags = parseTags(tags);
        return parsedTags.includes('decrypted') ? 'decrypted' : 'raw';
    };

    const getDataByTags = (request: RequestData, type: string) => {
        if (!request) return null;

        const tags = parseTags(request.tags);
        console.log('Request data:', {
            type,
            tags,
            hasDecryptedRequest: !!request.decrypted_request,
            hasDecryptedResponse: !!request.decrypted_response,
            hasRawRequest: !!request.raw_request,
            hasRawResponse: !!request.raw_response,
            hasRequestBody: !!request.request_body,
            hasResponseBody: !!request.response_body
        });

        // Special handling for URL
        if (type === 'url') {
            // For raw data entries, try to extract URL from request body first
            if ((request.url === 'raw://data' || request.url === '/') && request.request_body) {
                try {
                    const decodedBody = decodeBase64Data(request.request_body);
                    // Try to extract the first line which should contain the request line
                    const lines = decodedBody.split(/\r?\n/);
                    const requestLine = lines[0];
                    const hostLine = lines.find(line => line.toLowerCase().startsWith('host:'));

                    if (requestLine && requestLine.includes(' ')) {
                        // Extract path from "METHOD /path HTTP/1.1"
                        const [, path] = requestLine.split(' ');
                        if (path) {
                            console.log('Extracted path from request line:', path);
                            // If we have a host header, use it to construct the full URL
                            if (hostLine) {
                                const host = hostLine.split(':')[1].trim();
                                console.log('Extracted host from headers:', host);
                                return reconstructUrl({
                                    ...request,
                                    url: path,
                                    host: host
                                });
                            }
                            return reconstructUrl({
                                ...request,
                                url: path
                            });
                        }
                    }
                } catch (e) {
                    console.error('Failed to extract URL from request body:', e);
                }
            }
            return reconstructUrl(request);
        }

        // For request data
        if (type === 'request') {
            // First try decrypted data if available
            if (request.decrypted_request) {
                console.log('Using decrypted request data');
                return decodeBase64Data(request.decrypted_request);
            }
            // Then try raw request if raw tag is present
            if (tags.includes('raw') && request.raw_request) {
                console.log('Using raw request data');
                return decodeBase64Data(request.raw_request);
            }
            // Finally fall back to regular request body
            if (request.request_body) {
                console.log('Using regular request body');
                return decodeBase64Data(request.request_body);
            }
            return null;
        }

        // For response data
        if (type === 'response') {
            // First try decrypted data if available and decrypted tag is present
            if (tags.includes('decrypted') && request.decrypted_response) {
                console.log('Using decrypted response data');
                return request.decrypted_response;
            }
            // Then try raw response if raw tag is present
            if (tags.includes('raw') && request.raw_response) {
                console.log('Using raw response data');
                return decodeBase64Data(request.raw_response);
            }
            // Finally fall back to regular response body
            if (request.response_body) {
                console.log('Using regular response body');
                return decodeBase64Data(request.response_body);
            }
            return null;
        }

        if (type === 'request_headers') {
            try {
                if (request.request_headers) {
                    const headers = typeof request.request_headers === 'string' ?
                        JSON.parse(request.request_headers) :
                        request.request_headers;
                    return headers;
                }
                return null;
            } catch (e) {
                console.error('Failed to parse request headers:', e);
                return null;
            }
        }

        if (type === 'response_headers') {
            try {
                if (request.response_headers) {
                    const headers = typeof request.response_headers === 'string' ?
                        JSON.parse(request.response_headers) :
                        request.response_headers;
                    return headers;
                }
                return null;
            } catch (e) {
                console.error('Failed to parse response headers:', e);
                return null;
            }
        }

        return null;
    };

    const [editedRequest, setEditedRequest] = useState<{
        method: string;
        url: string;
        headers: Record<string, string> | string;
        body: string;
    }>(() => ({
        method: request?.method || 'GET',
        url: getDataByTags(request, 'url') || '',
        headers: request?.request_headers ? (
            typeof request.request_headers === 'string'
                ? JSON.parse(request.request_headers)
                : request.request_headers
        ) : {},
        body: request?.request_body || ''
    }));

    const [editedResponse, setEditedResponse] = useState<{
        headers: Record<string, string> | string;
        body: string;
    }>(() => ({
        headers: request?.response_headers ? (
            typeof request.response_headers === 'string'
                ? JSON.parse(request.response_headers)
                : request.response_headers
        ) : {},
        body: request?.response_body || ''
    }));

    // Update state when request changes
    React.useEffect(() => {
        if (request) {
            setEditedRequest({
                method: request.method,
                url: getDataByTags(request, 'url') || '',
                headers: typeof request.request_headers === 'string'
                    ? JSON.parse(request.request_headers)
                    : request.request_headers,
                body: request.request_body || ''
            });

            setEditedResponse({
                headers: typeof request.response_headers === 'string'
                    ? JSON.parse(request.response_headers)
                    : request.response_headers,
                body: request.response_body || ''
            });
        }
    }, [request]);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };

    const hasChanges = (): boolean => {
        if (!request) return false;

        // Get the original URL for comparison
        const originalUrl = getDataByTags(request, 'url') || '';

        // Normalize headers for comparison
        const normalizeHeaders = (headers: any) => {
            if (typeof headers === 'string') {
                try {
                    return JSON.parse(headers);
                } catch {
                    return {};
                }
            }
            return headers || {};
        };

        const originalHeaders = normalizeHeaders(request.request_headers);
        const currentHeaders = normalizeHeaders(editedRequest.headers);

        // Compare headers as stringified objects
        const headersEqual = JSON.stringify(originalHeaders) === JSON.stringify(currentHeaders);

        // Compare all fields
        return request.method !== editedRequest.method ||
            originalUrl !== editedRequest.url ||
            !headersEqual ||
            (request.request_body || '') !== (editedRequest.body || '');
    };

    const handleResend = async () => {
        try {
            let parsedHeaders;
            try {
                parsedHeaders = typeof editedRequest.headers === 'string'
                    ? JSON.parse(editedRequest.headers)
                    : editedRequest.headers;
            } catch (err) {
                console.error('Failed to parse headers:', err);
                return;
            }

            const result = await resendRequest({
                requestId: request.id,
                method: editedRequest.method,
                url: editedRequest.url,
                headers: parsedHeaders,
                body: editedRequest.body
            });
            if (onRequestResent) {
                onRequestResent();
            }
            return result;
        } catch (error) {
            console.error('Failed to resend request:', error);
        }
    };

    const formatData = (type: string) => {
        const data = getDataByTags(request, type);
        if (!data) return 'No data available';

        try {
            // For headers that are already JSON objects
            if (typeof data === 'object') {
                return JSON.stringify(data, null, 2);
            }

            // Handle base64 encoded data first
            const decodedData = decodeBase64Data(data);

            // Try parsing as JSON
            try {
                return JSON.stringify(JSON.parse(decodedData), null, 2);
            } catch {
                // If not JSON, return decoded data as is
                return decodedData;
            }
        } catch (error) {
            // If anything fails, return original data
            return String(data);
        }
    };

    if (!request) return null;

    return (
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth="lg"
            fullWidth
            PaperProps={{ sx: { height: '80vh' } }}
        >
            <DialogTitle sx={{ p: 2, position: 'relative' }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, pr: '160px' }}>
                    <Box sx={{ mb: 0.5 }}>
                        <Typography variant="h6" component="div" sx={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1.5,
                            mb: 0.5
                        }}>
                            <Box sx={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 1.5,
                                width: '100%'
                            }}>
                                <select
                                    value={editedRequest.method}
                                    onChange={(e) => setEditedRequest({
                                        ...editedRequest,
                                        method: e.target.value
                                    })}
                                    style={{
                                        backgroundColor: {
                                            'GET': '#2e7d32',
                                            'POST': '#0288d1',
                                            'PUT': '#ed6c02',
                                            'DELETE': '#d32f2f',
                                            'PATCH': '#ed6c02',
                                        }[editedRequest.method] || '#1976d2',
                                        color: '#fff',
                                        padding: '4px 8px',
                                        borderRadius: '4px',
                                        border: 'none',
                                        fontSize: '0.9rem',
                                        fontWeight: 'bold',
                                        minWidth: '70px',
                                        cursor: 'pointer'
                                    }}
                                >
                                    {['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'].map(method => (
                                        <option key={method} value={method}>{method}</option>
                                    ))}
                                </select>
                                <Box sx={{
                                    flex: 1,
                                    display: 'flex',
                                    alignItems: 'center',
                                    fontFamily: 'monospace',
                                    fontSize: '0.95rem',
                                    bgcolor: 'background.paper',
                                    borderRadius: 1,
                                    overflow: 'hidden',
                                    border: '1px solid',
                                    borderColor: 'divider',
                                }}>
                                    <input
                                        type="text"
                                        value={editedRequest.url}
                                        onChange={(e) => setEditedRequest({
                                            ...editedRequest,
                                            url: e.target.value
                                        })}
                                        style={{
                                            width: '100%',
                                            border: 'none',
                                            padding: '8px 12px',
                                            fontFamily: 'inherit',
                                            fontSize: 'inherit',
                                            backgroundColor: 'transparent',
                                            color: '#fff'
                                        }}
                                    />
                                </Box>
                            </Box>
                        </Typography>
                        <Box sx={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 2,
                            mt: 1.5,
                            py: 0.75,
                            px: 1.5,
                            borderRadius: 1,
                            bgcolor: 'rgba(0, 0, 0, 0.02)',
                        }}>
                            <Box sx={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 1
                            }}>
                                <Box sx={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 0.5,
                                    color: request.response_status ? (
                                        request.response_status < 300 ? 'success.main' :
                                            request.response_status < 400 ? 'info.main' :
                                                request.response_status < 500 ? 'warning.main' :
                                                    'error.main'
                                    ) : 'text.secondary'
                                }}>
                                    <Box component="span" sx={{
                                        fontWeight: 'bold',
                                        fontSize: '1rem'
                                    }}>
                                        {request.response_status || request.status_code || 'Pending'}
                                    </Box>
                                    <Box component="span" sx={{
                                        fontSize: '0.85rem',
                                        opacity: 0.9
                                    }}>
                                        {request.response_status ? (
                                            request.response_status < 300 ? 'Success' :
                                                request.response_status < 400 ? 'Redirect' :
                                                    request.response_status < 500 ? 'Client Error' :
                                                        'Server Error'
                                        ) : ''}
                                    </Box>
                                </Box>
                            </Box>
                            <Box sx={{ height: '1rem', borderLeft: 1, borderColor: 'divider' }} />
                            <Typography sx={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 0.5,
                                color: 'text.secondary',
                                fontSize: '0.9rem'
                            }}>
                                <Box component="span" sx={{ opacity: 0.7 }}>Time:</Box>
                                {request.duration ? (
                                    <Box component="span" sx={{
                                        fontFamily: 'monospace',
                                        color: request.duration > 1 ? 'warning.main' : 'text.primary'
                                    }}>
                                        {(request.duration * 1000).toFixed(2)}ms
                                    </Box>
                                ) : 'N/A'}
                            </Typography>
                            <Box sx={{ height: '1rem', borderLeft: 1, borderColor: 'divider' }} />
                            <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <Box component="span" sx={{ opacity: 0.7 }}>Session:</Box>
                                <Box component="span" sx={{ fontFamily: 'monospace', fontWeight: 'medium' }}>
                                    {request.session_id}
                                </Box>
                            </Typography>
                            <Box sx={{ height: '1rem', borderLeft: 1, borderColor: 'divider' }} />
                            <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <Box component="span" sx={{ opacity: 0.7 }}>Created:</Box>
                                <Box component="span" sx={{ fontFamily: 'monospace' }}>
                                    {new Date(request.timestamp).toLocaleString()}
                                </Box>
                            </Typography>
                            <Box sx={{ height: '1rem', borderLeft: 1, borderColor: 'divider' }} />
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography variant="caption" sx={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: 0.5,
                                    bgcolor: request.is_intercepted ? 'info.main' : 'success.main',
                                    color: 'white',
                                    px: 1,
                                    py: 0.5,
                                    borderRadius: 1,
                                    fontSize: '0.75rem'
                                }}>
                                    {request.is_intercepted ? '🔄 Intercepted' : '➡️ Passed'}
                                </Typography>
                                {request.is_encrypted && (
                                    <Typography variant="caption" sx={{
                                        display: 'inline-flex',
                                        alignItems: 'center',
                                        gap: 0.5,
                                        bgcolor: 'warning.main',
                                        color: 'warning.contrastText',
                                        px: 1,
                                        py: 0.5,
                                        borderRadius: 1,
                                        fontSize: '0.75rem'
                                    }}>
                                        🔒 Encrypted
                                    </Typography>
                                )}
                                {request.applied_rules && (
                                    <Typography variant="caption" sx={{
                                        display: 'inline-flex',
                                        alignItems: 'center',
                                        gap: 0.5,
                                        bgcolor: 'secondary.main',
                                        color: 'secondary.contrastText',
                                        px: 1,
                                        py: 0.5,
                                        borderRadius: 1,
                                        fontSize: '0.75rem'
                                    }}>
                                        📋 Rules Applied
                                    </Typography>
                                )}
                                {hasChanges() && (
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Typography variant="caption" sx={{
                                            display: 'inline-flex',
                                            alignItems: 'center',
                                            gap: 0.5,
                                            bgcolor: 'warning.light',
                                            color: 'warning.contrastText',
                                            px: 1,
                                            py: 0.5,
                                            borderRadius: 1,
                                            fontSize: '0.75rem'
                                        }}>
                                            ⚠️ Unsaved Changes
                                        </Typography>
                                        <Button
                                            size="small"
                                            variant="contained"
                                            onClick={handleResend}
                                            disabled={isResending}
                                            sx={{
                                                backgroundColor: 'warning.dark',
                                                '&:hover': {
                                                    backgroundColor: 'warning.dark',
                                                    opacity: 0.9
                                                },
                                                py: 0.5,
                                                minHeight: 0,
                                                fontSize: '0.75rem'
                                            }}
                                        >
                                            Resend Now
                                        </Button>
                                    </Box>
                                )}
                            </Box>
                        </Box>
                    </Box>
                </Box>
            </DialogTitle>
            <Box sx={{
                position: 'absolute',
                top: 16,
                right: 16,
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                background: 'rgba(0, 0, 0, 0.03)',
                borderRadius: 1,
                py: 0.5,
                px: 1,
                zIndex: 1
            }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mr: 1, borderRight: 1, borderColor: 'divider', pr: 1 }}>
                    <IconButton
                        size="small"
                        onClick={() => onNavigate(currentIndex - 1)}
                        disabled={currentIndex <= 0}
                    >
                        <KeyboardArrowLeftIcon fontSize="small" />
                    </IconButton>
                    <Typography variant="caption" sx={{ mx: 1, userSelect: 'none' }}>
                        {currentIndex + 1} / {history.length}
                    </Typography>
                    <IconButton
                        size="small"
                        onClick={() => onNavigate(currentIndex + 1)}
                        disabled={currentIndex >= history.length - 1}
                    >
                        <KeyboardArrowRightIcon fontSize="small" />
                    </IconButton>
                </Box>
                <IconButton
                    size="small"
                    onClick={onClose}
                    sx={{
                        '&:hover': {
                            color: 'error.main'
                        }
                    }}
                >
                    <CloseIcon fontSize="small" />
                </IconButton>
            </Box>
            <DialogContent dividers>
                <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
                    <Tabs value={tabValue} onChange={handleTabChange}>
                        <Tab label="Request" />
                        <Tab label="Response" />
                    </Tabs>
                </Box>

                <TabPanel value={tabValue} index={0}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="h6">Headers</Typography>
                        <Typography variant="caption" color="text.secondary">
                            {(request.tags && getDataVersion(request.tags) === 'decrypted') ? 'Decrypted' : 'Raw'} Data
                        </Typography>
                    </Box>
                    <Box sx={{ bgcolor: 'background.paper', borderRadius: 1, height: '200px', mb: 2 }}>
                        <Editor
                            height="200px"
                            defaultLanguage="json"
                            theme="vs-dark"
                            value={formatHeaders(editedRequest.headers)}
                            onChange={(value) => setEditedRequest({
                                ...editedRequest,
                                headers: value || ''
                            })}
                            options={{
                                minimap: { enabled: false },
                                scrollBeyondLastLine: false,
                                automaticLayout: true,
                                readOnly: false
                            }}
                        />
                    </Box>

                    {request.request_body && (
                        <>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                <Typography variant="h6">Body</Typography>
                                <Typography variant="caption" color="text.secondary">
                                    {(request.tags && getDataVersion(request.tags) === 'decrypted') ? 'Decrypted' : 'Raw'} Data
                                </Typography>
                            </Box>
                            <Box sx={{ bgcolor: 'background.paper', borderRadius: 1, height: '300px' }}>
                                <Editor
                                    height="300px"
                                    defaultLanguage="html"
                                    theme="vs-dark"
                                    value={editedRequest.body || formatData('request')}
                                    onChange={(value) => setEditedRequest({
                                        ...editedRequest,
                                        body: value || ''
                                    })}
                                    options={{
                                        minimap: { enabled: false },
                                        scrollBeyondLastLine: false,
                                        automaticLayout: true,
                                        readOnly: false
                                    }}
                                />
                            </Box>
                        </>
                    )}
                </TabPanel>

                <TabPanel value={tabValue} index={1}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="h6">Headers</Typography>
                        <Typography variant="caption" color="text.secondary">
                            {(request.tags && getDataVersion(request.tags) === 'decrypted') ? 'Decrypted' : 'Raw'} Data
                        </Typography>
                    </Box>
                    <Box sx={{ bgcolor: 'background.paper', borderRadius: 1, height: '200px', mb: 2 }}>
                        <Editor
                            height="200px"
                            defaultLanguage="json"
                            theme="vs-dark"
                            value={typeof editedResponse.headers === 'string' ?
                                editedResponse.headers :
                                JSON.stringify(editedResponse.headers, null, 2)
                            }
                            options={{
                                minimap: { enabled: false },
                                scrollBeyondLastLine: false,
                                automaticLayout: true,
                                readOnly: true
                            }}
                        />
                    </Box>

                    {request.response_body && (
                        <>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                <Typography variant="h6">Body</Typography>
                                <Typography variant="caption" color="text.secondary">
                                    {(request.tags && getDataVersion(request.tags) === 'decrypted') ? 'Decrypted' : 'Raw'} Data
                                </Typography>
                            </Box>
                            <Box sx={{ bgcolor: 'background.paper', borderRadius: 1, height: '300px' }}>
                                <Editor
                                    height="300px"
                                    defaultLanguage="html"
                                    theme="vs-dark"
                                    value={editedResponse.body || formatData('response')}
                                    options={{
                                        minimap: { enabled: false },
                                        scrollBeyondLastLine: false,
                                        automaticLayout: true,
                                        readOnly: true
                                    }}
                                />
                            </Box>
                        </>
                    )}
                </TabPanel>
            </DialogContent>
            <DialogContent dividers sx={{ maxHeight: '15vh', display: request.notes ? 'block' : 'none' }}>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                    Notes
                </Typography>
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                    {request.notes || ''}
                </pre>
            </DialogContent>
            <DialogActions sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                py: 1,
                px: 2
            }}>
                <Typography variant="caption" color="text.secondary">
                    ID: {request.id}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    {resendError && (
                        <Typography variant="caption" color="error">
                            Failed to resend request
                        </Typography>
                    )}
                    <Tooltip title="Resend Request">
                        <span>
                            <IconButton
                                onClick={handleResend}
                                disabled={isResending}
                                color="primary"
                                size="small"
                            >
                                {isResending ? (
                                    <CircularProgress size={20} />
                                ) : (
                                    <ReplayIcon />
                                )}
                            </IconButton>
                        </span>
                    </Tooltip>
                </Box>
            </DialogActions>
        </Dialog>
    );
};
