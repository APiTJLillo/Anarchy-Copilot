import React from 'react';
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
} from '@mui/material';
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
}

export const RequestModal: React.FC<RequestModalProps> = ({
    open,
    onClose,
    request,
    history,
    currentIndex,
    onNavigate
}) => {
    const [tabValue, setTabValue] = React.useState(0);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
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

    if (!request) return null;

    const getDataByTags = (request: RequestData, type: string) => {
        if (!request || !request.tags) return null;

        const tags = parseTags(request.tags);
        const decrypted = tags.includes('decrypted');
        const raw = tags.includes('raw');
        
        // Always prefer decrypted data if available
        if (type === 'request') {
            if (decrypted && request.decrypted_request) {
                return request.decrypted_request;
            } else if (raw && request.raw_request) {
                return request.raw_request;
            } else if (request.request_body) {
                return request.request_body;
            }
            return null;
        }
        
        if (type === 'response') {
            if (decrypted && request.decrypted_response) {
                return request.decrypted_response;
            } else if (raw && request.raw_response) {
                return request.raw_response;
            } else if (request.response_body) {
                return request.response_body;
            }
            return null;
        }
        
        if (type === 'request_headers') {
            try {
                if (decrypted && request.request_headers) {
                    return typeof request.request_headers === 'string' ? 
                        JSON.parse(request.request_headers) : 
                        request.request_headers;
                }
                return null;
            } catch (e) {
                console.error('Failed to parse request headers:', e);
                return null;
            }
        }
        
        if (type === 'response_headers') {
            try {
                if (decrypted && request.response_headers) {
                    return typeof request.response_headers === 'string' ? 
                        JSON.parse(request.response_headers) : 
                        request.response_headers;
                }
                return null;
            } catch (e) {
                console.error('Failed to parse response headers:', e);
                return null;
            }
        }
        
        return null;
    };

    const decodeBase64 = (data: string): string => {
        if (!data) return '';
        if (data.startsWith('base64://')) {
            try {
                return atob(data.substring('base64://'.length));
            } catch {
                return 'Failed to decode base64 data';
            }
        }
        return data;
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
            const decodedData = decodeBase64(data);

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

    return (
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth="lg"
            fullWidth
            PaperProps={{ sx: { height: '80vh' } }}
        >
            <DialogTitle sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                    <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 1 }}>
                        <Box sx={{ mb: 0.5 }}>
                            <Typography variant="h6" component="div" sx={{ 
                                display: 'flex', 
                                alignItems: 'center',
                                gap: 1.5,
                                mb: 0.5
                            }}>
                                <Box component="span" sx={{ 
                                    bgcolor: {
                                        'GET': 'success.main',
                                        'POST': 'info.main',
                                        'PUT': 'warning.main',
                                        'DELETE': 'error.main',
                                        'PATCH': 'warning.dark',
                                    }[request.method] || 'primary.main',
                                    color: '#fff',
                                    px: 1,
                                    py: 0.5,
                                    borderRadius: 1,
                                    fontSize: '0.9rem',
                                    fontWeight: 'bold',
                                    minWidth: '70px',
                                    textAlign: 'center',
                                    letterSpacing: '0.5px'
                                }}>
                                    {request.method}
                                </Box>
                                <Box sx={{ 
                                    display: 'flex', 
                                    alignItems: 'center',
                                    fontFamily: 'monospace',
                                    fontSize: '0.95rem',
                                    fontWeight: 'medium',
                                    bgcolor: 'background.paper',
                                    borderRadius: 1,
                                    overflow: 'hidden',
                                    border: '1px solid',
                                    borderColor: 'divider',
                                }}>
                                    {request.host && (
                                        <Box component="span" sx={{ 
                                            px: 1,
                                            py: 0.5,
                                            bgcolor: 'background.default',
                                            borderRight: '1px solid',
                                            borderColor: 'divider',
                                            color: 'text.secondary'
                                        }}>
                                            {request.host}
                                        </Box>
                                    )}
                                    <Box component="span" sx={{ 
                                        px: 1,
                                        py: 0.5,
                                        color: 'text.primary'
                                    }}>
                                        {request.path || request.url}
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
                            </Box>
                        </Box>
                        <Box sx={{ 
                            display: 'flex', 
                            gap: 2,
                            alignItems: 'center',
                            p: 1,
                            borderRadius: 1,
                            bgcolor: 'background.paper',
                            border: '1px solid',
                            borderColor: 'divider',
                        }}>
                            <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <Box component="span" sx={{ opacity: 0.7 }}>Session:</Box>
                                <Box component="span" sx={{ fontFamily: 'monospace', fontWeight: 'medium' }}>
                                    {request.session_id}
                                </Box>
                            </Typography>
                            <Box sx={{ mx: 2, height: '1rem', borderLeft: 1, borderColor: 'divider' }} />
                            <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <Box component="span" sx={{ opacity: 0.7 }}>Time:</Box>
                                <Box component="span" sx={{ fontFamily: 'monospace' }}>
                                    {new Date(request.timestamp).toLocaleString()}
                                </Box>
                            </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', gap: 1.5, mt: 0.5 }}>
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
                        </Box>
                    </Box>
                    <Box sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: 0.5, 
                        background: 'rgba(0, 0, 0, 0.03)',
                        borderRadius: 1,
                        py: 0.5,
                        px: 1
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
                </Box>
            </DialogTitle>
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
                    <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, mb: 2 }}>
                        <pre style={{ margin: 0, overflow: 'auto' }}>
                            {formatData('request_headers')}
                        </pre>
                    </Box>

                    {request.request_body && (
                        <>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                <Typography variant="h6">Body</Typography>
                                <Typography variant="caption" color="text.secondary">
                                    {(request.tags && getDataVersion(request.tags) === 'decrypted') ? 'Decrypted' : 'Raw'} Data
                                </Typography>
                            </Box>
                            <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1 }}>
                                <pre style={{ margin: 0, overflow: 'auto' }}>
                                    {formatData('request')}
                                </pre>
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
                    <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, mb: 2 }}>
                        <pre style={{ margin: 0, overflow: 'auto' }}>
                            {formatData('response_headers')}
                        </pre>
                    </Box>

                    {request.response_body && (
                        <>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                <Typography variant="h6">Body</Typography>
                                <Typography variant="caption" color="text.secondary">
                                    {(request.tags && getDataVersion(request.tags) === 'decrypted') ? 'Decrypted' : 'Raw'} Data
                                </Typography>
                            </Box>
                            <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1 }}>
                                <pre style={{ margin: 0, overflow: 'auto' }}>
                                    {formatData('response')}
                                </pre>
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
            <DialogActions sx={{ justifyContent: 'flex-start', py: 1, px: 2 }}>
                <Typography variant="caption" color="text.secondary">
                    ID: {request.id}
                </Typography>
            </DialogActions>
        </Dialog>
    );
};
