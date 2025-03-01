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

interface RequestModalProps {
    open: boolean;
    onClose: () => void;
    request: any;
    history: any[];
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

    if (!request) return null;

    const formatData = (data: any, headers?: Record<string, string>) => {
        // If data is a hex string, try to convert it back to readable format
        if (typeof data === 'string' && /^[0-9A-Fa-f]+$/.test(data)) {
            const bytes = new Uint8Array(data.match(/.{1,2}/g)?.map(byte => parseInt(byte, 16)) || []);
            const contentType = headers?.['content-type'] || '';

            // Handle text-based content types
            if (contentType.includes('text/') ||
                contentType.includes('application/json') ||
                contentType.includes('application/xml') ||
                contentType.includes('application/javascript')) {
                try {
                    const text = new TextDecoder().decode(bytes);
                    if (contentType.includes('json')) {
                        return JSON.stringify(JSON.parse(text), null, 2);
                    }
                    return text;
                } catch (e) {
                    return `[Binary data - ${bytes.length} bytes]`;
                }
            }

            // Handle other content types
            if (contentType.startsWith('image/')) {
                return `[Image data - ${bytes.length} bytes]`;
            }
            return `[Binary data - ${bytes.length} bytes (${contentType || 'unknown type'})]`;
        }

        // Handle regular string data
        if (typeof data === 'string') {
            try {
                return JSON.stringify(JSON.parse(data), null, 2);
            } catch {
                return data;
            }
        }
        return JSON.stringify(data, null, 2);
    };

    return (
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth="lg"
            fullWidth
            PaperProps={{ sx: { height: '80vh' } }}
        >
            <DialogTitle>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <IconButton
                        onClick={() => onNavigate(currentIndex - 1)}
                        disabled={currentIndex <= 0}
                    >
                        <KeyboardArrowLeftIcon />
                    </IconButton>
                    <Box sx={{ flex: 1 }}>
                        <Typography variant="h6" component="div">
                            <strong>{request.method}</strong> {request.url}
                        </Typography>
                        <Typography variant="subtitle1" color="text.secondary">
                            Status: {request.response_status} | Duration: {request.duration ? `${(request.duration * 1000).toFixed(2)}ms` : 'N/A'}
                        </Typography>
                    </Box>
                    <IconButton
                        onClick={() => onNavigate(currentIndex + 1)}
                        disabled={currentIndex >= history.length - 1}
                    >
                        <KeyboardArrowRightIcon />
                    </IconButton>
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
                    <Typography variant="h6" gutterBottom>Headers</Typography>
                    <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, mb: 2 }}>
                        <pre style={{ margin: 0, overflow: 'auto' }}>
                            {formatData(request.request_headers, request.request_headers)}
                        </pre>
                    </Box>

                    {request.request_body && (
                        <>
                            <Typography variant="h6" gutterBottom>Body</Typography>
                            <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1 }}>
                                <pre style={{ margin: 0, overflow: 'auto' }}>
                                    {formatData(request.request_body, request.request_headers)}
                                </pre>
                            </Box>
                        </>
                    )}
                </TabPanel>

                <TabPanel value={tabValue} index={1}>
                    <Typography variant="h6" gutterBottom>Headers</Typography>
                    <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, mb: 2 }}>
                        <pre style={{ margin: 0, overflow: 'auto' }}>
                            {formatData(request.response_headers, request.response_headers)}
                        </pre>
                    </Box>

                    {request.response_body && (
                        <>
                            <Typography variant="h6" gutterBottom>Body</Typography>
                            <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1 }}>
                                <pre style={{ margin: 0, overflow: 'auto' }}>
                                    {formatData(request.response_body, request.response_headers)}
                                </pre>
                            </Box>
                        </>
                    )}
                </TabPanel>
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Close</Button>
            </DialogActions>
        </Dialog>
    );
};
