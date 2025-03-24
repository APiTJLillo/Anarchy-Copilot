import React, { useState, useCallback } from 'react';
import {
    Box,
    Tabs,
    Tab,
    TextField,
    Button,
    Typography,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    IconButton,
} from '@mui/material';
import { Delete as DeleteIcon, Add as AddIcon } from '@mui/icons-material';

interface Header {
    name: string;
    value: string;
}

interface RequestEditorProps {
    method: string;
    url: string;
    headers: Header[];
    body: string;
    onMethodChange: (method: string) => void;
    onUrlChange: (url: string) => void;
    onHeadersChange: (headers: Header[]) => void;
    onBodyChange: (body: string) => void;
    onForward: () => void;
    onDrop: () => void;
}

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`request-tabpanel-${index}`}
            aria-labelledby={`request-tab-${index}`}
            {...other}
        >
            {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
        </div>
    );
}

export const RequestEditor: React.FC<RequestEditorProps> = ({
    method,
    url,
    headers,
    body,
    onMethodChange,
    onUrlChange,
    onHeadersChange,
    onBodyChange,
    onForward,
    onDrop,
}) => {
    const [currentTab, setCurrentTab] = useState(0);
    const [newHeaderName, setNewHeaderName] = useState('');
    const [newHeaderValue, setNewHeaderValue] = useState('');

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setCurrentTab(newValue);
    };

    const handleAddHeader = useCallback(() => {
        if (newHeaderName && newHeaderValue) {
            onHeadersChange([...headers, { name: newHeaderName, value: newHeaderValue }]);
            setNewHeaderName('');
            setNewHeaderValue('');
        }
    }, [newHeaderName, newHeaderValue, headers, onHeadersChange]);

    const handleDeleteHeader = useCallback((index: number) => {
        const newHeaders = headers.filter((_, i) => i !== index);
        onHeadersChange(newHeaders);
    }, [headers, onHeadersChange]);

    return (
        <Box sx={{ width: '100%' }}>
            <Box sx={{ mb: 2, display: 'flex', gap: 2 }}>
                <TextField
                    select
                    label="Method"
                    value={method}
                    onChange={(e) => onMethodChange(e.target.value)}
                    SelectProps={{ native: true }}
                    sx={{ width: 100 }}
                >
                    {['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'].map((m) => (
                        <option key={m} value={m}>{m}</option>
                    ))}
                </TextField>
                <TextField
                    fullWidth
                    label="URL"
                    value={url}
                    onChange={(e) => onUrlChange(e.target.value)}
                />
            </Box>

            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={currentTab} onChange={handleTabChange}>
                    <Tab label="Headers" />
                    <Tab label="Body" />
                </Tabs>
            </Box>

            <TabPanel value={currentTab} index={0}>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>Name</TableCell>
                                <TableCell>Value</TableCell>
                                <TableCell width={50}></TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {headers.map((header, index) => (
                                <TableRow key={index}>
                                    <TableCell>{header.name}</TableCell>
                                    <TableCell>{header.value}</TableCell>
                                    <TableCell>
                                        <IconButton size="small" onClick={() => handleDeleteHeader(index)}>
                                            <DeleteIcon />
                                        </IconButton>
                                    </TableCell>
                                </TableRow>
                            ))}
                            <TableRow>
                                <TableCell>
                                    <TextField
                                        fullWidth
                                        size="small"
                                        placeholder="Header name"
                                        value={newHeaderName}
                                        onChange={(e) => setNewHeaderName(e.target.value)}
                                    />
                                </TableCell>
                                <TableCell>
                                    <TextField
                                        fullWidth
                                        size="small"
                                        placeholder="Header value"
                                        value={newHeaderValue}
                                        onChange={(e) => setNewHeaderValue(e.target.value)}
                                    />
                                </TableCell>
                                <TableCell>
                                    <IconButton size="small" onClick={handleAddHeader}>
                                        <AddIcon />
                                    </IconButton>
                                </TableCell>
                            </TableRow>
                        </TableBody>
                    </Table>
                </TableContainer>
            </TabPanel>

            <TabPanel value={currentTab} index={1}>
                <TextField
                    fullWidth
                    multiline
                    minRows={10}
                    maxRows={20}
                    value={body}
                    onChange={(e) => onBodyChange(e.target.value)}
                    sx={{ fontFamily: 'monospace' }}
                />
            </TabPanel>

            <Box sx={{ mt: 2, display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                <Button variant="outlined" color="error" onClick={onDrop}>
                    Drop
                </Button>
                <Button variant="contained" color="primary" onClick={onForward}>
                    Forward
                </Button>
            </Box>
        </Box>
    );
};
