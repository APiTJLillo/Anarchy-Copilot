import React, { useState } from 'react';
import {
    Box,
    Tabs,
    Tab,
    TextField,
    Button,
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

interface ResponseEditorProps {
    statusCode: number;
    headers: Header[];
    body: string;
    onStatusCodeChange: (code: number) => void;
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
            id={`response-tabpanel-${index}`}
            aria-labelledby={`response-tab-${index}`}
            {...other}
        >
            {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
        </div>
    );
}

export const ResponseEditor: React.FC<ResponseEditorProps> = ({
    statusCode,
    headers,
    body,
    onStatusCodeChange,
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

    const handleAddHeader = () => {
        if (newHeaderName && newHeaderValue) {
            onHeadersChange([...headers, { name: newHeaderName, value: newHeaderValue }]);
            setNewHeaderName('');
            setNewHeaderValue('');
        }
    };

    const handleDeleteHeader = (index: number) => {
        const newHeaders = headers.filter((_, i) => i !== index);
        onHeadersChange(newHeaders);
    };

    const handleStatusCodeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const code = parseInt(e.target.value, 10);
        if (!isNaN(code) && code >= 100 && code < 600) {
            onStatusCodeChange(code);
        }
    };

    return (
        <Box sx={{ width: '100%' }}>
            <Box sx={{ mb: 2 }}>
                <TextField
                    label="Status Code"
                    type="number"
                    value={statusCode}
                    onChange={handleStatusCodeChange}
                    inputProps={{ min: 100, max: 599 }}
                    sx={{ width: 150 }}
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
