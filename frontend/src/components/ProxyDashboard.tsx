import React, { useState, useCallback } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';
import { WS_ENDPOINT } from '../config';
import { RequestModal } from './proxy/RequestModal';
import {
    Box,
    Grid,
    Paper,
    Typography,
    Button,
    CircularProgress,
    Alert,
    Tabs,
    Tab,
    Divider,
    FormControlLabel,
    Checkbox,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
} from '@mui/material';

interface ProxyHistoryEntry {
    id: number;
    timestamp: string;
    method: string;
    url: string;
    host?: string;
    path?: string;
    status_code?: number;
    response_status: number | null;
    duration: number | null;
    is_intercepted: boolean;
    is_encrypted: boolean;
    tags: string[];
    notes: string | null;
    request_headers: Record<string, string> | string;
    request_body: string | null;
    response_headers: Record<string, string> | string;
    response_body: string | null;
    raw_request: string;
    raw_response: string;
    decrypted_request?: string;
    decrypted_response?: string;
    applied_rules: any | null;
    session_id: number;
}

interface WebSocketMessage {
    type: string;
    data: {
        wsConnected?: boolean;
        interceptRequests?: boolean;
        interceptResponses?: boolean;
        history?: ProxyHistoryEntry[];
        [key: string]: any;
    };
}

interface ProxyHistoryMessage extends WebSocketMessage {
    type: 'proxy_history';
    data: ProxyHistoryEntry;
}

const ProxyDashboard: React.FC = () => {
    const [history, setHistory] = useState<ProxyHistoryEntry[]>([]);
    const [selectedEntry, setSelectedEntry] = useState<ProxyHistoryEntry | null>(null);
    const [wsConnected, setWsConnected] = useState(false);
    const [interceptRequests, setInterceptRequests] = useState(false);
    const [interceptResponses, setInterceptResponses] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleWebSocketMessage = useCallback((message: MessageEvent) => {
        try {
            console.log('Received WebSocket message:', message.data);
            const data: WebSocketMessage = JSON.parse(message.data);
            console.log('Parsed message:', data);

            if (data.type === "initial_data") {
                console.log('Processing initial data:', data.data);
                setWsConnected(data.data.wsConnected ?? false);
                setInterceptRequests(data.data.interceptRequests ?? false);
                setInterceptResponses(data.data.interceptResponses ?? false);
                if (data.data.history) {
                    console.log('Setting initial history:', data.data.history);
                    setHistory(data.data.history);
                }
            } else if (data.type === "proxy_history") {
                console.log('Processing history update:', data);
                const historyMessage = data as ProxyHistoryMessage;
                setHistory(prev => {
                    console.log('Previous history:', prev);
                    const newHistory = [historyMessage.data, ...prev];
                    console.log('New history:', newHistory);
                    return newHistory;
                });
            } else if (data.type === "state_update") {
                console.log('Processing state update:', data.data);
                setInterceptRequests(data.data.interceptRequests ?? interceptRequests);
                setInterceptResponses(data.data.interceptResponses ?? interceptResponses);
            }
            setError(null);
        } catch (err) {
            console.error('Error processing WebSocket message:', err);
            setError('Failed to process WebSocket message');
        }
    }, [interceptRequests, interceptResponses]);

    const { sendJsonMessage, readyState } = useWebSocket(WS_ENDPOINT, {
        onMessage: handleWebSocketMessage,
        onOpen: () => {
            console.log('WebSocket connected to:', WS_ENDPOINT);
            setWsConnected(true);
            setError(null);
        },
        onClose: () => {
            console.log('WebSocket disconnected');
            setWsConnected(false);
        },
        onError: (event: Event) => {
            console.error('WebSocket error:', event);
            setWsConnected(false);
            setError('Failed to connect to proxy server');
        },
        shouldReconnect: (closeEvent: CloseEvent) => {
            console.log('WebSocket closed, attempting to reconnect. Close event:', closeEvent);
            return true;
        },
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        share: true
    });

    const handleInterceptChange = useCallback((type: "requests" | "responses", value: boolean) => {
        const update: WebSocketMessage = {
            type: "state_update",
            data: {
                interceptRequests: type === "requests" ? value : interceptRequests,
                interceptResponses: type === "responses" ? value : interceptResponses
            }
        };
        sendJsonMessage(update);
    }, [interceptRequests, interceptResponses, sendJsonMessage]);

    const connectionStatus = {
        [ReadyState.CONNECTING]: 'Connecting',
        [ReadyState.OPEN]: 'Open',
        [ReadyState.CLOSING]: 'Closing',
        [ReadyState.CLOSED]: 'Closed',
        [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
    }[readyState];

    // Add debug render logging
    console.log('Current state:', {
        readyState,
        connectionStatus,
        wsConnected,
        historyLength: history.length,
        interceptRequests,
        interceptResponses
    });

    return (
        <Box sx={{ p: 3 }}>
            {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                </Alert>
            )}

            {/* Request Modal */}
            {selectedEntry && (
                <RequestModal
                    open={!!selectedEntry}
                    onClose={() => setSelectedEntry(null)}
                    request={selectedEntry}
                    history={history}
                    currentIndex={history.findIndex(entry => entry.id === selectedEntry.id)}
                    onNavigate={(index) => setSelectedEntry(history[index])}
                    onRequestResent={() => {
                        // Optionally refresh history after resending
                        setSelectedEntry(null);
                    }}
                />
            )}

            {/* Status and Controls */}
            <Grid container spacing={3}>
                <Grid item xs={12}>
                    <Paper sx={{ p: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                            <Typography variant="h6">
                                WebSocket Status: {connectionStatus}
                            </Typography>
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={interceptRequests}
                                        onChange={(e) => handleInterceptChange("requests", e.target.checked)}
                                        name="interceptRequests"
                                        disabled={readyState !== ReadyState.OPEN}
                                    />
                                }
                                label="Intercept Requests"
                            />
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={interceptResponses}
                                        onChange={(e) => handleInterceptChange("responses", e.target.checked)}
                                        name="interceptResponses"
                                        disabled={readyState !== ReadyState.OPEN}
                                    />
                                }
                                label="Intercept Responses"
                            />
                        </Box>
                    </Paper>
                </Grid>

                {/* History Table */}
                <Grid item xs={12}>
                    <Paper sx={{ p: 2, height: '60vh', display: 'flex', flexDirection: 'column' }}>
                        <Typography variant="h6" gutterBottom>
                            Proxy History
                        </Typography>
                        <TableContainer sx={{ flexGrow: 1, overflow: 'auto' }}>
                            <Table stickyHeader size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Method</TableCell>
                                        <TableCell>URL</TableCell>
                                        <TableCell>Status</TableCell>
                                        <TableCell>Duration</TableCell>
                                        <TableCell>Intercepted</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {history.map((entry) => (
                                        <TableRow
                                            key={entry.id}
                                            onClick={() => setSelectedEntry(entry)}
                                            sx={{ cursor: 'pointer' }}
                                        >
                                            <TableCell>{entry.method}</TableCell>
                                            <TableCell>{entry.url}</TableCell>
                                            <TableCell>{entry.status_code}</TableCell>
                                            <TableCell>
                                                {entry.duration ? `${(entry.duration * 1000).toFixed(2)}ms` : 'N/A'}
                                            </TableCell>
                                            <TableCell>{entry.is_intercepted ? 'Yes' : 'No'}</TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
};

export default ProxyDashboard; 