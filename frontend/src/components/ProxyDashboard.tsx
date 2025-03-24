import React, { useState, useCallback, useEffect } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';
import { getWebSocketUrl } from '../config';
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
import AnalysisResults from '../components/proxy/AnalysisResults';
import ErrorBoundary from '../components/proxy/ErrorBoundary';
import { useProxyApi, ProxySettings } from '../api/proxyApi';
import { User } from '../api/proxyApi';
import { Project } from '../api/proxyApi';
import { VersionInfo } from '../api/proxyApi';

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
        status?: { isRunning: boolean };
        isRunning?: boolean;
        version?: VersionInfo;
        [key: string]: any;
    };
}

interface ProxyHistoryMessage extends WebSocketMessage {
    type: 'proxy_history';
    data: ProxyHistoryEntry;
}

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

const CustomTabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
    <Box
        role="tabpanel"
        hidden={value !== index}
        id={`proxy-tabpanel-${index}`}
        aria-labelledby={`proxy-tab-${index}`}
        sx={{ py: 2 }}
    >
        {value === index && children}
    </Box>
);

const ProxyDashboard: React.FC = () => {
    const [history, setHistory] = useState<ProxyHistoryEntry[]>([]);
    const [selectedEntry, setSelectedEntry] = useState<ProxyHistoryEntry | null>(null);
    const [wsConnected, setWsConnected] = useState(false);
    const [interceptRequests, setInterceptRequests] = useState(false);
    const [interceptResponses, setInterceptResponses] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [analysisResults, setAnalysisResults] = useState<any[] | null>(null);
    const [tabValue, setTabValue] = useState(0);
    const [proxyStatus, setProxyStatus] = useState<{ isRunning: boolean }>({ isRunning: false });
    const [currentUser, setCurrentUser] = useState<User | null>(null);
    const [selectedProject, setSelectedProject] = useState<Project | null>(null);
    const [session, setSession] = useState<any>(null);
    const [localProxyEnabled, setLocalProxyEnabled] = useState(false);
    const [version, setVersion] = useState<VersionInfo | null>(null);
    const proxyApi = useProxyApi();

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };

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
                if (data.data.status) {
                    console.log('Setting initial proxy status:', data.data.status);
                    setProxyStatus(data.data.status);
                }
                if (data.data.version) {
                    console.log('Setting version info:', data.data.version);
                    setVersion(data.data.version);
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
                if (data.data.status) {
                    setProxyStatus(data.data.status);
                }
            } else if (data.type === "proxy_status") {
                console.log('Processing proxy status update:', data.data);
                setProxyStatus({ isRunning: data.data.isRunning ?? false });
            }
            setError(null);
        } catch (err) {
            console.error('Error processing WebSocket message:', err);
            setError('Failed to process WebSocket message');
        }
    }, [interceptRequests, interceptResponses]);

    const fetchStatus = useCallback(async () => {
        try {
            const status = await proxyApi.getStatus();
            setProxyStatus(status);
            setInterceptRequests(status.interceptRequests ?? false);
            setInterceptResponses(status.interceptResponses ?? false);
        } catch (err) {
            console.error('Failed to fetch status:', err);
            setError('Failed to fetch proxy status');
        }
    }, [proxyApi]);

    // Fetch status on mount
    useEffect(() => {
        fetchStatus();
    }, [fetchStatus]);

    const { sendJsonMessage, readyState } = useWebSocket(getWebSocketUrl(), {
        onMessage: handleWebSocketMessage,
        onOpen: () => {
            console.log('WebSocket connected to:', getWebSocketUrl());
            setWsConnected(true);
            setError(null);
            fetchStatus(); // Fetch status when WebSocket connects
        },
        onClose: () => {
            console.log('WebSocket disconnected');
            setWsConnected(false);
            // Don't set error on normal closure
            if (readyState !== ReadyState.CLOSING) {
                setError('Lost connection to proxy server');
            }
        },
        onError: (event: Event) => {
            console.error('WebSocket error:', event);
            setWsConnected(false);
            setError('Failed to connect to proxy server');
        },
        shouldReconnect: (closeEvent: CloseEvent) => {
            console.log('WebSocket closed, attempting to reconnect. Close event:', closeEvent);
            // Don't reconnect on normal closure
            return closeEvent.code !== 1000 && closeEvent.code !== 1001;
        },
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        share: true,
        retryOnError: true,
        filter: (message: MessageEvent) => {
            try {
                const data = JSON.parse(message.data);
                return data && typeof data === 'object';
            } catch {
                return false;
            }
        }
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

    const startProxy = useCallback(async () => {
        try {
            if (!currentUser || !selectedProject) {
                setError('Please select a user and project');
                return;
            }

            // Create settings with snake_case properties
            const settings: ProxySettings = {
                host: "127.0.0.1",
                port: 8083,
                intercept_requests: true,
                intercept_responses: true,
                allowed_hosts: [],
                excluded_hosts: [],
                max_connections: 100,
                max_keepalive_connections: 20,
                keepalive_timeout: 30
            };

            const newSession = await proxyApi.createSession("New Session", selectedProject.id, currentUser.id, settings);
            setSession(newSession);

            // Start the proxy with the same settings
            await proxyApi.startProxy(newSession.id, settings);
            await fetchStatus();
            if (localProxyEnabled) {
                console.log("Local proxy client started");
            }
            setError(null);
        } catch (err) {
            setError('Failed to start proxy');
            console.error(err);
        }
    }, [fetchStatus, currentUser, selectedProject, localProxyEnabled, proxyApi]);

    const stopProxy = useCallback(async () => {
        try {
            const update: WebSocketMessage = {
                type: "stop_proxy",
                data: {}
            };
            sendJsonMessage(update);
        } catch (err) {
            console.error('Failed to stop proxy:', err);
            setError('Failed to stop proxy');
        }
    }, [sendJsonMessage]);

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
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography variant="h6">
                                    Proxy Status: {proxyStatus.isRunning ? 'Running' : 'Stopped'}
                                </Typography>
                            </Box>
                            <Button
                                variant="contained"
                                color={proxyStatus.isRunning ? 'error' : 'primary'}
                                onClick={proxyStatus.isRunning ? stopProxy : startProxy}
                                disabled={!wsConnected}
                            >
                                {proxyStatus.isRunning ? 'Stop Proxy' : 'Start Proxy'}
                            </Button>
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

            <CustomTabPanel value={tabValue} index={2}>
                <Grid container spacing={3}>
                    <Grid item xs={12}>
                        <ErrorBoundary component="Analysis Results">
                            <AnalysisResults results={analysisResults || []} />
                        </ErrorBoundary>
                    </Grid>
                </Grid>
            </CustomTabPanel>
        </Box>
    );
};

export default ProxyDashboard; 