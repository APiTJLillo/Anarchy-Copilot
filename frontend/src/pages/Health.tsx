import React, { useEffect, useState, useCallback, useRef, useMemo } from 'react';
import {
    Box,
    Card,
    CardContent,
    Grid,
    Typography,
    CircularProgress,
    Alert,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
    IconButton,
    Tooltip,
} from '@mui/material';
import {
    Refresh as RefreshIcon,
    CheckCircle as CheckCircleIcon,
    Error as ErrorIcon,
    Warning as WarningIcon,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { formatDistanceToNow } from 'date-fns';
import { useApi } from '../hooks/useApi';
import { useWebSocket } from '../hooks/useWebSocket';
import { getWebSocketUrl } from '../config';

// Constants
const FETCH_INTERVAL = 30000; // 30 seconds between fetches
const FETCH_COOLDOWN = 2000; // Minimum time between manual fetches
const MAX_RETRIES = 3;
const INITIAL_RETRY_DELAY = 1000;

interface Connection {
    id: string;
    type: string;
    connectedAt: string;
    lastActivity: string;
    messageCount: number;
    errorCount: number;
}

interface ServiceStatus {
    name: string;
    status: 'healthy' | 'degraded' | 'down';
    lastCheck: string;
    details: string;
}

interface SystemMetrics {
    cpu: number;
    memory: number;
    disk: number;
    network: {
        in: number;
        out: number;
    };
}

interface HealthResponse {
    services: ServiceStatus[];
}

interface ProxyHealthResponse {
    metrics: SystemMetrics;
}

interface WebSocketStatusResponse {
    ui: {
        connected: boolean;
        connectionCount: number;
        lastMessage: string;
        messageCount: number;
        errorCount: number;
    };
    internal: {
        connected: boolean;
        connectionCount: number;
        lastMessage: string;
        messageCount: number;
        errorCount: number;
    };
    connections: Connection[];
}

interface HealthState {
    services: ServiceStatus[];
    metrics: SystemMetrics | null;
    wsStatus: WebSocketStatusResponse | null;
    connections: Connection[];
}

// Helper function to safely format dates
const formatDate = (date: string | Date): string => {
    try {
        const dateObj = typeof date === 'string' ? new Date(date) : date;
        return formatDistanceToNow(dateObj, { addSuffix: true });
    } catch (error) {
        console.error('Error formatting date:', error);
        return 'Invalid date';
    }
};

const Health: React.FC = () => {
    const theme = useTheme();
    const api = useApi();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [healthState, setHealthState] = useState<HealthState>({
        services: [],
        metrics: null,
        wsStatus: null,
        connections: []
    });

    // Refs for managing state
    const mountedRef = useRef(true);
    const lastFetchRef = useRef(Date.now() - FETCH_COOLDOWN); // Initialize to allow immediate fetch
    const fetchInProgressRef = useRef(false);
    const wsConnectedRef = useRef(false);
    const initialDataLoadedRef = useRef(false);

    // Fetch data with error handling and state management
    const fetchHealthData = useCallback(async (force = false) => {
        // Skip fetch if conditions aren't met
        if (
            !mountedRef.current ||
            fetchInProgressRef.current ||
            (!force && Date.now() - lastFetchRef.current < FETCH_COOLDOWN) ||
            (!force && initialDataLoadedRef.current && wsConnectedRef.current)
        ) {
            return;
        }

        try {
            fetchInProgressRef.current = true;
            if (!initialDataLoadedRef.current) {
                setLoading(true);
            }

            const [healthData, metricsData, wsData] = await Promise.all([
                api.get('/health/services'),
                api.get('/health/metrics'),
                api.get('/health/websocket-status')
            ]);

            if (!mountedRef.current) return;

            setHealthState(prev => ({
                services: [
                    ...healthData.data.services,
                    {
                        name: 'WebSocket',
                        status: wsData.data.ui.connected ? 'healthy' : 'down',
                        lastCheck: new Date().toISOString(),
                        details: wsData.data.ui.connected ? 'Connected' : 'Disconnected'
                    }
                ],
                metrics: metricsData.data,
                wsStatus: wsData.data,
                connections: wsData.data.connections || []
            }));

            initialDataLoadedRef.current = true;
            lastFetchRef.current = Date.now();
            setError(null);
        } catch (err) {
            if (!mountedRef.current) return;
            console.error('Health data fetch error:', err);
            setError('Failed to fetch health data');
        } finally {
            fetchInProgressRef.current = false;
            setLoading(false);
        }
    }, [api]);

    // Memoize WebSocket handlers
    const wsHandlers = useMemo(() => ({
        onMessage: (data: any) => {
            if (!mountedRef.current) return;

            if (data.type === 'health_update') {
                setHealthState(prev => ({
                    ...prev,
                    ...data.data
                }));
            }
        },
        onOpen: () => {
            if (!mountedRef.current) return;
            wsConnectedRef.current = true;
            setError(null);
            // Only fetch initial data if we haven't already
            if (!initialDataLoadedRef.current) {
                fetchHealthData(true);
            }
        },
        onClose: () => {
            if (!mountedRef.current) return;
            wsConnectedRef.current = false;
            // Don't update state if we're unmounted
            setHealthState(prev => ({
                ...prev,
                wsStatus: prev.wsStatus ? {
                    ...prev.wsStatus,
                    ui: { ...prev.wsStatus.ui, connected: false }
                } : null
            }));
        }
    }), [fetchHealthData]);

    // WebSocket integration
    const { isConnected, send: sendJsonMessage } = useWebSocket(getWebSocketUrl(), {
        ...wsHandlers,
        keepAlive: true,
        reconnectAttempts: 5,
        reconnectInterval: 3000,
        onOpen: () => {
            if (!mountedRef.current) return;
            wsConnectedRef.current = true;
            setError(null);
            // Send initial connection message
            sendJsonMessage({
                type: "test_connection"
            });
            // Only fetch initial data if we haven't already
            if (!initialDataLoadedRef.current) {
                fetchHealthData(true);
            }
        },
        onMessage: (data: any) => {
            if (!mountedRef.current) return;

            if (data.type === 'test_connection_response') {
                console.log('Test connection successful');
                return;
            }

            if (data.type === 'health_update') {
                setHealthState(prev => ({
                    ...prev,
                    ...data.data
                }));
            }
        }
    });

    // Component lifecycle
    useEffect(() => {
        mountedRef.current = true;
        wsConnectedRef.current = false;
        initialDataLoadedRef.current = false;

        // Initial fetch only if WebSocket isn't connected
        if (!isConnected) {
            fetchHealthData(true);
        }

        return () => {
            mountedRef.current = false;
            wsConnectedRef.current = false;
            initialDataLoadedRef.current = false;
        };
    }, [fetchHealthData, isConnected]);

    // Polling effect - only active when WebSocket is disconnected
    useEffect(() => {
        if (!mountedRef.current || isConnected) return;

        const intervalId = setInterval(() => {
            if (mountedRef.current && !wsConnectedRef.current) {
                fetchHealthData(false);
            }
        }, FETCH_INTERVAL);

        return () => clearInterval(intervalId);
    }, [fetchHealthData, isConnected]);

    // Manual refresh handler
    const handleRefresh = useCallback(async () => {
        if (loading) {
            return;
        }

        setLoading(true);
        fetchInProgressRef.current = true;

        try {
            const [healthData, metricsData, wsData] = await Promise.all([
                api.get('/health/services'),
                api.get('/health/metrics'),
                api.get('/health/websocket-status')
            ]);

            if (!mountedRef.current) return;

            setHealthState(prev => ({
                services: [
                    ...healthData.data.services,
                    {
                        name: 'WebSocket',
                        status: wsData.data.ui.connected ? 'healthy' : 'down',
                        lastCheck: new Date().toISOString(),
                        details: wsData.data.ui.connected ? 'Connected' : 'Disconnected'
                    }
                ],
                metrics: metricsData.data,
                wsStatus: wsData.data,
                connections: wsData.data.connections || []
            }));

            lastFetchRef.current = Date.now();
            setError(null);
        } catch (err) {
            console.error('Manual refresh error:', err);
            if (mountedRef.current) {
                setError('Failed to refresh health data');
            }
        } finally {
            if (mountedRef.current) {
                setLoading(false);
                fetchInProgressRef.current = false;
            }
        }
    }, [api, loading]);

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'healthy':
                return theme.palette.success.main;
            case 'degraded':
                return theme.palette.warning.main;
            case 'down':
                return theme.palette.error.main;
            default:
                return theme.palette.grey[500];
        }
    };

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'healthy':
                return <CheckCircleIcon color="success" />;
            case 'degraded':
                return <WarningIcon color="warning" />;
            case 'down':
                return <ErrorIcon color="error" />;
            default:
                return <ErrorIcon />;
        }
    };

    if (loading && !healthState.services.length) {
        return (
            <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
                <CircularProgress />
            </Box>
        );
    }

    return (
        <Box p={3}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
                <Typography variant="h4">System Health</Typography>
                <span>
                    <Tooltip title={loading ? 'Refreshing...' : 'Refresh'}>
                        <IconButton
                            onClick={handleRefresh}
                            disabled={loading}
                            sx={{ opacity: loading ? 0.7 : 1 }}
                        >
                            {loading ? <CircularProgress size={24} /> : <RefreshIcon />}
                        </IconButton>
                    </Tooltip>
                </span>
            </Box>

            {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                </Alert>
            )}

            <Grid container spacing={3}>
                {/* Service Status */}
                <Grid item xs={12} md={6}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Service Status
                            </Typography>
                            {healthState.services.length > 0 ? (
                                <Grid container spacing={2}>
                                    {healthState.services.map((service) => (
                                        <Grid item xs={12} key={service.name}>
                                            <Paper sx={{ p: 2 }}>
                                                <Box display="flex" alignItems="center" justifyContent="space-between">
                                                    <Box display="flex" alignItems="center" gap={1}>
                                                        {getStatusIcon(service.status)}
                                                        <Typography variant="subtitle1">
                                                            {service.name}
                                                        </Typography>
                                                    </Box>
                                                    <Chip
                                                        label={service.status}
                                                        size="small"
                                                        sx={{
                                                            backgroundColor: getStatusColor(service.status) + '20',
                                                            color: getStatusColor(service.status)
                                                        }}
                                                    />
                                                </Box>
                                                <Typography variant="body2" color="text.secondary" mt={1}>
                                                    Last checked: {formatDate(service.lastCheck)}
                                                </Typography>
                                                {service.details && (
                                                    <Typography variant="body2" mt={0.5}>
                                                        {service.details}
                                                    </Typography>
                                                )}
                                            </Paper>
                                        </Grid>
                                    ))}
                                </Grid>
                            ) : (
                                <Typography color="text.secondary">No services available</Typography>
                            )}
                        </CardContent>
                    </Card>
                </Grid>

                {/* System Metrics */}
                <Grid item xs={12} md={6}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                System Metrics
                            </Typography>
                            {healthState.metrics ? (
                                <Grid container spacing={2}>
                                    <Grid item xs={6}>
                                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                                            <Typography variant="subtitle2" color="textSecondary">
                                                CPU Usage
                                            </Typography>
                                            <Typography variant="h4">{healthState.metrics.cpu}%</Typography>
                                        </Paper>
                                    </Grid>
                                    <Grid item xs={6}>
                                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                                            <Typography variant="subtitle2" color="textSecondary">
                                                Memory Usage
                                            </Typography>
                                            <Typography variant="h4">{healthState.metrics.memory}%</Typography>
                                        </Paper>
                                    </Grid>
                                    <Grid item xs={6}>
                                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                                            <Typography variant="subtitle2" color="textSecondary">
                                                Disk Usage
                                            </Typography>
                                            <Typography variant="h4">{healthState.metrics.disk}%</Typography>
                                        </Paper>
                                    </Grid>
                                    <Grid item xs={6}>
                                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                                            <Typography variant="subtitle2" color="textSecondary">
                                                Network I/O
                                            </Typography>
                                            <Typography variant="body2">
                                                ↑ {healthState.metrics.network.out} MB/s
                                            </Typography>
                                            <Typography variant="body2">
                                                ↓ {healthState.metrics.network.in} MB/s
                                            </Typography>
                                        </Paper>
                                    </Grid>
                                </Grid>
                            ) : (
                                <Typography color="text.secondary">No metrics available</Typography>
                            )}
                        </CardContent>
                    </Card>
                </Grid>

                {/* WebSocket Status */}
                {healthState.wsStatus && (
                    <Grid item xs={12} md={6}>
                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    WebSocket Status
                                </Typography>
                                <Grid container spacing={2}>
                                    {/* UI Connection */}
                                    <Grid item xs={12}>
                                        <Typography variant="subtitle1" gutterBottom>
                                            UI Connection
                                        </Typography>
                                        <Grid container spacing={2}>
                                            <Grid item xs={6}>
                                                <Paper sx={{ p: 2, textAlign: 'center' }}>
                                                    <Typography variant="subtitle2" color="textSecondary">
                                                        Status
                                                    </Typography>
                                                    <Chip
                                                        icon={healthState.wsStatus.ui.connected ? <CheckCircleIcon /> : <ErrorIcon />}
                                                        label={healthState.wsStatus.ui.connected ? 'Connected' : 'Disconnected'}
                                                        color={healthState.wsStatus.ui.connected ? 'success' : 'error'}
                                                        sx={{ mt: 1 }}
                                                    />
                                                </Paper>
                                            </Grid>
                                            <Grid item xs={6}>
                                                <Paper sx={{ p: 2, textAlign: 'center' }}>
                                                    <Typography variant="subtitle2" color="textSecondary">
                                                        Messages
                                                    </Typography>
                                                    <Typography variant="h4">
                                                        {healthState.wsStatus.ui.messageCount}
                                                    </Typography>
                                                    <Typography variant="caption" color="text.secondary">
                                                        Errors: {healthState.wsStatus.ui.errorCount}
                                                    </Typography>
                                                </Paper>
                                            </Grid>
                                        </Grid>
                                    </Grid>

                                    {/* Internal Connection */}
                                    <Grid item xs={12}>
                                        <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>
                                            Dev-Proxy Connection
                                        </Typography>
                                        <Grid container spacing={2}>
                                            <Grid item xs={6}>
                                                <Paper sx={{ p: 2, textAlign: 'center' }}>
                                                    <Typography variant="subtitle2" color="textSecondary">
                                                        Status
                                                    </Typography>
                                                    <Chip
                                                        icon={healthState.wsStatus.internal.connected ? <CheckCircleIcon /> : <ErrorIcon />}
                                                        label={healthState.wsStatus.internal.connected ? 'Connected' : 'Disconnected'}
                                                        color={healthState.wsStatus.internal.connected ? 'success' : 'error'}
                                                        sx={{ mt: 1 }}
                                                    />
                                                </Paper>
                                            </Grid>
                                            <Grid item xs={6}>
                                                <Paper sx={{ p: 2, textAlign: 'center' }}>
                                                    <Typography variant="subtitle2" color="textSecondary">
                                                        Messages
                                                    </Typography>
                                                    <Typography variant="h4">
                                                        {healthState.wsStatus.internal.messageCount}
                                                    </Typography>
                                                    <Typography variant="caption" color="text.secondary">
                                                        Errors: {healthState.wsStatus.internal.errorCount}
                                                    </Typography>
                                                </Paper>
                                            </Grid>
                                        </Grid>
                                    </Grid>
                                </Grid>
                            </CardContent>
                        </Card>
                    </Grid>
                )}

                {/* Active Connections */}
                {healthState.connections.length > 0 && (
                    <Grid item xs={12} md={6}>
                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Active Connections
                                </Typography>
                                <Grid container spacing={2}>
                                    {healthState.connections.map((conn) => (
                                        <Grid item xs={12} key={conn.id}>
                                            <Paper sx={{ p: 2 }}>
                                                <Box display="flex" justifyContent="space-between" alignItems="center">
                                                    <Box>
                                                        <Typography variant="subtitle2">
                                                            {conn.type} Connection
                                                        </Typography>
                                                        <Typography variant="body2" color="text.secondary">
                                                            ID: {conn.id}
                                                        </Typography>
                                                    </Box>
                                                    <Box textAlign="right">
                                                        <Typography variant="body2">
                                                            Messages: {conn.messageCount}
                                                        </Typography>
                                                        <Typography variant="body2" color="error">
                                                            Errors: {conn.errorCount}
                                                        </Typography>
                                                    </Box>
                                                </Box>
                                            </Paper>
                                        </Grid>
                                    ))}
                                </Grid>
                            </CardContent>
                        </Card>
                    </Grid>
                )}
            </Grid>
        </Box>
    );
};

export default Health; 