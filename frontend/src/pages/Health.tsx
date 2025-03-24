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
import useWebSocketChannel from '../hooks/useWebSocketChannel';

// Constants
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
        active_connections?: Array<{
            id: string;
            connected_since: string;
            last_activity: string;
            message_count: number;
            error_count: number;
            state: string;
            last_error?: string;
            last_message_type?: string;
            connection_history: Array<{
                timestamp: string;
                event: string;
                details: string;
            }>;
        }>;
        connection_history?: Array<{
            timestamp: string;
            event: string;
            connection_id: string;
            details: string;
        }>;
    };
    internal: {
        connected: boolean;
        connectionCount: number;
        lastMessage: string;
        messageCount: number;
        errorCount: number;
        active_connections?: Array<{
            id: string;
            connected_since: string;
            last_activity: string;
            message_count: number;
            error_count: number;
            state: string;
            last_error?: string;
            last_message_type?: string;
            connection_history: Array<{
                timestamp: string;
                event: string;
                details: string;
            }>;
        }>;
        connection_history?: Array<{
            timestamp: string;
            event: string;
            connection_id: string;
            details: string;
        }>;
    };
    connections: Connection[];
}

interface HealthState {
    services: ServiceStatus[];
    metrics: SystemMetrics | null;
    wsStatus: WebSocketStatusResponse | null;
    connections: Connection[];
}

interface HealthData {
    status: string;
    active_connections: number;
    uptime: number;
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

// Add debug logging utility
const debugLog = (message: string, data?: any) => {
    const stack = new Error().stack?.split('\n').slice(2);
    console.debug(`[Health Debug] ${message}`, {
        timestamp: new Date().toISOString(),
        stack,
        ...data
    });
};

const Health: React.FC = React.memo(() => {
    const theme = useTheme();
    const api = useApi();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [healthState, setHealthState] = useState<HealthState>({
        services: [],
        metrics: null,
        wsStatus: {
            ui: {
                connected: false,
                connectionCount: 0,
                lastMessage: '',
                messageCount: 0,
                errorCount: 0,
                active_connections: [],
                connection_history: []
            },
            internal: {
                connected: false,
                connectionCount: 0,
                lastMessage: '',
                messageCount: 0,
                errorCount: 0,
                active_connections: [],
                connection_history: []
            },
            connections: []
        },
        connections: []
    });
    const [healthData, setHealthData] = useState<HealthData | null>(null);
    const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

    // WebSocket connection using our hook
    const { isConnected, error: wsError, send } = useWebSocketChannel('health', {
        onMessage: (data) => {
            if (!data) return;
            console.debug('[Health] Received WebSocket message:', data);

            try {
                if (data.type === 'health_update' || data.type === 'initial_data') {
                    console.debug('[Health] Processing health data:', data.data);
                    setHealthState(prevState => ({
                        ...prevState,
                        services: data.data.services || [],
                        metrics: data.data.metrics,
                        wsStatus: data.data.wsStatus,
                        connections: data.data.connections || []
                    }));
                    setLastUpdate(new Date());
                    if (data.type === 'initial_data') {
                        setLoading(false);
                    }
                }
            } catch (err) {
                console.error('Error handling WebSocket message:', err);
                setError('Failed to process health update');
            }
        }
    });

    // Request initial data when WebSocket connects
    useEffect(() => {
        if (isConnected) {
            setLoading(true);
            console.debug('[Health] Requesting initial data');
            send({
                type: 'get_initial_data',
                channel: 'health'
            });
        }
    }, [isConnected, send]);

    // Set up periodic health updates
    useEffect(() => {
        if (!isConnected) return;

        console.debug('[Health] Setting up periodic updates');
        const updateInterval = setInterval(() => {
            console.debug('[Health] Requesting health update');
            send({
                type: 'get_health_data',
                channel: 'health'
            });
        }, 5000); // Update every 5 seconds

        return () => {
            console.debug('[Health] Cleaning up periodic updates');
            clearInterval(updateInterval);
        };
    }, [isConnected, send]);

    // Manual refresh handler
    const handleRefresh = useCallback(() => {
        if (isConnected) {
            send({
                type: 'get_health_data',
                channel: 'health'
            });
        }
    }, [isConnected, send]);

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
                        <span>
                            <IconButton
                                onClick={handleRefresh}
                                disabled={loading || !isConnected}
                                sx={{ opacity: loading ? 0.7 : 1 }}
                            >
                                {loading ? <CircularProgress size={24} /> : <RefreshIcon />}
                            </IconButton>
                        </span>
                    </Tooltip>
                </span>
            </Box>

            {(error || wsError) && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error || (wsError && 'Unable to establish WebSocket connection. Some real-time updates may be unavailable.')}
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
                                                    {healthState.wsStatus.internal.active_connections?.map((conn: any) => (
                                                        <Box key={conn.id} sx={{ mt: 2, textAlign: 'left' }}>
                                                            <Typography variant="caption" display="block">
                                                                State: <Chip
                                                                    size="small"
                                                                    label={conn.state}
                                                                    color={conn.state === 'connected' ? 'success' :
                                                                        conn.state === 'error' ? 'error' : 'warning'}
                                                                />
                                                            </Typography>
                                                            <Typography variant="caption" display="block">
                                                                Connected since: {formatDate(conn.connected_since)}
                                                            </Typography>
                                                            <Typography variant="caption" display="block">
                                                                Last activity: {formatDate(conn.last_activity)}
                                                            </Typography>
                                                            {conn.last_error && (
                                                                <Typography variant="caption" display="block" color="error">
                                                                    Last error: {conn.last_error}
                                                                </Typography>
                                                            )}
                                                        </Box>
                                                    ))}
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
                                                    {healthState.wsStatus.internal.active_connections?.map((conn: any) => (
                                                        <Box key={conn.id} sx={{ mt: 2, textAlign: 'left' }}>
                                                            {conn.last_message_type && (
                                                                <Typography variant="caption" display="block">
                                                                    Last message type: {conn.last_message_type}
                                                                </Typography>
                                                            )}
                                                        </Box>
                                                    ))}
                                                </Paper>
                                            </Grid>
                                            {/* Connection History */}
                                            <Grid item xs={12}>
                                                <Paper sx={{ p: 2 }}>
                                                    <Typography variant="subtitle2" gutterBottom>
                                                        Recent Connection Events
                                                    </Typography>
                                                    <Box sx={{ maxHeight: '200px', overflow: 'auto' }}>
                                                        {healthState.wsStatus.internal.connection_history?.map((event: any, index: number) => (
                                                            <Box key={index} sx={{
                                                                py: 1,
                                                                borderBottom: '1px solid',
                                                                borderColor: 'divider'
                                                            }}>
                                                                <Typography variant="caption" display="block" color="text.secondary">
                                                                    {formatDate(event.timestamp)}
                                                                </Typography>
                                                                <Typography variant="body2">
                                                                    <Chip
                                                                        size="small"
                                                                        label={event.event}
                                                                        color={event.event === 'connected' ? 'success' :
                                                                            event.event === 'error' ? 'error' :
                                                                                event.event === 'disconnected' ? 'warning' :
                                                                                    'default'}
                                                                        sx={{ mr: 1 }}
                                                                    />
                                                                    {event.details}
                                                                </Typography>
                                                            </Box>
                                                        ))}
                                                    </Box>
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

            {lastUpdate && (
                <Typography variant="body2" color="text.secondary" mt={1}>
                    Last Update: {lastUpdate.toLocaleTimeString()}
                </Typography>
            )}

            {healthState && (
                <Box display="flex" justifyContent="space-between" alignItems="center" mt={3}>
                    <Typography variant="h6">System Status</Typography>
                    <div className="space-x-2">
                        <Typography variant="body2">Status: <span className="font-medium">{healthState.wsStatus?.ui.connected ? 'Connected' : 'Disconnected'}</span></Typography>
                        <Typography variant="body2">Active Connections: <span className="font-medium">{healthState.wsStatus?.ui.connectionCount || 0}</span></Typography>
                        <Typography variant="body2">Message Count: <span className="font-medium">{healthState.wsStatus?.ui.messageCount || 0}</span></Typography>
                    </div>
                </Box>
            )}
        </Box>
    );
});

export default Health; 