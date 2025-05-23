import React from 'react';
import {
    Box,
    Paper,
    Typography,
    List,
    ListItem,
    Collapse,
    IconButton,
    Card,
    CardContent,
    useTheme,
} from '@mui/material';
import {
    ExpandLess,
    ExpandMore,
    Computer as BrowserIcon,
    Storage as ProxyIcon,
    Language as WebIcon,
    ArrowRightAlt,
} from '@mui/icons-material';
import { useState, useCallback } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { WS_ENDPOINT } from '../../config';

// Utility function moved outside component for stability
const sortByLatestActivity = (connections: Connection[]) => {
    return [...connections].sort((a, b) => {
        const aTime = a.events[a.events.length - 1]?.timestamp || 0;
        const bTime = b.events[b.events.length - 1]?.timestamp || 0;
        return bTime - aTime;
    });
};

export interface ConnectionEvent {
    type: 'request' | 'response';
    direction: 'browser-proxy' | 'proxy-web' | 'web-proxy' | 'proxy-browser';
    timestamp: number;
    status: 'pending' | 'success' | 'error';
    bytesTransferred?: number;
}

interface Connection {
    id: string;
    host?: string;
    port?: number;
    status: string;
    events: ConnectionEvent[];
    bytes_sent: number;
    bytes_received: number;
    error?: string;
    tls_info?: {
        version?: string;
        cipher?: string;
        handshake_complete?: boolean;
    };
}

const ConnectionCard: React.FC<{ connection: Connection }> = ({ connection }) => {
    const [expanded, setExpanded] = useState(false);
    const theme = useTheme();

    const formatBytes = (bytes: number) => {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
    };

    return (
        <Card sx={{ mb: 1 }}>
            <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <BrowserIcon color="primary" />
                        <ArrowRightAlt />
                        <ProxyIcon color="secondary" />
                        <ArrowRightAlt />
                        <WebIcon color="primary" />
                        <Typography variant="body1">
                            {connection.host}:{connection.port}
                        </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Typography variant="body2" color="textSecondary">
                            {formatBytes(connection.bytes_sent)} sent
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                            {formatBytes(connection.bytes_received)} received
                        </Typography>
                        <IconButton size="small" onClick={() => setExpanded(!expanded)}>
                            {expanded ? <ExpandLess /> : <ExpandMore />}
                        </IconButton>
                    </Box>
                </Box>

                <Collapse in={expanded}>
                    <Box sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                            Status: {connection.status}
                            {connection.error && (
                                <Typography color="error" variant="body2">
                                    Error: {connection.error}
                                </Typography>
                            )}
                        </Typography>
                        {connection.tls_info && (
                            <Typography variant="body2" color="textSecondary">
                                TLS: {connection.tls_info.version} ({connection.tls_info.cipher})
                            </Typography>
                        )}
                        <List dense>
                            {connection.events.map((event, index) => (
                                <ListItem key={index}>
                                    <Typography variant="body2" sx={{
                                        color: event.status === 'error' ? 'error.main' :
                                            event.status === 'success' ? 'success.main' :
                                                'text.primary'
                                    }}>
                                        {new Date(event.timestamp * 1000).toLocaleTimeString()} - {event.type} ({event.direction})
                                        {event.bytesTransferred && ` - ${formatBytes(event.bytesTransferred)}`}
                                    </Typography>
                                </ListItem>
                            ))}
                        </List>
                    </Box>
                </Collapse>
            </CardContent>
        </Card>
    );
};

const ConnectionMonitor: React.FC = () => {
    const [connections, setConnections] = useState<Connection[]>([]);

    // WebSocket message types
    type ConnectionMonitorMessage =
        | { type: 'connection_update'; data: Connection }
        | { type: 'connection_closed'; data: { id: string } };

    // Memoized message handler
    const handleMessage = useCallback((message: ConnectionMonitorMessage) => {
        switch (message.type) {
            case 'connection_update': {
                setConnections(prev => {
                    // Filter out existing connection and add updated version
                    const updated = prev.filter(c => c.id !== message.data.id);
                    return sortByLatestActivity([...updated, message.data]);
                });
                break;
            }
            case 'connection_closed':
                setConnections(prev => prev.filter(c => c.id !== message.data.id));
                break;
        }
    }, []); // No dependencies needed since sortByLatestActivity is stable

    // WebSocket connection with type-safe messages and stable configuration
    const { isConnected, error } = useWebSocket<ConnectionMonitorMessage>(WS_ENDPOINT, {
        onMessage: handleMessage,
        reconnectAttempts: 5,
        reconnectInterval: 2000,
        debug: true
    });

    if (!isConnected) {
        return (
            <Paper sx={{ p: 2 }}>
                <Typography variant="body1" color="textSecondary" align="center">
                    Connecting to proxy server...
                </Typography>
            </Paper>
        );
    }

    if (error) {
        return (
            <Paper sx={{ p: 2 }}>
                <Typography variant="body1" color="error" align="center">
                    Failed to connect to proxy server
                </Typography>
            </Paper>
        );
    }

    if (connections.length === 0) {
        return (
            <Paper sx={{ p: 2 }}>
                <Typography variant="body1" color="textSecondary" align="center">
                    No active connections
                </Typography>
            </Paper>
        );
    }

    return (
        <Box>
            <Typography variant="h6" gutterBottom>
                Active Connections ({connections.length})
            </Typography>
            {connections.map(connection => (
                <ConnectionCard key={connection.id} connection={connection} />
            ))}
        </Box>
    );
}

export default ConnectionMonitor;
