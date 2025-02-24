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
    ArrowRightAlt as ArrowIcon,
} from '@mui/icons-material';
import { useState, useEffect } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';

export interface ConnectionEvent {
    type: 'request' | 'response';
    direction: 'browser-proxy' | 'proxy-web' | 'web-proxy' | 'proxy-browser';
    timestamp: number;
    status: 'pending' | 'success' | 'error';
    bytesTransferred?: number;
}

export interface Connection {
    id: string;
    host: string;
    port: number;
    start_time: number;
    end_time?: number;
    status: 'active' | 'closed' | 'error';
    events: ConnectionEvent[];
    bytes_received: number;
    bytes_sent: number;
    requests_processed: number;
    error?: string;
}

const ConnectionFlow: React.FC<{ event: ConnectionEvent }> = ({ event }) => {
    const theme = useTheme();

    const getColor = (status: string) => {
        switch (status) {
            case 'success':
                return theme.palette.success.main;
            case 'error':
                return theme.palette.error.main;
            default:
                return theme.palette.warning.main;
        }
    };

    const getFlowComponents = () => {
        const color = getColor(event.status);

        switch (event.direction) {
            case 'browser-proxy':
                return (
                    <>
                        <BrowserIcon sx={{ color: color }} />
                        <ArrowIcon sx={{ color: color }} />
                        <ProxyIcon sx={{ color: color }} />
                    </>
                );
            case 'proxy-web':
                return (
                    <>
                        <ProxyIcon sx={{ color: color }} />
                        <ArrowIcon sx={{ color: color }} />
                        <WebIcon sx={{ color: color }} />
                    </>
                );
            case 'web-proxy':
                return (
                    <>
                        <WebIcon sx={{ color: color }} />
                        <ArrowIcon sx={{ color: color }} />
                        <ProxyIcon sx={{ color: color }} />
                    </>
                );
            case 'proxy-browser':
                return (
                    <>
                        <ProxyIcon sx={{ color: color }} />
                        <ArrowIcon sx={{ color: color }} />
                        <BrowserIcon sx={{ color: color }} />
                    </>
                );
            default:
                return null;
        }
    };

    return (
        <Box sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            py: 0.5,
        }}>
            {getFlowComponents()}
            <Typography variant="caption" color="textSecondary">
                {new Date(event.timestamp).toLocaleTimeString()}
                {event.bytesTransferred && ` (${event.bytesTransferred} bytes)`}
            </Typography>
        </Box>
    );
};

const ConnectionItem: React.FC<{ connection: Connection }> = ({ connection }) => {
    const [expanded, setExpanded] = useState(false);
    const theme = useTheme();

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'active':
                return theme.palette.success.main;
            case 'closed':
                return theme.palette.info.main;
            case 'error':
                return theme.palette.error.main;
            default:
                return theme.palette.text.primary;
        }
    };

    return (
        <Card sx={{ mb: 1 }}>
            <ListItem
                button
                onClick={() => setExpanded(!expanded)}
                sx={{
                    borderLeft: 3,
                    borderColor: getStatusColor(connection.status),
                }}
            >
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                    <Typography>
                        {connection.host}:{connection.port}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Typography variant="caption" color="textSecondary">
                            {connection.requests_processed} requests •
                            {Math.round(connection.bytes_received / 1024)}KB received •
                            {Math.round(connection.bytes_sent / 1024)}KB sent
                        </Typography>
                        {expanded ? <ExpandLess /> : <ExpandMore />}
                    </Box>
                </Box>
            </ListItem>
            <Collapse in={expanded}>
                <CardContent>
                    <List sx={{ pl: 2 }}>
                        {connection.events.map((event, index) => (
                            <ListItem key={index} sx={{ py: 0 }}>
                                <ConnectionFlow event={event} />
                            </ListItem>
                        ))}
                    </List>
                </CardContent>
            </Collapse>
        </Card>
    );
};

const ConnectionMonitor: React.FC = () => {
    const [connections, setConnections] = useState<Connection[]>([]);
    const wsUrl = `${process.env.REACT_APP_API_BASE_URL?.replace('http', 'ws') || 'ws://localhost:8000'}/api/proxy/ws`;
    const { isConnected, error } = useWebSocket(wsUrl, {
        onMessage: (data) => {
            if (data.type === 'connection_update') {
                setConnections(prev => {
                    const updated = [...prev];
                    const index = updated.findIndex(c => c.id === data.data.id);
                    if (index >= 0) {
                        updated[index] = data.data;
                    } else {
                        updated.push(data.data);
                    }
                    return updated;
                });
            } else if (data.type === 'connection_closed') {
                setConnections(prev => prev.filter(c => c.id !== data.data.id));
            }
        }
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
        <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
                Active Connections ({connections.length})
            </Typography>
            <List>
                {connections.map((connection) => (
                    <ConnectionItem
                        key={connection.id}
                        connection={connection}
                    />
                ))}
            </List>
        </Paper>
    );
};

export default ConnectionMonitor;
