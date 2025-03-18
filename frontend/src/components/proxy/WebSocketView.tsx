import React, { useState, useEffect, useCallback } from 'react';
import { FuzzingConfigPanel } from './FuzzingConfigPanel';
import { SecurityIssueDetails } from './SecurityIssueDetails';
import {
    Box,
    Paper,
    Typography,
    List,
    ListItem,
    ListItemText,
    TextField,
    Button,
    Grid,
    Divider,
    IconButton,
    Chip,
    ButtonGroup,
    CircularProgress,
} from '@mui/material';
import { Send as SendIcon, Delete as DeleteIcon, Security as SecurityIcon } from '@mui/icons-material';
import axios from 'axios';
import { useWebSocket } from '../../hooks/useWebSocket';
import { WS_ENDPOINT } from '../../config';

interface WebSocketMessage {
    id: string;
    type: 'SEND' | 'RECEIVE';
    payload: string;
    timestamp: string;
    intercepted?: boolean;
    modified?: boolean;
    fuzzed?: boolean;
    securityIssues?: {
        severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
        description: string;
        evidence: string;
        rule_name: string;
    }[];
}

interface WebSocketConnection {
    id: string;
    url: string;
    status: 'ACTIVE' | 'CLOSED';
    timestamp: string;
    interceptorEnabled: boolean;
    fuzzingEnabled: boolean;
    securityAnalysisEnabled: boolean;
    conversationId?: string;
}

interface SecurityReport {
    finding_count: number;
    findings: {
        severity: string;
        description: string;
        evidence: string;
        rule_name: string;
        timestamp: string;
    }[];
    summary: {
        critical: number;
        high: number;
        medium: number;
        low: number;
    };
}

export const WebSocketView: React.FC = () => {
    const [connections, setConnections] = useState<WebSocketConnection[]>([]);
    const [messages, setMessages] = useState<WebSocketMessage[]>([]);
    const [selectedConnection, setSelectedConnection] = useState<string | null>(null);
    const [messageInput, setMessageInput] = useState('');
    const [selectedSecurityIssues, setSelectedSecurityIssues] = useState<WebSocketMessage['securityIssues']>(undefined);
    const [securityReport, setSecurityReport] = useState<SecurityReport | null>(null);
    const [loading, setLoading] = useState(true);

    // Handle real-time WebSocket messages
    interface WSConnectionListMessage {
        type: 'connection_list';
        data: WebSocketConnection[];
    }

    interface WSConnectionMessagesMessage {
        type: 'connection_messages';
        data: {
            connectionId: string;
            messages: WebSocketMessage[];
            securityReport?: SecurityReport;
        };
    }

    type WSMessage = WSConnectionListMessage | WSConnectionMessagesMessage;

    const { isConnected, error, send } = useWebSocket<WSMessage>(WS_ENDPOINT, {
        onMessage: (data: WSMessage) => {
            if (data.type === 'connection_list') {
                setConnections(data.data);
                setLoading(false);
            } else if (data.type === 'connection_messages' && selectedConnection && data.data.connectionId === selectedConnection) {
                setMessages(data.data.messages);
                if (data.data.securityReport) {
                    setSecurityReport(data.data.securityReport);
                }
            }
        }
    });

    const sendMessage = async () => {
        if (!selectedConnection || !messageInput.trim()) return;

        try {
            await axios.post('/api/proxy/websocket/send', {
                connectionId: selectedConnection,
                message: messageInput
            });
            setMessageInput('');
            // Message updates will come through WebSocket
        } catch (err) {
            console.error('Failed to send message:', err);
        }
    };

    const toggleFeature = async (connectionId: string, feature: 'interceptor' | 'fuzzing' | 'securityAnalysis') => {
        try {
            const connection = connections.find(c => c.id === connectionId);
            if (!connection) return;

            const config: Record<string, boolean> = {
                interceptorEnabled: feature === 'interceptor' ? !connection.interceptorEnabled : connection.interceptorEnabled,
                fuzzingEnabled: feature === 'fuzzing' ? !connection.fuzzingEnabled : connection.fuzzingEnabled,
                securityAnalysisEnabled: feature === 'securityAnalysis' ? !connection.securityAnalysisEnabled : connection.securityAnalysisEnabled
            };

            await axios.post(`/api/proxy/websocket/config/${connectionId}`, config);
            // Updates will come through WebSocket
        } catch (err) {
            console.error('Failed to toggle feature:', err);
        }
    };

    const closeConnection = async (connectionId: string) => {
        try {
            await axios.post(`/api/proxy/websocket/close/${connectionId}`);
            if (selectedConnection === connectionId) {
                setSelectedConnection(null);
                setMessages([]);
                setSecurityReport(null);
            }
            // Connection list update will come through WebSocket
        } catch (err) {
            console.error('Failed to close connection:', err);
        }
    };

    // Initial data loading is not needed as it will come through WebSocket

    // Show loading state while WebSocket connects
    if (!isConnected) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '70vh' }}>
                <Paper sx={{ p: 3, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                    <CircularProgress />
                    <Typography color="textSecondary">
                        Connecting to WebSocket server...
                    </Typography>
                </Paper>
            </Box>
        );
    }

    // Show error state
    if (error) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '70vh' }}>
                <Paper sx={{ p: 3, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                    <Typography color="error" variant="h6">
                        Connection Error
                    </Typography>
                    <Typography color="textSecondary">
                        Failed to connect to WebSocket server. Please try again later.
                    </Typography>
                </Paper>
            </Box>
        );
    }

    return (
        <Grid container spacing={2}>
            {/* Connections List */}
            <Grid item xs={12} md={4}>
                <Paper sx={{ p: 2, height: '70vh', overflow: 'auto' }}>
                    <Typography variant="h6" gutterBottom>
                        WebSocket Connections
                    </Typography>
                    {loading ? (
                        <CircularProgress />
                    ) : (
                        <List>
                            {connections.map(conn => (
                                <ListItem
                                    key={conn.id}
                                    onClick={() => setSelectedConnection(conn.id)}
                                    sx={{
                                        cursor: 'pointer',
                                        bgcolor: selectedConnection === conn.id ? 'action.selected' : 'inherit'
                                    }}
                                    secondaryAction={
                                        conn.status === 'ACTIVE' && (
                                            <IconButton edge="end" onClick={() => closeConnection(conn.id)}>
                                                <DeleteIcon />
                                            </IconButton>
                                        )
                                    }
                                >
                                    <ListItemText
                                        primary={conn.url}
                                        secondary={
                                            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
                                                <Chip
                                                    size="small"
                                                    label={conn.status}
                                                    color={conn.status === 'ACTIVE' ? 'success' : 'default'}
                                                />
                                                {conn.securityAnalysisEnabled && (
                                                    <Chip
                                                        size="small"
                                                        label="Security Analysis"
                                                        color="warning"
                                                        icon={<SecurityIcon />}
                                                    />
                                                )}
                                                <Typography variant="caption" display="block">
                                                    {new Date(conn.timestamp).toLocaleString()}
                                                </Typography>
                                            </Box>
                                        }
                                    />
                                </ListItem>
                            ))}
                        </List>
                    )}
                </Paper>
            </Grid>

            {selectedConnection && (
                <>
                    {/* Connection Controls */}
                    <Grid item xs={12}>
                        <Paper sx={{ p: 2, mb: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                WebSocket Controls
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                                <ButtonGroup variant="outlined">
                                    <Button
                                        color={connections.find(c => c.id === selectedConnection)?.interceptorEnabled ? 'success' : 'primary'}
                                        onClick={() => toggleFeature(selectedConnection, 'interceptor')}
                                    >
                                        Interceptor {connections.find(c => c.id === selectedConnection)?.interceptorEnabled ? 'ON' : 'OFF'}
                                    </Button>
                                    <Button
                                        color={connections.find(c => c.id === selectedConnection)?.securityAnalysisEnabled ? 'success' : 'primary'}
                                        onClick={() => toggleFeature(selectedConnection, 'securityAnalysis')}
                                        startIcon={<SecurityIcon />}
                                    >
                                        Security Analysis {connections.find(c => c.id === selectedConnection)?.securityAnalysisEnabled ? 'ON' : 'OFF'}
                                    </Button>
                                </ButtonGroup>
                            </Box>
                        </Paper>
                    </Grid>

                    {/* Messages Panel */}
                    <Grid item xs={12} md={8}>
                        <Paper sx={{ p: 2, height: '70vh', display: 'flex', flexDirection: 'column' }}>
                            {/* Messages List */}
                            <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 2 }}>
                                <List>
                                    {messages.map((msg) => (
                                        <ListItem key={msg.id}>
                                            <ListItemText
                                                primary={
                                                    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
                                                        <Chip
                                                            size="small"
                                                            label={msg.type}
                                                            color={msg.type === 'SEND' ? 'primary' : 'secondary'}
                                                        />
                                                        <Typography>{msg.payload}</Typography>
                                                        {msg.intercepted && (
                                                            <Chip
                                                                size="small"
                                                                label="Intercepted"
                                                                color="warning"
                                                            />
                                                        )}
                                                        {msg.modified && (
                                                            <Chip
                                                                size="small"
                                                                label="Modified"
                                                                color="info"
                                                            />
                                                        )}
                                                        {msg.fuzzed && (
                                                            <Chip
                                                                size="small"
                                                                label="Fuzzed"
                                                                color="secondary"
                                                            />
                                                        )}
                                                        {msg.securityIssues && msg.securityIssues.length > 0 && (
                                                            <Chip
                                                                size="small"
                                                                label={`Security Issues (${msg.securityIssues.length})`}
                                                                color="error"
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    setSelectedSecurityIssues(msg.securityIssues);
                                                                }}
                                                                sx={{ cursor: 'pointer' }}
                                                            />
                                                        )}
                                                    </Box>
                                                }
                                                secondary={new Date(msg.timestamp).toLocaleString()}
                                            />
                                        </ListItem>
                                    ))}
                                </List>
                            </Box>

                            {/* Message Input */}
                            <Divider />
                            <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                                <TextField
                                    fullWidth
                                    variant="outlined"
                                    size="small"
                                    placeholder="Enter message..."
                                    value={messageInput}
                                    onChange={(e) => setMessageInput(e.target.value)}
                                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                                />
                                <Button
                                    variant="contained"
                                    endIcon={<SendIcon />}
                                    onClick={sendMessage}
                                    disabled={!messageInput.trim()}
                                >
                                    Send
                                </Button>
                            </Box>
                        </Paper>
                    </Grid>

                    {/* Security Report */}
                    {securityReport && securityReport.finding_count > 0 && (
                        <Grid item xs={12} md={4}>
                            <Paper sx={{ p: 2, height: '70vh', overflow: 'auto' }}>
                                <Typography variant="h6" gutterBottom>
                                    Security Analysis
                                </Typography>
                                <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                                    <Chip
                                        label={`Critical: ${securityReport.summary.critical}`}
                                        color="error"
                                        size="small"
                                    />
                                    <Chip
                                        label={`High: ${securityReport.summary.high}`}
                                        color="warning"
                                        size="small"
                                    />
                                    <Chip
                                        label={`Medium: ${securityReport.summary.medium}`}
                                        color="info"
                                        size="small"
                                    />
                                    <Chip
                                        label={`Low: ${securityReport.summary.low}`}
                                        color="success"
                                        size="small"
                                    />
                                </Box>
                                <List>
                                    {securityReport.findings.map((finding, index) => (
                                        <ListItem key={index}>
                                            <ListItemText
                                                primary={
                                                    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                                                        <Chip
                                                            size="small"
                                                            label={finding.severity}
                                                            color={
                                                                finding.severity === 'CRITICAL' ? 'error' :
                                                                    finding.severity === 'HIGH' ? 'warning' :
                                                                        finding.severity === 'MEDIUM' ? 'info' : 'success'
                                                            }
                                                        />
                                                        <Typography>{finding.description}</Typography>
                                                    </Box>
                                                }
                                                secondary={
                                                    <>
                                                        <Typography variant="caption" component="div">
                                                            Rule: {finding.rule_name}
                                                        </Typography>
                                                        <Typography variant="caption" component="div">
                                                            Evidence: {finding.evidence}
                                                        </Typography>
                                                    </>
                                                }
                                            />
                                        </ListItem>
                                    ))}
                                </List>
                            </Paper>
                        </Grid>
                    )}
                </>
            )}

            {/* Security Issues Dialog */}
            {selectedSecurityIssues && selectedSecurityIssues.length > 0 && (
                <SecurityIssueDetails
                    open={true}
                    issues={selectedSecurityIssues}
                    onClose={() => setSelectedSecurityIssues(undefined)}
                />
            )}
        </Grid>
    );
};
