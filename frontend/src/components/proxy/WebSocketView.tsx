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
} from '@mui/material';
import { Send as SendIcon, Delete as DeleteIcon, Security as SecurityIcon } from '@mui/icons-material';
import { API_BASE_URL } from '../../config';
import axios from 'axios';

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

    const fetchConnections = useCallback(async () => {
        try {
            const response = await axios.get<WebSocketConnection[]>(`${API_BASE_URL}/api/proxy/websocket/connections`);
            setConnections(response.data);
        } catch (err) {
            console.error('Failed to fetch WebSocket connections:', err);
        }
    }, []);

    const fetchMessages = useCallback(async (connectionId: string) => {
        try {
            const [messagesResponse, reportResponse] = await Promise.all([
                axios.get<WebSocketMessage[]>(`${API_BASE_URL}/api/proxy/websocket/messages/${connectionId}`),
                axios.get<SecurityReport>(`${API_BASE_URL}/api/proxy/websocket/security/report/${connectionId}`)
            ]);
            setMessages(messagesResponse.data);
            setSecurityReport(reportResponse.data);
        } catch (err) {
            console.error('Failed to fetch WebSocket data:', err);
        }
    }, []);

    const sendMessage = async () => {
        if (!selectedConnection || !messageInput.trim()) return;

        try {
            await axios.post(`${API_BASE_URL}/api/proxy/websocket/send`, {
                connectionId: selectedConnection,
                message: messageInput
            });
            setMessageInput('');
            fetchMessages(selectedConnection);
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

            await axios.post(`${API_BASE_URL}/api/proxy/websocket/config/${connectionId}`, config);
            fetchConnections();
        } catch (err) {
            console.error('Failed to toggle feature:', err);
        }
    };

    const closeConnection = async (connectionId: string) => {
        try {
            await axios.post(`${API_BASE_URL}/api/proxy/websocket/close/${connectionId}`);
            fetchConnections();
            if (selectedConnection === connectionId) {
                setSelectedConnection(null);
                setMessages([]);
                setSecurityReport(null);
            }
        } catch (err) {
            console.error('Failed to close connection:', err);
        }
    };

    useEffect(() => {
        fetchConnections();
        const intervalId = setInterval(fetchConnections, 5000);
        return () => clearInterval(intervalId);
    }, [fetchConnections]);

    useEffect(() => {
        if (selectedConnection) {
            fetchMessages(selectedConnection);
            const intervalId = setInterval(() => fetchMessages(selectedConnection), 1000);
            return () => clearInterval(intervalId);
        }
    }, [selectedConnection, fetchMessages]);

    return (
        <Grid container spacing={2}>
            {/* Connections List */}
            <Grid item xs={12} md={4}>
                <Paper sx={{ p: 2, height: '70vh', overflow: 'auto' }}>
                    <Typography variant="h6" gutterBottom>
                        WebSocket Connections
                    </Typography>
                    <List>
                        {connections.map((conn) => (
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
