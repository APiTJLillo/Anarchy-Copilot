import React, { useEffect, useState, useCallback } from 'react';
import { Avatar, Menu, MenuItem } from '@mui/material';
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
  FormControlLabel,
  Checkbox,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import { InterceptorView } from './components/proxy/InterceptorView';
import { WebSocketView } from './components/proxy/WebSocketView';
import { RequestModal } from './components/proxy/RequestModal';
import { ProxyProvider } from './components/proxy/ProxyContext';
import AnalysisResults from './components/proxy/AnalysisResults';
import { proxyApi, ProxySession, ProxySettings } from './api/proxyApi';

interface ProxyStatus {
  isRunning: boolean;
  interceptRequests: boolean;
  interceptResponses: boolean;
  allowedHosts: string[];
  excludedHosts: string[];
  history: any[];
}

interface AnalysisResult {
  requestId: string;
  ruleName: string;
  severity: string;
  description: string;
  evidence: string;
  timestamp: string;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = (props) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`proxy-tabpanel-${index}`}
      aria-labelledby={`proxy-tab-${index}`}
      {...other}
    >
      <div style={{ padding: value === index ? 24 : 0, display: value === index ? 'block' : 'none' }}>
        {children}
      </div>
    </div>
  );
};

export const ProxyDashboard: React.FC = () => {
  const [selectedRequest, setSelectedRequest] = useState<any>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [currentRequestIndex, setCurrentRequestIndex] = useState<number>(0);
  const [status, setStatus] = useState<ProxyStatus | null>(null);
  const [session, setSession] = useState<ProxySession | null>(null);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tabValue, setTabValue] = useState(0);
  const [localProxyEnabled, setLocalProxyEnabled] = useState(false);
  const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [currentUser, setCurrentUser] = useState<string>('User1');

  const handleUserMenuClose = () => {
    setAnchorEl(null);
  };

  const handleUserSwitch = (user: string) => {
    setCurrentUser(user);
    handleUserMenuClose();
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleRequestNavigate = (index: number) => {
    setCurrentRequestIndex(index);
    setSelectedRequest(history[index]);
  };

  const fetchStatus = useCallback(async () => {
    try {
      const data = await proxyApi.getStatus();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch proxy status');
      console.error(err);
    }
  }, []);

  const fetchHistory = useCallback(async () => {
    try {
      console.log('Fetching proxy history...');
      const data = await proxyApi.getHistory();
      console.log('Received history data:', data);
      setHistory(data);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch proxy history:', err);
      setError('Failed to fetch proxy history');
    }
  }, []);

  const fetchAnalysisResults = useCallback(async () => {
    try {
      const data = await proxyApi.getAnalysisResults();
      setAnalysisResults(data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch analysis results');
      console.error(err);
    }
  }, []);

  const handleLocalProxyToggle = (event: React.ChangeEvent<HTMLInputElement>) => {
    setLocalProxyEnabled(event.target.checked);
    // Only toggle local proxy if Docker proxy is running
    if (status?.isRunning) {
      if (event.target.checked) {
        // Logic to start local proxy client
        console.log("Local proxy client started");
      } else {
        // Logic to stop local proxy client
        console.log("Local proxy client stopped");
      }
    }
  };

  const startProxy = useCallback(async () => {
    try {
      // Create a new session
      const newSession = await proxyApi.createSession("New Session", 1, 1, {
        host: "127.0.0.1",
        port: 8080,
        interceptRequests: true,
        interceptResponses: true,
        allowedHosts: [],
        excludedHosts: [],
        maxConnections: 100,
        maxKeepaliveConnections: 20,
        keepaliveTimeout: 30
      });
      setSession(newSession);

      // Start the proxy with the new session
      const settings: ProxySettings = {
        host: "127.0.0.1",
        port: 8080,
        interceptRequests: true,
        interceptResponses: true,
        allowedHosts: [],
        excludedHosts: [],
        maxConnections: 100,
        maxKeepaliveConnections: 20,
        keepaliveTimeout: 30
      };

      await proxyApi.startProxy(newSession.id, settings);
      await fetchStatus();
      if (localProxyEnabled) {
        // Start local proxy client if checkbox is checked
        console.log("Local proxy client started");
      }
      setError(null);
    } catch (err) {
      setError('Failed to start proxy');
      console.error(err);
    }
  }, [fetchStatus]);

  const stopProxy = useCallback(async () => {
    try {
      await proxyApi.stopProxy();
      await fetchStatus();
      setSession(null);
      // Stop local proxy client if it was running
      if (localProxyEnabled) {
        console.log("Local proxy client stopped");
      }
      setError(null);
    } catch (err) {
      setError('Failed to stop proxy');
      console.error(err);
    }
  }, [fetchStatus]);

  const clearAnalysisResults = useCallback(async () => {
    try {
      await proxyApi.clearAnalysisResults();
      setAnalysisResults([]);
      setError(null);
    } catch (err) {
      setError('Failed to clear analysis results');
      console.error(err);
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    let pollId: NodeJS.Timeout | null = null;

    const pollData = async () => {
      if (!mounted) return;
      console.log('Polling for updates...');

      try {
        await Promise.all([
          fetchStatus(),
          fetchHistory(),
          fetchAnalysisResults()
        ]);
      } catch (error) {
        console.error('Error during polling:', error);
      }

      if (mounted) {
        // Schedule next poll only if still mounted
        pollId = setTimeout(pollData, 5000);
      }
    };

    // Initial load
    const init = async () => {
      if (!mounted) return;
      setLoading(true);

      try {
        await Promise.all([
          fetchStatus(),
          fetchHistory(),
          fetchAnalysisResults()
        ]);
      } catch (error) {
        console.error('Error during initial load:', error);
      }

      if (mounted) {
        setLoading(false);
        // Start polling after initial load
        pollId = setTimeout(pollData, 5000);
      }
    };

    init();

    return () => {
      mounted = false;
      if (pollId) {
        clearTimeout(pollId);
      }
      console.log('Cleanup: polling stopped');
    };
  }, [fetchStatus, fetchHistory, fetchAnalysisResults]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Status and Controls */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="h6">
                Proxy Status: {status?.isRunning ? 'Running' : 'Stopped'}
              </Typography>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={localProxyEnabled}
                    disabled={isMobile}
                    onChange={handleLocalProxyToggle}
                    name="localProxy"
                    color="primary"
                  />
                }
                label="Enable Local Proxy"
              />
              <Button
                variant="contained"
                color={status?.isRunning ? 'error' : 'primary'}
                onClick={status?.isRunning ? stopProxy : startProxy}
              >
                {status?.isRunning ? 'Stop Proxy' : 'Start Proxy'}
              </Button>
              <Button
                variant="outlined"
                onClick={clearAnalysisResults}
                disabled={!status?.isRunning}
              >
                Clear Analysis Results
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Main Content */}
        <Grid item xs={12}>
          <Paper>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={tabValue} onChange={handleTabChange} aria-label="proxy tabs">
                <Tab label="HTTP/HTTPS" />
                <Tab label="WebSocket" />
                <Tab label="Analysis" />
              </Tabs>
            </Box>

            <TabPanel value={tabValue} index={0}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <ProxyProvider>
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="body1" align="center">
                        Waiting for requests to intercept...
                        The interceptor will open automatically when requests are captured.
                      </Typography>
                    </Paper>
                  </ProxyProvider>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" align="center">
                      Proxy History
                    </Typography>
                    {history.length === 0 ? (
                      <Typography variant="body1" align="center">
                        No history available.
                      </Typography>
                    ) : (
                      <TableContainer>
                        <Table size="small">
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
                            {history.map((entry, index) => (
                              <TableRow
                                key={index}
                                hover
                                onClick={() => {
                                  setSelectedRequest(entry);
                                  setModalOpen(true);
                                  setCurrentRequestIndex(index);
                                }}
                                sx={{ cursor: 'pointer' }}
                              >
                                <TableCell>{entry.method}</TableCell>
                                <TableCell sx={{ maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                  {entry.url}
                                </TableCell>
                                <TableCell>{entry.response_status}</TableCell>
                                <TableCell>{entry.duration ? `${(entry.duration * 1000).toFixed(2)}ms` : 'N/A'}</TableCell>
                                <TableCell>{entry.is_intercepted ? 'Yes' : 'No'}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    )}

                    <RequestModal
                      open={modalOpen}
                      onClose={() => setModalOpen(false)}
                      request={selectedRequest}
                      history={history}
                      currentIndex={currentRequestIndex}
                      onNavigate={handleRequestNavigate}
                    />
                  </Paper>
                </Grid>
              </Grid>
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <WebSocketView />
                </Grid>
              </Grid>
            </TabPanel>

            <TabPanel value={tabValue} index={2}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <AnalysisResults results={analysisResults} />
                </Grid>
              </Grid>
            </TabPanel>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};
