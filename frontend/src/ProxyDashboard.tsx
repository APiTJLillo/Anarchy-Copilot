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
  Checkbox
} from '@mui/material';
import { InterceptorView } from './components/proxy/InterceptorView';
import { WebSocketView } from './components/proxy/WebSocketView';
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
      const data = await proxyApi.getHistory();
      setHistory(data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch proxy history');
      console.error(err);
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
    const init = async () => {
      setLoading(true);
      await fetchStatus();
      await fetchHistory();
      await fetchAnalysisResults();
      setLoading(false);
    };
    init();

    // Poll for updates
    const intervalId = setInterval(() => {
      fetchStatus();
      fetchHistory();
      fetchAnalysisResults();
    }, 5000);

    return () => clearInterval(intervalId);
  }, [fetchStatus, fetchAnalysisResults]);

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
        {status?.isRunning && (
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
                        <ul>
                          {history.map((entry, index) => (
                            <li key={index}>
                              <strong>{entry.method}</strong> {entry.url}
                              <br />
                              <em>Status:</em> {entry.response_status}
                              <br />
                              <em>Request Headers:</em> {JSON.stringify(entry.request_headers)}
                              <br />
                              <em>Response Headers:</em> {JSON.stringify(entry.response_headers)}
                              <br />
                              <em>Request Body:</em> {entry.request_body}
                              <br />
                              <em>Response Body:</em> {entry.response_body}
                            </li>
                          ))}
                        </ul>
                      )}
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
        )}
      </Grid>
    </Box>
  );
};
