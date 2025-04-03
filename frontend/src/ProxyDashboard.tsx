import React, { useEffect, useState, useCallback } from 'react';
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
import { InterceptorView } from './components/proxy/InterceptorView';
import ConnectionMonitor from './components/proxy/ConnectionMonitor';
import WebSocketView from './components/proxy/WebSocketView';
import { RequestModal } from './components/proxy/RequestModal';
import { ProxyProvider } from './components/proxy/ProxyContext';
import { UserProvider, useUser } from './contexts/UserContext';
import AnalysisResults from './components/proxy/AnalysisResults';
import InterceptionRuleManager from './components/proxy/InterceptionRuleManager';
import { Version } from './components/proxy/Version';
import { useProxyApi } from './api/proxyApi';
import { useWebSocket } from './contexts/WebSocketContext';
import { reconstructUrl } from './utils/urlUtils';
import type { ProxySession, ProxySettings, VersionInfo } from './api/proxyApi';
import type { AnalysisResult } from './api/proxyApi';
import ErrorBoundary from './components/proxy/ErrorBoundary';

interface ProxyStatus {
  isRunning: boolean;
  interceptRequests: boolean;
  interceptResponses: boolean;
  allowedHosts: string[];
  excludedHosts: string[];
  history: any[];
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

// Base TabPanel component
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

// Custom tab panel component
interface CustomTabPanelProps {
  children: React.ReactNode;
  value: number;
  index: number;
}

const CustomTabPanel: React.FC<CustomTabPanelProps> = ({ children, value, index }) => (
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

const ProxyDashboardContent: React.FC = () => {
  const {
    currentUser,
    projects,
    loadingUsers,
    loadingProjects,
    error: userError,
    wsConnected,
    setWsConnected,
    isInitialized: userInitialized
  } = useUser();
  const proxyApi = useProxyApi();
  const { isConnected, error: wsError, send, subscribe } = useWebSocket();

  const selectedProject = projects.length ? projects[0].id : '';
  const [selectedRequest, setSelectedRequest] = useState<any>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [currentRequestIndex, setCurrentRequestIndex] = useState<number>(0);
  const [status, setStatus] = useState<ProxyStatus | null>(null);
  const [session, setSession] = useState<ProxySession | null>(null);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [initialized, setInitialized] = useState(false);
  const [dataLoaded, setDataLoaded] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  const [localProxyEnabled, setLocalProxyEnabled] = useState(false);
  const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

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
      setLoading(false);
      return data;
    } catch (err) {
      setError('Failed to fetch proxy status');
      console.error(err);
      setLoading(false);
      return null;
    }
  }, [proxyApi]);

  const handleLocalProxyToggle = (event: React.ChangeEvent<HTMLInputElement>) => {
    setLocalProxyEnabled(event.target.checked);
    if (status?.isRunning) {
      if (event.target.checked) {
        console.log("Local proxy client started");
      } else {
        console.log("Local proxy client stopped");
      }
    }
  };

  const startProxy = useCallback(async () => {
    try {
      if (!currentUser || !selectedProject) {
        setError('Please select a user and project');
        return;
      }

      // Create a new session
      const settings: ProxySettings = {
        host: "127.0.0.1",
        port: 8083,
        interceptRequests: true,
        interceptResponses: true,
        allowedHosts: [],
        excludedHosts: [],
        maxConnections: 100,
        maxKeepaliveConnections: 20,
        keepaliveTimeout: 30
      };

      const newSession = await proxyApi.createSession("New Session", selectedProject, currentUser.id, settings);
      setSession(newSession);

      // Start the proxy with the new session
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
      await proxyApi.stopProxy();
      await fetchStatus();
      setSession(null);
      if (localProxyEnabled) {
        console.log("Local proxy client stopped");
      }
      setError(null);
    } catch (err) {
      setError('Failed to stop proxy');
      console.error(err);
    }
  }, [fetchStatus, localProxyEnabled, proxyApi]);

  const clearAnalysisResults = useCallback(async () => {
    try {
      await proxyApi.clearAnalysisResults();
      setAnalysisResults([]);
      setError(null);
    } catch (err) {
      setError('Failed to clear analysis results');
      console.error(err);
    }
  }, [proxyApi]);

  type WSMessage =
    | { type: 'proxy_status'; data: ProxyStatus }
    | { type: 'proxy_history'; data: any[] }
    | { type: 'proxy_analysis'; data: AnalysisResult[] }
    | { type: 'version_info'; data: VersionInfo }
    | {
      type: 'initial_data'; data: {
        status: ProxyStatus;
        history: any[];
        analysis: AnalysisResult[];
        version: VersionInfo;
      }
    };

  // Track WebSocket and data loading states
  useEffect(() => {
    if (isConnected && dataLoaded && userInitialized) {
      setInitialized(true);
      setLoading(false);
    }
  }, [isConnected, dataLoaded, userInitialized]);

  // Subscribe to WebSocket messages
  useEffect(() => {
    const unsubscribe = subscribe('proxy', (message: WSMessage) => {
      console.debug('[ProxyDashboard] Received WebSocket message:', message);
      if (message.type === 'initial_data') {
        console.debug('[ProxyDashboard] Processing initial data:', message.data);
        setStatus(message.data.status);
        setHistory(message.data.history);
        setAnalysisResults(message.data.analysis);
        setDataLoaded(true);
      } else if (initialized) {
        // Only process updates after initialization
        switch (message.type) {
          case 'proxy_status':
            setStatus(prevStatus => {
              const statusChanged = JSON.stringify(message.data) !== JSON.stringify(prevStatus);
              return statusChanged ? message.data : prevStatus;
            });
            break;
          case 'proxy_history':
            setHistory(prevHistory => {
              const newEntry = message.data;
              return [newEntry, ...prevHistory];
            });
            break;
          case 'proxy_analysis':
            setAnalysisResults(prevResults => {
              const resultsChanged = JSON.stringify(message.data) !== JSON.stringify(prevResults);
              return resultsChanged ? message.data : prevResults;
            });
            break;
        }
      }
    });

    // Request initial data when WebSocket connects
    if (isConnected) {
      console.debug('[ProxyDashboard] Requesting initial data');
      send({
        type: 'get_initial_data',
        channel: 'proxy'  // Add channel to the message
      });
    }

    return () => unsubscribe();
  }, [subscribe, initialized, isConnected, send]);

  // Show loading state while initializing
  if (!initialized || loadingUsers || loadingProjects) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
        <Typography sx={{ mt: 2 }} color="textSecondary">
          {!userInitialized ? 'Loading user data...' : !isConnected ? 'Connecting to server...' : 'Loading proxy data...'}
        </Typography>
      </Box>
    );
  }

  // Show error state
  if (error || userError || wsError) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error || userError || wsError?.message}
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Grid container spacing={3}>
        {/* Status and Controls */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="h6">
                    Proxy Status: {status?.isRunning ? 'Running' : 'Stopped'}
                  </Typography>
                </Box>
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
                  disabled={!currentUser || !selectedProject}
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
              <Version />
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
                <Tab label="Rules" />
              </Tabs>
            </Box>

            <CustomTabPanel value={tabValue} index={0}>
              {/* Connections Monitor */}
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Box sx={{ mb: 3, minHeight: '200px' }}>
                    <ConnectionMonitor />
                  </Box>
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              {/* Existing Proxy History and InterceptorView */}
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Paper sx={{ p: 2 }}>
                    {history.length === 0 ? (
                      <Typography variant="body1" align="center">
                        No proxy history available
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
                                  {reconstructUrl(entry)}
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
            </CustomTabPanel>

            <CustomTabPanel value={tabValue} index={1}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <WebSocketView />
                </Grid>
              </Grid>
            </CustomTabPanel>

            <CustomTabPanel value={tabValue} index={2}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <ErrorBoundary component="Analysis Results">
                    <AnalysisResults results={analysisResults || []} />
                  </ErrorBoundary>
                </Grid>
              </Grid>
            </CustomTabPanel>

            <CustomTabPanel value={tabValue} index={3}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  {session && (
                    <ErrorBoundary component="Rule Manager">
                      <InterceptionRuleManager sessionId={session.id} />
                    </ErrorBoundary>
                  )}
                  {!session && (
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="body1" align="center">
                        Start a proxy session to manage interception rules.
                      </Typography>
                    </Paper>
                  )}
                </Grid>
              </Grid>
            </CustomTabPanel>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export const ProxyDashboard: React.FC = () => {
  return (
    <UserProvider>
      <ProxyDashboardContent />
    </UserProvider>
  );
};
