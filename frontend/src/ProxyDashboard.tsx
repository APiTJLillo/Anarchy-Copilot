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
} from '@mui/material';
import { InterceptorView } from './components/proxy/InterceptorView';
import { WebSocketView } from './components/proxy/WebSocketView';
import { ProxyProvider } from './components/proxy/ProxyContext';
import AnalysisResults from './components/proxy/AnalysisResults';
import axios from 'axios';
import { API_BASE_URL } from './config';

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
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const fetchStatus = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/proxy/status`);
      setStatus(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch proxy status');
      console.error(err);
    }
  }, []);

  const fetchAnalysisResults = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/proxy/analysis/results`);
      setAnalysisResults(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch analysis results');
      console.error(err);
    }
  }, []);

  const startProxy = useCallback(async () => {
    try {
      await axios.post(`${API_BASE_URL}/api/proxy/start`, {
        host: "127.0.0.1",
        port: 8080,
        interceptRequests: true,
        interceptResponses: true,
        allowedHosts: [],
        excludedHosts: []
      });
      await fetchStatus();
      setError(null);
    } catch (err) {
      setError('Failed to start proxy');
      console.error(err);
    }
  }, [fetchStatus]);

  const stopProxy = useCallback(async () => {
    try {
      await axios.post(`${API_BASE_URL}/api/proxy/stop`);
      await fetchStatus();
      setError(null);
    } catch (err) {
      setError('Failed to stop proxy');
      console.error(err);
    }
  }, [fetchStatus]);

  const clearAnalysisResults = useCallback(async () => {
    try {
      await axios.delete(`${API_BASE_URL}/api/proxy/analysis/results`);
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
      await fetchAnalysisResults();
      setLoading(false);
    };
    init();

    // Poll for updates
    const intervalId = setInterval(() => {
      fetchStatus();
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
                  <Grid item xs={12}>
                    <ProxyProvider>
                      <Paper sx={{ p: 2 }}>
                        <Typography variant="body1" align="center">
                          Waiting for requests to intercept...
                          The interceptor will open automatically when requests are captured.
                        </Typography>
                      </Paper>
                    </ProxyProvider>
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
