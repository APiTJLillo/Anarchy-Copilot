import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  Chip,
  IconButton,
  Switch,
  FormControlLabel,
  Grid,
  Tabs,
  Tab,
  Button,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import EditIcon from '@mui/icons-material/Edit';
import InfoIcon from '@mui/icons-material/Info';

interface ProxyRequest {
  id: string;
  timestamp: string;
  method: string;
  url: string;
  status?: number;
  duration?: number;
  tags: string[];
}

interface ProxyState {
  isRunning: boolean;
  interceptRequests: boolean;
  interceptResponses: boolean;
  history: ProxyRequest[];
}

export const ProxyDashboard: React.FC = () => {
  const [proxyState, setProxyState] = useState<ProxyState>({
    isRunning: false,
    interceptRequests: true,
    interceptResponses: true,
    history: [],
  });

  const [selectedTab, setSelectedTab] = useState(0);

  // Simulate fetching proxy status periodically
  useEffect(() => {
    const interval = setInterval(() => {
      // In real implementation, fetch from API
      fetch('/api/proxy/status')
        .then(res => res.json())
        .then(data => setProxyState(data))
        .catch(err => console.error('Failed to fetch proxy status:', err));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleToggleProxy = () => {
    const newState = !proxyState.isRunning;
    // In real implementation, call API to start/stop proxy
    fetch('/api/proxy/' + (newState ? 'start' : 'stop'), { method: 'POST' })
      .then(() => setProxyState(prev => ({ ...prev, isRunning: newState })))
      .catch(err => console.error('Failed to toggle proxy:', err));
  };

  const handleInterceptToggle = (type: 'requests' | 'responses') => {
    const updateField = type === 'requests' ? 'interceptRequests' : 'interceptResponses';
    // In real implementation, call API to update interception settings
    setProxyState(prev => ({ ...prev, [updateField]: !prev[updateField] }));
  };

  const handleEditRequest = (requestId: string) => {
    // In real implementation, open request editor dialog
    console.log('Edit request:', requestId);
  };

  const handleViewDetails = (requestId: string) => {
    // In real implementation, open request details dialog
    console.log('View details:', requestId);
  };

  return (
    <Box p={3}>
      <Paper elevation={2}>
        <Box p={2}>
          <Grid container spacing={2} alignItems="center">
            <Grid item>
              <Typography variant="h6">Proxy Status</Typography>
            </Grid>
            <Grid item>
              <Button
                variant="contained"
                color={proxyState.isRunning ? "error" : "success"}
                startIcon={proxyState.isRunning ? <StopIcon /> : <PlayArrowIcon />}
                onClick={handleToggleProxy}
              >
                {proxyState.isRunning ? "Stop Proxy" : "Start Proxy"}
              </Button>
            </Grid>
            <Grid item>
              <FormControlLabel
                control={
                  <Switch
                    checked={proxyState.interceptRequests}
                    onChange={() => handleInterceptToggle('requests')}
                  />
                }
                label="Intercept Requests"
              />
            </Grid>
            <Grid item>
              <FormControlLabel
                control={
                  <Switch
                    checked={proxyState.interceptResponses}
                    onChange={() => handleInterceptToggle('responses')}
                  />
                }
                label="Intercept Responses"
              />
            </Grid>
          </Grid>
        </Box>
      </Paper>

      <Box mt={3}>
        <Paper elevation={2}>
          <Tabs value={selectedTab} onChange={(_, val) => setSelectedTab(val)}>
            <Tab label="History" />
            <Tab label="Intercepted" />
            <Tab label="Settings" />
          </Tabs>

          <Box p={2}>
            {selectedTab === 0 && (
              <List>
                {proxyState.history.map((request) => (
                  <ListItem
                    key={request.id}
                    secondaryAction={
                      <Box>
                        <IconButton
                          edge="end"
                          aria-label="edit"
                          onClick={() => handleEditRequest(request.id)}
                        >
                          <EditIcon />
                        </IconButton>
                        <IconButton
                          edge="end"
                          aria-label="details"
                          onClick={() => handleViewDetails(request.id)}
                        >
                          <InfoIcon />
                        </IconButton>
                      </Box>
                    }
                  >
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center" gap={1}>
                          <Chip
                            label={request.method}
                            color={request.method === 'GET' ? 'primary' : 'secondary'}
                            size="small"
                          />
                          <Typography>{request.url}</Typography>
                        </Box>
                      }
                      secondary={
                        <Box display="flex" alignItems="center" gap={1}>
                          {request.status && (
                            <Chip
                              label={request.status}
                              color={request.status < 400 ? 'success' : 'error'}
                              size="small"
                            />
                          )}
                          {request.duration && (
                            <Typography variant="caption">
                              {request.duration}ms
                            </Typography>
                          )}
                          {request.tags.map((tag) => (
                            <Chip
                              key={tag}
                              label={tag}
                              variant="outlined"
                              size="small"
                            />
                          ))}
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            )}

            {selectedTab === 1 && (
              <Typography variant="body1">No requests currently intercepted</Typography>
            )}

            {selectedTab === 2 && (
              <Box p={2}>
                <Typography variant="h6">Proxy Settings</Typography>
                {/* Add proxy settings form here */}
              </Box>
            )}
          </Box>
        </Paper>
      </Box>
    </Box>
  );
};

export default ProxyDashboard;
