import React, { useState, useEffect } from 'react';
import { Box, Button, Card, CardContent, Divider, Grid, Tab, Tabs, Typography } from '@mui/material';
import { useProxyApi } from '../../contexts/ProxyApiContext';
import { FuzzingConfigPanel } from './FuzzingConfigPanel';
import ParameterDetectionPanel from './ParameterDetectionPanel';

const WebSocketView = ({ connectionId, messages, onSendMessage }) => {
  const [activeTab, setActiveTab] = useState(0);
  const [fuzzedMessages, setFuzzedMessages] = useState([]);
  const { api } = useProxyApi();

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleFuzz = (newFuzzedMessages) => {
    setFuzzedMessages(prev => [...prev, ...newFuzzedMessages]);
    setActiveTab(2); // Switch to the Fuzzing Results tab
  };

  const handleParametersFuzzed = (newFuzzedMessages) => {
    setFuzzedMessages(prev => [...prev, ...newFuzzedMessages]);
    setActiveTab(2); // Switch to the Fuzzing Results tab
  };

  const handleSendFuzzedMessage = (message) => {
    if (onSendMessage) {
      onSendMessage(message);
    }
  };

  return (
    <Card variant="outlined">
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={activeTab} onChange={handleTabChange} aria-label="websocket tabs">
          <Tab label="Messages" />
          <Tab label="Fuzzing" />
          <Tab label="Auto-Detect Parameters" />
          <Tab label="Fuzzing Results" />
        </Tabs>
      </Box>
      <CardContent>
        {activeTab === 0 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              WebSocket Messages
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Grid container spacing={2}>
              {messages.map((msg, index) => (
                <Grid item xs={12} key={index}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography color="textSecondary" gutterBottom>
                        {msg.direction} - {new Date(msg.timestamp).toLocaleTimeString()}
                      </Typography>
                      <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                        {typeof msg.data === 'object' ? JSON.stringify(msg.data, null, 2) : msg.data}
                      </Typography>
                      <Box sx={{ mt: 1 }}>
                        <Button 
                          size="small" 
                          variant="outlined" 
                          onClick={() => handleSendFuzzedMessage(msg)}
                        >
                          Resend
                        </Button>
                        <Button 
                          size="small" 
                          variant="outlined" 
                          sx={{ ml: 1 }}
                          onClick={() => {
                            // Open edit dialog
                          }}
                        >
                          Edit & Resend
                        </Button>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        )}
        
        {activeTab === 1 && (
          <FuzzingConfigPanel 
            connectionId={connectionId} 
            onFuzz={handleFuzz}
          />
        )}
        
        {activeTab === 2 && (
          <ParameterDetectionPanel 
            connectionId={connectionId}
            onParametersFuzzed={handleParametersFuzzed}
          />
        )}
        
        {activeTab === 3 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Fuzzing Results ({fuzzedMessages.length} messages)
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Grid container spacing={2}>
              {fuzzedMessages.map((msg, index) => (
                <Grid item xs={12} key={index}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography color="textSecondary" gutterBottom>
                        {msg.direction} - {new Date(msg.timestamp).toLocaleTimeString()}
                        {msg.metadata && msg.metadata.fuzz_type && (
                          <span> - Fuzz Type: {msg.metadata.fuzz_type}</span>
                        )}
                        {msg.metadata && msg.metadata.param_name && (
                          <span> - Parameter: {msg.metadata.param_name}</span>
                        )}
                      </Typography>
                      <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                        {typeof msg.data === 'object' ? JSON.stringify(msg.data, null, 2) : msg.data}
                      </Typography>
                      <Box sx={{ mt: 1 }}>
                        <Button 
                          size="small" 
                          variant="contained" 
                          color="primary"
                          onClick={() => handleSendFuzzedMessage(msg)}
                        >
                          Send
                        </Button>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default WebSocketView;
