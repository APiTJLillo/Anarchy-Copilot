import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Tabs, 
  Tab, 
  Paper, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Tooltip
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import RefreshIcon from '@mui/icons-material/Refresh';
import SecurityIcon from '@mui/icons-material/Security';
import TimelineIcon from '@mui/icons-material/Timeline';
import BugReportIcon from '@mui/icons-material/BugReport';
import SettingsIcon from '@mui/icons-material/Settings';
import { useProxyApi } from '../../contexts/ProxyApiContext';
import { useWebSocket } from '../../hooks/useWebSocket';

// Define severity color mapping
const severityColors = {
  high: 'error',
  medium: 'warning',
  low: 'info'
};

// TabPanel component for tab content
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analysis-tabpanel-${index}`}
      aria-labelledby={`analysis-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

// Traffic Analysis component
export default function TrafficAnalysisView() {
  const [tabValue, setTabValue] = useState(0);
  const [securityIssues, setSecurityIssues] = useState([]);
  const [behaviorPatterns, setBehaviorPatterns] = useState([]);
  const [analysisResults, setAnalysisResults] = useState({});
  const [selectedRequestId, setSelectedRequestId] = useState(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [detailsData, setDetailsData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [rulesDialogOpen, setRulesDialogOpen] = useState(false);
  const [rules, setRules] = useState([]);
  const [newRuleDialogOpen, setNewRuleDialogOpen] = useState(false);
  const [newRule, setNewRule] = useState({
    name: '',
    description: '',
    rule_type: 'security',
    priority: 50,
    conditions: {
      type: 'field',
      field: '',
      operator: 'equals',
      value: ''
    },
    actions: [{
      type: 'create_security_issue',
      severity: 'medium',
      type: 'Custom Rule',
      description: ''
    }]
  });

  const { proxyApi } = useProxyApi();

  // WebSocket for real-time updates
  const { connected, lastMessage } = useWebSocket({
    url: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws/analysis`,
    onMessage: (data) => {
      // Handle real-time analysis updates
      if (data.type === 'security_issue') {
        setSecurityIssues(prev => [...prev, data.issue]);
      } else if (data.type === 'behavior_pattern') {
        setBehaviorPatterns(prev => [...prev, data.pattern]);
      } else if (data.type === 'analysis_result') {
        setAnalysisResults(prev => ({
          ...prev,
          [data.request_id]: data.result
        }));
      }
    }
  });

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Fetch security issues
  const fetchSecurityIssues = async () => {
    setLoading(true);
    try {
      const response = await proxyApi.getSecurityIssues();
      if (response.success) {
        setSecurityIssues(response.issues || []);
      }
    } catch (error) {
      console.error('Error fetching security issues:', error);
    } finally {
      setLoading(false);
    }
  };

  // Fetch behavior patterns
  const fetchBehaviorPatterns = async () => {
    setLoading(true);
    try {
      const response = await proxyApi.getBehaviorPatterns();
      if (response.success) {
        setBehaviorPatterns(response.patterns || []);
      }
    } catch (error) {
      console.error('Error fetching behavior patterns:', error);
    } finally {
      setLoading(false);
    }
  };

  // Fetch analysis results
  const fetchAnalysisResults = async () => {
    setLoading(true);
    try {
      const response = await proxyApi.getAnalysisResults();
      if (response.success) {
        setAnalysisResults(response.results || {});
      }
    } catch (error) {
      console.error('Error fetching analysis results:', error);
    } finally {
      setLoading(false);
    }
  };

  // Fetch analysis rules
  const fetchRules = async () => {
    try {
      const response = await proxyApi.getAnalysisRules();
      if (response.success) {
        setRules(response.rules || []);
      }
    } catch (error) {
      console.error('Error fetching analysis rules:', error);
    }
  };

  // Handle view details
  const handleViewDetails = (data) => {
    setDetailsData(data);
    setDetailsOpen(true);
  };

  // Handle close details
  const handleCloseDetails = () => {
    setDetailsOpen(false);
    setDetailsData(null);
  };

  // Handle add rule
  const handleAddRule = async () => {
    try {
      const response = await proxyApi.addAnalysisRule(newRule);
      if (response.success) {
        setNewRuleDialogOpen(false);
        fetchRules();
      }
    } catch (error) {
      console.error('Error adding rule:', error);
    }
  };

  // Handle delete rule
  const handleDeleteRule = async (ruleId) => {
    try {
      const response = await proxyApi.deleteAnalysisRule(ruleId);
      if (response.success) {
        fetchRules();
      }
    } catch (error) {
      console.error('Error deleting rule:', error);
    }
  };

  // Initial data fetch
  useEffect(() => {
    fetchSecurityIssues();
    fetchBehaviorPatterns();
    fetchAnalysisResults();
    fetchRules();
  }, []);

  // Refresh data
  const handleRefresh = () => {
    fetchSecurityIssues();
    fetchBehaviorPatterns();
    fetchAnalysisResults();
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="analysis tabs">
          <Tab icon={<SecurityIcon />} label="Security Issues" />
          <Tab icon={<TimelineIcon />} label="Behavior Patterns" />
          <Tab icon={<BugReportIcon />} label="Analysis Results" />
          <Tab icon={<SettingsIcon />} label="Rules" />
        </Tabs>
        <Box sx={{ mr: 2 }}>
          <Tooltip title="Refresh Data">
            <IconButton onClick={handleRefresh} disabled={loading}>
              {loading ? <CircularProgress size={24} /> : <RefreshIcon />}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Security Issues Tab */}
      <TabPanel value={tabValue} index={0}>
        <Typography variant="h6" gutterBottom>
          Security Issues
        </Typography>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Severity</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Description</TableCell>
                <TableCell>Request ID</TableCell>
                <TableCell>Timestamp</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {securityIssues.length > 0 ? (
                securityIssues.map((issue, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      <Chip 
                        label={issue.severity} 
                        color={severityColors[issue.severity] || 'default'} 
                        size="small" 
                      />
                    </TableCell>
                    <TableCell>{issue.type}</TableCell>
                    <TableCell>{issue.description}</TableCell>
                    <TableCell>{issue.request_id}</TableCell>
                    <TableCell>{new Date(issue.timestamp).toLocaleString()}</TableCell>
                    <TableCell>
                      <Button 
                        size="small" 
                        variant="outlined" 
                        onClick={() => handleViewDetails(issue)}
                      >
                        Details
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={6} align="center">
                    No security issues found
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      {/* Behavior Patterns Tab */}
      <TabPanel value={tabValue} index={1}>
        <Typography variant="h6" gutterBottom>
          Behavior Patterns
        </Typography>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Pattern Type</TableCell>
                <TableCell>Confidence</TableCell>
                <TableCell>Description</TableCell>
                <TableCell>Timestamp</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {behaviorPatterns.length > 0 ? (
                behaviorPatterns.map((pattern, index) => (
                  <TableRow key={index}>
                    <TableCell>{pattern.pattern_type}</TableCell>
                    <TableCell>{(pattern.confidence * 100).toFixed(2)}%</TableCell>
                    <TableCell>{pattern.description}</TableCell>
                    <TableCell>{new Date(pattern.timestamp).toLocaleString()}</TableCell>
                    <TableCell>
                      <Button 
                        size="small" 
                        variant="outlined" 
                        onClick={() => handleViewDetails(pattern)}
                      >
                        Details
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={5} align="center">
                    No behavior patterns found
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      {/* Analysis Results Tab */}
      <TabPanel value={tabValue} index={2}>
        <Typography variant="h6" gutterBottom>
          Analysis Results
        </Typography>
        {Object.keys(analysisResults).length > 0 ? (
          Object.entries(analysisResults).map(([requestId, result]) => (
            <Accordion key={requestId}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>
                  Request ID: {requestId} - {result.metadata?.method} {result.metadata?.path}
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box>
                  <Typography variant="subtitle1">Security Issues: {result.security_issues?.length || 0}</Typography>
                  <Typography variant="subtitle1">Behavior Patterns: {result.behavior_patterns?.length || 0}</Typography>
                  <Typography variant="subtitle1">Timestamp: {new Date(result.timestamp).toLocaleString()}</Typography>
                  
                  {result.security_issues?.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2">Security Issues:</Typography>
                      <TableContainer component={Paper} sx={{ maxHeight: 200 }}>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>Severity</TableCell>
                              <TableCell>Type</TableCell>
                              <TableCell>Description</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {result.security_issues.map((issue, idx) => (
                              <TableRow key={idx}>
                                <TableCell>
                                  <Chip 
                                    label={issue.severity} 
                                    color={severityColors[issue.severity] || 'default'} 
                                    size="small" 
                                  />
                                </TableCell>
                                <TableCell>{issue.type}</TableCell>
                                <TableCell>{issue.description}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Box>
                  )}
                </Box>
              </AccordionDetails>
            </Accordion>
          ))
        ) : (
          <Paper sx={{ p: 2 }}>
            <Typography align="center">No analysis results found</Typography>
          </Paper>
        )}
      </TabPanel>

      {/* Rules Tab */}
      <TabPanel value={tabValue} index={3}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6">Analysis Rules</Typography>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={() => setNewRuleDialogOpen(true)}
          >
            Add Rule
          </Button>
        </Box>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Priority</TableCell>
                <TableCell>Enabled</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {rules.length > 0 ? (
                rules.map((rule) => (
                  <TableRow key={rule.id}>
                    <TableCell>{rule.name}</TableCell>
                    <TableCell>{rule.rule_type}</TableCell>
                    <TableCell>{rule.priority}</TableCell>
                    <TableCell>{rule.enabled ? 'Yes' : 'No'}</TableCell>
                    <TableCell>
                      <Button 
                        size="small" 
                        variant="outlined" 
                        onClick={() => handleViewDetails(rule)}
                        sx={{ mr: 1 }}
                      >
                        Details
                      </Button>
                      <Button 
                        size="small" 
                        variant="outlined" 
                        color="error"
                        onClick={() => handleDeleteRule(rule.id)}
                      >
                        Delete
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={5} align="center">
                    No rules found
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      {/* Details Dialog */}
      <Dialog open={detailsOpen} onClose={handleCloseDetails} maxWidth="md" fullWidth>
        <DialogTitle>Details</DialogTitle>
        <DialogContent>
          {detailsData && (
            <Box sx={{ mt: 2 }}>
              {Object.entries(detailsData).map(([key, value]) => (
                <Box key={key} sx={{ mb: 2 }}>
                  <Typography variant="subtitle2">{key}:</Typography>
                  {typeof value === 'object' ? (
                    <pre>{JSON.stringify(value, null, 2)}</pre>
                  ) : (
                    <Typography variant="body2">{String(value)}</Typography>
                  )}
                </Box>
              ))}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDetails}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* New Rule Dialog */}
      <Dialog open={newRuleDialogOpen} onClose={() => setNewRuleDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Add New Rule</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <TextField
              label="Rule Name"
              fullWidth
              margin="normal"
              value={newRule.name}
              onChange={(e) => setNewRule({...newRule, name: e.target.value})}
            />
            <TextField
              label="Description"
              fullWidth
              margin="normal"
              multiline
              rows={2}
              value={newRule.description}
              onChange={(e) => setNewRule({...newRule, description: e.target.value})}
            />
            <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
              <FormControl fullWidth>
                <InputLabel>Rule Type</InputLabel>
                <Select
                  value={newRule.rule_type}
                  label="Rule Type"
                  onChange={(e) => setNewRule({...newRule, rule_type: e.target.value})}
                >
                  <MenuItem value="security">Security</MenuItem>
                  <MenuItem value="performance">Performance</MenuItem>
                  <MenuItem value="behavior">Behavior</MenuItem>
                </Select>
              </FormControl>
              <TextField
                label="Priority"
                type="number"
                fullWidth
                margin="normal"
                value={newRule.priority}
                onChange={(e) => setNewRule({...newRule, priority: parseInt(e.target.value)})}
                InputProps={{ inputProps: { min: 1, max: 100 } }}
              />
            </Box>
            
            <Typography variant="h6" sx={{ mt: 3 }}>Condition</Typography>
            <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
              <TextField
                label="Field"
                fullWidth
                margin="normal"
                value={newRule.conditions.field}
                onChange={(e) => setNewRule({
                  ...newRule, 
                  conditions: {...newRule.conditions, field: e.target.value}
                })}
                placeholder="e.g., request.headers.user-agent"
              />
              <FormControl fullWidth margin="normal">
                <InputLabel>Operator</InputLabel>
                <Select
                  value={newRule.conditions.operator}
                  label="Operator"
                  onChange={(e) => setNewRule({
                    ...newRule, 
                    conditions: {...newRule.conditions, operator: e.target.value}
                  })}
                >
                  <MenuItem value="equals">Equals</MenuItem>
                  <MenuItem value="not_equals">Not Equals</MenuItem>
                  <MenuItem value="contains">Contains</MenuItem>
                  <MenuItem value="not_contains">Not Contains</MenuItem>
                  <MenuItem value="starts_with">Starts With</MenuItem>
                  <MenuItem value="ends_with">Ends With</MenuItem>
                  <MenuItem value="matches">Matches Regex</MenuItem>
                </Select>
              </FormControl>
              <TextField
                label="Value"
                fullWidth
                margin="normal"
                value={newRule.conditions.value}
                onChange={(e) => setNewRule({
                  ...newRule, 
                  conditions: {...newRule.conditions, value: e.target.value}
                })}
              />
            </Box>
            
            <Typography variant="h6" sx={{ mt: 3 }}>Action</Typography>
            <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
              <FormControl fullWidth margin="normal">
                <InputLabel>Severity</InputLabel>
                <Select
                  value={newRule.actions[0].severity}
                  label="Severity"
                  onChange={(e) => {
                    const newActions = [...newRule.actions];
                    newActions[0] = {...newActions[0], severity: e.target.value};
                    setNewRule({...newRule, actions: newActions});
                  }}
                >
                  <MenuItem value="high">High</MenuItem>
                  <MenuItem value="medium">Medium</MenuItem>
                  <MenuItem value="low">Low</MenuItem>
                </Select>
              </FormControl>
              <TextField
                label="Issue Type"
                fullWidth
                margin="normal"
                value={newRule.actions[0].type}
                onChange={(e) => {
                  const newActions = [...newRule.actions];
                  newActions[0] = {...newActions[0], type: e.target.value};
                  setNewRule({...newRule, actions: newActions});
                }}
              />
            </Box>
            <TextField
              label="Issue Description"
              fullWidth
              margin="normal"
              multiline
              rows={2}
              value={newRule.actions[0].description}
              onChange={(e) => {
                const newActions = [...newRule.actions];
                newActions[0] = {...newActions[0], description: e.target.value};
                setNewRule({...newRule, actions: newActions});
              }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setNewRuleDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleAddRule} variant="contained" color="primary">Add Rule</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
