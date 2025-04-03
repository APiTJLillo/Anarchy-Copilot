import React, { useState, useEffect, ChangeEvent } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Grid,
  Paper,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  CircularProgress,
  Alert,
  Snackbar,
} from '@mui/material';
import { Search as SearchIcon, Refresh as RefreshIcon } from '@mui/icons-material';
import axios from 'axios';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`recon-tabpanel-${index}`}
      aria-labelledby={`recon-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

interface Tool {
  id: string;
  name: string;
  description: string;
}

interface Scan {
  id: string;
  domain: string;
  tool: string;
  status: string;
  timestamp: string;
  error?: string;
}

interface ScanResult {
  type: string;
  host?: string;
  port?: number;
  service?: string;
  url?: string;
  status?: number;
  subdomain?: string;
  [key: string]: any;
}

const tools: Tool[] = [
  { id: 'subdomain', name: 'Subdomain Scanner', description: 'Discover subdomains' },
  { id: 'portscan', name: 'Port Scanner', description: 'Scan for open ports' },
  { id: 'service', name: 'Service Detection', description: 'Identify running services' },
];

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

export default function ReconDashboard() {
  const [domain, setDomain] = useState('');
  const [selectedTool, setSelectedTool] = useState('');
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [activeScans, setActiveScans] = useState<Scan[]>([]);
  const [scanHistory, setScanHistory] = useState<Scan[]>([]);
  const [scanResults, setScanResults] = useState<ScanResult[]>([]);
  const [selectedScanId, setSelectedScanId] = useState<string | null>(null);
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);

  // Fetch active scans and scan history on component mount
  useEffect(() => {
    fetchActiveScans();
    fetchScanHistory();

    // Set up refresh interval
    const interval = setInterval(() => {
      fetchActiveScans();
    }, 5000);
    setRefreshInterval(interval);

    // Clean up interval on component unmount
    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, []);

  // Fetch scan results when a scan is selected
  useEffect(() => {
    if (selectedScanId) {
      fetchScanResults(selectedScanId);
    }
  }, [selectedScanId]);

  const fetchActiveScans = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/recon/scans/active`);
      setActiveScans(response.data);
    } catch (err) {
      console.error('Error fetching active scans:', err);
    }
  };

  const fetchScanHistory = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/recon/scans/history`);
      setScanHistory(response.data);
    } catch (err) {
      console.error('Error fetching scan history:', err);
    }
  };

  const fetchScanResults = async (scanId: string) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/recon/scans/${scanId}/results`);
      setScanResults(response.data);
    } catch (err) {
      console.error('Error fetching scan results:', err);
      setScanResults([]);
    }
  };

  const handleDomainChange = (event: ChangeEvent<HTMLInputElement>) => {
    setDomain(event.target.value);
  };

  const handleToolChange = (event: SelectChangeEvent) => {
    setSelectedTool(event.target.value);
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleScanSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    
    if (!domain || !selectedTool) {
      setError('Please enter a domain and select a tool');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/api/recon/scan`, {
        domain,
        tool: selectedTool
      });
      
      setSuccess(`Scan started: ${response.data.message}`);
      fetchActiveScans();
      
      // Reset form
      setDomain('');
      setSelectedTool('');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start scan');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    fetchActiveScans();
    fetchScanHistory();
    if (selectedScanId) {
      fetchScanResults(selectedScanId);
    }
  };

  const handleViewResults = (scanId: string) => {
    setSelectedScanId(scanId);
    setTabValue(2); // Switch to Results tab
  };

  const renderResultsTable = () => {
    if (!selectedScanId) {
      return <Typography>Select a scan to view results</Typography>;
    }

    if (scanResults.length === 0) {
      return <Typography>No results available for this scan</Typography>;
    }

    // Determine columns based on the first result
    const firstResult = scanResults[0];
    const columns = Object.keys(firstResult).filter(key => key !== 'type');

    return (
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Type</TableCell>
              {columns.map(column => (
                <TableCell key={column}>{column}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {scanResults.map((result, index) => (
              <TableRow key={index}>
                <TableCell>{result.type}</TableCell>
                {columns.map(column => (
                  <TableCell key={column}>{result[column]?.toString() || '-'}</TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Snackbar 
        open={!!error} 
        autoHideDuration={6000} 
        onClose={() => setError(null)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert onClose={() => setError(null)} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>

      <Snackbar 
        open={!!success} 
        autoHideDuration={6000} 
        onClose={() => setSuccess(null)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert onClose={() => setSuccess(null)} severity="success" sx={{ width: '100%' }}>
          {success}
        </Alert>
      </Snackbar>

      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Reconnaissance Dashboard
        </Typography>
        <Button 
          variant="outlined" 
          startIcon={<RefreshIcon />} 
          onClick={handleRefresh}
        >
          Refresh
        </Button>
      </Box>

      <Grid container spacing={3}>
        {/* New Scan Form */}
        <Grid item xs={12} md={6}>
          <Card sx={{ minHeight: '100%', backgroundColor: 'background.paper' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                New Scan
              </Typography>
              <Box component="form" onSubmit={handleScanSubmit} sx={{ mt: 2 }}>
                <TextField
                  fullWidth
                  label="Domain"
                  variant="outlined"
                  value={domain}
                  onChange={handleDomainChange}
                  sx={{ mb: 2 }}
                  placeholder="example.com"
                  disabled={loading}
                />
                <FormControl fullWidth sx={{ mb: 2 }} disabled={loading}>
                  <InputLabel>Tool</InputLabel>
                  <Select
                    value={selectedTool}
                    label="Tool"
                    onChange={handleToolChange}
                  >
                    {tools.map((tool) => (
                      <MenuItem key={tool.id} value={tool.id}>
                        {tool.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Button
                  type="submit"
                  variant="contained"
                  fullWidth
                  startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
                  sx={{
                    backgroundColor: 'primary.main',
                    '&:hover': {
                      backgroundColor: 'primary.dark',
                    },
                  }}
                  disabled={loading}
                >
                  {loading ? 'Starting Scan...' : 'Start Scan'}
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Results and History */}
        <Grid item xs={12} md={6}>
          <Card sx={{ minHeight: '100%', backgroundColor: 'background.paper' }}>
            <CardContent>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs 
                  value={tabValue}
                  onChange={handleTabChange}
                  textColor="primary"
                  indicatorColor="primary"
                >
                  <Tab label="Active Scans" />
                  <Tab label="History" />
                  <Tab label="Results" />
                </Tabs>
              </Box>
              <TabPanel value={tabValue} index={0}>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Domain</TableCell>
                        <TableCell>Tool</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {activeScans.length > 0 ? (
                        activeScans.map((scan) => (
                          <TableRow key={scan.id}>
                            <TableCell>{scan.domain}</TableCell>
                            <TableCell>{scan.tool}</TableCell>
                            <TableCell>{scan.status}</TableCell>
                            <TableCell>
                              <Button 
                                size="small" 
                                onClick={() => handleViewResults(scan.id)}
                                disabled={scan.status !== 'completed'}
                              >
                                View Results
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))
                      ) : (
                        <TableRow>
                          <TableCell colSpan={4}>No active scans</TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              </TabPanel>
              <TabPanel value={tabValue} index={1}>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Domain</TableCell>
                        <TableCell>Tool</TableCell>
                        <TableCell>Date</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {scanHistory.length > 0 ? (
                        scanHistory.map((scan) => (
                          <TableRow key={scan.id}>
                            <TableCell>{scan.domain}</TableCell>
                            <TableCell>{scan.tool}</TableCell>
                            <TableCell>{new Date(scan.timestamp).toLocaleString()}</TableCell>
                            <TableCell>{scan.status}</TableCell>
                            <TableCell>
                              <Button 
                                size="small" 
                                onClick={() => handleViewResults(scan.id)}
                                disabled={scan.status !== 'completed'}
                              >
                                View Results
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))
                      ) : (
                        <TableRow>
                          <TableCell colSpan={5}>No scan history</TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              </TabPanel>
              <TabPanel value={tabValue} index={2}>
                {renderResultsTable()}
              </TabPanel>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
