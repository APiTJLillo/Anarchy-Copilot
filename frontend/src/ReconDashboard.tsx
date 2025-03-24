import React, { useState, ChangeEvent } from 'react';
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
} from '@mui/material';
import { Search as SearchIcon } from '@mui/icons-material';

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

const tools: Tool[] = [
  { id: 'subdomain', name: 'Subdomain Scanner', description: 'Discover subdomains' },
  { id: 'portscan', name: 'Port Scanner', description: 'Scan for open ports' },
  { id: 'service', name: 'Service Detection', description: 'Identify running services' },
];

export default function ReconDashboard() {
  const [domain, setDomain] = useState('');
  const [selectedTool, setSelectedTool] = useState('');
  const [tabValue, setTabValue] = useState(0);

  const handleDomainChange = (event: ChangeEvent<HTMLInputElement>) => {
    setDomain(event.target.value);
  };

  const handleToolChange = (event: SelectChangeEvent) => {
    setSelectedTool(event.target.value);
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleScanSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    // TODO: Implement scan submission
    console.log('Starting scan:', { domain, tool: selectedTool });
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
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
                />
                <FormControl fullWidth sx={{ mb: 2 }}>
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
                  startIcon={<SearchIcon />}
                  sx={{
                    backgroundColor: 'primary.main',
                    '&:hover': {
                      backgroundColor: 'primary.dark',
                    },
                  }}
                >
                  Start Scan
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
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>No active scans</TableCell>
                        <TableCell></TableCell>
                        <TableCell></TableCell>
                      </TableRow>
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
                        <TableCell>Results</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>No scan history</TableCell>
                        <TableCell></TableCell>
                        <TableCell></TableCell>
                        <TableCell></TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </TabPanel>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
