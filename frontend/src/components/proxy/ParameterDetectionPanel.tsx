import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  FormControlLabel,
  Grid,
  IconButton,
  InputLabel,
  List,
  ListItem,
  ListItemIcon,
  ListItemSecondaryAction,
  ListItemText,
  MenuItem,
  Paper,
  Select,
  Switch,
  Tab,
  Tabs,
  TextField,
  Tooltip,
  Typography
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Search as SearchIcon,
  Code as CodeIcon,
  Link as LinkIcon,
  TextFields as TextIcon,
  DataObject as JsonIcon,
  BugReport as FuzzIcon,
  AutoFixHigh as AutoDetectIcon,
  FilterList as FilterIcon,
  Save as SaveIcon
} from '@mui/icons-material';
import { useProxyApi } from '../../contexts/ProxyApiContext';
import { FuzzingConfigPanel } from './FuzzingConfigPanel';

// Parameter type icons
const getParameterTypeIcon = (type) => {
  switch (type.toLowerCase()) {
    case 'json_key':
    case 'json_value':
      return <JsonIcon />;
    case 'url_query':
      return <LinkIcon />;
    case 'form_data':
      return <CodeIcon />;
    case 'custom':
      return <TextIcon />;
    default:
      return <SearchIcon />;
  }
};

const ParameterDetectionPanel = ({ connectionId, onParametersFuzzed }) => {
  const { api } = useProxyApi();
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [parameters, setParameters] = useState([]);
  const [selectedParameters, setSelectedParameters] = useState([]);
  const [fuzzingLists, setFuzzingLists] = useState([]);
  const [selectedListId, setSelectedListId] = useState('');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.7);
  const [autoDetectEnabled, setAutoDetectEnabled] = useState(true);
  const [filterType, setFilterType] = useState('all');
  
  // Load fuzzing lists
  useEffect(() => {
    const loadFuzzingLists = async () => {
      try {
        setLoading(true);
        const lists = await api.getFuzzingLists();
        setFuzzingLists(lists);
        if (lists.length > 0) {
          setSelectedListId(lists[0].id);
        }
      } catch (error) {
        console.error('Error loading fuzzing lists:', error);
      } finally {
        setLoading(false);
      }
    };
    
    loadFuzzingLists();
  }, [api]);
  
  // Detect parameters
  const detectParameters = async () => {
    if (!connectionId) return;
    
    try {
      setAnalyzing(true);
      const result = await api.detectParameters(connectionId, { 
        confidence_threshold: confidenceThreshold 
      });
      
      if (result && result.parameters) {
        setParameters(result.parameters);
        // Auto-select parameters with high confidence
        const highConfidenceParams = result.parameters
          .filter(p => p.confidence > 0.8)
          .map(p => p.name);
        setSelectedParameters(highConfidenceParams);
      }
    } catch (error) {
      console.error('Error detecting parameters:', error);
    } finally {
      setAnalyzing(false);
    }
  };
  
  // Start fuzzing with detected parameters
  const startFuzzing = async () => {
    if (!connectionId || selectedParameters.length === 0) return;
    
    try {
      setLoading(true);
      const result = await api.fuzzDetectedParameters(connectionId, {
        parameters: selectedParameters,
        list_id: selectedListId || undefined,
        auto_detect: autoDetectEnabled
      });
      
      if (result && result.fuzzed_messages && onParametersFuzzed) {
        onParametersFuzzed(result.fuzzed_messages);
      }
    } catch (error) {
      console.error('Error fuzzing parameters:', error);
    } finally {
      setLoading(false);
    }
  };
  
  // Toggle parameter selection
  const toggleParameter = (paramName) => {
    setSelectedParameters(prev => 
      prev.includes(paramName)
        ? prev.filter(p => p !== paramName)
        : [...prev, paramName]
    );
  };
  
  // Filter parameters by type
  const filteredParameters = parameters.filter(param => {
    if (filterType === 'all') return true;
    return param.type.toLowerCase() === filterType.toLowerCase();
  });
  
  // Get unique parameter types
  const parameterTypes = [...new Set(parameters.map(p => p.type.toLowerCase()))];
  
  return (
    <Card variant="outlined">
      <CardHeader 
        title={
          <Box display="flex" alignItems="center">
            <AutoDetectIcon sx={{ mr: 1 }} />
            <Typography variant="h6">Parameter Auto-Detection</Typography>
          </Box>
        }
        action={
          <Tooltip title="Refresh Parameters">
            <IconButton onClick={detectParameters} disabled={analyzing || !connectionId}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        }
      />
      <Divider />
      <CardContent>
        <Grid container spacing={2}>
          {/* Configuration Section */}
          <Grid item xs={12}>
            <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Detection Settings
              </Typography>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Confidence Threshold"
                    type="number"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                    inputProps={{ min: 0, max: 1, step: 0.1 }}
                    fullWidth
                    disabled={analyzing}
                    size="small"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Filter by Type</InputLabel>
                    <Select
                      value={filterType}
                      onChange={(e) => setFilterType(e.target.value)}
                      label="Filter by Type"
                    >
                      <MenuItem value="all">All Types</MenuItem>
                      {parameterTypes.map(type => (
                        <MenuItem key={type} value={type}>
                          {type.replace('_', ' ').toUpperCase()}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={analyzing ? <CircularProgress size={20} /> : <SearchIcon />}
                    onClick={detectParameters}
                    disabled={analyzing || !connectionId}
                    fullWidth
                  >
                    {analyzing ? 'Analyzing...' : 'Detect Parameters'}
                  </Button>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
          
          {/* Parameters List */}
          <Grid item xs={12}>
            <Paper variant="outlined" sx={{ p: 0, mb: 2 }}>
              <List dense sx={{ maxHeight: '300px', overflow: 'auto' }}>
                {filteredParameters.length === 0 ? (
                  <ListItem>
                    <ListItemText 
                      primary="No parameters detected" 
                      secondary="Click 'Detect Parameters' to analyze the WebSocket messages"
                    />
                  </ListItem>
                ) : (
                  filteredParameters.map((param, index) => (
                    <ListItem key={`${param.name}-${index}`} divider={index < filteredParameters.length - 1}>
                      <ListItemIcon>
                        {getParameterTypeIcon(param.type)}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box display="flex" alignItems="center">
                            <Typography variant="body1" fontWeight="medium">
                              {param.name}
                            </Typography>
                            <Chip 
                              label={param.type.replace('_', ' ')} 
                              size="small" 
                              variant="outlined"
                              sx={{ ml: 1, fontSize: '0.7rem' }}
                            />
                          </Box>
                        }
                        secondary={
                          <React.Fragment>
                            <Typography variant="body2" component="span" color="text.secondary">
                              Value: {String(param.value).substring(0, 30)}
                              {String(param.value).length > 30 ? '...' : ''}
                            </Typography>
                            <br />
                            <Typography variant="body2" component="span" color="text.secondary">
                              Path: {param.path}
                            </Typography>
                            <br />
                            <Typography variant="body2" component="span" color="text.secondary">
                              Confidence: {(param.confidence * 100).toFixed(1)}%
                            </Typography>
                          </React.Fragment>
                        }
                      />
                      <ListItemSecondaryAction>
                        <FormControlLabel
                          control={
                            <Switch
                              edge="end"
                              checked={selectedParameters.includes(param.name)}
                              onChange={() => toggleParameter(param.name)}
                            />
                          }
                          label=""
                        />
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))
                )}
              </List>
            </Paper>
          </Grid>
          
          {/* Fuzzing Configuration */}
          <Grid item xs={12}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Fuzzing Configuration
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Fuzzing List</InputLabel>
                    <Select
                      value={selectedListId}
                      onChange={(e) => setSelectedListId(e.target.value)}
                      label="Fuzzing List"
                      disabled={loading || fuzzingLists.length === 0}
                    >
                      {fuzzingLists.map(list => (
                        <MenuItem key={list.id} value={list.id}>
                          {list.name} ({list.payload_count} payloads)
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={autoDetectEnabled}
                        onChange={(e) => setAutoDetectEnabled(e.target.checked)}
                      />
                    }
                    label="Auto-detect best injection points"
                  />
                </Grid>
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    color="secondary"
                    startIcon={loading ? <CircularProgress size={20} /> : <FuzzIcon />}
                    onClick={startFuzzing}
                    disabled={loading || selectedParameters.length === 0 || !connectionId}
                    fullWidth
                  >
                    {loading ? 'Fuzzing...' : `Fuzz ${selectedParameters.length} Selected Parameters`}
                  </Button>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default ParameterDetectionPanel;
