import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Collapse,
  TextField,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Tooltip,
  CircularProgress,
  Pagination,
} from '@mui/material';
import {
  FilterList,
  KeyboardArrowDown,
  KeyboardArrowUp,
  ContentCopy,
  Add,
  Refresh,
  Search,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { format } from 'date-fns';

/**
 * Component to display traffic history with the ability to add items to filters
 */
const TrafficHistoryView = ({ 
  trafficHistory = [], 
  isLoading = false, 
  onRefresh, 
  onAddToFilter 
}) => {
  const theme = useTheme();
  const [page, setPage] = useState(1);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [expandedRow, setExpandedRow] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredTraffic, setFilteredTraffic] = useState([]);
  
  // Filter traffic based on search term
  useEffect(() => {
    if (!searchTerm) {
      setFilteredTraffic(trafficHistory);
      return;
    }
    
    const filtered = trafficHistory.filter(item => {
      const searchString = searchTerm.toLowerCase();
      
      // Search in path
      if (item.path && item.path.toLowerCase().includes(searchString)) {
        return true;
      }
      
      // Search in request path
      if (item.request_path && item.request_path.toLowerCase().includes(searchString)) {
        return true;
      }
      
      // Search in method
      if (item.method && item.method.toLowerCase().includes(searchString)) {
        return true;
      }
      
      // Search in request method
      if (item.request_method && item.request_method.toLowerCase().includes(searchString)) {
        return true;
      }
      
      // Search in status code
      if (item.status_code && item.status_code.toString().includes(searchString)) {
        return true;
      }
      
      return false;
    });
    
    setFilteredTraffic(filtered);
  }, [searchTerm, trafficHistory]);
  
  // Reset to first page when filtered traffic changes
  useEffect(() => {
    setPage(1);
  }, [filteredTraffic]);
  
  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };
  
  const handleRowClick = (index) => {
    setExpandedRow(expandedRow === index ? null : index);
  };
  
  const handleAddToFilter = (item) => {
    if (onAddToFilter) {
      onAddToFilter(item);
    }
  };
  
  const handleCopyContent = (content) => {
    navigator.clipboard.writeText(content);
  };
  
  // Calculate pagination
  const startIndex = (page - 1) * rowsPerPage;
  const endIndex = startIndex + rowsPerPage;
  const paginatedTraffic = filteredTraffic.slice(startIndex, endIndex);
  const totalPages = Math.ceil(filteredTraffic.length / rowsPerPage);
  
  // Format timestamp
  const formatTimestamp = (timestamp) => {
    try {
      return format(new Date(timestamp), 'yyyy-MM-dd HH:mm:ss');
    } catch (e) {
      return timestamp;
    }
  };
  
  return (
    <Card variant="outlined">
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h6" component="div">
            Traffic History
          </Typography>
          
          <Box display="flex" alignItems="center">
            <TextField
              size="small"
              placeholder="Search traffic..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: <Search fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />,
              }}
              sx={{ mr: 2, width: 250 }}
            />
            
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={onRefresh}
              disabled={isLoading}
              size="small"
            >
              Refresh
            </Button>
          </Box>
        </Box>
        
        {isLoading ? (
          <Box display="flex" justifyContent="center" my={4}>
            <CircularProgress />
          </Box>
        ) : filteredTraffic.length === 0 ? (
          <Typography variant="body1" color="text.secondary" align="center" my={4}>
            No traffic history available
          </Typography>
        ) : (
          <>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell padding="checkbox" />
                    <TableCell>Type</TableCell>
                    <TableCell>Method/Status</TableCell>
                    <TableCell>Path</TableCell>
                    <TableCell>Timestamp</TableCell>
                    <TableCell>Filtered</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {paginatedTraffic.map((item, index) => {
                    const actualIndex = startIndex + index;
                    const isExpanded = expandedRow === actualIndex;
                    
                    return (
                      <React.Fragment key={actualIndex}>
                        <TableRow 
                          hover 
                          onClick={() => handleRowClick(actualIndex)}
                          sx={{ 
                            cursor: 'pointer',
                            '&.MuiTableRow-root:hover': {
                              backgroundColor: theme.palette.action.hover,
                            },
                          }}
                        >
                          <TableCell padding="checkbox">
                            <IconButton size="small">
                              {isExpanded ? <KeyboardArrowUp /> : <KeyboardArrowDown />}
                            </IconButton>
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label={item.type} 
                              size="small"
                              color={item.type === 'request' ? 'primary' : 'secondary'}
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell>
                            {item.type === 'request' ? (
                              <Typography variant="body2" fontWeight="medium">
                                {item.method}
                              </Typography>
                            ) : (
                              <Typography 
                                variant="body2" 
                                fontWeight="medium"
                                color={item.status_code >= 400 ? 'error.main' : 'success.main'}
                              >
                                {item.status_code}
                              </Typography>
                            )}
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" noWrap sx={{ maxWidth: 300 }}>
                              {item.type === 'request' ? item.path : item.request_path}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" color="text.secondary">
                              {formatTimestamp(item.timestamp)}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            {item.filtered !== undefined && (
                              <Chip 
                                label={item.filtered ? 'Yes' : 'No'} 
                                size="small"
                                color={item.filtered ? 'error' : 'success'}
                                variant="outlined"
                              />
                            )}
                          </TableCell>
                          <TableCell align="right">
                            <Tooltip title="Add to filter">
                              <IconButton 
                                size="small" 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleAddToFilter(item);
                                }}
                              >
                                <Add />
                              </IconButton>
                            </Tooltip>
                          </TableCell>
                        </TableRow>
                        
                        <TableRow>
                          <TableCell colSpan={7} padding="none" sx={{ borderBottom: 'none' }}>
                            <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                              <Box p={2} bgcolor={theme.palette.background.default}>
                                <Typography variant="subtitle2" gutterBottom component="div">
                                  Details
                                </Typography>
                                
                                {/* Headers */}
                                <Box mb={2}>
                                  <Typography variant="body2" fontWeight="medium" gutterBottom>
                                    Headers
                                  </Typography>
                                  <Paper variant="outlined" sx={{ p: 1, bgcolor: theme.palette.background.paper }}>
                                    <Box display="flex" justifyContent="flex-end" mb={1}>
                                      <IconButton 
                                        size="small" 
                                        onClick={() => handleCopyContent(JSON.stringify(item.headers, null, 2))}
                                      >
                                        <ContentCopy fontSize="small" />
                                      </IconButton>
                                    </Box>
                                    <pre style={{ margin: 0, overflow: 'auto', maxHeight: 150 }}>
                                      {JSON.stringify(item.headers, null, 2)}
                                    </pre>
                                  </Paper>
                                </Box>
                                
                                {/* Body (if exists) */}
                                {item.body && (
                                  <Box mb={2}>
                                    <Typography variant="body2" fontWeight="medium" gutterBottom>
                                      Body
                                    </Typography>
                                    <Paper variant="outlined" sx={{ p: 1, bgcolor: theme.palette.background.paper }}>
                                      <Box display="flex" justifyContent="flex-end" mb={1}>
                                        <IconButton 
                                          size="small" 
                                          onClick={() => handleCopyContent(item.body)}
                                        >
                                          <ContentCopy fontSize="small" />
                                        </IconButton>
                                      </Box>
                                      <pre style={{ margin: 0, overflow: 'auto', maxHeight: 200 }}>
                                        {item.body}
                                      </pre>
                                    </Paper>
                                  </Box>
                                )}
                                
                                {/* Matched Rules (if any) */}
                                {item.matched_rules && item.matched_rules.length > 0 && (
                                  <Box>
                                    <Typography variant="body2" fontWeight="medium" gutterBottom>
                                      Matched Filter Rules
                                    </Typography>
                                    <Box display="flex" flexWrap="wrap" gap={1}>
                                      {item.matched_rules.map((ruleId, idx) => (
                                        <Chip 
                                          key={idx} 
                                          label={ruleId} 
                                          size="small" 
                                          color="error" 
                                        />
                                      ))}
                                    </Box>
                                  </Box>
                                )}
                                
                                <Box display="flex" justifyContent="flex-end" mt={2}>
                                  <Button
                                    variant="contained"
                                    startIcon={<FilterList />}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleAddToFilter(item);
                                    }}
                                    size="small"
                                  >
                                    Add to Filter
                                  </Button>
                                </Box>
                              </Box>
                            </Collapse>
                          </TableCell>
                        </TableRow>
                      </React.Fragment>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
            
            <Box display="flex" justifyContent="center" mt={2}>
              <Pagination 
                count={totalPages} 
                page={page} 
                onChange={handleChangePage} 
                color="primary" 
              />
            </Box>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default TrafficHistoryView;
