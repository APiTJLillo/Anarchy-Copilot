import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  Paper,
  Chip,
  IconButton,
  Divider,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Tooltip,
  Alert,
  Collapse,
} from '@mui/material';
import {
  Add,
  Refresh,
  Check,
  Info,
  ExpandMore,
  ExpandLess,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

/**
 * Component to display suggested filter conditions based on traffic patterns
 */
const FilterSuggestionPanel = ({ 
  suggestions = [], 
  isLoading = false, 
  onRefresh, 
  onAddSuggestion,
  onAddAllConditions,
}) => {
  const theme = useTheme();
  const [expandedRule, setExpandedRule] = useState(null);
  
  const handleToggleExpand = (index) => {
    setExpandedRule(expandedRule === index ? null : index);
  };
  
  const handleAddSuggestion = (suggestion) => {
    if (onAddSuggestion) {
      onAddSuggestion(suggestion);
    }
  };
  
  const handleAddAllConditions = (suggestion) => {
    if (onAddAllConditions) {
      onAddAllConditions(suggestion);
    }
  };
  
  return (
    <Card variant="outlined">
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h6" component="div">
            Suggested Filters
          </Typography>
          
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
        
        {isLoading ? (
          <Box display="flex" justifyContent="center" my={4}>
            <CircularProgress />
          </Box>
        ) : suggestions.length === 0 ? (
          <Alert severity="info" sx={{ mb: 2 }}>
            No filter suggestions available. Collect more traffic in bypass mode to generate suggestions.
          </Alert>
        ) : (
          <List>
            {suggestions.map((suggestion, index) => (
              <Paper 
                key={index} 
                variant="outlined" 
                sx={{ 
                  mb: 2, 
                  overflow: 'hidden',
                  border: expandedRule === index 
                    ? `1px solid ${theme.palette.primary.main}` 
                    : undefined
                }}
              >
                <ListItem 
                  button 
                  onClick={() => handleToggleExpand(index)}
                  sx={{ 
                    bgcolor: theme.palette.background.default,
                    '&:hover': {
                      bgcolor: theme.palette.action.hover,
                    },
                  }}
                >
                  <ListItemText
                    primary={
                      <Typography variant="subtitle2">
                        {suggestion.name}
                      </Typography>
                    }
                    secondary={
                      <Box component="span" display="flex" alignItems="center" mt={0.5}>
                        <Chip 
                          label={suggestion.tags[0] || 'filter'} 
                          size="small" 
                          color="primary"
                          variant="outlined"
                          sx={{ mr: 1 }}
                        />
                        <Typography variant="body2" color="text.secondary">
                          {suggestion.description}
                        </Typography>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton 
                      edge="end" 
                      onClick={(e) => {
                        e.stopPropagation();
                        handleToggleExpand(index);
                      }}
                    >
                      {expandedRule === index ? <ExpandLess /> : <ExpandMore />}
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
                
                <Collapse in={expandedRule === index} timeout="auto" unmountOnExit>
                  <Box p={2}>
                    <Typography variant="subtitle2" gutterBottom>
                      Conditions
                    </Typography>
                    
                    <List dense>
                      {suggestion.conditions.map((condition, condIndex) => (
                        <ListItem key={condIndex} sx={{ py: 0.5 }}>
                          <ListItemText
                            primary={
                              <Box display="flex" alignItems="center">
                                <Chip 
                                  label={condition.field} 
                                  size="small" 
                                  sx={{ mr: 1 }}
                                />
                                <Typography variant="body2" color="text.secondary">
                                  {condition.operator}
                                </Typography>
                                <Chip 
                                  label={condition.value} 
                                  size="small" 
                                  variant="outlined"
                                  sx={{ ml: 1 }}
                                />
                              </Box>
                            }
                          />
                          <ListItemSecondaryAction>
                            <Tooltip title="Add this condition">
                              <IconButton 
                                edge="end" 
                                size="small"
                                onClick={() => handleAddSuggestion(condition)}
                              >
                                <Add fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </ListItemSecondaryAction>
                        </ListItem>
                      ))}
                    </List>
                    
                    <Divider sx={{ my: 1.5 }} />
                    
                    <Box display="flex" justifyContent="flex-end">
                      <Button
                        variant="contained"
                        size="small"
                        startIcon={<Check />}
                        onClick={() => handleAddAllConditions(suggestion)}
                      >
                        Use All Conditions
                      </Button>
                    </Box>
                  </Box>
                </Collapse>
              </Paper>
            ))}
          </List>
        )}
        
        {!isLoading && suggestions.length > 0 && (
          <Box display="flex" alignItems="center" mt={2}>
            <Info fontSize="small" color="info" sx={{ mr: 1 }} />
            <Typography variant="body2" color="text.secondary">
              These suggestions are based on traffic patterns detected in bypass mode.
              Click on a suggestion to view its conditions.
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default FilterSuggestionPanel;
