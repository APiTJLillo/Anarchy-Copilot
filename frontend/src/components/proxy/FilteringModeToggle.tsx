import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  FormControlLabel,
  Switch,
  Typography,
  Tooltip,
  IconButton,
  Chip,
  Divider,
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import WarningIcon from '@mui/icons-material/Warning';
import { useTheme } from '@mui/material/styles';

/**
 * Component for toggling between Active and Bypass filtering modes
 */
const FilteringModeToggle = ({ mode, onModeChange, isLoading }) => {
  const theme = useTheme();
  
  const handleChange = (event) => {
    const newMode = event.target.checked ? 'ACTIVE' : 'BYPASS';
    onModeChange(newMode);
  };
  
  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box display="flex" alignItems="center">
            <Typography variant="h6" component="div" sx={{ mr: 1 }}>
              Filtering Mode
            </Typography>
            <Tooltip title="In Active mode, traffic matching filter rules will be blocked. In Bypass mode, all traffic is allowed but still recorded for analysis.">
              <IconButton size="small">
                <InfoIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
          
          <Box display="flex" alignItems="center">
            {mode === 'BYPASS' && (
              <Tooltip title="Filters are not being applied in Bypass mode">
                <WarningIcon 
                  color="warning" 
                  fontSize="small" 
                  sx={{ mr: 1 }} 
                />
              </Tooltip>
            )}
            
            <FormControlLabel
              control={
                <Switch
                  checked={mode === 'ACTIVE'}
                  onChange={handleChange}
                  disabled={isLoading}
                  color="primary"
                />
              }
              label={
                <Box display="flex" alignItems="center">
                  <Typography 
                    variant="body1" 
                    color={mode === 'BYPASS' ? theme.palette.text.secondary : theme.palette.primary.main}
                  >
                    Bypass
                  </Typography>
                  <Divider orientation="vertical" flexItem sx={{ mx: 1, height: 16 }} />
                  <Typography 
                    variant="body1"
                    color={mode === 'ACTIVE' ? theme.palette.primary.main : theme.palette.text.secondary}
                  >
                    Active
                  </Typography>
                </Box>
              }
            />
            
            <Chip
              label={mode}
              color={mode === 'ACTIVE' ? 'primary' : 'default'}
              size="small"
              sx={{ ml: 1 }}
            />
          </Box>
        </Box>
        
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          {mode === 'ACTIVE' 
            ? 'Filters are active and will block matching traffic' 
            : 'Filters are bypassed, all traffic is allowed but still recorded for analysis'}
        </Typography>
      </CardContent>
    </Card>
  );
};

export default FilteringModeToggle;
