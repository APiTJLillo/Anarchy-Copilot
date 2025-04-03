import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import TrafficAnalysisView from '../components/analysis/TrafficAnalysisView';

export default function AnalysisDashboard() {
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>Traffic Analysis Dashboard</Typography>
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="body1">
          The Traffic Analysis Dashboard provides comprehensive analysis of proxy traffic to identify security issues, 
          behavior patterns, and anomalies. Use the tabs below to explore different aspects of the analysis.
        </Typography>
      </Paper>
      <TrafficAnalysisView />
    </Box>
  );
}
