import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Box, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import AppLayout from './components/layout/AppLayout';
import ReconDashboard from './ReconDashboard';
import ProxyDashboard from './ProxyDashboard';

// Create base theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ff00',
    },
    secondary: {
      main: '#424242',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <AppLayout>
          <Routes>
            <Route path="/" element={<ReconDashboard />} />
            <Route path="/recon" element={<ReconDashboard />} />
            <Route path="/proxy" element={<ProxyDashboard />} />
          </Routes>
        </AppLayout>
      </Router>
    </ThemeProvider>
  );
}

export default App;
