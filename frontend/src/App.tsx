import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import AppLayout from './components/layout/AppLayout';
import ReconDashboard from './ReconDashboard';
import { ProxyDashboard } from './ProxyDashboard';
import { ProjectManager } from './components/projects/ProjectManager';
import { ProjectProvider } from './contexts/ProjectContext';
import AISettings from './components/settings/AISettings';
import { aiReducer } from './store/ai/reducer';
import type { RootState } from './store/types';
import { AIRoutes } from './routes/ai';
import Health from './pages/Health';

// Create Redux store with proper typing
const store = configureStore({
  reducer: {
    ai: aiReducer
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // It's safe to ignore these cases since we handle them properly
        ignoredActionPaths: ['error', 'payload.error'],
        ignoredPaths: ['ai.error']
      }
    })
});

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
    <Provider store={store}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <ProjectProvider>
          <Router>
            <AppLayout>
              <Routes>
                <Route path="/" element={<ReconDashboard />} />
                <Route path="/recon" element={<ReconDashboard />} />
                <Route path="/proxy" element={<ProxyDashboard />} />
                <Route path="/projects" element={<ProjectManager />} />
                <Route path="/health" element={<Health />} />
                {/* AI Routes */}
                <Route path="/ai/*" element={AIRoutes} />
                <Route path="/settings/ai" element={<AISettings />} />
              </Routes>
            </AppLayout>
          </Router>
        </ProjectProvider>
      </ThemeProvider>
    </Provider>
  );
}

export default App;
