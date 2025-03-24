import React from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';
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
import { WebSocketProvider } from './contexts/WebSocketContext';

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

// Wrap routes with transition handling
const RouteWrapper: React.FC<{ element: React.ReactElement }> = ({ element }) => {
  const location = useLocation();
  const [transitioning, setTransitioning] = React.useState(false);

  React.useEffect(() => {
    setTransitioning(true);
    const timer = setTimeout(() => setTransitioning(false), 100);
    return () => clearTimeout(timer);
  }, [location.pathname]);

  if (transitioning) {
    return null;
  }

  return element;
};

function App() {
  return (
    <Provider store={store}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <WebSocketProvider options={{ debug: process.env.NODE_ENV === 'development' }}>
          <ProjectProvider>
            <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
              <AppLayout>
                <Routes>
                  <Route path="/" element={<RouteWrapper element={<ReconDashboard />} />} />
                  <Route path="/recon" element={<RouteWrapper element={<ReconDashboard />} />} />
                  <Route path="/proxy" element={<RouteWrapper element={<ProxyDashboard />} />} />
                  <Route path="/projects" element={<RouteWrapper element={<ProjectManager />} />} />
                  <Route path="/health" element={<RouteWrapper element={<Health />} />} />
                  {/* AI Routes */}
                  <Route path="/ai/*" element={<RouteWrapper element={AIRoutes} />} />
                  <Route path="/settings/ai" element={<RouteWrapper element={<AISettings />} />} />
                </Routes>
              </AppLayout>
            </Router>
          </ProjectProvider>
        </WebSocketProvider>
      </ThemeProvider>
    </Provider>
  );
}

export default App;
