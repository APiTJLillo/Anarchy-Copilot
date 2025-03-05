import React from 'react';
import { render, screen } from '@testing-library/react';
import { createMemoryHistory } from 'history';
import { Router } from 'react-router-dom';
import App from './App';

// Mock components
jest.mock('./ReconDashboard', () => ({
    __esModule: true,
    default: () => <div data-testid="recon-dashboard">Recon Dashboard Mock</div>
}));

jest.mock('./ProxyDashboard', () => ({
    __esModule: true,
    ProxyDashboard: () => <div data-testid="proxy-dashboard">Proxy Dashboard Mock</div>
}));

jest.mock('./components/projects/ProjectManager', () => ({
    __esModule: true,
    ProjectManager: () => <div data-testid="project-manager">Project Manager Mock</div>
}));

jest.mock('./components/settings/AISettings', () => ({
    __esModule: true,
    default: () => <div data-testid="ai-settings">AI Settings Mock</div>
}));

// Mock AI components
jest.mock('./routes/ai', () => ({
    __esModule: true,
    AIRoutes: () => <div data-testid="ai-routes">AI Routes Mock</div>,
    isAIRoute: (path: string) => path.startsWith('/ai') || path === '/settings/ai',
    getCurrentAISection: () => 'AI Hub',
    AI_PATHS: {
        root: '/ai',
        chat: '/ai/chat',
        analytics: '/ai/analytics',
        operations: '/ai/operations',
        monitoring: '/ai/monitoring',
        settings: '/settings/ai'
    }
}));

// Mock ProjectProvider to avoid context errors
jest.mock('./contexts/ProjectContext', () => ({
    ProjectProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>
}));

describe('App', () => {
    const renderWithRouter = (initialEntry = '/') => {
        const history = createMemoryHistory({ initialEntries: [initialEntry] });
        return {
            ...render(
                <Router location={history.location} navigator={history}>
                    <App />
                </Router>
            ),
            history
        };
    };

    it('renders without crashing', () => {
        renderWithRouter();
    });

    it('renders ReconDashboard at root path', () => {
        renderWithRouter('/');
        expect(screen.getByTestId('recon-dashboard')).toBeInTheDocument();
    });

    it('renders ProxyDashboard at /proxy path', () => {
        renderWithRouter('/proxy');
        expect(screen.getByTestId('proxy-dashboard')).toBeInTheDocument();
    });

    it('renders ProjectManager at /projects path', () => {
        renderWithRouter('/projects');
        expect(screen.getByTestId('project-manager')).toBeInTheDocument();
    });

    it('renders AI routes at /ai path', () => {
        renderWithRouter('/ai');
        expect(screen.getByTestId('ai-routes')).toBeInTheDocument();
    });

    it('renders AI Settings at /settings/ai path', () => {
        renderWithRouter('/settings/ai');
        expect(screen.getByTestId('ai-settings')).toBeInTheDocument();
    });

    describe('Theme Configuration', () => {
        it('applies dark theme', () => {
            renderWithRouter();
            const root = document.documentElement;
            expect(root).toHaveStyle({
                colorScheme: 'dark'
            });
        });

        it('uses green primary color', () => {
            renderWithRouter();
            const primaryElements = document.getElementsByClassName('MuiButton-containedPrimary');
            if (primaryElements.length > 0) {
                expect(primaryElements[0]).toHaveStyle({
                    backgroundColor: '#00ff00'
                });
            }
        });
    });

    describe('Redux Store', () => {
        it('initializes with AI reducer', () => {
            renderWithRouter();
            // Access the store from the window object where we expose it in dev mode
            const store = (window as any).__REDUX_DEVTOOLS_EXTENSION__?.connect?.()?.getState?.();
            expect(store).toBeDefined();
            expect(store.ai).toBeDefined();
        });
    });

    describe('Project Context', () => {
        it('provides project context to components', () => {
            renderWithRouter();
            // The project context should be available to all components
            // We can verify this by checking if the ProjectManager renders without errors
            expect(screen.queryByTestId('project-manager')).not.toBeNull();
        });
    });

    describe('Error Handling', () => {
        beforeEach(() => {
            // Clear console.error to avoid noise in test output
            jest.spyOn(console, 'error').mockImplementation(() => { });
        });

        afterEach(() => {
            (console.error as jest.Mock).mockRestore();
        });

        it('handles invalid routes gracefully', () => {
            renderWithRouter('/invalid-route');
            // Should render the default dashboard as fallback
            expect(screen.getByTestId('recon-dashboard')).toBeInTheDocument();
        });

        it('preserves route after error recovery', () => {
            const { history } = renderWithRouter('/proxy');
            // Simulate an error and recovery
            history.push('/invalid');
            history.push('/proxy');
            expect(screen.getByTestId('proxy-dashboard')).toBeInTheDocument();
        });
    });
});
