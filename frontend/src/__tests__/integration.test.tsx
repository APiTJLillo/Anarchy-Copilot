import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryHistory } from 'history';
import { Router } from 'react-router-dom';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { aiReducer } from '../store/ai/reducer';
import { defaultReasoningConfig, AIState, initialState } from '../store/ai/types';
import App from '../App';

// Create a custom renderer with all providers
const renderWithProviders = (
    ui: React.ReactElement,
    {
        initialRoute = '/',
        preloadedState = initialState,
        store = configureStore({
            reducer: {
                ai: aiReducer
            },
            preloadedState: {
                ai: preloadedState
            }
        })
    } = {}
) => {
    const history = createMemoryHistory({ initialEntries: [initialRoute] });
    return {
        ...render(
            <Provider store={store}>
                <Router location={history.location} navigator={history}>
                    {ui}
                </Router>
            </Provider>
        ),
        history,
        store
    };
};

describe('Integration Tests', () => {
    describe('Navigation Flow', () => {
        it('navigates between AI features correctly', async () => {
            const { history } = renderWithProviders(<App />, { initialRoute: '/ai' });

            // Should start at AI Hub
            expect(screen.getByTestId('ai-routes')).toBeInTheDocument();

            // Navigate to Chat
            history.push('/ai/chat');
            await waitFor(() => {
                expect(screen.getByText('Start a conversation with the AI assistant')).toBeInTheDocument();
            });

            // Navigate to Analytics
            history.push('/ai/analytics');
            await waitFor(() => {
                expect(screen.getByText('AI Analytics')).toBeInTheDocument();
            });

            // Navigate to Operations
            history.push('/ai/operations');
            await waitFor(() => {
                expect(screen.getByText('Model Training & Deployment')).toBeInTheDocument();
            });

            // Navigate to Monitoring
            history.push('/ai/monitoring');
            await waitFor(() => {
                expect(screen.getByText('System Status')).toBeInTheDocument();
            });
        });

        it('preserves navigation state during page refreshes', () => {
            const { history } = renderWithProviders(<App />, { initialRoute: '/ai/chat' });
            expect(screen.getByText('Start a conversation with the AI assistant')).toBeInTheDocument();

            // Simulate page refresh by re-rendering at the same route
            history.replace(history.location);
            expect(screen.getByText('Start a conversation with the AI assistant')).toBeInTheDocument();
        });
    });

    describe('State Management', () => {
        it('maintains AI settings across navigation', async () => {
            const customState: AIState = {
                ...initialState,
                settings: {
                    ...initialState.settings,
                    models: {
                        ...initialState.settings.models,
                        "gpt-4": {
                            ...defaultReasoningConfig,
                            maxTokens: 4096
                        }
                    }
                }
            };

            const { history, store } = renderWithProviders(<App />, {
                preloadedState: customState
            });

            // Go to AI Settings
            history.push('/settings/ai');
            await waitFor(() => {
                expect(screen.getByTestId('ai-settings')).toBeInTheDocument();
            });

            // Navigate away and back
            history.push('/ai/chat');
            history.push('/settings/ai');

            // Settings should be preserved
            expect(store.getState().ai.settings.models['gpt-4'].maxTokens).toBe(4096);
        });
    });

    describe('Error Boundaries', () => {
        beforeEach(() => {
            jest.spyOn(console, 'error').mockImplementation(() => { });
        });

        afterEach(() => {
            (console.error as jest.Mock).mockRestore();
        });

        it('recovers from component errors', async () => {
            const { history } = renderWithProviders(<App />);

            // Force an error by navigating to an invalid route
            history.push('/invalid-route');

            // Should recover by showing the default route
            await waitFor(() => {
                expect(screen.getByTestId('recon-dashboard')).toBeInTheDocument();
            });
        });

        it('maintains state after error recovery', async () => {
            const customState: AIState = {
                ...initialState,
                settings: {
                    ...initialState.settings,
                    defaultModel: "gpt-4"
                }
            };

            const { store, history } = renderWithProviders(<App />, {
                preloadedState: customState
            });

            // Force an error and recover
            history.push('/invalid-route');
            history.push('/ai');

            // State should be preserved
            expect(store.getState().ai.settings.defaultModel).toBe('gpt-4');
        });
    });

    describe('Theme and Styling', () => {
        it('applies theme consistently across routes', async () => {
            const { history } = renderWithProviders(<App />);

            // Check theme on multiple routes
            const routes = ['/ai', '/ai/chat', '/ai/analytics', '/settings/ai'];
            for (const route of routes) {
                history.push(route);
                await waitFor(() => {
                    const root = document.documentElement;
                    expect(root).toHaveStyle({ colorScheme: 'dark' });
                });
            }
        });

        it('maintains responsive layout across routes', async () => {
            const { history } = renderWithProviders(<App />);

            // Check layout on different routes
            const routes = ['/ai', '/ai/chat', '/ai/analytics'];
            for (const route of routes) {
                history.push(route);
                await waitFor(() => {
                    const mainContent = document.querySelector('main');
                    expect(mainContent).toHaveStyle({
                        flexGrow: 1,
                        padding: '24px'
                    });
                });
            }
        });
    });

    describe('Performance', () => {
        it('lazy loads AI components', async () => {
            const { history } = renderWithProviders(<App />);

            // Navigate to AI route
            history.push('/ai');

            // Component should load without blocking
            await waitFor(() => {
                expect(screen.getByTestId('ai-routes')).toBeInTheDocument();
            });
        });
    });
});
