import React from 'react';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { aiReducer } from '../../store/ai/reducer';
import { AIRoutes, isAIRoute, getCurrentAISection, AI_PATHS } from '../ai';

// Create test store
const createTestStore = () => configureStore({
    reducer: {
        ai: aiReducer
    }
});

// Wrapper component for testing
const TestWrapper: React.FC<{ children: React.ReactNode, initialEntry?: string }> = ({
    children,
    initialEntry = '/'
}) => (
    <Provider store={createTestStore()}>
        <MemoryRouter initialEntries={[initialEntry]}>
            {children}
        </MemoryRouter>
    </Provider>
);

describe('AI Routes', () => {
    describe('Route Rendering', () => {
        it('renders AI Hub at root path', () => {
            render(AIRoutes, { wrapper: (props) => <TestWrapper {...props} initialEntry="/ai" /> });
            expect(screen.getByText('AI Hub')).toBeInTheDocument();
        });

        it('renders Chat component at /chat path', () => {
            render(AIRoutes, { wrapper: (props) => <TestWrapper {...props} initialEntry="/ai/chat" /> });
            expect(screen.getByText('Start a conversation with the AI assistant')).toBeInTheDocument();
        });

        it('renders Analytics component at /analytics path', () => {
            render(AIRoutes, { wrapper: (props) => <TestWrapper {...props} initialEntry="/ai/analytics" /> });
            expect(screen.getByText('AI Analytics')).toBeInTheDocument();
        });

        it('renders Operations component at /operations path', () => {
            render(AIRoutes, { wrapper: (props) => <TestWrapper {...props} initialEntry="/ai/operations" /> });
            expect(screen.getByText('Model Training & Deployment')).toBeInTheDocument();
        });

        it('renders Monitoring component at /monitoring path', () => {
            render(AIRoutes, { wrapper: (props) => <TestWrapper {...props} initialEntry="/ai/monitoring" /> });
            expect(screen.getByText('System Status')).toBeInTheDocument();
        });
    });

    describe('Route Helpers', () => {
        describe('isAIRoute', () => {
            it('identifies AI routes correctly', () => {
                expect(isAIRoute('/ai')).toBe(true);
                expect(isAIRoute('/ai/chat')).toBe(true);
                expect(isAIRoute('/ai/analytics')).toBe(true);
                expect(isAIRoute('/settings/ai')).toBe(true);
                expect(isAIRoute('/projects')).toBe(false);
                expect(isAIRoute('/recon')).toBe(false);
            });
        });

        describe('getCurrentAISection', () => {
            it('returns correct section names', () => {
                expect(getCurrentAISection(AI_PATHS.root)).toBe('AI Hub');
                expect(getCurrentAISection(AI_PATHS.chat)).toBe('Chat');
                expect(getCurrentAISection(AI_PATHS.analytics)).toBe('Analytics');
                expect(getCurrentAISection(AI_PATHS.operations)).toBe('Operations');
                expect(getCurrentAISection(AI_PATHS.monitoring)).toBe('Monitoring');
                expect(getCurrentAISection(AI_PATHS.settings)).toBe('Settings');
                expect(getCurrentAISection('/unknown')).toBe('AI Hub');
            });
        });
    });

    describe('Path Constants', () => {
        it('has correct path definitions', () => {
            expect(AI_PATHS.root).toBe('/ai');
            expect(AI_PATHS.chat).toBe('/ai/chat');
            expect(AI_PATHS.analytics).toBe('/ai/analytics');
            expect(AI_PATHS.operations).toBe('/ai/operations');
            expect(AI_PATHS.monitoring).toBe('/ai/monitoring');
            expect(AI_PATHS.settings).toBe('/settings/ai');
        });
    });
});
