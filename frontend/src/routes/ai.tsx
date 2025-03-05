import React from 'react';
import { Route, Routes } from 'react-router-dom';
import AIHub from '../components/ai/AIHub';
import AIChat from '../components/ai/chat/AIChat';
import AIAnalytics from '../components/ai/analytics/AIAnalytics';
import AIOperations from '../components/ai/operations/AIOperations';
import AIMonitoring from '../components/ai/monitoring/AIMonitoring';
import AISettings from '../components/settings/AISettings';

export const AIRoutes = (
    <Routes>
        <Route path="/" element={<AIHub />} />
        <Route path="/chat" element={<AIChat />} />
        <Route path="/analytics" element={<AIAnalytics />} />
        <Route path="/operations" element={<AIOperations />} />
        <Route path="/monitoring" element={<AIMonitoring />} />
        <Route path="/settings" element={<AISettings />} />
    </Routes>
);

// Path configuration for AI routes
export const AI_PATHS = {
    root: '/ai',
    chat: '/ai/chat',
    analytics: '/ai/analytics',
    operations: '/ai/operations',
    monitoring: '/ai/monitoring',
    settings: '/settings/ai',
} as const;

// Helper to check if a path is an AI route
export const isAIRoute = (path: string) => {
    return path.startsWith('/ai') || path === '/settings/ai';
};

// Current section helper
export const getCurrentAISection = (path: string) => {
    switch (path) {
        case AI_PATHS.chat:
            return 'Chat';
        case AI_PATHS.analytics:
            return 'Analytics';
        case AI_PATHS.operations:
            return 'Operations';
        case AI_PATHS.monitoring:
            return 'Monitoring';
        case AI_PATHS.settings:
            return 'Settings';
        default:
            return 'AI Hub';
    }
};
