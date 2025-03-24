import React from 'react';
import { render as rtlRender } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { aiReducer } from '../store/ai/reducer';
import type { AIState } from '../store/ai/types';
import type { RenderOptions, CustomRenderResult } from '../types/test-utils';

// Helper function to safely get attribute value
export const getAttributeSafe = (element: Element | null, attribute: string): string => {
    const value = element?.getAttribute(attribute);
    return value || '';
};

function render(
    ui: React.ReactElement,
    {
        initialState = {},
        store = configureStore({
            reducer: {
                ai: aiReducer
            },
            preloadedState: {
                ai: {
                    ...aiReducer(undefined, { type: 'INIT' }),
                    ...(initialState as Partial<AIState>)
                }
            }
        }),
        ...renderOptions
    }: RenderOptions = {}
): CustomRenderResult {
    function Wrapper({ children }: { children: React.ReactNode }) {
        return <Provider store={store}>{children}</Provider>;
    }

    const rendered = rtlRender(ui, {
        wrapper: Wrapper,
        ...renderOptions
    }) as CustomRenderResult;

    return {
        ...rendered,
        store,
        rerender: (ui: React.ReactElement) =>
            render(ui, {
                container: rendered.container,
                store,
                ...renderOptions
            })
    };
}

// Re-export everything
export * from '@testing-library/react';

// Override render method
export { render };

// Test Data Generators
export const createMockAIState = (overrides: Partial<AIState> = {}): AIState => ({
    settings: {
        defaultModel: "gpt-4",
        models: {
            "gpt-4": {
                model: "gpt-4",
                maxTokens: 2048,
                temperature: 0.7,
                topP: 1,
                frequencyPenalty: 0,
                presencePenalty: 0,
                reasoningCapability: true,
                reasoningEffort: 80,
                chainOfThought: true,
                selfReflection: true,
                responseFormat: 'structured',
                streamResponse: false,
                maxResponseSegments: 5,
                contextWindow: 8192,
                multilingualCapability: true,
                codeGeneration: true,
                cacheResults: true,
                timeoutMs: 30000,
                retryAttempts: 3,
                costPerToken: 0.0002,
                reasoningStrategy: 'step-by-step',
                verificationSteps: true,
                uncertaintyThreshold: 0.8,
                maxReasoningSteps: 5,
                feedbackLoop: true
            }
        },
        translationModel: "neural",
        autoDetectLanguage: true,
        enableCulturalContext: true,
        defaultRegion: "US",
        enableCache: true
    },
    loading: false,
    error: null,
    ...overrides
} as AIState);

// Mock API Response Generator
export const createMockAPIResponse = <T extends Record<string, unknown>>(data: T): Response => {
    const responseInit: ResponseInit = {
        status: 200,
        statusText: "OK",
        headers: new Headers({
            'Content-Type': 'application/json'
        })
    };

    const body = JSON.stringify({ status: "success", ...data });
    return new Response(body, responseInit);
};

// Mock Error Response Generator
export const createMockErrorResponse = (message: string): Response => {
    const responseInit: ResponseInit = {
        status: 400,
        statusText: "Bad Request",
        headers: new Headers({
            'Content-Type': 'application/json'
        })
    };

    const body = JSON.stringify({ status: "error", message });
    return new Response(body, responseInit);
};

// Wait for Loading State
export const waitForLoadingEnd = async (): Promise<void> => {
    await new Promise(resolve => setTimeout(resolve, 0));
};

// Mock Redux Store Type Helper
export type RootState = ReturnType<typeof aiReducer>;
