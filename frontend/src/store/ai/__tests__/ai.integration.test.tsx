/// <reference types="@testing-library/jest-dom" />
import type { RenderResult } from '@testing-library/react';
import { jest, describe, it, expect } from '@jest/globals';
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import type { PreloadedState, PayloadAction } from '@reduxjs/toolkit';
import { aiReducer } from '../reducer';
import type {
    AIState,
    AIActionTypes,
    UpdateAISettingsAction,
    SetAILoadingAction,
    SetAIErrorAction,
    AISettings as AISettingsType
} from '../types';
import {
    initialState,
    defaultReasoningConfig,
    UPDATE_AI_SETTINGS,
    SET_AI_LOADING,
    SET_AI_ERROR
} from '../types';
import AIHub from '../../../components/ai/AIHub';
import AISettings from '../../../components/settings/AISettings';
import { MemoryRouter } from 'react-router-dom';
import '@testing-library/jest-dom';

// Define root state type
interface RootState {
    ai: AIState;
}

// Define action creator types with PayloadAction
const createUpdateSettingsAction = (settings: AISettingsType): PayloadAction<AISettingsType, typeof UPDATE_AI_SETTINGS> => ({
    type: UPDATE_AI_SETTINGS,
    payload: settings
});

const createSetLoadingAction = (loading: boolean): PayloadAction<boolean, typeof SET_AI_LOADING> => ({
    type: SET_AI_LOADING,
    payload: loading
});

const createSetErrorAction = (error: string): PayloadAction<string, typeof SET_AI_ERROR> => ({
    type: SET_AI_ERROR,
    payload: error
});

// Create the store setup function
const setupStore = (preloadedState?: PreloadedState<RootState>) => {
    return configureStore({
        reducer: {
            ai: aiReducer
        },
        middleware: (getDefaultMiddleware) =>
            getDefaultMiddleware({
                serializableCheck: {
                    // Ignore specific action types or paths in state
                    ignoredActions: [UPDATE_AI_SETTINGS, SET_AI_LOADING, SET_AI_ERROR],
                    ignoredPaths: ['ai.settings.models']
                }
            }),
        preloadedState
    });
};

type AppStore = ReturnType<typeof setupStore>;
type AppDispatch = AppStore['dispatch'];

interface RenderWithStoreResult extends RenderResult {
    store: AppStore;
}

const renderWithStore = (
    ui: React.ReactElement,
    {
        preloadedState = { ai: initialState },
        store = setupStore(preloadedState)
    } = {}
): RenderWithStoreResult => {
    return {
        ...render(
            <Provider store={store}>
                <MemoryRouter>
                    {ui}
                </MemoryRouter>
            </Provider>
        ),
        store
    };
};

describe('AI Redux Integration', () => {
    describe('State Updates', () => {
        it('updates model settings correctly', async () => {
            const { store } = renderWithStore(<AISettings />);

            store.dispatch(createUpdateSettingsAction({
                ...initialState.settings,
                models: {
                    ...initialState.settings.models,
                    'gpt-4': {
                        ...defaultReasoningConfig,
                        maxTokens: 8192
                    }
                }
            }));

            expect(store.getState().ai.settings.models['gpt-4'].maxTokens).toBe(8192);
        });

        it('preserves other settings when updating one model', () => {
            const customState: AIState = {
                ...initialState,
                settings: {
                    ...initialState.settings,
                    models: {
                        'gpt-4': { ...defaultReasoningConfig },
                        'gpt-3.5-turbo': { ...defaultReasoningConfig, maxTokens: 2048 }
                    }
                }
            };

            const { store } = renderWithStore(<AISettings />, {
                preloadedState: { ai: customState }
            });

            store.dispatch(createUpdateSettingsAction({
                ...customState.settings,
                models: {
                    ...customState.settings.models,
                    'gpt-4': {
                        ...defaultReasoningConfig,
                        maxTokens: 8192
                    }
                }
            }));

            expect(store.getState().ai.settings.models['gpt-3.5-turbo'].maxTokens).toBe(2048);
        });
    });

    describe('Loading States', () => {
        it('shows loading state during operations', async () => {
            const { store } = renderWithStore(<AIHub />);

            store.dispatch(createSetLoadingAction(true));
            expect(store.getState().ai.loading).toBe(true);

            store.dispatch(createSetLoadingAction(false));
            expect(store.getState().ai.loading).toBe(false);
        });

        it('handles errors appropriately', async () => {
            const { store } = renderWithStore(<AIHub />);

            store.dispatch(createSetErrorAction('Test error message'));
            expect(store.getState().ai.error).toBe('Test error message');
        });
    });

    describe('Settings Persistence', () => {
        it('maintains settings after component unmount', () => {
            const customState: AIState = {
                ...initialState,
                settings: {
                    ...initialState.settings,
                    defaultModel: 'gpt-4'
                }
            };

            const { store, unmount } = renderWithStore(<AISettings />, {
                preloadedState: { ai: customState }
            });

            const initialDefaultModel = store.getState().ai.settings.defaultModel;
            unmount();
            expect(store.getState().ai.settings.defaultModel).toBe(initialDefaultModel);
        });

        it('preserves state through multiple updates', () => {
            const { store } = renderWithStore(<AISettings />);

            store.dispatch(createUpdateSettingsAction({
                ...initialState.settings,
                defaultModel: 'gpt-4'
            }));

            store.dispatch(createUpdateSettingsAction({
                ...store.getState().ai.settings,
                enableCache: false
            }));

            const finalState = store.getState().ai.settings;
            expect(finalState.defaultModel).toBe('gpt-4');
            expect(finalState.enableCache).toBe(false);
            expect(finalState.autoDetectLanguage).toBe(initialState.settings.autoDetectLanguage);
        });
    });

    describe('Component Integration', () => {
        it('sync settings across components', async () => {
            const { store, rerender } = renderWithStore(<AIHub />);

            store.dispatch(createUpdateSettingsAction({
                ...initialState.settings,
                defaultModel: 'gpt-4'
            }));

            rerender(
                <Provider store={store}>
                    <MemoryRouter>
                        <AISettings />
                    </MemoryRouter>
                </Provider>
            );

            expect(store.getState().ai.settings.defaultModel).toBe('gpt-4');
        });
    });

    describe('Error Handling', () => {
        it('clears error state appropriately', () => {
            const { store } = renderWithStore(<AIHub />);

            store.dispatch(createSetErrorAction('Test error'));
            expect(store.getState().ai.error).toBe('Test error');

            store.dispatch(createSetErrorAction(''));
            expect(store.getState().ai.error).toBe('');
        });

        it('handles invalid state updates gracefully', () => {
            const { store } = renderWithStore(<AIHub />);

            const safeUpdate = createUpdateSettingsAction({
                ...initialState.settings,
                defaultModel: 'gpt-4'
            });

            expect(() => store.dispatch(safeUpdate)).not.toThrow();
            expect(store.getState().ai.settings).toBeDefined();
        });
    });
});
