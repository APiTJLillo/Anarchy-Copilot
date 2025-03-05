import { configureStore } from '@reduxjs/toolkit';
import { aiReducer } from '../reducer';
import {
    updateAISettingsAsync,
    resetAISettingsAsync,
    updateModelConfigAsync,
    addModelConfigAsync,
    deleteModelConfigAsync
} from '../actions';
import { AISettings, defaultReasoningConfig, defaultCompletionConfig } from '../types';

const mockStore = configureStore({
    reducer: {
        ai: aiReducer
    },
    middleware: (getDefaultMiddleware) => getDefaultMiddleware()
});

type AppStore = typeof mockStore;
type RootState = ReturnType<typeof mockStore.getState>;

describe('AI Settings API Integration', () => {
    let store: AppStore;

    beforeEach(() => {
        fetchMock.resetMocks();
        store = configureStore({
            reducer: {
                ai: aiReducer
            },
            middleware: (getDefaultMiddleware) => getDefaultMiddleware()
        });
    });

    it('should update AI settings', async () => {
        const mockSettings: AISettings = {
            defaultModel: "gpt-4",
            models: {
                "gpt-4": defaultReasoningConfig,
                "gpt-3.5-turbo": defaultCompletionConfig
            },
            translationModel: "neural",
            autoDetectLanguage: true,
            enableCulturalContext: true,
            defaultRegion: "US",
            enableCache: true
        };

        global.fetch = jest.fn().mockImplementationOnce(() =>
            Promise.resolve({
                ok: true,
                json: () => Promise.resolve({ status: "success", settings: mockSettings })
            } as Response)
        );

        const resultAction = await store.dispatch(updateAISettingsAsync(mockSettings));
        expect(updateAISettingsAsync.fulfilled.match(resultAction)).toBeTruthy();

        const state = store.getState().ai;
        expect(state.settings).toEqual(mockSettings);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
    });

    it('should add new model configuration', async () => {
        const mockModelId = "claude-2";
        const mockConfig = {
            ...defaultCompletionConfig,
            model: mockModelId,
            temperature: 0.9
        };

        global.fetch = jest.fn().mockImplementationOnce(() =>
            Promise.resolve({
                ok: true,
                json: () => Promise.resolve({ status: "success", config: mockConfig })
            } as Response)
        );

        const resultAction = await store.dispatch(addModelConfigAsync({ modelId: mockModelId, config: mockConfig }));
        expect(addModelConfigAsync.fulfilled.match(resultAction)).toBeTruthy();

        const state = store.getState().ai;
        expect(state.settings.models[mockModelId]).toEqual(mockConfig);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
    });

    it('should handle API errors gracefully', async () => {
        global.fetch = jest.fn().mockImplementationOnce(() =>
            Promise.reject(new Error('Network error'))
        );

        const resultAction = await store.dispatch(updateAISettingsAsync(store.getState().ai.settings));
        expect(updateAISettingsAsync.rejected.match(resultAction)).toBeTruthy();

        const state = store.getState().ai;
        expect(state.loading).toBe(false);
        expect(state.error).toBe('Failed to update settings');
    });

    it('should handle model deletion with fallback default', async () => {
        const mockModelId = "gpt-3.5-turbo";
        global.fetch = jest.fn().mockImplementationOnce(() =>
            Promise.resolve({
                ok: true,
                json: () => Promise.resolve({ status: "success" })
            } as Response)
        );

        const resultAction = await store.dispatch(deleteModelConfigAsync(mockModelId));
        expect(deleteModelConfigAsync.fulfilled.match(resultAction)).toBeTruthy();

        const state = store.getState().ai;
        expect(state.settings.models[mockModelId]).toBeUndefined();

        if (state.settings.defaultModel === mockModelId) {
            expect(Object.keys(state.settings.models)).toContain(state.settings.defaultModel);
        }
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
    });

    it('should handle concurrent requests properly', async () => {
        const mockConfig1 = {
            ...defaultReasoningConfig,
            temperature: 0.8
        };
        const mockConfig2 = {
            ...defaultReasoningConfig,
            temperature: 0.9
        };

        global.fetch = jest.fn()
            .mockImplementationOnce(() => Promise.resolve({
                ok: true,
                json: () => Promise.resolve({ status: "success", config: mockConfig1 })
            } as Response))
            .mockImplementationOnce(() => Promise.resolve({
                ok: true,
                json: () => Promise.resolve({ status: "success", config: mockConfig2 })
            } as Response));

        const [action1, action2] = await Promise.all([
            store.dispatch(updateModelConfigAsync({ modelId: "gpt-4", config: mockConfig1 })),
            store.dispatch(updateModelConfigAsync({ modelId: "gpt-4", config: mockConfig2 }))
        ]);

        expect(updateModelConfigAsync.fulfilled.match(action1)).toBeTruthy();
        expect(updateModelConfigAsync.fulfilled.match(action2)).toBeTruthy();

        const state = store.getState().ai;
        expect(state.settings.models["gpt-4"]).toEqual(mockConfig2);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
    });
});
