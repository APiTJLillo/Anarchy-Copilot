import { aiReducer } from '../reducer';
import {
    updateAISettingsAsync,
    resetAISettingsAsync,
    updateModelConfigAsync,
    addModelConfigAsync,
    deleteModelConfigAsync
} from '../actions';
import { initialState, defaultReasoningConfig, defaultCompletionConfig } from '../types';

describe('AI Settings Reducer', () => {
    it('should handle initial state', () => {
        expect(aiReducer(undefined, { type: 'unknown' })).toEqual(initialState);
    });

    describe('updateAISettingsAsync', () => {
        const mockSettings = {
            ...initialState.settings,
            defaultModel: "gpt-3.5-turbo"
        };

        it('should handle pending', () => {
            const nextState = aiReducer(initialState, updateAISettingsAsync.pending('', mockSettings));
            expect(nextState.loading).toBe(true);
            expect(nextState.error).toBeNull();
        });

        it('should handle fulfilled', () => {
            const nextState = aiReducer(initialState, updateAISettingsAsync.fulfilled(mockSettings, '', mockSettings));
            expect(nextState.settings).toEqual(mockSettings);
            expect(nextState.loading).toBe(false);
            expect(nextState.error).toBeNull();
        });

        it('should handle rejected', () => {
            const error = new Error('Failed to update');
            const nextState = aiReducer(initialState, updateAISettingsAsync.rejected(error, '', mockSettings));
            expect(nextState.loading).toBe(false);
            expect(nextState.error).toBe(error.message);
        });
    });

    describe('updateModelConfigAsync', () => {
        const mockModelId = "gpt-4";
        const mockConfig = {
            ...defaultReasoningConfig,
            temperature: 0.8
        };

        it('should handle fulfilled', () => {
            const nextState = aiReducer(initialState, updateModelConfigAsync.fulfilled(
                { modelId: mockModelId, config: mockConfig },
                '',
                { modelId: mockModelId, config: mockConfig }
            ));
            expect(nextState.settings.models[mockModelId]).toEqual(mockConfig);
            expect(nextState.loading).toBe(false);
            expect(nextState.error).toBeNull();
        });
    });

    describe('addModelConfigAsync', () => {
        const mockModelId = "claude-2";
        const mockConfig = {
            ...defaultCompletionConfig,
            model: "claude-2",
            temperature: 0.9
        };

        it('should handle fulfilled', () => {
            const nextState = aiReducer(initialState, addModelConfigAsync.fulfilled(
                { modelId: mockModelId, config: mockConfig },
                '',
                { modelId: mockModelId, config: mockConfig }
            ));
            expect(nextState.settings.models[mockModelId]).toEqual(mockConfig);
            expect(nextState.loading).toBe(false);
            expect(nextState.error).toBeNull();
        });
    });

    describe('deleteModelConfigAsync', () => {
        const mockModelId = "gpt-3.5-turbo";

        it('should handle fulfilled', () => {
            const nextState = aiReducer(initialState, deleteModelConfigAsync.fulfilled(
                mockModelId,
                '',
                mockModelId
            ));
            expect(nextState.settings.models[mockModelId]).toBeUndefined();
            expect(nextState.loading).toBe(false);
            expect(nextState.error).toBeNull();

            // Should set new default if deleted model was default
            if (initialState.settings.defaultModel === mockModelId) {
                expect(nextState.settings.defaultModel).not.toBe(mockModelId);
                expect(Object.keys(nextState.settings.models)).toContain(nextState.settings.defaultModel);
            }
        });

        it('should not allow deleting last model', () => {
            const stateWithOneModel = {
                ...initialState,
                settings: {
                    ...initialState.settings,
                    models: {
                        "gpt-4": defaultReasoningConfig
                    }
                }
            };

            const error = new Error('Cannot delete last model');
            const nextState = aiReducer(stateWithOneModel, deleteModelConfigAsync.rejected(error, '', "gpt-4"));
            expect(nextState.error).toBe(error.message);
            expect(Object.keys(nextState.settings.models)).toHaveLength(1);
        });
    });
});
