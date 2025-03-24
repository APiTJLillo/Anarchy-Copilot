import { createReducer } from '@reduxjs/toolkit';
import { AIState, initialState } from "./types";
import {
    updateAISettingsAsync,
    resetAISettingsAsync,
    updateModelConfigAsync,
    addModelConfigAsync,
    deleteModelConfigAsync
} from './actions';

export const aiReducer = createReducer(initialState, (builder) => {
    builder
        // Update Settings
        .addCase(updateAISettingsAsync.pending, (state) => {
            state.loading = true;
            state.error = null;
        })
        .addCase(updateAISettingsAsync.fulfilled, (state, action) => {
            state.settings = action.payload;
            state.loading = false;
            state.error = null;
        })
        .addCase(updateAISettingsAsync.rejected, (state, action) => {
            state.loading = false;
            state.error = action.error.message || "Failed to update settings";
        })

        // Reset Settings
        .addCase(resetAISettingsAsync.pending, (state) => {
            state.loading = true;
            state.error = null;
        })
        .addCase(resetAISettingsAsync.fulfilled, (state, action) => {
            state.settings = action.payload;
            state.loading = false;
            state.error = null;
        })
        .addCase(resetAISettingsAsync.rejected, (state, action) => {
            state.loading = false;
            state.error = action.error.message || "Failed to reset settings";
        })

        // Update Model Config
        .addCase(updateModelConfigAsync.pending, (state) => {
            state.loading = true;
            state.error = null;
        })
        .addCase(updateModelConfigAsync.fulfilled, (state, action) => {
            const { modelId, config } = action.payload;
            state.settings.models[modelId] = config;
            state.loading = false;
            state.error = null;
        })
        .addCase(updateModelConfigAsync.rejected, (state, action) => {
            state.loading = false;
            state.error = action.error.message || "Failed to update model configuration";
        })

        // Add Model Config
        .addCase(addModelConfigAsync.pending, (state) => {
            state.loading = true;
            state.error = null;
        })
        .addCase(addModelConfigAsync.fulfilled, (state, action) => {
            const { modelId, config } = action.payload;
            state.settings.models[modelId] = config;
            state.loading = false;
            state.error = null;
        })
        .addCase(addModelConfigAsync.rejected, (state, action) => {
            state.loading = false;
            state.error = action.error.message || "Failed to add model configuration";
        })

        // Delete Model Config
        .addCase(deleteModelConfigAsync.pending, (state) => {
            state.loading = true;
            state.error = null;
        })
        .addCase(deleteModelConfigAsync.fulfilled, (state, action) => {
            const modelId = action.payload;
            delete state.settings.models[modelId];

            // If the deleted model was the default, set a new default
            if (state.settings.defaultModel === modelId) {
                const remainingModels = Object.keys(state.settings.models);
                if (remainingModels.length > 0) {
                    state.settings.defaultModel = remainingModels[0];
                }
            }

            state.loading = false;
            state.error = null;
        })
        .addCase(deleteModelConfigAsync.rejected, (state, action) => {
            state.loading = false;
            state.error = action.error.message || "Failed to delete model configuration";
        });
});
