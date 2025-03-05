import { createAsyncThunk } from '@reduxjs/toolkit';
import { AISettings, BaseModelConfig, ReasoningModelConfig, CompletionModelConfig } from "./types";

export const updateAISettingsAsync = createAsyncThunk(
    'ai/updateSettings',
    async (settings: AISettings) => {
        const response = await fetch("/api/settings/ai", {
            method: "PUT",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(settings)
        });

        if (!response.ok) {
            throw new Error("Failed to update AI settings");
        }

        return settings;
    }
);

export const resetAISettingsAsync = createAsyncThunk(
    'ai/resetSettings',
    async () => {
        const response = await fetch("/api/settings/ai/reset", {
            method: "POST"
        });

        if (!response.ok) {
            throw new Error("Failed to reset AI settings");
        }

        const data = await response.json();
        return data.settings;
    }
);

export const updateModelConfigAsync = createAsyncThunk(
    'ai/updateModelConfig',
    async ({ modelId, config }: {
        modelId: string,
        config: BaseModelConfig | ReasoningModelConfig | CompletionModelConfig
    }) => {
        const response = await fetch(`/api/settings/ai/models/${modelId}`, {
            method: "PUT",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(config)
        });

        if (!response.ok) {
            throw new Error("Failed to update model configuration");
        }

        const data = await response.json();
        return { modelId, config: data.config };
    }
);

export const addModelConfigAsync = createAsyncThunk(
    'ai/addModelConfig',
    async ({ modelId, config }: {
        modelId: string,
        config: BaseModelConfig | ReasoningModelConfig | CompletionModelConfig
    }) => {
        const response = await fetch(`/api/settings/ai/models`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                model_id: modelId,
                ...config
            })
        });

        if (!response.ok) {
            throw new Error("Failed to add model configuration");
        }

        const data = await response.json();
        return { modelId, config: data.config };
    }
);

export const deleteModelConfigAsync = createAsyncThunk(
    'ai/deleteModelConfig',
    async (modelId: string) => {
        const response = await fetch(`/api/settings/ai/models/${modelId}`, {
            method: "DELETE"
        });

        if (!response.ok) {
            throw new Error("Failed to delete model configuration");
        }

        return modelId;
    }
);
