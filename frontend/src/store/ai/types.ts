// Base Model Configuration
export interface BaseModelConfig {
    model: string;
    apiKey?: string;
    maxTokens: number;
    temperature: number;
    topP: number;
    frequencyPenalty: number;
    presencePenalty: number;

    // Reasoning settings
    reasoningCapability: boolean;
    reasoningEffort: number;  // 0-100 scale
    chainOfThought: boolean;
    selfReflection: boolean;

    // Response settings
    responseFormat: 'text' | 'json' | 'structured';
    streamResponse: boolean;
    maxResponseSegments: number;

    // Specialized capabilities
    contextWindow: number;
    embeddingDimension?: number;
    multilingualCapability: boolean;
    codeGeneration: boolean;

    // Performance settings
    cacheResults: boolean;
    timeoutMs: number;
    retryAttempts: number;

    // Cost management  
    costPerToken: number;
    budgetLimit?: number;

    // Task-specific tuning
    defaultPersona?: string;
    domainExpertise?: string[];
    customPromptPrefix?: string;
}

export interface ReasoningModelConfig extends BaseModelConfig {
    reasoningStrategy: 'step-by-step' | 'tree-of-thought' | 'parallel';
    verificationSteps: boolean;
    uncertaintyThreshold: number;
    maxReasoningSteps: number;
    feedbackLoop: boolean;
}

export interface CompletionModelConfig extends BaseModelConfig {
    completionStyle: 'creative' | 'precise' | 'balanced';
    stopSequences: string[];
    biasTokens: Record<string, number>;
    logitBias?: Record<string, number>;
}

// AI Settings State
export interface AISettings {
    defaultModel: string;
    models: Record<string, BaseModelConfig | ReasoningModelConfig | CompletionModelConfig>;
    translationModel: "neural" | "basic";
    autoDetectLanguage: boolean;
    enableCulturalContext: boolean;
    defaultRegion: string;
    enableCache: boolean;
}

export interface AIState {
    settings: AISettings;
    loading: boolean;
    error: string | null;
}

// Action Types
export const UPDATE_AI_SETTINGS = "UPDATE_AI_SETTINGS";
export const UPDATE_MODEL_CONFIG = "UPDATE_MODEL_CONFIG";
export const ADD_MODEL_CONFIG = "ADD_MODEL_CONFIG";
export const REMOVE_MODEL_CONFIG = "REMOVE_MODEL_CONFIG";
export const RESET_AI_SETTINGS = "RESET_AI_SETTINGS";
export const SET_AI_LOADING = "SET_AI_LOADING";
export const SET_AI_ERROR = "SET_AI_ERROR";

// Action Interfaces
export interface UpdateAISettingsAction {
    type: typeof UPDATE_AI_SETTINGS;
    payload: AISettings;
}

export interface UpdateModelConfigAction {
    type: typeof UPDATE_MODEL_CONFIG;
    payload: {
        modelId: string;
        config: BaseModelConfig | ReasoningModelConfig | CompletionModelConfig;
    };
}

export interface AddModelConfigAction {
    type: typeof ADD_MODEL_CONFIG;
    payload: {
        modelId: string;
        config: BaseModelConfig | ReasoningModelConfig | CompletionModelConfig;
    };
}

export interface RemoveModelConfigAction {
    type: typeof REMOVE_MODEL_CONFIG;
    payload: string; // modelId
}

export interface ResetAISettingsAction {
    type: typeof RESET_AI_SETTINGS;
}

export interface SetAILoadingAction {
    type: typeof SET_AI_LOADING;
    payload: boolean;
}

export interface SetAIErrorAction {
    type: typeof SET_AI_ERROR;
    payload: string;
}

// Default configurations
export const defaultReasoningConfig: ReasoningModelConfig = {
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
};

export const defaultCompletionConfig: CompletionModelConfig = {
    model: "gpt-3.5-turbo",
    maxTokens: 1024,
    temperature: 0.9,
    topP: 0.9,
    frequencyPenalty: 0.2,
    presencePenalty: 0.2,
    reasoningCapability: false,
    reasoningEffort: 40,
    chainOfThought: false,
    selfReflection: false,
    responseFormat: 'text',
    streamResponse: true,
    maxResponseSegments: 3,
    contextWindow: 4096,
    multilingualCapability: true,
    codeGeneration: false,
    cacheResults: true,
    timeoutMs: 15000,
    retryAttempts: 2,
    costPerToken: 0.00002,
    completionStyle: 'creative',
    stopSequences: ["\n\n", "###"],
    biasTokens: {}
};

// Initial State
export const initialState: AIState = {
    settings: {
        defaultModel: "gpt-4",
        models: {
            "gpt-4": defaultReasoningConfig,
            "gpt-3.5-turbo": defaultCompletionConfig,
        },
        translationModel: "neural",
        autoDetectLanguage: true,
        enableCulturalContext: true,
        defaultRegion: "US",
        enableCache: true
    },
    loading: false,
    error: null
};

export type AIActionTypes =
    | UpdateAISettingsAction
    | UpdateModelConfigAction
    | AddModelConfigAction
    | RemoveModelConfigAction
    | ResetAISettingsAction
    | SetAILoadingAction
    | SetAIErrorAction;
