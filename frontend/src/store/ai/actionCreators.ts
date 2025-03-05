import { createAction } from '@reduxjs/toolkit';
import { AISettings } from './types';

// Create strongly-typed action creators
export const updateAISettings = createAction<AISettings>('UPDATE_AI_SETTINGS');
export const resetAISettings = createAction<AISettings>('RESET_AI_SETTINGS');
export const setAILoading = createAction<boolean>('SET_AI_LOADING');
export const setAIError = createAction<string>('SET_AI_ERROR');
