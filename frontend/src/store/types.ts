import { ThunkAction, ThunkDispatch } from 'redux-thunk';
import { AnyAction } from 'redux';
import { AIState } from './ai/types';

export interface RootState {
    ai: AIState;
    // Add other state slices here
}

export type AppThunk<ReturnType = void> = ThunkAction<
    ReturnType,
    RootState,
    unknown,
    AnyAction
>;

export type AppDispatch = ThunkDispatch<RootState, unknown, AnyAction>;
