import { Store } from '@reduxjs/toolkit';
import { RenderResult } from '@testing-library/react';
import { AIState } from '../store/ai/types';

declare global {
    interface Window {
        store: Store;
    }

    namespace NodeJS {
        interface Global {
            fetch: typeof fetch;
            fetchMock: FetchMock;
        }
    }

    namespace jest {
        interface Matchers<R> {
            toBeInTheDocument(): R;
            toHaveTextContent(text: string): R;
            toHaveValue(value: string | number | string[]): R;
            toHaveBeenCalledWith(expected: unknown): R;
            toBeTruthy(): R;
            toBeNull(): R;
            toBe(expected: unknown): R;
            toEqual(expected: unknown): R;
        }
    }

    interface FetchMock extends jest.Mock {
        mockResponse: (body: string) => FetchMock;
        mockResponseOnce: (body: string) => FetchMock;
        mockReject: (error: Error) => FetchMock;
        mockRejectOnce: (error: Error) => FetchMock;
        resetMocks: () => void;
        mockImplementation: (fn: () => any) => FetchMock;
        mockImplementationOnce: (fn: () => any) => FetchMock;
    }

    var fetchMock: FetchMock;

    interface Response {
        ok: boolean;
        json(): Promise<any>;
        text(): Promise<string>;
        blob(): Promise<Blob>;
        arrayBuffer(): Promise<ArrayBuffer>;
        formData(): Promise<FormData>;
        headers: Headers;
        status: number;
        statusText: string;
        url: string;
        type: ResponseType;
        body: ReadableStream<Uint8Array> | null;
        bodyUsed: boolean;
        clone(): Response;
    }
}

// Test Utilities
interface RenderOptions {
    initialState?: Partial<AIState>;
    store?: Store;
    baseElement?: HTMLElement;
    container?: HTMLElement;
    hydrate?: boolean;
    wrapper?: React.ComponentType;
}

interface CustomRenderResult extends RenderResult {
    store: Store;
    rerender: (ui: React.ReactElement) => void;
}

declare module '@testing-library/react' {
    function render(
        ui: React.ReactElement,
        options?: RenderOptions
    ): CustomRenderResult;
}

// Asset modules
declare module '*.css' {
    const content: { [className: string]: string };
    export default content;
}

declare module '*.scss' {
    const content: { [className: string]: string };
    export default content;
}

declare module '*.svg' {
    import * as React from 'react';
    export const ReactComponent: React.FunctionComponent<React.SVGProps<SVGSVGElement>>;
    const src: string;
    export default src;
}

// Add any additional module declarations here
declare module '@reduxjs/toolkit' {
    export interface SerializedError {
        name?: string;
        message?: string;
        code?: string;
        stack?: string;
    }
}

export type { RenderOptions, CustomRenderResult };
