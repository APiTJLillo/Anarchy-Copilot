/// <reference types="@testing-library/jest-dom" />

declare namespace jest {
    interface Matchers<R> {
        toBeInTheDocument(): R;
        toHaveStyle(style: { [key: string]: any }): R;
        toHaveClass(className: string): R;
        toBeVisible(): R;
        toBeDisabled(): R;
        toHaveValue(value: string | number | string[]): R;
        toHaveTextContent(text: string | RegExp): R;
        toBeChecked(): R;
        toHaveAttribute(attr: string, value?: string): R;
    }
}

// Mock WebSocket for testing
declare class MockWebSocket implements WebSocket {
    static CONNECTING: 0;
    static OPEN: 1;
    static CLOSING: 2;
    static CLOSED: 3;

    CONNECTING: 0;
    OPEN: 1;
    CLOSING: 2;
    CLOSED: 3;

    url: string;
    readyState: number;
    bufferedAmount: number;
    extensions: string;
    protocol: string;
    binaryType: BinaryType;

    onopen: ((this: WebSocket, ev: Event) => any) | null;
    onclose: ((this: WebSocket, ev: CloseEvent) => any) | null;
    onmessage: ((this: WebSocket, ev: MessageEvent) => any) | null;
    onerror: ((this: WebSocket, ev: Event) => any) | null;

    constructor(url: string, protocols?: string | string[]);
    close(code?: number, reason?: string): void;
    send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void;
    addEventListener<K extends keyof WebSocketEventMap>(
        type: K,
        listener: (this: WebSocket, ev: WebSocketEventMap[K]) => any,
        options?: boolean | AddEventListenerOptions
    ): void;
    addEventListener(
        type: string,
        listener: EventListenerOrEventListenerObject,
        options?: boolean | AddEventListenerOptions
    ): void;
    removeEventListener<K extends keyof WebSocketEventMap>(
        type: K,
        listener: (this: WebSocket, ev: WebSocketEventMap[K]) => any,
        options?: boolean | EventListenerOptions
    ): void;
    removeEventListener(
        type: string,
        listener: EventListenerOrEventListenerObject,
        options?: boolean | EventListenerOptions
    ): void;
}

declare global {
    interface Window {
        WebSocket: typeof MockWebSocket;
    }

    // Add any other global interfaces needed for testing
    interface Console {
        warn(...data: any[]): void;
        group(...data: any[]): void;
        groupEnd(): void;
    }
}

// Module declarations for asset imports in tests
declare module '*.svg' {
    import * as React from 'react';
    export const ReactComponent: React.FunctionComponent<React.SVGProps<SVGElement>>;
    const src: string;
    export default src;
}

declare module '*.png' {
    const content: string;
    export default content;
}

declare module '*.jpg' {
    const content: string;
    export default content;
}

declare module '*.jpeg' {
    const content: string;
    export default content;
}

declare module '*.gif' {
    const content: string;
    export default content;
}

declare module '*.json' {
    const content: any;
    export default content;
}
