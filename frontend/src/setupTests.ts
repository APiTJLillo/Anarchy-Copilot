import '@testing-library/jest-dom';
import { jest, expect } from '@jest/globals';

// Ambient declarations
declare global {
    namespace jest {
        interface Matchers<R> {
            toHaveContrastRatio(options: { against: string; ratio: number }): R;
        }
    }
}

// Create mock response
const createResponse = (body: any = {}): Response => ({
    ok: true,
    status: 200,
    statusText: "OK",
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
    blob: () => Promise.resolve(new Blob()),
    arrayBuffer: () => Promise.resolve(new ArrayBuffer(0)),
    formData: () => Promise.resolve(new FormData()),
    headers: new Headers(),
    body: null,
    bodyUsed: false,
    url: "http://localhost",
    type: "basic",
    redirected: false,
    clone() { return createResponse(body); }
} as Response);

// Create fetch mock
const mockFetch = Object.assign(jest.fn(() => Promise.resolve(createResponse())), {
    mockResponse(body: string) {
        this.mockImplementation(() => Promise.resolve(createResponse(JSON.parse(body))));
        return this;
    },
    mockResponseOnce(body: string) {
        this.mockImplementationOnce(() => Promise.resolve(createResponse(JSON.parse(body))));
        return this;
    },
    mockReject(error: Error) {
        this.mockImplementation(() => Promise.reject(error));
        return this;
    },
    mockRejectOnce(error: Error) {
        this.mockImplementationOnce(() => Promise.reject(error));
        return this;
    },
    resetMocks() {
        this.mockReset();
    }
});

// Assign mock to globals
const g = globalThis as any;
g.fetch = mockFetch;
g.fetchMock = mockFetch;

// Helper functions for color contrast
const hexToRgb = (hex: string) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
};

const getLuminance = (r: number, g: number, b: number) => {
    const [rs, gs, bs] = [r / 255, g / 255, b / 255].map(c => {
        return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
};

const getContrastRatio = (color1: string, color2: string) => {
    const rgb1 = hexToRgb(color1);
    const rgb2 = hexToRgb(color2);

    if (!rgb1 || !rgb2) return 1;

    const l1 = getLuminance(rgb1.r, rgb1.g, rgb1.b);
    const l2 = getLuminance(rgb2.r, rgb2.g, rgb2.b);

    const lighter = Math.max(l1, l2);
    const darker = Math.min(l1, l2);

    return (lighter + 0.05) / (darker + 0.05);
};

// Add custom matcher
expect.extend({
    toHaveContrastRatio(received: string, { against, ratio }: { against: string; ratio: number }) {
        const actualRatio = getContrastRatio(received, against);
        const pass = actualRatio >= ratio;

        return {
            pass,
            message: () =>
                `Expected color ${received} to ${pass ? 'not ' : ''}have contrast ratio of at least ${ratio} with ${against}. Actual ratio: ${actualRatio.toFixed(2)}`,
        };
    }
});
