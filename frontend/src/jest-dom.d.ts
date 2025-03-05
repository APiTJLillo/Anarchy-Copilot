/// <reference types="@testing-library/jest-dom" />

declare namespace jest {
    interface Matchers<R> {
        toBeInTheDocument(): R;
        toHaveTextContent(text: string | RegExp): R;
        toHaveAttribute(attr: string, value?: string): R;
        toBeVisible(): R;
        toBeDisabled(): R;
        toBeEnabled(): R;
        toHaveClass(className: string): R;
        toHaveValue(value: string | string[] | number): R;
        toBeChecked(): R;
    }
}
