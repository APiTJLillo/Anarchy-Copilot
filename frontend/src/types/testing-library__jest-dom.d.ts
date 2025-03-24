import '@testing-library/jest-dom';

declare global {
    namespace jest {
        interface Matchers<R> {
            toBeInTheDocument(): R;
            toHaveTextContent(text: string | RegExp): R;
            toHaveAttribute(attr: string, value?: string | number): R;
            toBeVisible(): R;
            toBeDisabled(): R;
            toBeEnabled(): R;
            toHaveClass(...classNames: string[]): R;
            toHaveStyle(css: Record<string, any>): R;
            toHaveValue(value: string | string[] | number): R;
            toBeChecked(): R;
            toBeEmpty(): R;
            toContainElement(element: HTMLElement | null): R;
            toContainHTML(htmlText: string): R;
            toBeRequired(): R;
            toBeValid(): R;
            toBeInvalid(): R;
            toHaveFocus(): R;
            toBePartiallyChecked(): R;
        }
    }
}
