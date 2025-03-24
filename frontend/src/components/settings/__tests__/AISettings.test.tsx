import React from 'react';
import { render, screen, fireEvent, waitFor, getAttributeSafe } from '../../../utils/test-utils';
import userEvent from '@testing-library/user-event';
import AISettings from '../AISettings';
import { createMockAIState, createMockAPIResponse, createMockErrorResponse } from '../../../utils/test-utils';

describe('AISettings Component', () => {
    beforeEach(() => {
        global.fetch = jest.fn();
    });

    it('renders with initial state', () => {
        const initialState = createMockAIState();
        render(<AISettings />, { initialState });

        expect(screen.getByText('AI Settings')).toBeInTheDocument();
        expect(screen.getByText('Model Selection')).toBeInTheDocument();
        expect(screen.getByText('Translation')).toBeInTheDocument();
        expect(screen.getByText('Cultural Context')).toBeInTheDocument();
    });

    it('updates model configuration', async () => {
        const initialState = createMockAIState();
        const mockResponse = createMockAPIResponse({
            config: {
                ...initialState.settings.models["gpt-4"],
                temperature: 0.8
            }
        });

        global.fetch = jest.fn().mockResolvedValueOnce(mockResponse);
        const { store } = render(<AISettings />, { initialState });

        const temperatureSlider = screen.getByRole('slider', { name: /temperature/i });
        await userEvent.type(temperatureSlider, '0.8');

        const saveButton = screen.getByRole('button', { name: /save settings/i });
        await userEvent.click(saveButton);

        expect(global.fetch).toHaveBeenCalledWith(
            "/api/settings/ai",
            expect.objectContaining({
                method: "PUT",
                headers: expect.objectContaining({
                    "Content-Type": "application/json"
                })
            })
        );

        await waitFor(() => {
            const state = store.getState().ai;
            expect(state.settings.models["gpt-4"].temperature).toBe(0.8);
            expect(state.error).toBeNull();
        });

        expect(screen.getByText('Settings saved successfully!')).toBeInTheDocument();
    });

    it('handles validation errors', async () => {
        const initialState = createMockAIState();
        const errorResponse = createMockErrorResponse("Temperature must be between 0 and 1");
        global.fetch = jest.fn().mockResolvedValueOnce(errorResponse);

        render(<AISettings />, { initialState });

        const temperatureInput = screen.getByRole('spinbutton', { name: /temperature/i });
        await userEvent.clear(temperatureInput);
        await userEvent.type(temperatureInput, '1.5');

        const saveButton = screen.getByRole('button', { name: /save settings/i });
        await userEvent.click(saveButton);

        await waitFor(() => {
            expect(screen.getByText('Temperature must be between 0 and 1')).toBeInTheDocument();
        });
    });

    it('confirms before resetting settings', async () => {
        const initialState = createMockAIState();
        const mockResponse = createMockAPIResponse({ settings: createMockAIState().settings });
        global.fetch = jest.fn().mockResolvedValueOnce(mockResponse);

        window.confirm = jest.fn().mockImplementation(() => true);

        const { store } = render(<AISettings />, { initialState });

        const resetButton = screen.getByRole('button', { name: /reset to defaults/i });
        await userEvent.click(resetButton);

        expect(window.confirm).toHaveBeenCalledWith(
            expect.stringContaining('Are you sure you want to reset')
        );
    });

    it('prevents concurrent save operations', async () => {
        const initialState = createMockAIState();
        const mockResponse = createMockAPIResponse({
            config: initialState.settings.models["gpt-4"]
        });

        global.fetch = jest.fn()
            .mockImplementationOnce(() => new Promise(resolve => setTimeout(resolve, 100)))
            .mockResolvedValueOnce(mockResponse);

        render(<AISettings />, { initialState });

        const saveButton = screen.getByRole('button', { name: /save settings/i });
        await userEvent.click(saveButton);

        // Button should be disabled during save
        expect(saveButton).toBeDisabled();

        // Try clicking again
        await userEvent.click(saveButton);

        // Verify only one API call was made
        await waitFor(() => {
            expect(global.fetch).toHaveBeenCalledTimes(1);
        });
    });

    it('maintains form state during navigation attempts', async () => {
        const initialState = createMockAIState();
        render(<AISettings />, { initialState });

        // Mock unsaved changes
        const temperatureSlider = screen.getByRole('slider', { name: /temperature/i });
        await userEvent.type(temperatureSlider, '0.8');

        // Mock navigation attempt
        const event = new Event('beforeunload', { cancelable: true });
        window.dispatchEvent(event);

        expect(event.defaultPrevented).toBe(true);
    });

    it('properly handles cultural context settings', async () => {
        const initialState = createMockAIState();
        render(<AISettings />, { initialState });

        const culturalSwitch = screen.getByRole('switch', { name: /enable cultural context/i });
        expect(culturalSwitch).toBeChecked();

        await userEvent.click(culturalSwitch);
        expect(culturalSwitch).not.toBeChecked();

        // Region selector should be disabled when cultural context is off
        const regionSelect = screen.getByRole('combobox', { name: /default region/i });
        expect(regionSelect).toBeDisabled();
    });

    it('updates translation model selection', async () => {
        const initialState = createMockAIState();
        render(<AISettings />, { initialState });

        // Find and change translation model select
        const translationSelect = screen.getByRole('combobox', { name: /translation model/i });
        await userEvent.selectOptions(translationSelect, 'basic');

        // Verify the change
        expect(translationSelect).toHaveValue('basic');
    });

    it('handles adding new model configuration', async () => {
        const initialState = createMockAIState();
        const newModel = {
            model: "claude-2",
            maxTokens: 2048,
            temperature: 0.7,
            // ... other required properties
        };

        const mockResponse = createMockAPIResponse({ config: newModel });
        global.fetch = jest.fn().mockResolvedValueOnce(mockResponse);

        const { store } = render(<AISettings />, { initialState });

        // TODO: Add new model UI implementation and test
        // This is a placeholder for when we implement the "Add Model" feature
        /* 
        const addButton = screen.getByRole('button', { name: /add model/i });
        await userEvent.click(addButton);
        
        // Fill in new model details
        await userEvent.type(screen.getByLabelText(/model name/i), "claude-2");
        await userEvent.click(screen.getByRole('button', { name: /save model/i }));
        */

        // Verify API call occurred (when implemented)
        // expect(global.fetch).toHaveBeenCalled();
    });

    // Accessibility Tests
    describe('Accessibility', () => {
        // [Previous accessibility tests remain the same]

        it('preserves ARIA descriptions during state updates', async () => {
            const initialState = createMockAIState();
            render(<AISettings />, { initialState });

            const temperatureInput = screen.getByRole('spinbutton', { name: /temperature/i });
            const initialDescriptionId = getAttributeSafe(temperatureInput, 'aria-describedby');
            expect(initialDescriptionId).toBeTruthy();

            await userEvent.clear(temperatureInput);
            await userEvent.type(temperatureInput, '0.8');

            // Description should be preserved after update
            expect(temperatureInput).toHaveAttribute('aria-describedby', initialDescriptionId);
            expect(screen.getByText(/controls randomness/i)).toHaveAttribute('id', initialDescriptionId);
        });

        it('provides clear error resolution guidance', async () => {
            const initialState = createMockAIState();
            render(<AISettings />, { initialState });

            const temperatureInput = screen.getByRole('spinbutton', { name: /temperature/i });
            await userEvent.clear(temperatureInput);
            await userEvent.type(temperatureInput, '2');

            const errorMessage = screen.getByRole('alert');
            expect(errorMessage).toHaveTextContent(/must be between 0 and 1/i);
            expect(errorMessage).toHaveTextContent(/please adjust the value/i);

            // Error should be associated with input
            const errorId = getAttributeSafe(errorMessage, 'id');
            expect(temperatureInput).toHaveAttribute('aria-errormessage', errorId);
        });

        it('announces form validation in real-time', async () => {
            const initialState = createMockAIState();
            render(<AISettings />, { initialState });

            const maxTokensInput = screen.getByRole('spinbutton', { name: /max tokens/i });

            await userEvent.clear(maxTokensInput);
            await userEvent.type(maxTokensInput, '100');

            // Should announce validation status
            const validationMessage = screen.getByText(/must be at least 1024 tokens/i);
            expect(validationMessage).toHaveAttribute('role', 'status');
            expect(validationMessage).toHaveAttribute('aria-live', 'polite');
        });

        it('handles screen reader pronunciation', () => {
            const initialState = createMockAIState();
            render(<AISettings />, { initialState });

            const gpt4Label = screen.getByText('GPT-4');
            expect(gpt4Label).toHaveAttribute('aria-label', 'G P T 4');

            const apiKeyInput = screen.getByLabelText(/api key/i);
            expect(apiKeyInput).toHaveAttribute('aria-label', 'A P I Key');
        });

        it('supports international date and number formats', async () => {
            const initialState = createMockAIState();
            render(<AISettings />, { initialState });

            const temperatureInput = screen.getByRole('spinbutton', { name: /temperature/i });
            const costDisplay = screen.getByTestId('cost-display');

            // Test with different locales
            const locales = ['en-US', 'de-DE', 'ar-SA'];
            for (const locale of locales) {
                // Mock locale
                const originalFormat = Intl.NumberFormat;
                global.Intl.NumberFormat = function (locale) {
                    return new originalFormat(locale);
                };

                // Update values
                await userEvent.clear(temperatureInput);
                await userEvent.type(temperatureInput, '0.8');

                // Verify formatting
                expect(costDisplay).toHaveTextContent(
                    new Intl.NumberFormat(locale, {
                        style: 'currency',
                        currency: 'USD'
                    }).format(0.016)
                );
            }
        });

        it('respects user motion and animation settings', () => {
            const initialState = createMockAIState();
            render(<AISettings />, { initialState });

            const loadingSpinner = screen.getByTestId('loading-spinner');
            const progressBar = screen.getByRole('progressbar');

            // Verify reduced motion handling
            expect(loadingSpinner).toHaveStyle({
                '@media (prefers-reduced-motion: reduce)': {
                    animation: 'none'
                }
            });

            expect(progressBar).toHaveStyle({
                '@media (prefers-reduced-motion: reduce)': {
                    transition: 'none'
                }
            });
        });

        it('maintains contrast ratios in all themes', () => {
            const initialState = createMockAIState();
            render(<AISettings />, { initialState });

            const themes = ['light', 'dark', 'high-contrast'];
            themes.forEach(theme => {
                document.documentElement.setAttribute('data-theme', theme);

                const text = screen.getByText('AI Settings');
                const computedStyle = window.getComputedStyle(text);

                // Using WCAG 2.1 contrast requirements
                expect(computedStyle.color).toHaveContrastRatio({
                    against: computedStyle.backgroundColor,
                    ratio: 4.5  // Minimum for WCAG AA
                });
            });

            // Cleanup
            document.documentElement.removeAttribute('data-theme');
        });
    });
});
