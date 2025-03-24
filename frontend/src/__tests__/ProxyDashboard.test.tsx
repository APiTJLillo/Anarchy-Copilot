import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import { ProxyDashboard } from '../ProxyDashboard';
import { API_BASE_URL } from '../config';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('ProxyDashboard', () => {
    const mockStatus = {
        isRunning: false,
        interceptRequests: true,
        interceptResponses: true,
        allowedHosts: [],
        excludedHosts: [],
        history: []
    };

    const mockAnalysisResults = [
        {
            requestId: 'test-1',
            ruleName: 'SQL Injection',
            severity: 'high',
            description: 'Possible SQL injection detected',
            evidence: "UNION SELECT",
            timestamp: new Date().toISOString()
        }
    ];

    beforeEach(() => {
        // Reset mock status before each test
        mockedAxios.get.mockImplementation((url) => {
            if (url === `${API_BASE_URL}/api/proxy/status`) {
                return Promise.resolve({ data: mockStatus });
            }
            if (url === `${API_BASE_URL}/analysis/results`) {
                return Promise.resolve({ data: mockAnalysisResults });
            }
            return Promise.reject(new Error('Not found'));
        });
    });

    it('renders loading state initially', () => {
        render(<ProxyDashboard />);
        expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });

    it('displays proxy status correctly', async () => {
        render(<ProxyDashboard />);
        await waitFor(() => {
            expect(screen.getByText('Proxy Status: Stopped')).toBeInTheDocument();
        });
    });

    it('handles start proxy action', async () => {
        mockedAxios.post.mockResolvedValueOnce({ data: { status: 'success' } });

        render(<ProxyDashboard />);

        // Wait for loading to complete
        await waitFor(() => {
            expect(screen.getByText('Start Proxy')).toBeInTheDocument();
        });

        // Click start button
        fireEvent.click(screen.getByText('Start Proxy'));

        // Verify API call
        expect(mockedAxios.post).toHaveBeenCalledWith(
            `${API_BASE_URL}/api/proxy/start`,
            expect.any(Object)
        );
    });

    it('handles stop proxy action', async () => {
        // Mock proxy as running
        mockedAxios.get.mockImplementation((url) => {
            if (url === `${API_BASE_URL}/api/proxy/status`) {
                return Promise.resolve({
                    data: { ...mockStatus, isRunning: true }
                });
            }
            if (url === `${API_BASE_URL}/analysis/results`) {
                return Promise.resolve({ data: mockAnalysisResults });
            }
            return Promise.reject(new Error('Not found'));
        });

        mockedAxios.post.mockResolvedValueOnce({ data: { status: 'success' } });

        render(<ProxyDashboard />);

        // Wait for loading to complete
        await waitFor(() => {
            expect(screen.getByText('Stop Proxy')).toBeInTheDocument();
        });

        // Click stop button
        fireEvent.click(screen.getByText('Stop Proxy'));

        // Verify API call
        expect(mockedAxios.post).toHaveBeenCalledWith(
            `${API_BASE_URL}/api/proxy/stop`
        );
    });

    it('displays analysis results when proxy is running', async () => {
        // Mock proxy as running
        mockedAxios.get.mockImplementation((url) => {
            if (url === `${API_BASE_URL}/api/proxy/status`) {
                return Promise.resolve({
                    data: { ...mockStatus, isRunning: true }
                });
            }
            if (url === `${API_BASE_URL}/analysis/results`) {
                return Promise.resolve({ data: mockAnalysisResults });
            }
            return Promise.reject(new Error('Not found'));
        });

        render(<ProxyDashboard />);

        // Wait for analysis results to be displayed
        await waitFor(() => {
            expect(screen.getByText('SQL Injection')).toBeInTheDocument();
        });
    });

    it('handles clear analysis results action', async () => {
        // Mock proxy as running
        mockedAxios.get.mockImplementation((url) => {
            if (url === `${API_BASE_URL}/api/proxy/status`) {
                return Promise.resolve({
                    data: { ...mockStatus, isRunning: true }
                });
            }
            if (url === `${API_BASE_URL}/analysis/results`) {
                return Promise.resolve({ data: mockAnalysisResults });
            }
            return Promise.reject(new Error('Not found'));
        });

        mockedAxios.delete.mockResolvedValueOnce({ data: { status: 'success' } });

        render(<ProxyDashboard />);

        // Wait for clear button to be available
        await waitFor(() => {
            expect(screen.getByText('Clear Analysis Results')).toBeInTheDocument();
        });

        // Click clear button
        fireEvent.click(screen.getByText('Clear Analysis Results'));

        // Verify API call
        expect(mockedAxios.delete).toHaveBeenCalledWith(
            `${API_BASE_URL}/analysis/results`
        );
    });

    it('displays error message on API failure', async () => {
        // Mock API error
        mockedAxios.get.mockRejectedValueOnce(new Error('API Error'));

        render(<ProxyDashboard />);

        // Wait for error message
        await waitFor(() => {
            expect(screen.getByText('Failed to fetch proxy status')).toBeInTheDocument();
        });
    });
});
