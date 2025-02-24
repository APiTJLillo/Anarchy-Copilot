import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Box, Typography, Button, Paper } from '@mui/material';
import { Warning as WarningIcon } from '@mui/icons-material';

interface Props {
    children: ReactNode;
    component?: string;
}

interface State {
    hasError: boolean;
    error: Error | null;
    errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null,
        errorInfo: null
    };

    public static getDerivedStateFromError(error: Error): State {
        return {
            hasError: true,
            error,
            errorInfo: null
        };
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        this.setState({
            error,
            errorInfo
        });

        // Log error to an error reporting service
        console.error('Component Error:', error, errorInfo);
    }

    private handleReset = () => {
        this.setState({
            hasError: false,
            error: null,
            errorInfo: null
        });
    };

    public render() {
        if (this.state.hasError) {
            return (
                <Paper sx={{ p: 3, m: 2, backgroundColor: 'error.light' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                        <WarningIcon color="error" sx={{ fontSize: 40 }} />
                        <Typography variant="h6" color="error">
                            {this.props.component || 'Component'} Error
                        </Typography>
                    </Box>
                    <Typography variant="body1" color="error.dark" gutterBottom>
                        {this.state.error?.message || 'An unexpected error occurred.'}
                    </Typography>
                    {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
                        <Box sx={{ mt: 2 }}>
                            <Typography variant="caption" component="pre" sx={{
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-word',
                                color: 'error.dark',
                                fontSize: '0.75rem'
                            }}>
                                {this.state.errorInfo.componentStack}
                            </Typography>
                        </Box>
                    )}
                    <Box sx={{ mt: 2 }}>
                        <Button
                            variant="contained"
                            color="primary"
                            onClick={this.handleReset}
                        >
                            Try Again
                        </Button>
                    </Box>
                </Paper>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
