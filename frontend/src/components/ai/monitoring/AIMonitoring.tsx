import React from 'react';
import {
    Box,
    Grid,
    Card,
    CardContent,
    Typography,
    List,
    ListItem,
    ListItemText,
    Alert,
    AlertTitle,
    LinearProgress,
    Chip,
} from '@mui/material';
import {
    Warning as WarningIcon,
    CheckCircle as SuccessIcon,
    Error as ErrorIcon,
    Info as InfoIcon,
} from '@mui/icons-material';

interface SystemStatus {
    component: string;
    status: 'healthy' | 'warning' | 'error' | 'info';
    message: string;
    timestamp: string;
    metric?: number;
}

interface PerformanceMetric {
    name: string;
    value: number;
    unit: string;
    threshold: number;
    trend: 'up' | 'down' | 'stable';
}

const AIMonitoring: React.FC = () => {
    // Placeholder data - would come from backend
    const systemStatus: SystemStatus[] = [
        {
            component: "Inference Engine",
            status: "healthy",
            message: "Operating normally",
            timestamp: "Just now",
            metric: 98
        },
        {
            component: "Model Serving",
            status: "warning",
            message: "High latency detected",
            timestamp: "5 minutes ago",
            metric: 65
        },
        {
            component: "Training Pipeline",
            status: "error",
            message: "Resource allocation failed",
            timestamp: "15 minutes ago"
        },
        {
            component: "Data Pipeline",
            status: "info",
            message: "Processing new dataset",
            timestamp: "1 hour ago",
            metric: 45
        }
    ];

    const metrics: PerformanceMetric[] = [
        {
            name: "Response Time",
            value: 245,
            unit: "ms",
            threshold: 500,
            trend: "up"
        },
        {
            name: "Success Rate",
            value: 99.2,
            unit: "%",
            threshold: 99,
            trend: "stable"
        },
        {
            name: "Token Usage",
            value: 75,
            unit: "%",
            threshold: 80,
            trend: "up"
        },
        {
            name: "Error Rate",
            value: 0.5,
            unit: "%",
            threshold: 1,
            trend: "down"
        }
    ];

    const getStatusIcon = (status: SystemStatus['status']) => {
        switch (status) {
            case 'healthy': return <SuccessIcon color="success" />;
            case 'warning': return <WarningIcon color="warning" />;
            case 'error': return <ErrorIcon color="error" />;
            case 'info': return <InfoIcon color="info" />;
        }
    };

    const getTrendIcon = (trend: PerformanceMetric['trend']) => {
        switch (trend) {
            case 'up': return '↑';
            case 'down': return '↓';
            case 'stable': return '→';
        }
    };

    const getTrendColor = (trend: PerformanceMetric['trend'], isGoodMetric: boolean) => {
        if (trend === 'stable') return 'default';
        return (trend === 'up') === isGoodMetric ? 'success' : 'error';
    };

    return (
        <Box>
            <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                System Status
                            </Typography>
                            <List>
                                {systemStatus.map((status) => (
                                    <ListItem key={status.component}>
                                        <Alert
                                            icon={getStatusIcon(status.status)}
                                            severity={status.status === 'healthy' ? 'success' : status.status}
                                            sx={{ width: '100%' }}
                                        >
                                            <AlertTitle>{status.component}</AlertTitle>
                                            {status.message}
                                            <Typography variant="caption" display="block">
                                                {status.timestamp}
                                            </Typography>
                                            {status.metric !== undefined && (
                                                <Box sx={{ mt: 1 }}>
                                                    <LinearProgress
                                                        variant="determinate"
                                                        value={status.metric}
                                                        color={status.status === 'healthy' ? 'success' : status.status === 'warning' ? 'warning' : 'error'}
                                                    />
                                                </Box>
                                            )}
                                        </Alert>
                                    </ListItem>
                                ))}
                            </List>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} md={6}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Performance Metrics
                            </Typography>
                            <List>
                                {metrics.map((metric) => (
                                    <ListItem key={metric.name}>
                                        <ListItemText
                                            primary={
                                                <Box display="flex" justifyContent="space-between" alignItems="center">
                                                    <Typography variant="body1">
                                                        {metric.name}
                                                    </Typography>
                                                    <Box display="flex" alignItems="center" gap={1}>
                                                        <Typography variant="h6">
                                                            {metric.value}{metric.unit}
                                                        </Typography>
                                                        <Chip
                                                            size="small"
                                                            label={getTrendIcon(metric.trend)}
                                                            color={getTrendColor(metric.trend, metric.name !== 'Error Rate')}
                                                        />
                                                    </Box>
                                                </Box>
                                            }
                                            secondary={
                                                <Box sx={{ mt: 1 }}>
                                                    <LinearProgress
                                                        variant="determinate"
                                                        value={(metric.value / metric.threshold) * 100}
                                                        color={metric.value <= metric.threshold ? 'success' : 'error'}
                                                    />
                                                </Box>
                                            }
                                        />
                                    </ListItem>
                                ))}
                            </List>
                        </CardContent>
                    </Card>

                    <Card sx={{ mt: 2 }}>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Real-time Monitoring
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                Coming soon:
                            </Typography>
                            <ul>
                                <li>Live inference tracking</li>
                                <li>Resource utilization graphs</li>
                                <li>Anomaly detection alerts</li>
                                <li>Quality metrics visualization</li>
                            </ul>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>
        </Box>
    );
};

export default AIMonitoring;
