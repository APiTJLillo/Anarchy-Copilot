import React from 'react';
import {
    Box,
    Grid,
    Card,
    CardContent,
    Typography,
    LinearProgress,
    Paper,
} from '@mui/material';
import {
    Timeline as TimelineIcon,
    Speed as PerformanceIcon,
    MonetizationOn as CostIcon,
    Psychology as UsageIcon,
} from '@mui/icons-material';

interface MetricCardProps {
    title: string;
    value: string;
    icon: React.ReactNode;
    trend?: number;
    progress?: number;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, icon, trend, progress }) => (
    <Card sx={{ height: '100%' }}>
        <CardContent>
            <Box display="flex" alignItems="center" mb={2}>
                <Box sx={{ mr: 2, color: 'primary.main' }}>{icon}</Box>
                <Typography variant="h6" component="div">
                    {title}
                </Typography>
            </Box>
            <Typography variant="h4" component="div" sx={{ mb: 1 }}>
                {value}
            </Typography>
            {trend !== undefined && (
                <Typography
                    variant="body2"
                    color={trend >= 0 ? "success.main" : "error.main"}
                >
                    {trend >= 0 ? "↑" : "↓"} {Math.abs(trend)}% from last period
                </Typography>
            )}
            {progress !== undefined && (
                <Box sx={{ mt: 2 }}>
                    <LinearProgress variant="determinate" value={progress} />
                </Box>
            )}
        </CardContent>
    </Card>
);

const AIAnalytics: React.FC = () => {
    // Placeholder data - would come from backend in real implementation
    const metrics = {
        requests: {
            total: "1,234",
            trend: 12.5,
            progress: 75
        },
        performance: {
            avgLatency: "245ms",
            trend: -8.3,
            progress: 85
        },
        cost: {
            monthly: "$123.45",
            trend: 5.2,
            progress: 45
        },
        usage: {
            tokens: "89.2K",
            trend: 15.7,
            progress: 60
        }
    };

    return (
        <Box>
            <Grid container spacing={3}>
                <Grid item xs={12} md={6} lg={3}>
                    <MetricCard
                        title="Total Requests"
                        value={metrics.requests.total}
                        icon={<TimelineIcon />}
                        trend={metrics.requests.trend}
                        progress={metrics.requests.progress}
                    />
                </Grid>
                <Grid item xs={12} md={6} lg={3}>
                    <MetricCard
                        title="Avg. Latency"
                        value={metrics.performance.avgLatency}
                        icon={<PerformanceIcon />}
                        trend={metrics.performance.trend}
                        progress={metrics.performance.progress}
                    />
                </Grid>
                <Grid item xs={12} md={6} lg={3}>
                    <MetricCard
                        title="Monthly Cost"
                        value={metrics.cost.monthly}
                        icon={<CostIcon />}
                        trend={metrics.cost.trend}
                        progress={metrics.cost.progress}
                    />
                </Grid>
                <Grid item xs={12} md={6} lg={3}>
                    <MetricCard
                        title="Token Usage"
                        value={metrics.usage.tokens}
                        icon={<UsageIcon />}
                        trend={metrics.usage.trend}
                        progress={metrics.usage.progress}
                    />
                </Grid>
                <Grid item xs={12}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>
                            Detailed Analytics Coming Soon
                        </Typography>
                        <Typography variant="body1">
                            Future features will include:
                        </Typography>
                        <ul>
                            <li>Request volume trends</li>
                            <li>Response quality metrics</li>
                            <li>Cost optimization suggestions</li>
                            <li>Performance analysis</li>
                            <li>Token usage breakdown</li>
                            <li>Model comparison stats</li>
                        </ul>
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
};

export default AIAnalytics;
