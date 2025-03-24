import React from 'react';
import {
    Box,
    Grid,
    Card,
    CardContent,
    Typography,
    Button,
    CircularProgress,
    List,
    ListItem,
    ListItemText,
    ListItemIcon,
    Chip,
    Divider,
} from '@mui/material';
import {
    PlayArrow as StartIcon,
    Stop as StopIcon,
    Check as SuccessIcon,
    Error as ErrorIcon,
    Pause as PauseIcon,
    FileUpload as DeployIcon,
    Storage as DatasetIcon,
    Settings as TuningIcon,
    Upload as UploadIcon,
} from '@mui/icons-material';

interface ModelStatus {
    id: string;
    name: string;
    status: 'training' | 'stopped' | 'deployed' | 'failed';
    progress: number;
    lastUpdated: string;
    type: string;
}

interface DatasetInfo {
    id: string;
    name: string;
    records: number;
    lastUpdated: string;
    status: 'ready' | 'processing' | 'error';
}

const StatusChip: React.FC<{ status: ModelStatus['status'] }> = ({ status }) => {
    const statusConfig = {
        training: { color: 'warning' as const, label: 'Training' },
        stopped: { color: 'default' as const, label: 'Stopped' },
        deployed: { color: 'success' as const, label: 'Deployed' },
        failed: { color: 'error' as const, label: 'Failed' },
    };

    const config = statusConfig[status];
    return <Chip size="small" color={config.color} label={config.label} />;
};

const AIOperations: React.FC = () => {
    // Placeholder data - would come from backend
    const models: ModelStatus[] = [
        {
            id: '1',
            name: 'Custom GPT-4 Fine-tune',
            status: 'training',
            progress: 45,
            lastUpdated: '2 minutes ago',
            type: 'Fine-tuned GPT-4'
        },
        {
            id: '2',
            name: 'Production Model v2',
            status: 'deployed',
            progress: 100,
            lastUpdated: '2 days ago',
            type: 'GPT-3.5 Turbo'
        },
        {
            id: '3',
            name: 'Experimental Model',
            status: 'stopped',
            progress: 67,
            lastUpdated: '5 hours ago',
            type: 'Custom ML Model'
        }
    ];

    const datasets: DatasetInfo[] = [
        {
            id: '1',
            name: 'Training Dataset 2025',
            records: 50000,
            lastUpdated: '1 day ago',
            status: 'ready'
        },
        {
            id: '2',
            name: 'Validation Set',
            records: 10000,
            lastUpdated: '1 day ago',
            status: 'processing'
        }
    ];

    return (
        <Box>
            <Grid container spacing={3}>
                <Grid item xs={12} md={8}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Model Training & Deployment
                            </Typography>
                            <List>
                                {models.map((model) => (
                                    <React.Fragment key={model.id}>
                                        <ListItem
                                            secondaryAction={
                                                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                                                    <StatusChip status={model.status} />
                                                    {model.status === 'training' && (
                                                        <CircularProgress
                                                            variant="determinate"
                                                            value={model.progress}
                                                            size={24}
                                                        />
                                                    )}
                                                    <Button
                                                        variant="outlined"
                                                        size="small"
                                                        startIcon={model.status === 'training' ? <StopIcon /> : <StartIcon />}
                                                    >
                                                        {model.status === 'training' ? 'Stop' : 'Start'}
                                                    </Button>
                                                    <Button
                                                        variant="contained"
                                                        size="small"
                                                        startIcon={<DeployIcon />}
                                                        disabled={model.status !== 'stopped'}
                                                    >
                                                        Deploy
                                                    </Button>
                                                </Box>
                                            }
                                        >
                                            <ListItemText
                                                primary={model.name}
                                                secondary={`${model.type} • Last updated ${model.lastUpdated}`}
                                            />
                                        </ListItem>
                                        <Divider component="li" />
                                    </React.Fragment>
                                ))}
                            </List>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} md={4}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Training Datasets
                            </Typography>
                            <List>
                                {datasets.map((dataset) => (
                                    <React.Fragment key={dataset.id}>
                                        <ListItem>
                                            <ListItemIcon>
                                                <DatasetIcon />
                                            </ListItemIcon>
                                            <ListItemText
                                                primary={dataset.name}
                                                secondary={`${dataset.records.toLocaleString()} records • ${dataset.lastUpdated}`}
                                            />
                                            <Chip
                                                size="small"
                                                color={dataset.status === 'ready' ? 'success' : 'warning'}
                                                label={dataset.status}
                                            />
                                        </ListItem>
                                        <Divider component="li" />
                                    </React.Fragment>
                                ))}
                            </List>
                            <Box sx={{ mt: 2 }}>
                                <Button
                                    variant="outlined"
                                    startIcon={<UploadIcon />}
                                    fullWidth
                                >
                                    Upload New Dataset
                                </Button>
                            </Box>
                        </CardContent>
                    </Card>

                    <Card sx={{ mt: 2 }}>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Fine-tuning Configuration
                            </Typography>
                            <Typography variant="body2" color="text.secondary" paragraph>
                                Configure hyperparameters, training objectives, and validation settings
                                for model fine-tuning.
                            </Typography>
                            <Button
                                variant="outlined"
                                startIcon={<TuningIcon />}
                                fullWidth
                            >
                                Configure Fine-tuning
                            </Button>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>
        </Box>
    );
};

export default AIOperations;
