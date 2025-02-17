import React from 'react';
import {
    Box,
    Paper,
    Typography,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    TextField,
    Switch,
    FormControlLabel,
    SelectChangeEvent,
} from '@mui/material';

export interface FuzzingConfig {
    enabled: boolean;
    payloadType: 'XSS' | 'SQL' | 'NOSQL' | 'CUSTOM';
    customPayload?: string;
    frequency: number;
}

interface FuzzingConfigPanelProps {
    config: FuzzingConfig;
    onChange: (config: FuzzingConfig) => void;
}

export const FuzzingConfigPanel: React.FC<FuzzingConfigPanelProps> = ({ config, onChange }) => {
    const handleChange = (field: keyof FuzzingConfig, value: any) => {
        onChange({
            ...config,
            [field]: value
        });
    };

    return (
        <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
                Fuzzing Configuration
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <FormControlLabel
                    control={
                        <Switch
                            checked={config.enabled}
                            onChange={(e) => handleChange('enabled', e.target.checked)}
                        />
                    }
                    label="Enable Fuzzing"
                />

                <FormControl fullWidth disabled={!config.enabled}>
                    <InputLabel>Payload Type</InputLabel>
                    <Select
                        value={config.payloadType}
                        label="Payload Type"
                        onChange={(e: SelectChangeEvent<string>) =>
                            handleChange('payloadType', e.target.value as FuzzingConfig['payloadType'])
                        }
                    >
                        <MenuItem value="XSS">Cross-Site Scripting (XSS)</MenuItem>
                        <MenuItem value="SQL">SQL Injection</MenuItem>
                        <MenuItem value="NOSQL">NoSQL Injection</MenuItem>
                        <MenuItem value="CUSTOM">Custom Payload</MenuItem>
                    </Select>
                </FormControl>

                {config.payloadType === 'CUSTOM' && (
                    <TextField
                        fullWidth
                        label="Custom Payload"
                        value={config.customPayload || ''}
                        onChange={(e) => handleChange('customPayload', e.target.value)}
                        disabled={!config.enabled}
                        multiline
                        rows={3}
                        placeholder="Enter your custom payload here..."
                    />
                )}

                <TextField
                    fullWidth
                    label="Frequency (ms)"
                    type="number"
                    value={config.frequency}
                    onChange={(e) => handleChange('frequency', parseInt(e.target.value) || 1000)}
                    disabled={!config.enabled}
                    inputProps={{ min: 100, max: 10000 }}
                    helperText="Time between fuzzing attempts (100ms - 10000ms)"
                />
            </Box>
        </Paper>
    );
};
