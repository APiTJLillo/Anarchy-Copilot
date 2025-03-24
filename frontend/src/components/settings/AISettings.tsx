import React, { useEffect, useState } from "react";
import {
    Box,
    Card,
    CardContent,
    Typography,
    FormControl,
    Select,
    MenuItem,
    Switch,
    TextField,
    Button,
    Divider,
    Alert,
    Tooltip,
    IconButton,
    SelectChangeEvent,
    Slider,
} from "@mui/material";
import { InfoOutlined, SaveOutlined, RestartAlt } from "@mui/icons-material";
import { updateAISettingsAsync, resetAISettingsAsync } from "../../store/ai/actions";
import { useAppDispatch, useAppSelector } from "../../store/hooks";
import type { AISettings as AISettingsType, BaseModelConfig } from "../../store/ai/types";

interface AISettingsProps {
    onSave?: () => void;
}

const AISettings: React.FC<AISettingsProps> = ({ onSave }) => {
    const dispatch = useAppDispatch();
    const settings = useAppSelector(state => state.ai.settings);
    const loading = useAppSelector(state => state.ai.loading);
    const error = useAppSelector(state => state.ai.error);
    const [localSettings, setLocalSettings] = useState<AISettingsType>(settings);
    const [saveStatus, setSaveStatus] = useState<"success" | "error" | null>(null);

    useEffect(() => {
        setLocalSettings(settings);
    }, [settings]);

    const handleSelectChange = (field: keyof AISettingsType) => (
        event: SelectChangeEvent<string>
    ) => {
        setLocalSettings({
            ...localSettings,
            [field]: event.target.value,
        });
    };

    const handleSwitchChange = (field: keyof AISettingsType) => (
        event: React.ChangeEvent<HTMLInputElement>
    ) => {
        setLocalSettings({
            ...localSettings,
            [field]: event.target.checked,
        });
    };

    const handleModelConfigChange = (modelId: string, field: keyof BaseModelConfig) => (
        event: React.ChangeEvent<HTMLInputElement>
    ) => {
        const value = event.target.type === 'checkbox' ? event.target.checked : event.target.value;
        setLocalSettings({
            ...localSettings,
            models: {
                ...localSettings.models,
                [modelId]: {
                    ...localSettings.models[modelId],
                    [field]: value,
                },
            },
        });
    };

    const handleModelSliderChange = (modelId: string, field: keyof BaseModelConfig) => (
        event: Event,
        newValue: number | number[]
    ) => {
        setLocalSettings({
            ...localSettings,
            models: {
                ...localSettings.models,
                [modelId]: {
                    ...localSettings.models[modelId],
                    [field]: newValue,
                },
            },
        });
    };

    const handleSave = async () => {
        try {
            const result = await dispatch(updateAISettingsAsync(localSettings));
            if (updateAISettingsAsync.fulfilled.match(result)) {
                setSaveStatus("success");
                onSave?.();
            } else {
                setSaveStatus("error");
            }
        } catch (error) {
            setSaveStatus("error");
            console.error("Failed to save AI settings:", error);
        }

        setTimeout(() => setSaveStatus(null), 3000);
    };

    const handleReset = async () => {
        try {
            await dispatch(resetAISettingsAsync());
        } catch (error) {
            console.error("Failed to reset settings:", error);
        }
    };

    const currentModel = localSettings.models[localSettings.defaultModel];

    return (
        <Card>
            <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Typography variant="h5" component="h2">
                        AI Settings
                    </Typography>
                    <Box>
                        <Tooltip title="Reset to defaults">
                            <IconButton onClick={handleReset} size="small" disabled={loading}>
                                <RestartAlt />
                            </IconButton>
                        </Tooltip>
                    </Box>
                </Box>

                <Divider sx={{ mb: 3 }} />

                {/* Model Selection */}
                <Box mb={3}>
                    <Typography variant="h6" gutterBottom>
                        Model Selection
                    </Typography>
                    <FormControl fullWidth margin="normal">
                        <Select
                            value={localSettings.defaultModel}
                            onChange={handleSelectChange("defaultModel")}
                            displayEmpty
                            disabled={loading}
                        >
                            {Object.keys(localSettings.models).map(modelId => (
                                <MenuItem key={modelId} value={modelId}>{modelId}</MenuItem>
                            ))}
                        </Select>
                    </FormControl>

                    {/* Model Configuration */}
                    {currentModel && (
                        <>
                            <Typography variant="subtitle1" gutterBottom>
                                Model Configuration
                            </Typography>

                            <FormControl fullWidth margin="normal">
                                <Typography component="div" variant="body2" gutterBottom>
                                    Temperature
                                </Typography>
                                <Slider
                                    value={currentModel.temperature}
                                    onChange={(e, v) => handleModelSliderChange(localSettings.defaultModel, "temperature")(e, v)}
                                    min={0}
                                    max={1}
                                    step={0.1}
                                    marks
                                    disabled={loading}
                                />
                            </FormControl>

                            <FormControl fullWidth margin="normal">
                                <Typography component="div" variant="body2" gutterBottom>
                                    Max Tokens
                                </Typography>
                                <TextField
                                    type="number"
                                    value={currentModel.maxTokens}
                                    onChange={handleModelConfigChange(localSettings.defaultModel, "maxTokens")}
                                    disabled={loading}
                                />
                            </FormControl>

                            <FormControl fullWidth margin="normal">
                                <Typography component="div" variant="body2" gutterBottom>
                                    API Key
                                </Typography>
                                <TextField
                                    type="password"
                                    value={currentModel.apiKey || ""}
                                    onChange={handleModelConfigChange(localSettings.defaultModel, "apiKey")}
                                    disabled={loading}
                                />
                            </FormControl>

                            {/* Reasoning Settings */}
                            <Box mt={2}>
                                <Typography variant="subtitle2" gutterBottom>
                                    Reasoning Settings
                                </Typography>
                                <FormControl fullWidth>
                                    <Typography component="div" variant="body2">
                                        Enable Reasoning
                                        <Switch
                                            checked={currentModel.reasoningCapability}
                                            onChange={handleModelConfigChange(localSettings.defaultModel, "reasoningCapability")}
                                            disabled={loading}
                                        />
                                    </Typography>
                                </FormControl>

                                {currentModel.reasoningCapability && (
                                    <FormControl fullWidth margin="normal">
                                        <Typography component="div" variant="body2" gutterBottom>
                                            Reasoning Effort
                                        </Typography>
                                        <Slider
                                            value={currentModel.reasoningEffort}
                                            onChange={(e, v) => handleModelSliderChange(localSettings.defaultModel, "reasoningEffort")(e, v)}
                                            min={0}
                                            max={100}
                                            step={10}
                                            marks
                                            disabled={loading}
                                        />
                                    </FormControl>
                                )}
                            </Box>
                        </>
                    )}
                </Box>

                {/* Translation Settings */}
                <Box mb={3}>
                    <Typography variant="h6" gutterBottom>
                        Translation
                    </Typography>
                    <FormControl fullWidth margin="normal">
                        <Select
                            value={localSettings.translationModel}
                            onChange={handleSelectChange("translationModel")}
                            displayEmpty
                            disabled={loading}
                        >
                            <MenuItem value="neural">Neural Translation</MenuItem>
                            <MenuItem value="basic">Basic Translation</MenuItem>
                        </Select>
                    </FormControl>

                    <FormControl fullWidth margin="normal">
                        <Typography component="div" variant="body2" gutterBottom>
                            Auto-detect Language
                            <Tooltip title="Automatically detect and translate non-English text">
                                <IconButton size="small">
                                    <InfoOutlined />
                                </IconButton>
                            </Tooltip>
                        </Typography>
                        <Switch
                            checked={localSettings.autoDetectLanguage}
                            onChange={handleSwitchChange("autoDetectLanguage")}
                            disabled={loading}
                        />
                    </FormControl>
                </Box>

                {/* Cultural Context Settings */}
                <Box mb={3}>
                    <Typography variant="h6" gutterBottom>
                        Cultural Context
                    </Typography>
                    <FormControl fullWidth margin="normal">
                        <Typography component="div" variant="body2" gutterBottom>
                            Enable Cultural Adaptation
                            <Tooltip title="Adjust content based on cultural context">
                                <IconButton size="small">
                                    <InfoOutlined />
                                </IconButton>
                            </Tooltip>
                        </Typography>
                        <Switch
                            checked={localSettings.enableCulturalContext}
                            onChange={handleSwitchChange("enableCulturalContext")}
                            disabled={loading}
                        />
                    </FormControl>

                    {localSettings.enableCulturalContext && (
                        <FormControl fullWidth margin="normal">
                            <Select
                                value={localSettings.defaultRegion}
                                onChange={handleSelectChange("defaultRegion")}
                                displayEmpty
                                disabled={loading}
                            >
                                <MenuItem value="US">United States</MenuItem>
                                <MenuItem value="UK">United Kingdom</MenuItem>
                                <MenuItem value="EU">European Union</MenuItem>
                                <MenuItem value="AS">Asia</MenuItem>
                            </Select>
                        </FormControl>
                    )}
                </Box>

                {/* Performance Settings */}
                <Box mb={3}>
                    <Typography variant="h6" gutterBottom>
                        Performance
                    </Typography>
                    <FormControl fullWidth margin="normal">
                        <Typography component="div" variant="body2" gutterBottom>
                            Enable Caching
                            <Tooltip title="Cache responses to improve performance">
                                <IconButton size="small">
                                    <InfoOutlined />
                                </IconButton>
                            </Tooltip>
                        </Typography>
                        <Switch
                            checked={localSettings.enableCache}
                            onChange={handleSwitchChange("enableCache")}
                            disabled={loading}
                        />
                    </FormControl>
                </Box>

                {(saveStatus || error) && (
                    <Alert severity={saveStatus || "error"} sx={{ mb: 2 }}>
                        {saveStatus === "success"
                            ? "Settings saved successfully!"
                            : error || "Failed to save settings"}
                    </Alert>
                )}

                <Box display="flex" justifyContent="flex-end">
                    <Button
                        variant="contained"
                        color="primary"
                        onClick={handleSave}
                        startIcon={<SaveOutlined />}
                        disabled={loading}
                    >
                        {loading ? "Saving..." : "Save Settings"}
                    </Button>
                </Box>
            </CardContent>
        </Card>
    );
};

export default AISettings;
