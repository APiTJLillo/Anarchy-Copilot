import React from 'react';
import {
    Box,
    Paper,
    Typography,
    Chip,
    List,
    ListItem,
    ListItemText,
    ListItemIcon,
    Collapse,
} from '@mui/material';
import {
    Warning as WarningIcon,
    Error as ErrorIcon,
    Info as InfoIcon,
    ExpandLess,
    ExpandMore,
} from '@mui/icons-material';
import { useState } from 'react';

interface AnalysisResult {
    requestId: string;
    ruleName: string;
    severity: string;
    description: string;
    evidence: string;
    timestamp: string;
}

interface AnalysisResultsProps {
    results: AnalysisResult[];
}

const severityColors = {
    high: 'error',
    medium: 'warning',
    low: 'info',
} as const;

const severityIcons = {
    high: <ErrorIcon color="error" />,
    medium: <WarningIcon color="warning" />,
    low: <InfoIcon color="info" />,
};

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ results }) => {
    const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());

    const toggleExpand = (resultId: string) => {
        setExpandedItems(prev => {
            const newSet = new Set(prev);
            if (prev.has(resultId)) {
                newSet.delete(resultId);
            } else {
                newSet.add(resultId);
            }
            return newSet;
        });
    };

    if (results.length === 0) {
        return (
            <Box sx={{ p: 2 }}>
                <Typography variant="body1" color="text.secondary">
                    No security issues detected
                </Typography>
            </Box>
        );
    }

    // Group results by severity
    const groupedResults = results.reduce((acc, result) => {
        const severity = result.severity.toLowerCase();
        if (!acc[severity]) {
            acc[severity] = [];
        }
        acc[severity].push(result);
        return acc;
    }, {} as Record<string, AnalysisResult[]>);

    return (
        <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
                Security Analysis Results
            </Typography>

            {/* Summary chips */}
            <Box sx={{ mb: 2, display: 'flex', gap: 1 }}>
                {Object.entries(groupedResults).map(([severity, items]) => (
                    <Chip
                        key={severity}
                        label={`${severity}: ${items.length}`}
                        color={severityColors[severity as keyof typeof severityColors] || 'default'}
                        icon={severityIcons[severity as keyof typeof severityIcons]}
                        variant="outlined"
                    />
                ))}
            </Box>

            {/* Results list */}
            <List>
                {Object.entries(groupedResults).map(([severity, items]) =>
                    items.map((result, index) => {
                        const resultId = `${result.requestId}-${index}`;
                        const isExpanded = expandedItems.has(resultId);

                        return (
                            <React.Fragment key={resultId}>
                                <ListItem
                                    button
                                    onClick={() => toggleExpand(resultId)}
                                    sx={{
                                        borderLeft: 3,
                                        borderColor: `${severityColors[severity as keyof typeof severityColors]}.main`,
                                    }}
                                >
                                    <ListItemIcon>
                                        {severityIcons[severity as keyof typeof severityIcons]}
                                    </ListItemIcon>
                                    <ListItemText
                                        primary={result.ruleName}
                                        secondary={result.description}
                                    />
                                    {isExpanded ? <ExpandLess /> : <ExpandMore />}
                                </ListItem>
                                <Collapse in={isExpanded}>
                                    <Box sx={{ pl: 4, pr: 2, pb: 2 }}>
                                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                            Evidence:
                                        </Typography>
                                        <Paper
                                            variant="outlined"
                                            sx={{
                                                p: 1,
                                                backgroundColor: 'background.default',
                                                fontFamily: 'monospace',
                                                overflowX: 'auto',
                                                maxWidth: '100%',
                                            }}
                                        >
                                            <code>{result.evidence}</code>
                                        </Paper>
                                        <Typography
                                            variant="caption"
                                            color="text.secondary"
                                            sx={{ display: 'block', mt: 1 }}
                                        >
                                            Request ID: {result.requestId}
                                            <br />
                                            Detected: {new Date(result.timestamp).toLocaleString()}
                                        </Typography>
                                    </Box>
                                </Collapse>
                            </React.Fragment>
                        );
                    })
                )}
            </List>
        </Paper>
    );
};

export default AnalysisResults;
