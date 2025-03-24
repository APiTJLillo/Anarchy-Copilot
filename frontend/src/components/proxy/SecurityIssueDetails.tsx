import React from 'react';
import {
    Box,
    Dialog,
    DialogTitle,
    DialogContent,
    Typography,
    List,
    ListItem,
    ListItemText,
    Chip,
} from '@mui/material';

interface SecurityIssue {
    severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    description: string;
}

interface SecurityIssueDetailsProps {
    open: boolean;
    onClose: () => void;
    issues: SecurityIssue[];
}

const severityColors = {
    LOW: 'info',
    MEDIUM: 'warning',
    HIGH: 'error',
    CRITICAL: 'error'
} as const;

export const SecurityIssueDetails: React.FC<SecurityIssueDetailsProps> = ({ open, onClose, issues }) => {
    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
            <DialogTitle>Security Issues</DialogTitle>
            <DialogContent>
                <List>
                    {issues.map((issue, index) => (
                        <ListItem key={index}>
                            <ListItemText
                                primary={
                                    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 1 }}>
                                        <Chip
                                            size="small"
                                            label={issue.severity}
                                            color={severityColors[issue.severity]}
                                        />
                                        <Typography variant="subtitle1">
                                            {issue.description}
                                        </Typography>
                                    </Box>
                                }
                            />
                        </ListItem>
                    ))}
                </List>
            </DialogContent>
        </Dialog>
    );
};
