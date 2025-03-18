import React, { useState } from 'react';
import { Typography, Box, Chip, Tooltip, CircularProgress } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import { useWebSocket } from '../../hooks/useWebSocket';
import { WS_ENDPOINT } from '../../config';
import type { VersionInfo } from '../../api/proxyApi';

type WSMessage = 
  | { type: 'version_info'; data: VersionInfo }
  | { type: 'initial_data'; data: { version: VersionInfo; [key: string]: any } };

export const Version: React.FC = () => {
    const [version, setVersion] = useState<VersionInfo | null>(null);
    const { isConnected, error } = useWebSocket<WSMessage>(WS_ENDPOINT, {
        onMessage: (message) => {
            if (message.type === 'version_info') {
                setVersion(message.data);
            } else if (message.type === 'initial_data') {
                setVersion(message.data.version);
            }
        }
    });

    if (error) return null;
    if (!isConnected || !version) {
        return (
            <Box display="flex" alignItems="center">
                <CircularProgress size={20} sx={{ mr: 1 }} />
                <Typography variant="body2" color="textSecondary">
                    Loading version...
                </Typography>
            </Box>
        );
    }

    return (
        <Box display="flex" alignItems="center" gap={1}>
            <Tooltip
                title={
                    <>
                        <Typography variant="body2">
                            {version.name}
                        </Typography>
                        <Typography variant="caption">
                            API Compatibility: {version.api_compatibility}
                        </Typography>
                    </>
                }
            >
                <Box display="flex" alignItems="center" gap={0.5}>
                    <Chip
                        label={`v${version.version}`}
                        size="small"
                        color="primary"
                        variant="outlined"
                    />
                    <InfoIcon fontSize="small" color="action" />
                </Box>
            </Tooltip>
        </Box>
    );
};
