import React, { useEffect, useState } from 'react';
import { Typography, Box, Chip, Tooltip } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import proxyApi from '../../api/proxyApi';
import type { VersionInfo } from '../../api/proxyApi';

export const Version: React.FC = () => {
    const [version, setVersion] = useState<VersionInfo | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchVersion = async () => {
            try {
                const data = await proxyApi.getVersion();
                setVersion(data);
            } catch (err) {
                setError('Failed to fetch version info');
                console.error(err);
            }
        };

        fetchVersion();
    }, []);

    if (error) {
        return null; // Don't show anything if there's an error
    }

    if (!version) {
        return null;
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
