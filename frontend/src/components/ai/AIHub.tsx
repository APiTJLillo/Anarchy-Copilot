import React from 'react';
import {
    Box,
    Card,
    CardContent,
    Grid,
    Tab,
    Tabs,
    Typography,
} from '@mui/material';
import {
    Chat as ChatIcon,
    Analytics as AnalyticsIcon,
    Science as MLIcon,
    Speed as MonitorIcon,
} from '@mui/icons-material';

// Sub-components (to be implemented)
import AIChat from './chat/AIChat';
import AIAnalytics from './analytics/AIAnalytics';
import AIOperations from './operations/AIOperations';
import AIMonitoring from './monitoring/AIMonitoring';

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`ai-hub-tabpanel-${index}`}
            aria-labelledby={`ai-hub-tab-${index}`}
            {...other}
        >
            {value === index && (
                <Box sx={{ p: 3 }}>
                    {children}
                </Box>
            )}
        </div>
    );
}

function a11yProps(index: number) {
    return {
        id: `ai-hub-tab-${index}`,
        'aria-controls': `ai-hub-tabpanel-${index}`,
    };
}

const AIHub: React.FC = () => {
    const [activeTab, setActiveTab] = React.useState(0);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setActiveTab(newValue);
    };

    return (
        <Box sx={{ width: '100%' }}>
            <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
                AI Hub
            </Typography>

            <Card>
                <CardContent>
                    <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                        <Tabs
                            value={activeTab}
                            onChange={handleTabChange}
                            variant="fullWidth"
                        >
                            <Tab
                                icon={<ChatIcon />}
                                label="Chat & Interact"
                                {...a11yProps(0)}
                            />
                            <Tab
                                icon={<AnalyticsIcon />}
                                label="Analytics"
                                {...a11yProps(1)}
                            />
                            <Tab
                                icon={<MLIcon />}
                                label="ML Operations"
                                {...a11yProps(2)}
                            />
                            <Tab
                                icon={<MonitorIcon />}
                                label="Monitoring"
                                {...a11yProps(3)}
                            />
                        </Tabs>
                    </Box>

                    <TabPanel value={activeTab} index={0}>
                        <AIChat />
                    </TabPanel>
                    <TabPanel value={activeTab} index={1}>
                        <AIAnalytics />
                    </TabPanel>
                    <TabPanel value={activeTab} index={2}>
                        <AIOperations />
                    </TabPanel>
                    <TabPanel value={activeTab} index={3}>
                        <AIMonitoring />
                    </TabPanel>
                </CardContent>
            </Card>
        </Box>
    );
};

export default AIHub;
