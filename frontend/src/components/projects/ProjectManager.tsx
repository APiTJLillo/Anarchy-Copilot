import React, { useState } from 'react';
import { Box, Container, Divider, Paper, Tab, Tabs, Typography } from '@mui/material';
import { ProjectList } from './ProjectList';
import { ProjectSelector } from './ProjectSelector';
import { useProject } from '../../contexts/ProjectContext';

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
            id={`project-tabpanel-${index}`}
            aria-labelledby={`project-tab-${index}`}
            {...other}
        >
            {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
        </div>
    );
}

export const ProjectManager: React.FC = () => {
    const { activeProject } = useProject();
    const [selectedTab, setSelectedTab] = useState(0);
    const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);

    const handleCreateClick = () => {
        setSelectedTab(1); // Switch to "All Projects" tab
    };

    return (
        <Container maxWidth="xl">
            <Box py={3}>
                {/* Header with project selector */}
                <Box mb={3} display="flex" justifyContent="space-between" alignItems="center">
                    <Typography variant="h4">Project Management</Typography>
                    <ProjectSelector onCreateClick={handleCreateClick} />
                </Box>

                <Paper>
                    <Tabs
                        value={selectedTab}
                        onChange={(_, newValue) => setSelectedTab(newValue)}
                        sx={{ borderBottom: 1, borderColor: 'divider' }}
                    >
                        <Tab label="Active Project" />
                        <Tab label="All Projects" />
                    </Tabs>

                    <TabPanel value={selectedTab} index={0}>
                        {activeProject ? (
                            <Box>
                                <Typography variant="h6" gutterBottom>Project Details</Typography>
                                <Typography variant="h5" gutterBottom>
                                    {activeProject.name}
                                </Typography>
                                {activeProject.description && (
                                    <Typography color="textSecondary" gutterBottom>
                                        {activeProject.description}
                                    </Typography>
                                )}
                                <Box mt={3}>
                                    <Typography variant="h6" gutterBottom>
                                        Project Settings and Collaborators
                                    </Typography>
                                    <Typography>
                                        Project management features like collaborators, settings, etc. will go here.
                                    </Typography>
                                </Box>
                            </Box>
                        ) : (
                            <Typography color="textSecondary">
                                No project selected. Please select or create a project to get started.
                            </Typography>
                        )}
                    </TabPanel>

                    <TabPanel value={selectedTab} index={1}>
                        <ProjectList />
                    </TabPanel>
                </Paper>
            </Box>
        </Container>
    );
};
