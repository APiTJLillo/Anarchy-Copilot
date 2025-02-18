import React from 'react';
import {
    Box,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    IconButton,
    SelectChangeEvent,
    CircularProgress,
} from '@mui/material';
import { Add as AddIcon } from '@mui/icons-material';
import { useProject } from '../../contexts/ProjectContext';

interface ProjectSelectorProps {
    onCreateClick?: () => void;
}

export const ProjectSelector: React.FC<ProjectSelectorProps> = ({
    onCreateClick,
}) => {
    const { activeProject, setActiveProject, loading, projects } = useProject();

    const handleChange = (event: SelectChangeEvent<number>) => {
        const selectedProject = projects.find(p => p.id === event.target.value);
        if (selectedProject) {
            setActiveProject(selectedProject);
        }
    };

    return (
        <Box display="flex" alignItems="center" gap={1}>
            <FormControl size="small" sx={{ minWidth: 200 }}>
                <InputLabel id="project-select-label">Active Project</InputLabel>
                <Select
                    labelId="project-select-label"
                    value={activeProject?.id || ''}
                    label="Active Project"
                    onChange={handleChange}
                >
                    {loading ? (
                        <MenuItem disabled>
                            <CircularProgress size={20} />
                        </MenuItem>
                    ) : projects.length === 0 ? (
                        <MenuItem disabled>No projects available</MenuItem>
                    ) : (
                        projects.map((project) => (
                            <MenuItem key={project.id} value={project.id}>
                                {project.name}
                            </MenuItem>
                        ))
                    )}
                </Select>
            </FormControl>
            {onCreateClick && (
                <IconButton
                    size="small"
                    onClick={onCreateClick}
                    sx={{
                        bgcolor: 'primary.main',
                        color: 'white',
                        '&:hover': {
                            bgcolor: 'primary.dark',
                        }
                    }}
                >
                    <AddIcon />
                </IconButton>
            )}
        </Box>
    );
};
