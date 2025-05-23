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

    const handleChange = (event: SelectChangeEvent<number | ''>) => {
        const value = event.target.value;
        if (value === '') return;
        const selectedProject = projects.find(p => p.id === value);
        if (selectedProject) {
            setActiveProject(selectedProject);
        }
    };

    // Only set a value if we have projects and an active project
    const selectValue = projects.length > 0 && activeProject?.id ? activeProject.id : '';

    return (
        <Box display="flex" alignItems="center" gap={1}>
            <FormControl size="small" sx={{ minWidth: 200 }}>
                <InputLabel id="project-select-label">Active Project</InputLabel>
                <Select
                    labelId="project-select-label"
                    value={selectValue}
                    label="Active Project"
                    onChange={handleChange}
                >
                    {loading ? (
                        <MenuItem disabled>
                            <CircularProgress size={20} />
                        </MenuItem>
                    ) : projects.length === 0 ? (
                        <MenuItem disabled value="">No projects available</MenuItem>
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
