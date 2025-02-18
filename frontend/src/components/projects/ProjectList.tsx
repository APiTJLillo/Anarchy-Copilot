import React, { useState } from 'react';
import {
    Box,
    Button,
    Card,
    CardContent,
    Typography,
    IconButton,
    Grid,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    TextField,
    Chip,
} from '@mui/material';
import { Add as AddIcon, Edit as EditIcon, Delete as DeleteIcon } from '@mui/icons-material';
import { Project, CreateProjectData, UpdateProjectData } from '../../types/project';
import { projectApi } from '../../api/projectApi';
import { useProject } from '../../contexts/ProjectContext';

export const ProjectList: React.FC = () => {
    const { projects, loadProjects, activeProject, setActiveProject } = useProject();
    const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
    const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
    const [selectedProject, setSelectedProject] = useState<Project | null>(null);
    const [formData, setFormData] = useState<CreateProjectData | UpdateProjectData>({
        name: '',
        description: '',
        scope: {},
    });

    const handleCreateProject = async () => {
        if (!formData.name) return;

        try {
            const newProject = await projectApi.createProject({
                ...formData as CreateProjectData,
                owner_id: 1, // TODO: Get from auth context
            });
            setIsCreateDialogOpen(false);
            setFormData({ name: '', description: '', scope: {} });
            await loadProjects();

            // Set as active project if it's the first one
            if (!activeProject) {
                setActiveProject(newProject);
            }
        } catch (error) {
            console.error('Failed to create project:', error);
        }
    };

    const handleEditProject = async () => {
        if (!selectedProject || !formData.name) return;

        try {
            const updatedProject = await projectApi.updateProject(selectedProject.id, formData);
            setIsEditDialogOpen(false);
            setSelectedProject(null);
            setFormData({ name: '', description: '', scope: {} });
            await loadProjects();

            // Update active project if it was the one being edited
            if (activeProject?.id === selectedProject.id) {
                setActiveProject(updatedProject);
            }
        } catch (error) {
            console.error('Failed to update project:', error);
        }
    };

    const handleDeleteProject = async (project: Project) => {
        if (!window.confirm('Are you sure you want to delete this project?')) {
            return;
        }

        try {
            await projectApi.deleteProject(project.id);

            // If deleting active project, clear it
            if (activeProject?.id === project.id) {
                setActiveProject(null);
            }

            await loadProjects();
        } catch (error) {
            console.error('Failed to delete project:', error);
        }
    };

    const openEditDialog = (project: Project) => {
        setSelectedProject(project);
        setFormData({
            name: project.name,
            description: project.description || '',
            scope: project.scope,
        });
        setIsEditDialogOpen(true);
    };

    return (
        <Box p={3}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
                <Typography variant="h5">Projects</Typography>
                <Button
                    variant="contained"
                    color="primary"
                    startIcon={<AddIcon />}
                    onClick={() => setIsCreateDialogOpen(true)}
                >
                    New Project
                </Button>
            </Box>

            <Grid container spacing={3}>
                {projects.map((project) => (
                    <Grid item xs={12} sm={6} md={4} key={project.id}>
                        <Card
                            sx={{
                                border: activeProject?.id === project.id ? '2px solid #00ff00' : 'none',
                                cursor: 'pointer',
                                '&:hover': {
                                    bgcolor: 'rgba(0, 255, 0, 0.05)',
                                },
                            }}
                            onClick={() => setActiveProject(project)}
                        >
                            <CardContent>
                                <Box display="flex" justifyContent="space-between" alignItems="start">
                                    <Typography variant="h6" gutterBottom>
                                        {project.name}
                                        {activeProject?.id === project.id && (
                                            <Chip
                                                label="Active"
                                                color="primary"
                                                size="small"
                                                sx={{ ml: 1 }}
                                            />
                                        )}
                                    </Typography>
                                    <Box>
                                        <IconButton
                                            size="small"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                openEditDialog(project);
                                            }}
                                        >
                                            <EditIcon />
                                        </IconButton>
                                        <IconButton
                                            size="small"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleDeleteProject(project);
                                            }}
                                        >
                                            <DeleteIcon />
                                        </IconButton>
                                    </Box>
                                </Box>
                                <Typography color="textSecondary" gutterBottom>
                                    {project.description}
                                </Typography>
                                {project.is_archived && (
                                    <Chip label="Archived" color="default" size="small" />
                                )}
                                <Typography variant="caption" display="block" gutterBottom>
                                    Created: {new Date(project.created_at).toLocaleDateString()}
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>
                ))}
            </Grid>

            {/* Create Project Dialog */}
            <Dialog open={isCreateDialogOpen} onClose={() => setIsCreateDialogOpen(false)} fullWidth>
                <DialogTitle>Create New Project</DialogTitle>
                <DialogContent>
                    <TextField
                        autoFocus
                        margin="dense"
                        label="Project Name"
                        fullWidth
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    />
                    <TextField
                        margin="dense"
                        label="Description"
                        fullWidth
                        multiline
                        rows={4}
                        value={formData.description}
                        onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setIsCreateDialogOpen(false)}>Cancel</Button>
                    <Button
                        onClick={handleCreateProject}
                        color="primary"
                        disabled={!formData.name}
                    >
                        Create
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Edit Project Dialog */}
            <Dialog open={isEditDialogOpen} onClose={() => setIsEditDialogOpen(false)} fullWidth>
                <DialogTitle>Edit Project</DialogTitle>
                <DialogContent>
                    <TextField
                        autoFocus
                        margin="dense"
                        label="Project Name"
                        fullWidth
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    />
                    <TextField
                        margin="dense"
                        label="Description"
                        fullWidth
                        multiline
                        rows={4}
                        value={formData.description}
                        onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setIsEditDialogOpen(false)}>Cancel</Button>
                    <Button
                        onClick={handleEditProject}
                        color="primary"
                        disabled={!formData.name}
                    >
                        Save
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
};
