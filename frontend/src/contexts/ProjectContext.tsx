import React, { createContext, useContext, useState, useEffect } from 'react';
import { Project } from '../types/project';
import { projectApi } from '../api/projectApi';

interface ProjectContextType {
    activeProject: Project | null;
    setActiveProject: (project: Project | null) => void;
    loading: boolean;
    loadProjects: () => Promise<void>;
    projects: Project[];
}

const ProjectContext = createContext<ProjectContextType>({
    activeProject: null,
    setActiveProject: () => { },
    loading: true,
    loadProjects: async () => { },
    projects: [],
});

export const useProject = () => useContext(ProjectContext);

export const ProjectProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [activeProject, setActiveProject] = useState<Project | null>(null);
    const [projects, setProjects] = useState<Project[]>([]);
    const [loading, setLoading] = useState(true);

    // Load projects from local storage on mount
    useEffect(() => {
        const savedProjectId = localStorage.getItem('activeProjectId');
        if (savedProjectId) {
            projectApi.getProject(parseInt(savedProjectId))
                .then(project => setActiveProject(project))
                .catch(console.error);
        }
        loadProjects();
    }, []);

    // Save active project to local storage when it changes
    useEffect(() => {
        if (activeProject) {
            localStorage.setItem('activeProjectId', activeProject.id.toString());
        } else {
            localStorage.removeItem('activeProjectId');
        }
    }, [activeProject]);

    const loadProjects = async () => {
        try {
            setLoading(true);
            const data = await projectApi.getProjects();
            setProjects(data);

            // If no active project is set and we have projects, set the first one as active
            if (!activeProject && data.length > 0) {
                setActiveProject(data[0]);
            }
        } catch (error) {
            console.error('Failed to load projects:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <ProjectContext.Provider
            value={{
                activeProject,
                setActiveProject,
                loading,
                loadProjects,
                projects,
            }}
        >
            {children}
        </ProjectContext.Provider>
    );
};
