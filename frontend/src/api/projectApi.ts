import { Project, CreateProjectData, UpdateProjectData, ProjectCollaborator, AddCollaboratorData } from '../types/project';

export class ProjectApi {
    private baseUrl = 'http://localhost:8000/api/projects';

    async createProject(data: CreateProjectData): Promise<Project> {
        const response = await fetch(this.baseUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error(`Failed to create project: ${response.statusText}`);
        }

        return response.json();
    }

    async getProjects(): Promise<Project[]> {
        const response = await fetch(this.baseUrl);

        if (!response.ok) {
            throw new Error(`Failed to fetch projects: ${response.statusText}`);
        }

        return response.json();
    }

    async getProject(id: number): Promise<Project> {
        const response = await fetch(`${this.baseUrl}/${id}`);

        if (!response.ok) {
            throw new Error(`Failed to fetch project: ${response.statusText}`);
        }

        return response.json();
    }

    async updateProject(id: number, data: UpdateProjectData): Promise<Project> {
        const response = await fetch(`${this.baseUrl}/${id}`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error(`Failed to update project: ${response.statusText}`);
        }

        return response.json();
    }

    async deleteProject(id: number): Promise<void> {
        const response = await fetch(`${this.baseUrl}/${id}`, {
            method: 'DELETE',
        });

        if (!response.ok) {
            throw new Error(`Failed to delete project: ${response.statusText}`);
        }
    }

    async addCollaborator(projectId: number, data: AddCollaboratorData): Promise<ProjectCollaborator> {
        const response = await fetch(`${this.baseUrl}/${projectId}/collaborators`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error(`Failed to add collaborator: ${response.statusText}`);
        }

        return response.json();
    }

    async removeCollaborator(projectId: number, userId: number): Promise<void> {
        const response = await fetch(`${this.baseUrl}/${projectId}/collaborators/${userId}`, {
            method: 'DELETE',
        });

        if (!response.ok) {
            throw new Error(`Failed to remove collaborator: ${response.statusText}`);
        }
    }

    async getCollaborators(projectId: number): Promise<ProjectCollaborator[]> {
        const response = await fetch(`${this.baseUrl}/${projectId}/collaborators`);

        if (!response.ok) {
            throw new Error(`Failed to fetch collaborators: ${response.statusText}`);
        }

        return response.json();
    }
}

export const projectApi = new ProjectApi();
