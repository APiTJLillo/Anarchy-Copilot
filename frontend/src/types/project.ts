/**
 * Project-related type definitions
 */

export interface Project {
    id: number;
    name: string;
    description?: string;
    scope: Record<string, any>;
    owner_id: number;
    created_at: string;
    updated_at: string;
    is_archived: boolean;
}

export interface ProjectCollaborator {
    user_id: number;
    username: string;
    email: string;
    role: string;
}

export interface CreateProjectData {
    name: string;
    description?: string;
    scope?: Record<string, any>;
    owner_id: number;
}

export interface UpdateProjectData {
    name?: string;
    description?: string;
    scope?: Record<string, any>;
    is_archived?: boolean;
}

export interface AddCollaboratorData {
    user_id: number;
    role: string;
}
