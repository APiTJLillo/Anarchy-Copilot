import React, { createContext, useContext, useState, useEffect } from 'react';
import proxyApi from '../api/proxyApi';

export interface User {
    id: number;
    username: string;
    email: string;
}

interface Project {
    id: number;
    name: string;
}

interface UserContextType {
    currentUser: User | null;
    setCurrentUser: (user: User | null) => void;
    users: User[];
    projects: Project[];
    loadingUsers: boolean;
    loadingProjects: boolean;
    error: string | null;
}

const UserContext = createContext<UserContextType | null>(null);

export const UserProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [currentUser, setCurrentUser] = useState<User | null>(null);
    const [users, setUsers] = useState<User[]>([]);
    const [projects, setProjects] = useState<Project[]>([]);
    const [loadingUsers, setLoadingUsers] = useState(true);
    const [loadingProjects, setLoadingProjects] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchUsers = async () => {
            try {
                const data = await proxyApi.getUsers();
                setUsers(data);
                // Set first user as current if none selected
                if (!currentUser && data.length > 0) {
                    setCurrentUser(data[0]);
                }
            } catch (err) {
                setError('Failed to load users');
                console.error(err);
            } finally {
                setLoadingUsers(false);
            }
        };

        const fetchProjects = async () => {
            try {
                const data = await proxyApi.getProjects();
                setProjects(data);
            } catch (err) {
                setError('Failed to load projects');
                console.error(err);
            } finally {
                setLoadingProjects(false);
            }
        };

        fetchUsers();
        fetchProjects();
    }, []);

    return (
        <UserContext.Provider
            value={{
                currentUser,
                setCurrentUser,
                users,
                projects,
                loadingUsers,
                loadingProjects,
                error
            }}
        >
            {children}
        </UserContext.Provider>
    );
};

export const useUser = () => {
    const context = useContext(UserContext);
    if (!context) {
        throw new Error('useUser must be used within a UserProvider');
    }
    return context;
};
