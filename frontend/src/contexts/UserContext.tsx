import React, { createContext, useContext, useState, useEffect } from 'react';
import { useProxyApi } from '../api/proxyApi';
import type { User, Project } from '../api/proxyApi';

interface UserContextType {
    users: User[];
    projects: Project[];
    currentUser: User | null;
    setCurrentUser: (user: User | null) => void;
    loadingUsers: boolean;
    loadingProjects: boolean;
    error: string | null;
}

const UserContext = createContext<UserContextType>({
    users: [],
    projects: [],
    currentUser: null,
    setCurrentUser: () => { },
    loadingUsers: true,
    loadingProjects: true,
    error: null
});

export const UserProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [users, setUsers] = useState<User[]>([]);
    const [projects, setProjects] = useState<Project[]>([]);
    const [currentUser, setCurrentUser] = useState<User | null>(null);
    const [loadingUsers, setLoadingUsers] = useState(true);
    const [loadingProjects, setLoadingProjects] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const proxyApi = useProxyApi();

    useEffect(() => {
        let mounted = true;
        let timeoutId: NodeJS.Timeout | null = null;

        async function fetchData() {
            if (!mounted) return;

            try {
                const [userData, projectData] = await Promise.all([
                    proxyApi.getUsers(),
                    proxyApi.getProjects()
                ]);

                if (!mounted) return;

                setUsers(userData);
                if (!currentUser && userData.length > 0) {
                    setCurrentUser(userData[0]);
                }
                setProjects(projectData);
                setLoadingUsers(false);
                setLoadingProjects(false);
                setError(null);
            } catch (err) {
                if (!mounted) return;
                console.error('Failed to fetch data:', err);
                setError('Failed to fetch data');
            }
        }

        // Initial fetch
        fetchData();

        return () => {
            mounted = false;
            if (timeoutId) {
                clearTimeout(timeoutId);
            }
        };
    }, []); // Empty dependency array - only run once on mount

    return (
        <UserContext.Provider value={{
            users,
            projects,
            currentUser,
            setCurrentUser,
            loadingUsers,
            loadingProjects,
            error
        }}>
            {children}
        </UserContext.Provider>
    );
};

export const useUser = () => useContext(UserContext);
