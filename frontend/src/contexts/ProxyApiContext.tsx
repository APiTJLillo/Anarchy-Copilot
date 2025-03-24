import React, { createContext, useContext } from 'react';
import { useProxyApi as useProxyApiHook } from '../api/proxyApi';
import type { ProxySession, ProxySettings } from '../api/proxyApi';

interface ProxyApiContextType {
    createSession: (name: string, projectId: number, userId: number, settings: ProxySettings) => Promise<ProxySession>;
    startProxy: (sessionId: number, settings: ProxySettings) => Promise<any>;
    stopProxy: () => Promise<any>;
    // ... other methods
}

const ProxyApiContext = createContext<ProxyApiContextType | null>(null);

export const ProxyApiProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const proxyApi = useProxyApiHook();

    return (
        <ProxyApiContext.Provider value={proxyApi}>
            {children}
        </ProxyApiContext.Provider>
    );
};

export const useProxyApi = () => {
    const context = useContext(ProxyApiContext);
    if (!context) {
        throw new Error('useProxyApi must be used within a ProxyApiProvider');
    }
    return context;
}; 