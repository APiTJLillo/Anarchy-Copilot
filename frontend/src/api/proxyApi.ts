import { useApi } from '../hooks/useApi';

// Remove hardcoded API_BASE_URL
// const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

export interface ProxyConfig {
    host: string;
    port: number;
    intercept_requests: boolean;
    intercept_responses: boolean;
    allowed_hosts: string[];
    excluded_hosts: string[];
}

export interface ProxySettings {
    host: string;
    port: number;
    interceptRequests: boolean;
    interceptResponses: boolean;
    allowedHosts: string[];
    excludedHosts: string[];
    maxConnections: number;
    maxKeepaliveConnections: number;
    keepaliveTimeout: number;
}

export interface ProxySession {
    id: number;
    name: string;
    project_id: number;
    user_id: number;
    settings: ProxySettings;
    is_active: boolean;
    start_time: string;
    end_time?: string;
}

export interface ConnectionEvent {
    type: 'request' | 'response';
    direction: 'browser-proxy' | 'proxy-web' | 'web-proxy' | 'proxy-browser';
    timestamp: number;
    status: 'pending' | 'success' | 'error';
    bytes_transferred?: number;
}

export interface Connection {
    id: string;
    host: string;
    port: number;
    start_time: number;
    end_time?: number;
    status: 'active' | 'closed' | 'error';
    events: ConnectionEvent[];
    bytes_received: number;
    bytes_sent: number;
    requests_processed: number;
    error?: string;
}

export interface AnalysisResult {
    id: number;
    timestamp: string;
    category: string;
    severity: 'info' | 'low' | 'medium' | 'high' | 'critical';
    description: string;
    details: any;
    request_id?: string;
    requestId?: string;  // Alternative field name
    false_positive?: boolean;
    notes?: string;
    ruleName?: string;
    evidence?: string;
    findings?: any[];
    analysis_type?: string;
    history_entry_id?: number;
    analysis_metadata?: Record<string, any>;
}

export interface Condition {
    field: string;
    operator: string;
    value: string;
    use_regex?: boolean;
}

export interface Modification {
    field: string;
    value: string;
    headers?: Record<string, string>;
    body?: string;
    status_code?: number;
}

export type ActionType = 'FORWARD' | 'BLOCK' | 'MODIFY';

export interface RuleAction {
    type: ActionType;
    modifications: Modification[];
}

// Helper constants for rule actions
export const RuleActions: Record<ActionType, ActionType> = {
    FORWARD: 'FORWARD',
    BLOCK: 'BLOCK',
    MODIFY: 'MODIFY'
};

export interface CreateRuleRequest {
    name: string;
    conditions: Condition[];
    modifications: Modification[];
    action?: RuleAction;
    enabled: boolean;
    order: number;
    session_id: number;
}

export interface InterceptionRule extends CreateRuleRequest {
    id: number;
    created_at: string;
    updated_at: string;
    action: RuleAction;
}

export interface VersionInfo {
    version: string;
    name: string;
    api_compatibility: string;
}

export interface User {
    id: number;
    username: string;
    email: string;
}

export interface Project {
    id: number;
    name: string;
    description?: string;
}

export const useProxyApi = () => {
    const api = useApi();

    return {
        // User and Project Management
        async getUsers(): Promise<User[]> {
            const response = await api.get('/users');
            return response.data;
        },

        async getProjects(): Promise<Project[]> {
            const response = await api.get('/projects');
            return response.data;
        },

        async getCurrentProject(): Promise<Project> {
            const response = await api.get('/projects/current');
            return response.data;
        },

        // Proxy Management
        async getStatus() {
            const response = await api.get('/proxy/status');
            return response.data;
        },

        async getVersion(): Promise<VersionInfo> {
            const response = await api.get('/version');
            return response.data;
        },

        async getHealth() {
            const response = await api.get('/health');
            return response.data;
        },

        async getConnections(): Promise<Connection[]> {
            const response = await api.get('/proxy/connections');
            return response.data;
        },

        async createSession(
            name: string,
            projectId: number,
            userId: number,
            settings: ProxySettings
        ): Promise<ProxySession> {
            const response = await api.post('/proxy/sessions', {
                name,
                project_id: projectId,
                user_id: userId,
                settings
            });
            return response.data;
        },

        async startProxy(sessionId: number, settings: ProxySettings): Promise<any> {
            const response = await api.post(`/proxy/sessions/${sessionId}/start`, settings);
            return response.data;
        },

        async stopProxy() {
            const response = await api.post('/proxy/stop');
            return response.data;
        },

        async getHistory() {
            const response = await api.get('/proxy/history');
            return response.data;
        },

        async getAnalysisResults() {
            const response = await api.get('/proxy/analysis');
            return response.data;
        },

        async clearAnalysisResults() {
            const response = await api.post('/proxy/analysis/clear');
            return response.data;
        },

        // Rule management methods
        async getRules(sessionId: number): Promise<InterceptionRule[]> {
            const response = await api.get(`/proxy/sessions/${sessionId}/rules`);
            return response.data;
        },

        async createRule(sessionId: number, rule: CreateRuleRequest): Promise<InterceptionRule> {
            const response = await api.post(`/proxy/sessions/${sessionId}/rules`, rule);
            return response.data;
        },

        async updateRule(sessionId: number, ruleId: number, rule: Partial<CreateRuleRequest>): Promise<InterceptionRule> {
            const response = await api.patch(`/proxy/sessions/${sessionId}/rules/${ruleId}`, rule);
            return response.data;
        },

        async deleteRule(sessionId: number, ruleId: number): Promise<void> {
            await api.delete(`/proxy/sessions/${sessionId}/rules/${ruleId}`);
        },

        async reorderRules(sessionId: number, ruleIds: number[]): Promise<InterceptionRule[]> {
            const response = await api.post(`/proxy/sessions/${sessionId}/rules/reorder`, {
                rule_ids: ruleIds
            });
            return response.data;
        }
    };
};

// For backward compatibility, export a default object that throws an error if used
export default {
    getUsers: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    getProjects: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    getCurrentProject: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    getStatus: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    getVersion: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    getHealth: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    getConnections: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    createSession: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    startProxy: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    stopProxy: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    getHistory: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    getAnalysisResults: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    clearAnalysisResults: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    getRules: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    createRule: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    updateRule: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    deleteRule: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    },
    reorderRules: () => {
        throw new Error('Please use the useProxyApi hook instead of the default export');
    }
} as const;
