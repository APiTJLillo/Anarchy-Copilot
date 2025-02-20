import axios from 'axios';
import { API_BASE_URL } from '../config';

export interface ProxySettings {
    host: string;
    port: number;
    interceptRequests: boolean;
    interceptResponses: boolean;
    allowedHosts: string[];
    excludedHosts: string[];
    maxConnections?: number;
    maxKeepaliveConnections?: number;
    keepaliveTimeout?: number;
}

export interface ProxySession {
    id: number;
    project_id?: number;
    name: string;
    settings?: ProxySettings;
    is_active: boolean;
    created_at: string;
}

class ProxyApi {
    private baseUrl = `${API_BASE_URL}/api/proxy`;

    async createSession(name: string, createdBy: number, projectId?: number, settings?: ProxySettings): Promise<ProxySession> {
        const response = await axios.post(`${this.baseUrl}/sessions`, {
            name: name,
            created_by: createdBy,
            project_id: projectId,
            settings: settings
        });
        return response.data;
    }

    async startProxy(sessionId: number, settings: ProxySettings): Promise<void> {
        await axios.post(`${this.baseUrl}/sessions/${sessionId}/start`, settings);
    }

    async stopProxy(): Promise<void> {
        await axios.post(`${this.baseUrl}/stop`);
    }

    async getStatus(): Promise<any> {
        const response = await axios.get(`${this.baseUrl}/status`);
        return response.data;
    }

    async getAnalysisResults(): Promise<any[]> {
        const response = await axios.get(`${this.baseUrl}/analysis/results`);
        return response.data;
    }

    async getHistory(): Promise<any[]> {
        const response = await axios.get(`${this.baseUrl}/history`);
        return response.data;
    }

    async clearAnalysisResults(): Promise<void> {
        await axios.delete(`${this.baseUrl}/analysis/results`);
    }
}

export const proxyApi = new ProxyApi();
