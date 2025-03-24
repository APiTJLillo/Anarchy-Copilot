import { EventEmitter } from 'events';

export interface WebSocketOptions {
    reconnectAttempts?: number;
    reconnectInterval?: number;
    heartbeatInterval?: number;
    debug?: boolean;
}

export interface WebSocketState {
    isConnected: boolean;
    error: Error | null;
    lastAttempt: Date | null;
    failedAttempts: number;
    protocol: string | null;
}

type MessageHandler = (data: any) => void;
type StateChangeListener = (state: WebSocketState) => void;

export class WebSocketManager {
    private static instance: WebSocketManager | null = null;
    private socket: WebSocket | null = null;
    private url: string;
    private options: WebSocketOptions;
    private readonly INTERNAL_HEADERS = {
        'x-connection-type': 'internal',
        'x-proxy-version': '0.1.0',
        'x-client-id': crypto.randomUUID()
    };
    private state: WebSocketState = {
        isConnected: false,
        error: null,
        lastAttempt: null,
        failedAttempts: 0,
        protocol: null
    };
    private stateChangeListeners: StateChangeListener[] = [];
    private channelHandlers: { [key: string]: MessageHandler[] } = {};
    private typeHandlers: { [key: string]: MessageHandler[] } = {};
    private heartbeatInterval: NodeJS.Timeout | null = null;
    private lastHeartbeatResponse: number | null = null;
    private reconnectTimeout: NodeJS.Timeout | null = null;
    private reconnectAttempts: number = 0;
    private debug: boolean = false;
    private isReconnecting: boolean = false;

    private constructor(url: string, options: WebSocketOptions = {}) {
        this.url = url;
        this.options = {
            reconnectAttempts: 10,
            reconnectInterval: 2000,
            heartbeatInterval: 15000,
            debug: true,
            ...options
        };
        this.debug = this.options.debug || false;
        this.log(`Creating new WebSocket instance with URL: ${url}`);
    }

    private log(message: string, data?: any): void {
        if (this.debug) {
            if (data !== undefined) {
                console.log(`[WebSocket] ${message}`, data);
            } else {
                console.log(`[WebSocket] ${message}`);
            }
        }
    }

    public static getInstance(url: string, options: WebSocketOptions = {}): WebSocketManager {
        if (!WebSocketManager.instance || WebSocketManager.instance.url !== url) {
            if (WebSocketManager.instance) {
                WebSocketManager.instance.disconnect();
            }
            WebSocketManager.instance = new WebSocketManager(url, options);
        }
        return WebSocketManager.instance;
    }

    private buildConnectionUrl(isInternal: boolean): URL {
        const connectionUrl = new URL(this.url);

        if (isInternal) {
            this.log('Building internal connection URL with headers');
            Object.entries(this.INTERNAL_HEADERS).forEach(([key, value]) => {
                const paramKey = key.toLowerCase();
                connectionUrl.searchParams.append(paramKey, String(value));
                this.log(`Added header as parameter: ${paramKey}=${value}`);
            });
            this.log(`Final internal connection URL: ${connectionUrl.toString()}`);
        }

        return connectionUrl;
    }

    private getConnectionDetails(url: string): { url: URL; isInternal: boolean; protocol: string | undefined } {
        const isInternal = url.includes('/internal');
        const connectionUrl = this.buildConnectionUrl(isInternal);
        const protocol = isInternal ? 'proxy-internal' : undefined;

        return {
            url: connectionUrl,
            isInternal,
            protocol
        };
    }

    public connect(): void {
        if (this.socket?.readyState === WebSocket.CONNECTING || this.socket?.readyState === WebSocket.OPEN) {
            this.log('WebSocket is already connecting or connected');
            return;
        }

        this.state.lastAttempt = new Date();
        this.log(`Connecting to ${this.url}`);

        try {
            if (this.socket) {
                this.log('Cleaning up existing socket');
                this.socket.onclose = null;
                this.socket.onerror = null;
                this.socket.close();
            }

            const { url: connectionUrl, isInternal, protocol } = this.getConnectionDetails(this.url);
            
            // Update state with protocol before connection
            this.updateState({ protocol: protocol || null });

            // Create WebSocket with explicit protocol for internal connections
            this.socket = new WebSocket(
                connectionUrl.toString(),
                protocol ? [protocol] : undefined
            );

            this.log('Connection details', {
                isInternal,
                url: connectionUrl.toString(),
                protocol: protocol || 'none',
                parameters: Object.fromEntries(connectionUrl.searchParams.entries()),
                readyState: this.socket.readyState,
                binaryType: this.socket.binaryType,
                extensions: this.socket.extensions || 'none',
                clientId: this.INTERNAL_HEADERS['x-client-id']
            });

            this.setupSocket();

            this.socket.onopen = () => {
                this.log('WebSocket connected successfully');
                this.updateState({ 
                    isConnected: true, 
                    error: null,
                    failedAttempts: 0
                });
                this.reconnectAttempts = 0;
                this.isReconnecting = false;
                this.startHeartbeat();
            };

            this.socket.onclose = (event) => {
                this.log(`WebSocket closed with code ${event.code}, reason: ${event.reason || 'No reason provided'}`);
                const failedAttempts = event.code === 1000 ? 0 : this.state.failedAttempts + 1;
                
                this.updateState({
                    isConnected: false,
                    error: event.code === 1000 ? null : new Error(`Connection closed: ${event.reason || 'Unknown reason'}`),
                    failedAttempts
                });
                
                this.stopHeartbeat();

                if (event.code !== 1000 && !this.isReconnecting) {
                    this.log('Abnormal closure, attempting reconnect');
                    this.attemptReconnect();
                }
            };

            this.socket.onerror = (error) => {
                this.log('WebSocket error occurred', error);
                this.updateState({ 
                    failedAttempts: this.state.failedAttempts + 1
                });
                
                if (!this.isReconnecting) {
                    this.attemptReconnect();
                }
            };

        } catch (error) {
            this.log(`Error creating WebSocket connection: ${error}`);
            this.updateState({
                isConnected: false,
                error: new Error(`Failed to create WebSocket connection: ${error}`),
                failedAttempts: this.state.failedAttempts + 1
            });
            if (!this.isReconnecting) {
                this.attemptReconnect();
            }
        }
    }

    private setupSocket() {
        if (!this.socket) return;

        this.socket.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.log('Received message', message);

                if (message.type === 'heartbeat_response' || message.type === 'heartbeat') {
                    this.lastHeartbeatResponse = Date.now();
                    this.log('Heartbeat response received');
                    return;
                }

                this.handleMessage(message);
            } catch (error) {
                this.log(`Error processing message: ${error}`);
            }
        };
    }

    private startHeartbeat(): void {
        this.stopHeartbeat();
        this.lastHeartbeatResponse = Date.now();
        this.log('Starting heartbeat');

        this.heartbeatInterval = setInterval(() => {
            if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
                this.log('Socket not ready, stopping heartbeat');
                this.stopHeartbeat();
                return;
            }

            const now = Date.now();
            if (this.lastHeartbeatResponse && now - this.lastHeartbeatResponse > this.options.heartbeatInterval! * 2) {
                this.log('No heartbeat response received, reconnecting...');
                this.disconnect();
                if (!this.isReconnecting) {
                    this.attemptReconnect();
                }
                return;
            }

            try {
                this.log('Sending heartbeat');
                this.socket.send(JSON.stringify({
                    type: 'heartbeat',
                    timestamp: Math.floor(Date.now() / 1000),
                    clientId: this.INTERNAL_HEADERS['x-client-id']
                }));
            } catch (error) {
                this.log(`Error sending heartbeat: ${error}`);
                this.disconnect();
                if (!this.isReconnecting) {
                    this.attemptReconnect();
                }
            }
        }, this.options.heartbeatInterval);
    }

    private stopHeartbeat(): void {
        if (this.heartbeatInterval) {
            this.log('Stopping heartbeat');
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
        this.lastHeartbeatResponse = null;
    }

    private attemptReconnect(): void {
        if (this.isReconnecting || this.reconnectTimeout) {
            this.log('Reconnection already in progress');
            return;
        }

        this.isReconnecting = true;
        if (this.reconnectAttempts >= (this.options.reconnectAttempts || 10)) {
            this.log('Max reconnection attempts reached');
            this.updateState({
                isConnected: false,
                error: new Error('Max reconnection attempts reached'),
                protocol: null
            });
            this.isReconnecting = false;
            return;
        }

        const delay = Math.min(
            this.options.reconnectInterval! * Math.pow(1.5, this.reconnectAttempts),
            30000
        );
        this.reconnectAttempts++;

        this.log(`Scheduling reconnection attempt ${this.reconnectAttempts} in ${delay}ms`);
        this.reconnectTimeout = setTimeout(() => {
            this.reconnectTimeout = null;
            this.log(`Executing reconnection attempt ${this.reconnectAttempts}`);
            this.connect();
        }, delay);
    }

    public disconnect(): void {
        this.log('Disconnecting WebSocket');
        this.stopHeartbeat();

        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }

        if (this.socket) {
            this.socket.onclose = null;
            this.socket.onerror = null;
            try {
                this.socket.close(1000);
            } catch (error) {
                this.log(`Error closing socket: ${error}`);
            }
            this.socket = null;
        }

        this.updateState({ 
            isConnected: false, 
            error: null,
            protocol: null
        });
    }

    public subscribe(channel: string, handler: MessageHandler): () => void {
        if (!this.channelHandlers[channel]) {
            this.channelHandlers[channel] = [];
        }
        this.channelHandlers[channel].push(handler);
        this.log(`Subscribed handler to channel: ${channel}`);

        return () => {
            const handlers = this.channelHandlers[channel];
            if (handlers) {
                const index = handlers.indexOf(handler);
                if (index !== -1) {
                    handlers.splice(index, 1);
                }
                if (handlers.length === 0) {
                    delete this.channelHandlers[channel];
                }
            }
        };
    }

    private handleMessage(message: any) {
        try {
            this.log('Handling message', message);
            const channel = message.channel;
            const type = message.type;

            if (channel && this.channelHandlers[channel]) {
                this.log(`Dispatching message to ${this.channelHandlers[channel].length} handlers for channel ${channel}`);
                this.channelHandlers[channel].forEach(handler => handler(message));
                return;
            }

            if (type && this.typeHandlers[type]) {
                this.log(`Dispatching message to ${this.typeHandlers[type].length} handlers for type ${type}`);
                this.typeHandlers[type].forEach(handler => handler(message));
                return;
            }

            this.log(`No handlers registered for message: ${JSON.stringify(message)}`);
        } catch (error) {
            this.log(`Error handling message: ${error}`);
        }
    }

    public send(data: any): void {
        if (this.socket?.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({
                ...data,
                clientId: this.INTERNAL_HEADERS['x-client-id']
            }));
        } else {
            this.log('Cannot send message - socket not connected');
        }
    }

    public getState(): WebSocketState {
        return { ...this.state };
    }

    public onStateChange(callback: (state: WebSocketState) => void): () => void {
        this.stateChangeListeners.push(callback);
        return () => this.stateChangeListeners.splice(this.stateChangeListeners.indexOf(callback), 1);
    }

    private updateState(newState: Partial<WebSocketState>): void {
        this.state = { ...this.state, ...newState };
        this.stateChangeListeners.forEach(callback => callback(this.state));
    }
}

export default WebSocketManager;
