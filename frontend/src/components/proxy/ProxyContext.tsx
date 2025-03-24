import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { Modal, Box } from '@mui/material';
import { InterceptorView } from './InterceptorView';

interface Header {
    name: string;
    value: string;
}

interface InterceptedRequest {
    id: string;
    method: string;
    url: string;
    headers: Header[];
    body: string;
}

interface InterceptedResponse {
    statusCode: number;
    headers: Header[];
    body: string;
}

interface ProxyContextType {
    interceptRequest: (request: InterceptedRequest) => Promise<InterceptedRequest>;
    interceptResponse: (requestId: string, response: InterceptedResponse) => Promise<InterceptedResponse>;
}

const ProxyContext = createContext<ProxyContextType | null>(null);

interface PendingRequest {
    request: InterceptedRequest;
    resolve: (request: InterceptedRequest) => void;
    reject: (error: Error) => void;
}

interface PendingResponse {
    requestId: string;
    response: InterceptedResponse;
    resolve: (response: InterceptedResponse) => void;
    reject: (error: Error) => void;
}

export const ProxyProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [openModal, setOpenModal] = useState(false);
    const [currentRequest, setCurrentRequest] = useState<PendingRequest | null>(null);
    const [currentResponse, setCurrentResponse] = useState<PendingResponse | null>(null);

    const interceptRequest = useCallback((request: InterceptedRequest): Promise<InterceptedRequest> => {
        return new Promise((resolve, reject) => {
            setCurrentRequest({ request, resolve, reject });
            setOpenModal(true);
        });
    }, []);

    const interceptResponse = useCallback((requestId: string, response: InterceptedResponse): Promise<InterceptedResponse> => {
        return new Promise((resolve, reject) => {
            setCurrentResponse({ requestId, response, resolve, reject });
            setOpenModal(true);
        });
    }, []);

    const handleForwardRequest = useCallback(() => {
        if (currentRequest) {
            currentRequest.resolve(currentRequest.request);
            setCurrentRequest(null);
            setOpenModal(false);
        }
    }, [currentRequest]);

    const handleDropRequest = useCallback((requestId: string) => {
        if (currentRequest && currentRequest.request.id === requestId) {
            currentRequest.reject(new Error('Request dropped by user'));
            setCurrentRequest(null);
            setOpenModal(false);
        }
    }, [currentRequest]);

    const handleForwardResponse = useCallback((requestId: string, response: InterceptedResponse) => {
        if (currentResponse && currentResponse.requestId === requestId) {
            currentResponse.resolve(response);
            setCurrentResponse(null);
            setOpenModal(false);
        }
    }, [currentResponse]);

    const handleDropResponse = useCallback((requestId: string) => {
        if (currentResponse && currentResponse.requestId === requestId) {
            currentResponse.reject(new Error('Response dropped by user'));
            setCurrentResponse(null);
            setOpenModal(false);
        }
    }, [currentResponse]);

    const handleModifyRequest = useCallback((request: InterceptedRequest) => {
        if (currentRequest) {
            setCurrentRequest(prev => prev ? { ...prev, request } : null);
        }
    }, [currentRequest]);

    const handleModifyResponse = useCallback((response: InterceptedResponse) => {
        if (currentResponse) {
            setCurrentResponse(prev => prev ? { ...prev, response } : null);
        }
    }, [currentResponse]);

    return (
        <ProxyContext.Provider value={{ interceptRequest, interceptResponse }}>
            {children}
            <Modal
                open={openModal}
                onClose={() => {/* Prevent closing by clicking outside */ }}
                aria-labelledby="proxy-interceptor-modal"
            >
                <Box sx={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    width: '80%',
                    maxHeight: '90vh',
                    bgcolor: 'background.paper',
                    borderRadius: 1,
                    boxShadow: 24,
                    overflowY: 'auto',
                    p: 4,
                }}>
                    {currentRequest && (
                        <InterceptorView
                            request={currentRequest.request}
                            response={null}
                            onForwardRequest={handleForwardRequest}
                            onDropRequest={handleDropRequest}
                            onForwardResponse={handleForwardResponse}
                            onDropResponse={handleDropResponse}
                        />
                    )}
                    {currentResponse && currentRequest && (
                        <InterceptorView
                            request={currentRequest.request}
                            response={currentResponse.response}
                            onForwardRequest={handleForwardRequest}
                            onDropRequest={handleDropRequest}
                            onForwardResponse={handleForwardResponse}
                            onDropResponse={handleDropResponse}
                        />
                    )}
                </Box>
            </Modal>
        </ProxyContext.Provider>
    );
};

export const useProxy = () => {
    const context = useContext(ProxyContext);
    if (!context) {
        throw new Error('useProxy must be used within a ProxyProvider');
    }
    return context;
};
