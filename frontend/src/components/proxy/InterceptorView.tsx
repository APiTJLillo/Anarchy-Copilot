import React, { useState, useCallback } from 'react';
import { Box, Paper, Typography, Divider } from '@mui/material';
import { RequestEditor } from './RequestEditor';
import { ResponseEditor } from './ResponseEditor';

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

interface InterceptorViewProps {
    request: InterceptedRequest;
    response: InterceptedResponse | null;
    onForwardRequest: (request: InterceptedRequest) => void;
    onDropRequest: (requestId: string) => void;
    onForwardResponse: (requestId: string, response: InterceptedResponse) => void;
    onDropResponse: (requestId: string) => void;
}

export const InterceptorView: React.FC<InterceptorViewProps> = ({
    request,
    response,
    onForwardRequest,
    onDropRequest,
    onForwardResponse,
    onDropResponse,
}) => {
    const [modifiedRequest, setModifiedRequest] = useState(request);
    const [modifiedResponse, setModifiedResponse] = useState(response);

    // Request modification handlers
    const handleRequestMethodChange = useCallback((method: string) => {
        setModifiedRequest(prev => ({ ...prev, method }));
    }, []);

    const handleRequestUrlChange = useCallback((url: string) => {
        setModifiedRequest(prev => ({ ...prev, url }));
    }, []);

    const handleRequestHeadersChange = useCallback((headers: Header[]) => {
        setModifiedRequest(prev => ({ ...prev, headers }));
    }, []);

    const handleRequestBodyChange = useCallback((body: string) => {
        setModifiedRequest(prev => ({ ...prev, body }));
    }, []);

    const handleRequestForward = useCallback(() => {
        onForwardRequest(modifiedRequest);
    }, [modifiedRequest, onForwardRequest]);

    const handleRequestDrop = useCallback(() => {
        onDropRequest(request.id);
    }, [request.id, onDropRequest]);

    // Response modification handlers
    const handleResponseStatusCodeChange = useCallback((statusCode: number) => {
        setModifiedResponse(prev => prev ? { ...prev, statusCode } : null);
    }, []);

    const handleResponseHeadersChange = useCallback((headers: Header[]) => {
        setModifiedResponse(prev => prev ? { ...prev, headers } : null);
    }, []);

    const handleResponseBodyChange = useCallback((body: string) => {
        setModifiedResponse(prev => prev ? { ...prev, body } : null);
    }, []);

    const handleResponseForward = useCallback(() => {
        if (modifiedResponse) {
            onForwardResponse(request.id, modifiedResponse);
        }
    }, [request.id, modifiedResponse, onForwardResponse]);

    const handleResponseDrop = useCallback(() => {
        onDropResponse(request.id);
    }, [request.id, onDropResponse]);

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, p: 2 }}>
            {/* Request Section */}
            <Paper elevation={2} sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                    Request
                </Typography>
                <RequestEditor
                    method={modifiedRequest.method}
                    url={modifiedRequest.url}
                    headers={modifiedRequest.headers}
                    body={modifiedRequest.body}
                    onMethodChange={handleRequestMethodChange}
                    onUrlChange={handleRequestUrlChange}
                    onHeadersChange={handleRequestHeadersChange}
                    onBodyChange={handleRequestBodyChange}
                    onForward={handleRequestForward}
                    onDrop={handleRequestDrop}
                />
            </Paper>

            {/* Response Section */}
            {modifiedResponse && (
                <Paper elevation={2} sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                        Response
                    </Typography>
                    <ResponseEditor
                        statusCode={modifiedResponse.statusCode}
                        headers={modifiedResponse.headers}
                        body={modifiedResponse.body}
                        onStatusCodeChange={handleResponseStatusCodeChange}
                        onHeadersChange={handleResponseHeadersChange}
                        onBodyChange={handleResponseBodyChange}
                        onForward={handleResponseForward}
                        onDrop={handleResponseDrop}
                    />
                </Paper>
            )}
        </Box>
    );
};
