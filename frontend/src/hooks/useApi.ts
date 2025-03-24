import axios, { AxiosInstance } from 'axios';

export const useApi = (): AxiosInstance => {
    const baseURL = `${process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000'}/api`;

    const instance = axios.create({
        baseURL,
        headers: {
            'Content-Type': 'application/json',
        },
        withCredentials: true
    });

    instance.interceptors.response.use(
        response => response,
        error => {
            if (error.response) {
                // The request was made and the server responded with a status code
                // that falls out of the range of 2xx
                console.error('API Error:', error.response.data);
            } else if (error.request) {
                // The request was made but no response was received
                console.error('Network Error:', error.request);
            } else {
                // Something happened in setting up the request that triggered an Error
                console.error('Request Error:', error.message);
            }
            return Promise.reject(error);
        }
    );

    return instance;
};
