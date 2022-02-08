import request from 'axios';

export const GET = 'GET';
export const POST = 'POST';
export const PUT = 'PUT';
export const DELETE = 'DELETE';

const service = (requestType, url, data = {}, config = {}) => {
    config = {
        withCredentials: true,
        ...config,
        headers: {
            Accept: 'application/json',
            ...config.headers,
        },
    };

    switch (requestType) {
        case GET: {
            return request.get(url, config);
        }
        case POST: {
            return request.post(url, data, config);
        }
        case PUT: {
            return request.put(url, data, config);
        }
        case DELETE: {
            return request.delete(url, config);
        }
        default: {
            throw new TypeError('No valid request type provided');
        }
    }
};

export default service;
