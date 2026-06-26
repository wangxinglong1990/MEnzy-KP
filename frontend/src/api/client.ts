import axios from "axios";

export const api = axios.create({
  baseURL: "/api",
  timeout: 300000, // 5 min for batch prediction
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    const msg = err.response?.data?.detail || err.message || "Unknown error";
    return Promise.reject(new Error(msg));
  }
);
