// src/api/client.js
import axios from "axios";

export const BASE_URL = "http://localhost:8000";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 300000, // 5 min — inference can take time on CPU
});

export const predictFromFile = (formData) =>
  api.post("/api/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

export const predictFromPreset = (presetId, patientContext = {}) => {
  const fd = new FormData();
  fd.append("preset_id", presetId);
  Object.entries(patientContext).forEach(([k, v]) => v && fd.append(k, v));
  return api.post("/api/predict", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const fetchPresets = () => api.get("/api/presets");
export const fetchConfig = () => api.get("/api/config");
export const updateConfig = (cfg) => api.post("/api/config", cfg);
export const fetchProfiles = () => api.get("/api/config/profiles");
export const applyProfile = (id) => api.post(`/api/config/profiles/${id}`);
export const simulate = (data) => api.post("/api/simulate", data);

export const fetchCases = (status) =>
  api.get("/api/cases", { params: status ? { status } : {} });
export const fetchCase = (id) => api.get(`/api/cases/${id}`);
export const fetchStats = () => api.get("/api/cases/stats");
export const submitReview = (caseId, review) =>
  api.post(`/api/cases/${caseId}/review`, review);

export const healthCheck = () => api.get("/api/health");

export default api;
