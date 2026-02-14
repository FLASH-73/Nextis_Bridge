import { api } from "./client";
import type { PolicyInfo } from "./types";

export const policiesApi = {
  list: () => api.get<PolicyInfo[]>("/policies"),

  get: (policyId: string) => api.get<PolicyInfo>(`/policies/${policyId}`),

  update: (policyId: string, data: { name?: string }) =>
    api.put<void>(`/policies/${policyId}`, data),

  delete: (policyId: string) => api.delete<void>(`/policies/${policyId}`),

  deploy: (policyId: string) =>
    api.post<{ success: boolean; error?: string }>(
      `/policies/${policyId}/deploy`
    ),

  resume: (policyId: string) =>
    api.post<{ job_id: string }>(`/policies/${policyId}/resume`),
};
