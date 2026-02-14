import { api } from "./client";
import type { TrainingJob, TrainingValidationResult } from "./types";

export const trainingApi = {
  validate: (data: { dataset_repo_id: string; policy_type: string }) =>
    api.post<TrainingValidationResult>("/training/validate", data),

  start: (config: Record<string, unknown>) =>
    api.post<{ job_id: string }>("/training/start", config),

  jobStatus: (jobId: string) =>
    api.get<TrainingJob>(`/training/jobs/${jobId}`),

  jobLogs: (jobId: string, limit = 1000) =>
    api.get<{ logs: string[] }>(`/training/jobs/${jobId}/logs?limit=${limit}`),

  cancelJob: (jobId: string) =>
    api.post<void>(`/training/jobs/${jobId}/cancel`),

  hardware: () =>
    api.get<Record<string, unknown>>("/training/hardware"),

  datasetQuantiles: (datasetId: string) =>
    api.get<{ has_quantiles: boolean; missing_features: string[]; message: string }>(
      `/training/dataset/${datasetId}/quantiles`
    ),

  computeQuantiles: (datasetId: string) =>
    api.post<void>(`/training/dataset/${datasetId}/compute-quantiles`),
};
