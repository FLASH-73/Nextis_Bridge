import { api, API_BASE } from "./client";
import type { DatasetInfo, EpisodeSummary, MergeValidationResult, MergeProgress } from "./types";

export const datasetsApi = {
  list: () => api.get<DatasetInfo[]>("/datasets"),

  episodes: (repoId: string) =>
    api.get<{ episodes: EpisodeSummary[] }>(
      `/datasets/${encodeURIComponent(repoId)}/episodes`
    ),

  episodeDetail: (repoId: string, index: number) =>
    api.get<Record<string, unknown>>(
      `/datasets/${encodeURIComponent(repoId)}/episode/${index}`
    ),

  deleteEpisode: (repoId: string, index: number) =>
    api.post<void>(
      `/datasets/${encodeURIComponent(repoId)}/episode/${index}`
    ),

  metadata: (repoId: string) =>
    api.get<Record<string, unknown>>(
      `/datasets/${encodeURIComponent(repoId)}`
    ),

  files: (repoId: string) =>
    api.get<Record<string, unknown>>(
      `/datasets/${encodeURIComponent(repoId)}/files`
    ),

  // Merge operations
  mergeValidate: (repos: string[]) =>
    api.post<MergeValidationResult>("/datasets/merge/validate", {
      repo_ids: repos,
    }),

  mergeStart: (data: { repo_ids: string[]; output_name: string }) =>
    api.post<{ job_id: string }>("/datasets/merge/start", data),

  mergeStatus: (jobId: string) =>
    api.get<{ status: string; progress: MergeProgress }>(
      `/datasets/merge/status/${jobId}`
    ),

  // Video feed URL (not a fetch — used for <video> src)
  videoFeedUrl: (camId: string) => `${API_BASE}/video_feed/${camId}`,
};
