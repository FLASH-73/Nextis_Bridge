import { api } from "./client";

export const recordingApi = {
  status: () => api.get<Record<string, unknown>>("/recording/status"),

  options: () => api.get<Record<string, unknown>>("/recording/options"),

  startSession: (opts: Record<string, unknown>) =>
    api.post<{ success: boolean; error?: string }>(
      "/recording/session/start",
      opts
    ),

  stopSession: () => api.post<void>("/recording/session/stop"),

  startEpisode: () =>
    api.post<{ success: boolean; error?: string }>("/recording/episode/start"),

  stopEpisode: () => api.post<void>("/recording/episode/stop"),

  deleteLastEpisode: () => api.delete<void>("/recording/episode/last"),
};
