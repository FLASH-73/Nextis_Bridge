import { api } from "./client";
import type { HILStatus } from "./types";

export const hilApi = {
  status: () => api.get<HILStatus>("/hil/status"),

  startSession: (opts: Record<string, unknown>) =>
    api.post<{ success: boolean; error?: string }>("/hil/session/start", opts),

  stopSession: () => api.post<void>("/hil/session/stop"),

  startEpisode: () => api.post<void>("/hil/episode/start"),

  stopEpisode: () => api.post<void>("/hil/episode/stop"),

  nextEpisode: () => api.post<void>("/hil/episode/next"),

  resume: (opts: Record<string, unknown>) =>
    api.post<void>("/hil/resume", opts),

  retrain: (opts: Record<string, unknown>) =>
    api.post<void>("/hil/retrain", opts),
};
