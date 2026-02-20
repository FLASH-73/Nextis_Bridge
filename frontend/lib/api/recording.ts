import { api } from "./client";
import type {
  RecordingSessionConfig,
  RecordingStatus,
  RecordingOptions,
} from "./types";

export const recordingApi = {
  status: () => api.get<RecordingStatus>("/recording/status"),

  options: () => api.get<RecordingOptions>("/recording/options"),

  startSession: (opts: RecordingSessionConfig) =>
    api.post<{ status: string; message: string; episode_count?: number }>(
      "/recording/session/start",
      opts
    ),

  stopSession: () =>
    api.post<{ status: string; message: string }>("/recording/session/stop"),

  startEpisode: () =>
    api.post<{ status: string; message: string }>("/recording/episode/start"),

  stopEpisode: () =>
    api.post<{ status: string; message: string; episode_count?: number }>(
      "/recording/episode/stop"
    ),

  deleteLastEpisode: () =>
    api.delete<{ status: string; message: string; episode_count?: number }>(
      "/recording/episode/last"
    ),
};
