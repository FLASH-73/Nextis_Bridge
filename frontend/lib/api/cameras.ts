import { api, API_BASE } from "./client";
import type { CameraConfig, CameraDevice, CameraStatusEntry } from "./types";

export const camerasApi = {
  config: () => api.get<CameraConfig[]>("/cameras/config"),

  updateConfig: (config: CameraConfig[]) =>
    api.post<void>("/cameras/config", config),

  scan: () => api.get<{ devices: CameraDevice[] }>("/cameras/scan"),

  status: () =>
    api.get<Record<string, CameraStatusEntry>>("/cameras/status"),

  connect: (cameraKey: string) =>
    api.post<void>(`/cameras/${cameraKey}/connect`),

  disconnect: (cameraKey: string) =>
    api.post<void>(`/cameras/${cameraKey}/disconnect`),

  // Stream URL (not a fetch — used for <img> src)
  videoFeedUrl: (camId: string) => `${API_BASE}/video_feed/${camId}`,
};
