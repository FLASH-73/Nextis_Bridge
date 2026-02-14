import { api, apiFetchSafe } from "./client";

export const systemApi = {
  status: () =>
    apiFetchSafe<{
      connection: string;
      execution: string;
      status: string;
    }>("/status"),

  reconnect: () =>
    api.post<{ success: boolean; error?: string }>("/system/reconnect"),

  restart: () =>
    api.post<{ success: boolean; error?: string }>("/system/reset"),

  emergencyStop: () => api.post<void>("/emergency/stop"),

  velocityLimit: () =>
    apiFetchSafe<{ velocity_limit: number; has_damiao: boolean }>(
      "/robot/velocity-limit"
    ),

  setVelocityLimit: (limit: number) =>
    api.post<void>("/robot/velocity-limit", { velocity_limit: limit }),
};
