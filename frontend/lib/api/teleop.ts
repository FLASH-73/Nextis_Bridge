import { api } from "./client";

export const teleopApi = {
  status: () => api.get<{ running: boolean }>("/teleop/status"),

  start: (opts: { force?: boolean; active_arms?: string[] }) =>
    api.post<{ success: boolean; error?: string }>("/teleop/start", opts),

  stop: () => api.post<void>("/teleop/stop"),

  data: () => api.get<Record<string, unknown>>("/teleop/data"),

  tune: (params: Record<string, number>) =>
    api.post<void>("/teleop/tune", params),

  getForceFeedback: () =>
    api.get<{ gripper: boolean; joint: boolean }>("/teleop/force-feedback"),

  setForceFeedback: (opts: { gripper?: boolean; joint?: boolean }) =>
    api.post<void>("/teleop/force-feedback", opts),

  setAssist: (params: Record<string, unknown>) =>
    api.post<void>("/teleop/assist/set", params),
};
