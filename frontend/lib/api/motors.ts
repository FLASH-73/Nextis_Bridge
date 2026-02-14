import { api } from "./client";

export const motorsApi = {
  scan: (port: string, motorType: string) =>
    api.post<{ found_ids: number[]; motor_info?: Record<string, unknown> }>(
      "/motors/scan",
      { port, motor_type: motorType }
    ),

  setId: (data: {
    port: string;
    motor_type: string;
    current_id: number;
    new_id: number;
  }) => api.post<{ success: boolean; error?: string }>("/motors/set-id", data),

  recover: (data: { port: string; motor_type: string; motor_id: number }) =>
    api.post<{
      success: boolean;
      log: Array<{ step: string; action: string; status: string; detail?: string }>;
      error?: string;
    }>("/motors/recover", data),
};
