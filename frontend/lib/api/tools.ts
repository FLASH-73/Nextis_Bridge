import { api } from "./client";
import type { Tool, Trigger, ToolPairing } from "./types";

export const toolsApi = {
  list: () => api.get<Tool[]>("/tools"),

  create: (tool: {
    id: string;
    name: string;
    motor_type: string;
    port: string;
    motor_id: number;
    tool_type: string;
    enabled?: boolean;
    config?: Record<string, unknown>;
  }) => api.post<{ success: boolean; tool?: Tool; error?: string }>("/tools", tool),

  remove: (toolId: string) =>
    api.delete<{ success: boolean; error?: string }>(`/tools/${toolId}`),

  connect: (toolId: string) =>
    api.post<{ success: boolean; status?: string; error?: string }>(`/tools/${toolId}/connect`),

  disconnect: (toolId: string) =>
    api.post<{ success: boolean; status?: string; error?: string }>(`/tools/${toolId}/disconnect`),

  activate: (toolId: string, opts?: { speed?: number; direction?: number }) =>
    api.post<{ success: boolean; error?: string }>(`/tools/${toolId}/activate`, opts),

  deactivate: (toolId: string) =>
    api.post<{ success: boolean; error?: string }>(`/tools/${toolId}/deactivate`),

  scan: (port: string, motorType: string) =>
    api.post<{ success: boolean; found_ids: number[]; baudrate_info?: string }>(
      "/tools/scan",
      { port, motor_type: motorType }
    ),
};

export const triggersApi = {
  list: () => api.get<Trigger[]>("/triggers"),

  create: (trigger: {
    id: string;
    name: string;
    trigger_type: string;
    port: string;
    pin: number;
    active_low?: boolean;
    enabled?: boolean;
    config?: Record<string, unknown>;
  }) => api.post<{ success: boolean; trigger?: Trigger; error?: string }>("/triggers", trigger),

  remove: (triggerId: string) =>
    api.delete<{ success: boolean; error?: string }>(`/triggers/${triggerId}`),

  identify: (port: string) =>
    api.post<{ success: boolean; is_trigger: boolean; version?: string; port?: string }>(
      "/triggers/identify",
      { port }
    ),
};

export const toolPairingsApi = {
  list: () => api.get<ToolPairing[]>("/tool-pairings"),

  create: (pairing: {
    trigger_id: string;
    tool_id: string;
    name?: string;
    action: string;
    config?: Record<string, unknown>;
  }) => api.post<{ success: boolean; pairing?: ToolPairing; error?: string }>("/tool-pairings", pairing),

  remove: (triggerId: string, toolId: string) =>
    api.delete<{ success: boolean; error?: string }>("/tool-pairings", {
      trigger_id: triggerId,
      tool_id: toolId,
    }),

  startListener: () =>
    api.post<{ success: boolean; message?: string }>("/tool-pairings/listener/start"),

  stopListener: () =>
    api.post<{ success: boolean; message?: string }>("/tool-pairings/listener/stop"),

  listenerStatus: () =>
    api.get<{ running: boolean; trigger_states: Record<string, boolean>; tool_states: Record<string, boolean> }>(
      "/tool-pairings/listener/status"
    ),
};
