import { api } from "./client";
import type { Arm, Pairing, Port } from "./types";

export const armsApi = {
  list: () =>
    api.get<{ arms: Arm[]; summary: { total_arms: number; connected: number } }>(
      "/arms"
    ),

  create: (arm: {
    id: string;
    name: string;
    role: string;
    motor_type: string;
    port: string;
    structural_design?: string;
  }) => api.post<{ success: boolean; error?: string }>("/arms", arm),

  update: (armId: string, data: { name?: string; port?: string }) =>
    api.put<void>(`/arms/${armId}`, data),

  delete: (armId: string) => api.delete<void>(`/arms/${armId}`),

  connect: (armId: string) =>
    api.post<{ success: boolean; error?: string }>(`/arms/${armId}/connect`),

  disconnect: (armId: string) =>
    api.post<void>(`/arms/${armId}/disconnect`),

  setHome: (armId: string) =>
    api.post<{
      success: boolean;
      home_position?: Record<string, number>;
      error?: string;
    }>(`/arms/${armId}/set-home`),

  scanPorts: () => api.get<{ ports: Port[] }>("/arms/scan-ports"),

  listPairings: () => api.get<{ pairings: Pairing[] }>("/arms/pairings"),

  createPairing: (pairing: {
    leader_id: string;
    follower_id: string;
    name: string;
  }) =>
    api.post<{ success: boolean; error?: string; warning?: string }>(
      "/arms/pairings",
      pairing
    ),

  removePairing: (leaderId: string, followerId: string) =>
    api.delete<void>("/arms/pairings", {
      leader_id: leaderId,
      follower_id: followerId,
    }),

  motorDiagnostics: (armId: string) =>
    api.get<{ motors: import("./types").MotorData[] }>(
      `/arms/${armId}/motors/diagnostics`
    ),
};
