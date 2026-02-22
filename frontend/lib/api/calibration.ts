import { api } from "./client";
import type { ArmCalibrationInfo, MotorState } from "./types";

export const calibrationApi = {
  arms: () =>
    api.get<{ arms: ArmCalibrationInfo[] }>("/calibration/arms"),

  state: (armId: string) =>
    api.get<{ motors: MotorState[] }>(`/calibration/${armId}/state`),

  torque: (armId: string, enabled: boolean) =>
    api.post<void>(`/calibration/${armId}/torque`, { enable: enabled }),

  homing: (armId: string) =>
    api.post<void>(`/calibration/${armId}/homing`),

  startDiscovery: (armId: string) =>
    api.post<void>(`/calibration/${armId}/discovery/start`),

  stopDiscovery: (armId: string) =>
    api.post<void>(`/calibration/${armId}/discovery/stop`),

  save: (armId: string) =>
    api.post<void>(`/calibration/${armId}/save`),

  saveNamed: (armId: string, name: string) =>
    api.post<void>(`/calibration/${armId}/save_named`, { name }),

  files: (armId: string) =>
    api.get<{ files: string[] }>(`/calibration/${armId}/files`),

  load: (armId: string, filename: string) =>
    api.post<void>(`/calibration/${armId}/load`, { filename }),

  deleteFile: (armId: string, filename: string) =>
    api.post<void>(`/calibration/${armId}/delete`, { filename }),

  inversions: (armId: string) =>
    api.get<Record<string, boolean>>(`/calibration/${armId}/inversions`),

  setInversions: (armId: string, inversions: Record<string, boolean>) =>
    api.post<void>(`/calibration/${armId}/inversions`, inversions),

  setZero: (armId: string) =>
    api.post<void>(`/calibration/${armId}/set-zero`),

  autoAlign: (armId: string) =>
    api.post<void>(`/calibration/${armId}/auto-align`),
};
