import { api } from "./client";

// ---------------------------------------------------------------------------
// Types (mirrors app/core/deployment/pipeline_types.py 1:1)
// ---------------------------------------------------------------------------

export type TransitionTrigger =
  | "frame_count"
  | "timeout"
  | "gripper_closed"
  | "gripper_opened"
  | "torque_spike"
  | "position_reached"
  | "manual";

export type PipelineState =
  | "idle"
  | "loading"
  | "ready"
  | "running"
  | "transitioning"
  | "completed"
  | "estop"
  | "error";

export interface TransitionCondition {
  trigger: TransitionTrigger;
  threshold_value: number;
  threshold_position: Record<string, number>;
  timeout_seconds: number;
  debounce_frames: number;
}

export interface PipelineStep {
  policy_id: string;
  name: string;
  transition: TransitionCondition | null;
  warmup_frames: number;
  speed_scale: number;
  temporal_ensemble_coeff: number | null;
}

export interface PipelineConfig {
  name: string;
  steps: PipelineStep[];
  active_arms: string[];
  loop_hz: number;
  safety_overrides: Record<string, unknown> | null;
}

export interface TransitionProgress {
  current_value: number;
  threshold_value: number;
  label: string;
  debounce_current: number;
  debounce_required: number;
}

export interface AlignmentWarning {
  step_from: string;
  step_to: string;
  joint_name: string;
  delta_rad: number;
  message: string;
}

export interface PipelineStatus {
  state: PipelineState;
  current_step_index: number;
  current_step_name: string;
  total_steps: number;
  step_frame_count: number;
  total_frame_count: number;
  step_elapsed_seconds: number;
  total_elapsed_seconds: number;
  transition_progress: TransitionProgress | null;
  error_message: string;
}

// ---------------------------------------------------------------------------
// API client
// ---------------------------------------------------------------------------

export const pipelineApi = {
  load: (config: PipelineConfig) =>
    api.post<{ status: string; total_steps: number; alignment_warnings: AlignmentWarning[] }>(
      "/pipeline/load",
      config
    ),

  start: () =>
    api.post<{ status: string }>("/pipeline/start"),

  stop: () =>
    api.post<{ status: string }>("/pipeline/stop"),

  estop: () =>
    api.post<{ status: string }>("/pipeline/estop"),

  trigger: () =>
    api.post<{ status: string }>("/pipeline/trigger"),

  status: () =>
    api.get<PipelineStatus>("/pipeline/status"),

  saveConfig: (config: PipelineConfig) =>
    api.post<{ status: string }>("/pipeline/configs/save", config),

  listConfigs: () =>
    api.get<{ name: string; path: string }[]>("/pipeline/configs"),

  loadConfig: (name: string) =>
    api.get<PipelineConfig>(`/pipeline/configs/${encodeURIComponent(name)}`),

  deleteConfig: (name: string) =>
    api.delete<{ status: string }>(`/pipeline/configs/${encodeURIComponent(name)}`),
};
