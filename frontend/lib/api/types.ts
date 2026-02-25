// ─── Arms ────────────────────────────────────────────────────────────────────

export interface Arm {
  id: string;
  name: string;
  role: "leader" | "follower";
  motor_type: string;
  port: string;
  enabled: boolean;
  status: "connected" | "disconnected" | "connecting" | "error";
  calibrated: boolean;
  structural_design?: string;
}

export interface Pairing {
  leader_id: string;
  follower_id: string;
  name: string;
}

export interface Port {
  device: string;
  name: string;
  description: string;
  manufacturer?: string;
  in_use: boolean;
}

// ─── Datasets ────────────────────────────────────────────────────────────────

export interface DatasetInfo {
  repo_id: string;
  root: string;
  total_episodes: number;
  total_frames: number;
  fps?: number;
  robot_type?: string;
  features?: string[];
}

export interface EpisodeSummary {
  index: number;
  episode_index?: number;
  length?: number;
  timestamp?: number;
}

export interface VideoMetadata {
  is_depth: boolean;
  from_timestamp?: number;
}

// ─── Policies ────────────────────────────────────────────────────────────────

export interface PolicyInfo {
  id: string;
  name: string;
  policy_type: string;
  status: "completed" | "training" | "failed" | string;
  steps: number;
  total_steps: number;
  dataset_repo_id: string;
  created_at: string;
  final_loss: number | null;
  checkpoint_path: string;
  loss_history: [number, number][];
  output_dir: string;
}

// ─── Training ────────────────────────────────────────────────────────────────

export interface TrainingDataset {
  repo_id: string;
  root: string;
  fps: number;
  robot_type: string;
  total_episodes: number;
  total_frames: number;
}

export interface TrainingValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  features: {
    detected: string[];
    total_episodes: number;
    total_frames: number;
    fps: number;
    robot_type: string;
  };
}

export interface TrainingProgress {
  step: number;
  total_steps: number;
  loss: number | null;
  learning_rate: number | null;
  eta_seconds: number | null;
  epoch: number;
  loss_history: [number, number][];
}

export interface TrainingJob {
  id: string;
  status:
    | "pending"
    | "validating"
    | "training"
    | "completed"
    | "failed"
    | "cancelled";
  policy_type: string;
  dataset_repo_id: string;
  progress?: TrainingProgress;
}

// ─── RL Training ─────────────────────────────────────────────────────────────

export interface RewardClassifier {
  name: string;
  dataset_repo_id: string;
  accuracy: number;
  created_at: string;
}

export interface SARMModel {
  name: string;
  dataset_repo_id: string;
  created_at: string;
  annotation_mode: string;
}

export interface SARMTrainingStatus {
  status: string;
  epoch: number;
  total_epochs: number;
  loss: number;
  error: string;
}

export interface RLStatus {
  status: string;
  episode: number;
  total_episodes: number;
  episode_step: number;
  training_step: number;
  loss_critic: number;
  loss_actor: number;
  avg_reward: number;
  intervention_rate: number;
  online_buffer_size: number;
  offline_buffer_size: number;
  current_reward: number;
  is_human_intervening: boolean;
  error: string;
  total_interventions: number;
  total_autonomous_steps: number;
  gvl_queries: number;
  gvl_avg_latency_ms: number;
  episode_rewards: number[];
}

// ─── HIL ─────────────────────────────────────────────────────────────────────

export interface HILStatus {
  active: boolean;
  mode: "idle" | "autonomous" | "human" | "paused";
  policy_id: string;
  intervention_dataset: string;
  task: string;
  episode_active: boolean;
  episode_count: number;
  intervention_count: number;
  current_episode_interventions: number;
  autonomous_frames: number;
  human_frames: number;
  policy_config?: {
    cameras: string[];
    arms: string[];
    type: string;
  };
  movement_scale?: number;
}

// ─── Calibration ─────────────────────────────────────────────────────────────

export interface ArmCalibrationInfo {
  id: string;
  name: string;
  calibrated: boolean;
  type: "leader" | "follower";
  motor_type?: "sts3215" | "damiao";
}

export interface MotorState {
  name: string;
  id: number;
  min: number;
  max: number;
  pos: number;
  visited_min?: number;
  visited_max?: number;
}

// ─── Cameras ─────────────────────────────────────────────────────────────────

export interface CameraConfig {
  id: string;
  video_device_id: number | string;
  width: number;
  height: number;
  fps: number;
  type?: string;
  use_depth?: boolean;
}

export interface CameraDevice {
  id: number | string;
  name: string;
  path: string;
  type: "opencv" | "intelrealsense";
}

export type CameraConnectionStatus =
  | "connected"
  | "disconnected"
  | "connecting"
  | "disconnecting"
  | "error";

export interface CameraStatusEntry {
  status: CameraConnectionStatus;
  error: string;
  actual_width?: number | null;
  actual_height?: number | null;
  actual_fps?: number | null;
}

export interface CameraResolutionOption {
  width: number;
  height: number;
  fps: number[];
  label?: string;
}

export interface CameraCapabilities {
  resolutions: CameraResolutionOption[];
  native: { width: number; height: number } | null;
  connected?: boolean;
  error?: string;
}

// ─── Motor Diagnostics ──────────────────────────────────────────────────────

export interface MotorData {
  name: string;
  id: number;
  model: string;
  position: number | null;
  velocity: number | null;
  current: number | null;
  temperature: number | null;
  voltage: number | null;
  load: number | null;
  error: number;
  error_names: string[];
}

// ─── Dataset Merge ───────────────────────────────────────────────────────────

export interface MergeDatasetInfo {
  repo_id: string;
  fps: number;
  robot_type: string;
  features: string[];
  total_episodes: number;
  total_frames: number;
}

export interface MergeValidationResult {
  compatible: boolean;
  datasets: MergeDatasetInfo[];
  merged_info: {
    total_episodes: number;
    total_frames: number;
    fps: number;
    robot_type: string;
    features: string[];
  } | null;
  errors: { type: string; message: string }[];
  warnings: string[];
}

export interface MergeProgress {
  percent: number;
  message: string;
}

export type MergeStatus = "idle" | "validating" | "merging" | "completed" | "error";

// ─── File Upload ─────────────────────────────────────────────────────────────

export type UploadState = "idle" | "uploading" | "paused" | "complete" | "error";

export interface FileEntry {
  file: File;
  relativePath: string;
}

export interface FolderStructure {
  name: string;
  type: "folder" | "file";
  size?: number;
  children?: FolderStructure[];
}

// ─── Common Modal Props ──────────────────────────────────────────────────────

export interface BaseModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export interface ResizableModalProps extends BaseModalProps {
  maximizedWindow: string | null;
  setMaximizedWindow: (window: string | null) => void;
}

// ─── Tools ──────────────────────────────────────────────────────────────────

export interface Tool {
  id: string;
  name: string;
  motor_type: string;
  port: string;
  motor_id: number;
  tool_type: string;
  enabled: boolean;
  status: "connected" | "disconnected" | "error";
  config: Record<string, unknown>;
}

export interface Trigger {
  id: string;
  name: string;
  trigger_type: string;
  port: string;
  pin: number;
  active_low: boolean;
  enabled: boolean;
  config: Record<string, unknown>;
}

export interface ToolPairing {
  trigger_id: string;
  tool_id: string;
  name: string;
  action: "toggle" | "hold" | "pulse";
  config: Record<string, unknown>;
}

// ─── Recording ──────────────────────────────────────────────────────────────

export interface RecordingStatus {
  session_active: boolean;
  episode_active: boolean;
  episode_count: number;
}

export interface EpisodeRecord {
  id: number;
  index: number;
  duration: number;
  status: "saved" | "discarded";
}

export interface RecordingSessionConfig {
  repo_id: string;
  task: string;
  selected_cameras: string[] | null;
  selected_arms: string[] | null;
}

export interface RecordingOptions {
  cameras: { id: string; name: string }[];
  arms: { id: string; name: string; joints: number; status: string }[];
}

export interface ListenerStatus {
  running: boolean;
  trigger_states: Record<string, boolean>;
  tool_states: Record<string, boolean>;
  trigger_count: number;
  tool_pairing_count: number;
  ports: string[];
}
