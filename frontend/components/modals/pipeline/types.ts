import type { TransitionCondition, PipelineStep, PipelineConfig } from "../../../lib/api/pipeline";

export const JOINT_OPTIONS = ["link1", "link2", "link3", "link4", "link5", "gripper"] as const;

export function createDefaultTransition(): TransitionCondition {
  return {
    trigger: "frame_count",
    threshold_value: 150,
    threshold_position: {},
    timeout_seconds: 5,
    debounce_frames: 8,
  };
}

export function createDefaultStep(index: number): PipelineStep {
  return {
    policy_id: "",
    name: `Step ${index + 1}`,
    transition: createDefaultTransition(),
    warmup_frames: 12,
    speed_scale: 1.0,
    temporal_ensemble_coeff: null,
  };
}

export function createDefaultConfig(): PipelineConfig {
  return {
    name: "",
    steps: [createDefaultStep(0)],
    active_arms: [],
    loop_hz: 30,
    safety_overrides: null,
  };
}
