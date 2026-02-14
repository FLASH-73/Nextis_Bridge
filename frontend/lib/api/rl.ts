import { api } from "./client";
import type {
  RLStatus,
  RewardClassifier,
  SARMModel,
  SARMTrainingStatus,
} from "./types";

export const rlApi = {
  // RL training
  trainingStatus: () => api.get<RLStatus>("/rl/training/status"),

  startTraining: (config: Record<string, unknown>) =>
    api.post<void>("/rl/training/start", config),

  stopTraining: () => api.post<void>("/rl/training/stop"),

  pauseTraining: () => api.post<void>("/rl/training/pause"),

  resumeTraining: () => api.post<void>("/rl/training/resume"),

  updateSettings: (settings: Record<string, unknown>) =>
    api.patch<void>("/rl/training/settings", settings),

  // Reward classifiers
  listClassifiers: () =>
    api.get<{ classifiers: RewardClassifier[] }>("/rl/reward-classifier/list"),

  trainClassifier: (opts: Record<string, unknown>) =>
    api.post<void>("/rl/reward-classifier/train", opts),

  classifierStatus: () =>
    api.get<{ status: string; accuracy?: number; error?: string }>(
      "/rl/reward-classifier/training-status"
    ),

  // SARM models
  listSarmModels: () =>
    api.get<{ models: SARMModel[] }>("/rl/sarm/list"),

  trainSarm: (opts: Record<string, unknown>) =>
    api.post<void>("/rl/sarm/train", opts),

  sarmStatus: () =>
    api.get<SARMTrainingStatus>("/rl/sarm/training-status"),
};
