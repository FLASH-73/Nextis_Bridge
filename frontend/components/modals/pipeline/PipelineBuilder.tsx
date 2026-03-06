import { useState, useMemo } from "react";
import {
  Plus,
  Loader2,
  AlertCircle,
  ChevronDown,
  Save,
  FolderOpen,
  Trash2,
  Workflow,
} from "lucide-react";
import useSWR from "swr";
import { policiesApi } from "../../../lib/api";
import {
  pipelineApi,
  type PipelineConfig,
  type PipelineStep,
  type AlignmentWarning,
} from "../../../lib/api/pipeline";
import type { PolicyInfo } from "../../../lib/api/types";
import StepCard from "./StepCard";
import ArmSelector from "./ArmSelector";
import { createDefaultStep, createDefaultConfig } from "./types";

interface PipelineBuilderProps {
  onLoad: (warnings: AlignmentWarning[]) => void;
}

export default function PipelineBuilder({ onLoad }: PipelineBuilderProps) {
  const [config, setConfig] = useState<PipelineConfig>(createDefaultConfig());
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  // Data fetching
  const { data: allPolicies } = useSWR<PolicyInfo[]>("/policies", () =>
    policiesApi.list()
  );
  const { data: savedConfigs, mutate: mutateSaved } = useSWR(
    "/pipeline/configs",
    () => pipelineApi.listConfigs()
  );

  const completedPolicies = useMemo(
    () => (allPolicies ?? []).filter((p) => p.status === "completed"),
    [allPolicies]
  );

  // Step manipulation
  const addStep = () => {
    setConfig((prev) => ({
      ...prev,
      steps: [...prev.steps, createDefaultStep(prev.steps.length)],
    }));
  };

  const updateStep = (index: number, updated: PipelineStep) => {
    setConfig((prev) => ({
      ...prev,
      steps: prev.steps.map((s, i) => (i === index ? updated : s)),
    }));
  };

  const moveStep = (index: number, direction: "up" | "down") => {
    setConfig((prev) => {
      const steps = [...prev.steps];
      const target = direction === "up" ? index - 1 : index + 1;
      if (target < 0 || target >= steps.length) return prev;
      [steps[index], steps[target]] = [steps[target], steps[index]];
      return { ...prev, steps };
    });
  };

  const removeStep = (index: number) => {
    setConfig((prev) => ({
      ...prev,
      steps: prev.steps.filter((_, i) => i !== index),
    }));
  };

  // Validation
  const canLoad =
    config.name.trim().length > 0 &&
    config.steps.length > 0 &&
    config.steps.every((s) => s.policy_id) &&
    config.active_arms.length > 0 &&
    !isLoading;

  let disabledReason = "";
  if (!config.name.trim()) disabledReason = "Enter a pipeline name";
  else if (config.steps.length === 0) disabledReason = "Add at least one step";
  else if (!config.steps.every((s) => s.policy_id))
    disabledReason = "Select a policy for every step";
  else if (config.active_arms.length === 0) disabledReason = "Select arm pairs";

  // Load & Validate
  const handleLoad = async () => {
    setError("");
    setIsLoading(true);
    try {
      const result = await pipelineApi.load(config);
      onLoad(result.alignment_warnings ?? []);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load pipeline");
    } finally {
      setIsLoading(false);
    }
  };

  // Save / Load configs
  const handleSave = async () => {
    if (!config.name.trim()) return;
    try {
      await pipelineApi.saveConfig(config);
      mutateSaved();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to save config");
    }
  };

  const handleLoadSaved = async (name: string) => {
    try {
      const loaded = await pipelineApi.loadConfig(name);
      setConfig(loaded as PipelineConfig);
      setError("");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load config");
    }
  };

  const handleDeleteSaved = async (name: string) => {
    try {
      await pipelineApi.deleteConfig(name);
      mutateSaved();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to delete config");
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-6 py-4 border-b border-neutral-100 dark:border-zinc-800 flex-shrink-0">
        <Workflow className="w-5 h-5 text-neutral-500 dark:text-zinc-400" />
        <h2 className="text-lg font-semibold text-neutral-900 dark:text-zinc-100">
          Pipeline Builder
        </h2>
      </div>

      {/* Two-column layout */}
      <div className="flex-1 overflow-hidden flex gap-6 p-6">
        {/* Left column — 60% — Step list */}
        <div className="w-3/5 flex flex-col min-h-0">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-zinc-400 mb-3 flex items-center gap-2">
            Steps
            <span className="bg-neutral-200 dark:bg-zinc-700 text-neutral-600 dark:text-zinc-300 px-1.5 py-0.5 rounded-full text-[10px] font-bold">
              {config.steps.length}
            </span>
          </h3>

          <div className="flex-1 overflow-y-auto space-y-0 pr-1">
            {config.steps.map((step, i) => {
              const isLast = i === config.steps.length - 1;
              return (
                <div key={i}>
                  <StepCard
                    step={step}
                    index={i}
                    isLast={isLast}
                    onChange={(updated) => updateStep(i, updated)}
                    onMove={(dir) => moveStep(i, dir)}
                    onRemove={() => removeStep(i)}
                    policies={completedPolicies}
                  />
                  {/* Connector line between steps */}
                  {!isLast ? (
                    <div className="flex justify-center py-1">
                      <div className="w-px h-6 bg-neutral-300 dark:bg-zinc-600 relative">
                        <ChevronDown className="w-3 h-3 absolute -bottom-1.5 -left-[5px] text-neutral-400 dark:text-zinc-500" />
                      </div>
                    </div>
                  ) : (
                    <div className="flex justify-center py-2">
                      <span className="px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider bg-neutral-200 dark:bg-zinc-700 text-neutral-500 dark:text-zinc-400">
                        END
                      </span>
                    </div>
                  )}
                </div>
              );
            })}

            {/* Add step button */}
            <button
              onClick={addStep}
              className="w-full py-3 border-2 border-dashed border-neutral-200 dark:border-zinc-700 rounded-xl text-sm text-neutral-400 dark:text-zinc-500 hover:border-neutral-400 dark:hover:border-zinc-500 hover:text-neutral-600 dark:hover:text-zinc-300 transition-colors flex items-center justify-center gap-1.5"
            >
              <Plus className="w-4 h-4" />
              Add Step
            </button>
          </div>
        </div>

        {/* Right column — 40% — Config panel */}
        <div className="w-2/5 flex flex-col gap-4 min-h-0 overflow-y-auto">
          {/* Pipeline name */}
          <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700 p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-zinc-400 mb-2">
              Pipeline Name
            </h3>
            <input
              type="text"
              value={config.name}
              onChange={(e) => setConfig((prev) => ({ ...prev, name: e.target.value }))}
              placeholder="e.g. pick-and-place"
              className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-black/20 dark:focus:ring-white/20"
            />
          </div>

          {/* Arm selector */}
          <ArmSelector
            selectedArms={config.active_arms}
            onChange={(arms) => setConfig((prev) => ({ ...prev, active_arms: arms }))}
          />

          {/* Save / Load configs */}
          <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700 p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-zinc-400 mb-3">
              Saved Configs
            </h3>
            <div className="space-y-2">
              <button
                onClick={handleSave}
                disabled={!config.name.trim()}
                className="w-full flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium bg-neutral-200 dark:bg-zinc-700 text-neutral-700 dark:text-zinc-300 hover:bg-neutral-300 dark:hover:bg-zinc-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                <Save className="w-3.5 h-3.5" />
                Save Current
              </button>

              {(savedConfigs ?? []).length > 0 && (
                <div className="space-y-1">
                  {(savedConfigs ?? []).map((cfg) => (
                    <div
                      key={cfg.name}
                      className="flex items-center gap-2 group"
                    >
                      <button
                        onClick={() => handleLoadSaved(cfg.name)}
                        className="flex-1 flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs text-neutral-600 dark:text-zinc-400 hover:bg-neutral-100 dark:hover:bg-zinc-700 transition-colors text-left"
                      >
                        <FolderOpen className="w-3 h-3 flex-shrink-0" />
                        {cfg.name}
                      </button>
                      <button
                        onClick={() => handleDeleteSaved(cfg.name)}
                        className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-red-50 dark:hover:bg-red-950/30 transition-all"
                      >
                        <Trash2 className="w-3 h-3 text-neutral-400 hover:text-red-500" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 px-6 py-4 border-t border-neutral-100 dark:border-zinc-800 space-y-2">
        {error && (
          <div className="bg-red-50 dark:bg-red-950/50 text-red-600 dark:text-red-400 px-3 py-2 rounded-lg text-xs flex items-center gap-2 border border-red-100 dark:border-red-900">
            <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
            {error}
          </div>
        )}

        <button
          onClick={handleLoad}
          disabled={!canLoad}
          className={`w-full py-3 rounded-xl font-medium text-sm transition-all ${
            canLoad
              ? "bg-black dark:bg-white text-white dark:text-black hover:bg-neutral-800 dark:hover:bg-zinc-200 hover:scale-[1.01] active:scale-[0.99] shadow-lg"
              : "bg-neutral-200 dark:bg-zinc-700 text-neutral-400 dark:text-zinc-500 cursor-not-allowed"
          }`}
        >
          {isLoading ? (
            <span className="flex items-center justify-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading policies...
            </span>
          ) : canLoad ? (
            "Load & Validate"
          ) : (
            disabledReason || "Load & Validate"
          )}
        </button>
      </div>
    </div>
  );
}
