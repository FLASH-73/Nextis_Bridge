import { useState } from "react";
import { ChevronUp, ChevronDown, Trash2, ChevronRight } from "lucide-react";
import type { PipelineStep } from "../../../lib/api/pipeline";
import type { PolicyInfo } from "../../../lib/api/types";
import TransitionEditor from "./TransitionEditor";

interface StepCardProps {
  step: PipelineStep;
  index: number;
  isLast: boolean;
  onChange: (updated: PipelineStep) => void;
  onMove: (direction: "up" | "down") => void;
  onRemove: () => void;
  policies: PolicyInfo[];
}

export default function StepCard({
  step,
  index,
  isLast,
  onChange,
  onMove,
  onRemove,
  policies,
}: StepCardProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const speedColor =
    step.speed_scale >= 0.8
      ? "text-red-500"
      : step.speed_scale >= 0.5
      ? "text-amber-500"
      : "text-green-500";

  return (
    <div className="bg-white dark:bg-zinc-800 rounded-xl border border-neutral-200 dark:border-zinc-700 p-4 transition-all">
      {/* Header: index pill + name + action buttons */}
      <div className="flex items-center gap-3">
        <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-black dark:bg-white text-white dark:text-black text-xs font-bold flex-shrink-0">
          {index + 1}
        </span>
        <input
          type="text"
          value={step.name}
          onChange={(e) => onChange({ ...step, name: e.target.value })}
          className="flex-1 min-w-0 px-2 py-1 text-sm bg-transparent border-b border-neutral-200 dark:border-zinc-700 focus:border-black dark:focus:border-white outline-none text-neutral-900 dark:text-zinc-100"
          placeholder="Step name"
        />
        <div className="flex items-center gap-0.5 flex-shrink-0">
          <button
            onClick={() => onMove("up")}
            disabled={index === 0}
            className="p-1.5 rounded-lg hover:bg-neutral-100 dark:hover:bg-zinc-700 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            title="Move up"
          >
            <ChevronUp className="w-4 h-4 text-neutral-500 dark:text-zinc-400" />
          </button>
          <button
            onClick={() => onMove("down")}
            disabled={isLast}
            className="p-1.5 rounded-lg hover:bg-neutral-100 dark:hover:bg-zinc-700 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            title="Move down"
          >
            <ChevronDown className="w-4 h-4 text-neutral-500 dark:text-zinc-400" />
          </button>
          <button
            onClick={onRemove}
            className="p-1.5 rounded-lg hover:bg-red-50 dark:hover:bg-red-950/30 transition-colors"
            title="Remove step"
          >
            <Trash2 className="w-4 h-4 text-neutral-400 dark:text-zinc-500 hover:text-red-500" />
          </button>
        </div>
      </div>

      {/* Policy selector */}
      <div className="mt-3">
        <select
          value={step.policy_id}
          onChange={(e) => onChange({ ...step, policy_id: e.target.value })}
          className="w-full px-3 py-2 bg-neutral-50 dark:bg-zinc-800/50 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-black/20 dark:focus:ring-white/20"
        >
          <option value="">Select policy...</option>
          {policies.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name} ({p.policy_type})
            </option>
          ))}
        </select>
      </div>

      {/* Advanced toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="flex items-center gap-1 mt-3 text-xs text-neutral-400 dark:text-zinc-500 hover:text-neutral-600 dark:hover:text-zinc-300 transition-colors"
      >
        <ChevronRight
          className={`w-3 h-3 transition-transform ${showAdvanced ? "rotate-90" : ""}`}
        />
        Advanced
      </button>

      {/* Advanced settings */}
      {showAdvanced && (
        <div className="mt-2 space-y-3 pl-4 border-l-2 border-neutral-100 dark:border-zinc-700">
          {/* Speed scale */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs font-medium text-neutral-500 dark:text-zinc-400">
                Speed Scale
              </label>
              <span className={`text-xs font-bold ${speedColor}`}>
                {Math.round(step.speed_scale * 100)}%
              </span>
            </div>
            <input
              type="range"
              value={step.speed_scale}
              onChange={(e) => onChange({ ...step, speed_scale: parseFloat(e.target.value) })}
              min={0.1}
              max={1.0}
              step={0.05}
              className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-neutral-200 dark:bg-zinc-700"
            />
          </div>

          {/* Warmup frames */}
          <div>
            <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">
              Warmup Frames
            </label>
            <input
              type="number"
              value={step.warmup_frames}
              onChange={(e) =>
                onChange({ ...step, warmup_frames: parseInt(e.target.value) || 0 })
              }
              min={0}
              className="w-24 px-2 py-1 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-black/20 dark:focus:ring-white/20"
            />
          </div>

          {/* Temporal ensemble coefficient */}
          <div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={step.temporal_ensemble_coeff !== null}
                onChange={(e) =>
                  onChange({
                    ...step,
                    temporal_ensemble_coeff: e.target.checked ? 0.01 : null,
                  })
                }
                className="w-3.5 h-3.5 rounded border-neutral-300 dark:border-zinc-600 text-black dark:text-white focus:ring-black/20 dark:focus:ring-white/20"
              />
              <span className="text-xs font-medium text-neutral-500 dark:text-zinc-400">
                Temporal Ensemble
              </span>
            </label>
            {step.temporal_ensemble_coeff !== null && (
              <input
                type="number"
                value={step.temporal_ensemble_coeff}
                onChange={(e) =>
                  onChange({
                    ...step,
                    temporal_ensemble_coeff: parseFloat(e.target.value) || 0.01,
                  })
                }
                min={0.001}
                max={1.0}
                step={0.001}
                className="mt-1 w-24 px-2 py-1 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-black/20 dark:focus:ring-white/20"
              />
            )}
          </div>
        </div>
      )}

      {/* Transition editor — hidden for last step */}
      {!isLast && (
        <TransitionEditor
          condition={step.transition}
          onChange={(t) => onChange({ ...step, transition: t })}
        />
      )}
    </div>
  );
}
