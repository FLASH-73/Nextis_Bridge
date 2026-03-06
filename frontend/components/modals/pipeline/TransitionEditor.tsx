import type { TransitionCondition, TransitionTrigger } from "../../../lib/api/pipeline";
import { JOINT_OPTIONS, createDefaultTransition } from "./types";

const TRIGGER_LABELS: Record<TransitionTrigger, string> = {
  frame_count: "Frame Count",
  timeout: "Timeout",
  gripper_closed: "Gripper Closed",
  gripper_opened: "Gripper Opened",
  torque_spike: "Torque Spike",
  position_reached: "Position Reached",
  manual: "Manual Trigger",
};

interface TransitionEditorProps {
  condition: TransitionCondition | null;
  onChange: (condition: TransitionCondition) => void;
}

export default function TransitionEditor({ condition, onChange }: TransitionEditorProps) {
  const c = condition ?? createDefaultTransition();

  const update = (patch: Partial<TransitionCondition>) => {
    onChange({ ...c, ...patch });
  };

  const selectedJoint = Object.keys(c.threshold_position)[0] ?? "link1";

  return (
    <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-lg p-3 mt-3 border border-neutral-100 dark:border-zinc-700">
      <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-zinc-400 mb-2">
        Transition
      </h4>

      {/* Trigger type selector */}
      <select
        value={c.trigger}
        onChange={(e) => update({ trigger: e.target.value as TransitionTrigger })}
        className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-black/20 dark:focus:ring-white/20"
      >
        {Object.entries(TRIGGER_LABELS).map(([value, label]) => (
          <option key={value} value={value}>
            {label}
          </option>
        ))}
      </select>

      {/* Conditional inputs */}
      <div className="mt-3 space-y-2">
        {c.trigger === "frame_count" && (
          <div>
            <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">
              Frame count
            </label>
            <input
              type="number"
              value={c.threshold_value}
              onChange={(e) => update({ threshold_value: parseInt(e.target.value) || 1 })}
              min={1}
              className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-black/20 dark:focus:ring-white/20"
            />
            <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-1">
              = {(c.threshold_value / 30).toFixed(1)} seconds at 30 Hz
            </p>
          </div>
        )}

        {c.trigger === "timeout" && (
          <div>
            <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">
              Timeout (seconds)
            </label>
            <input
              type="number"
              value={c.timeout_seconds}
              onChange={(e) => update({ timeout_seconds: parseFloat(e.target.value) || 0.5 })}
              min={0.5}
              step={0.5}
              className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-black/20 dark:focus:ring-white/20"
            />
          </div>
        )}

        {(c.trigger === "gripper_closed" || c.trigger === "gripper_opened") && (
          <div>
            <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">
              Gripper position threshold
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                value={c.threshold_value}
                onChange={(e) => update({ threshold_value: parseFloat(e.target.value) })}
                min={0}
                max={1}
                step={0.01}
                className="flex-1 h-2 rounded-full appearance-none cursor-pointer bg-neutral-200 dark:bg-zinc-700"
              />
              <span className="text-sm font-mono text-neutral-700 dark:text-zinc-300 w-12 text-right">
                {c.threshold_value.toFixed(2)}
              </span>
            </div>
          </div>
        )}

        {c.trigger === "torque_spike" && (
          <>
            <div>
              <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">
                Joint
              </label>
              <select
                value={selectedJoint}
                onChange={(e) =>
                  update({
                    threshold_position: { [e.target.value]: c.threshold_value },
                  })
                }
                className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-black/20 dark:focus:ring-white/20"
              >
                {JOINT_OPTIONS.map((j) => (
                  <option key={j} value={j}>
                    {j}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">
                Torque threshold (Nm)
              </label>
              <input
                type="number"
                value={c.threshold_value}
                onChange={(e) => update({ threshold_value: parseFloat(e.target.value) || 0 })}
                min={0}
                step={0.1}
                className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-black/20 dark:focus:ring-white/20"
              />
            </div>
          </>
        )}

        {c.trigger === "position_reached" && (
          <>
            <div>
              <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">
                Joint
              </label>
              <select
                value={selectedJoint}
                onChange={(e) =>
                  update({
                    threshold_position: {
                      [e.target.value]: c.threshold_position[selectedJoint] ?? 0,
                    },
                  })
                }
                className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-black/20 dark:focus:ring-white/20"
              >
                {JOINT_OPTIONS.map((j) => (
                  <option key={j} value={j}>
                    {j}
                  </option>
                ))}
              </select>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">
                  Target (rad)
                </label>
                <input
                  type="number"
                  value={c.threshold_position[selectedJoint] ?? 0}
                  onChange={(e) =>
                    update({
                      threshold_position: {
                        [selectedJoint]: parseFloat(e.target.value) || 0,
                      },
                    })
                  }
                  step={0.01}
                  className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-black/20 dark:focus:ring-white/20"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">
                  Tolerance (rad)
                </label>
                <input
                  type="number"
                  value={c.threshold_value || 0.05}
                  onChange={(e) => update({ threshold_value: parseFloat(e.target.value) || 0.05 })}
                  min={0.001}
                  step={0.01}
                  className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-black/20 dark:focus:ring-white/20"
                />
              </div>
            </div>
          </>
        )}

        {c.trigger === "manual" && (
          <p className="text-xs text-neutral-400 dark:text-zinc-500 italic py-2">
            Step advances when the operator clicks &ldquo;Next&rdquo; or sends a trigger API call.
          </p>
        )}
      </div>

      {/* Debounce — always shown */}
      <div className="mt-3 pt-3 border-t border-neutral-100 dark:border-zinc-700">
        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">
          Debounce Frames
        </label>
        <input
          type="number"
          value={c.debounce_frames}
          onChange={(e) => update({ debounce_frames: parseInt(e.target.value) || 0 })}
          min={0}
          max={60}
          className="w-20 px-2 py-1 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm text-neutral-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-black/20 dark:focus:ring-white/20"
        />
        <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-1">
          Condition must hold for this many consecutive frames
        </p>
      </div>
    </div>
  );
}
