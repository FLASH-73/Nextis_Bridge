import { Check, ChevronRight } from "lucide-react";
import type { PipelineStep, PipelineState } from "../../../lib/api/pipeline";

interface StepProgressBarProps {
  steps: PipelineStep[];
  currentIndex: number;
  state: PipelineState;
  stepElapsed?: number;
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export default function StepProgressBar({
  steps,
  currentIndex,
  state,
  stepElapsed = 0,
}: StepProgressBarProps) {
  if (steps.length === 0) return null;

  const isTerminal = state === "completed" || state === "estop" || state === "error";

  return (
    <div className="flex items-stretch gap-1 w-full">
      {steps.map((step, i) => {
        const isCompleted = i < currentIndex || (isTerminal && state === "completed");
        const isCurrent = i === currentIndex && !isTerminal;
        const isUpcoming = !isCompleted && !isCurrent;

        let bg: string;
        let text: string;
        if (isCompleted) {
          bg = "bg-neutral-200 dark:bg-zinc-700";
          text = "text-neutral-500 dark:text-zinc-400";
        } else if (isCurrent) {
          bg = "bg-black dark:bg-white";
          text = "text-white dark:text-black";
        } else {
          bg = "bg-neutral-100 dark:bg-zinc-800";
          text = "text-neutral-400 dark:text-zinc-500";
        }

        return (
          <div key={i} className="flex items-center flex-1 min-w-0">
            <div
              className={`flex-1 flex items-center gap-2 px-3 py-2.5 rounded-lg transition-colors ${bg} ${text}`}
            >
              {isCompleted ? (
                <Check className="w-3.5 h-3.5 flex-shrink-0" />
              ) : (
                <span className="text-[10px] font-bold flex-shrink-0 opacity-60">
                  {i + 1}
                </span>
              )}
              <span className="text-xs font-medium truncate">
                {step.name || `Step ${i + 1}`}
              </span>
              {isCurrent && stepElapsed > 0 && (
                <span className="text-[10px] font-mono opacity-70 flex-shrink-0 ml-auto">
                  {formatElapsed(stepElapsed)}
                </span>
              )}
            </div>
            {i < steps.length - 1 && (
              <ChevronRight
                className={`w-3.5 h-3.5 flex-shrink-0 mx-0.5 ${
                  isUpcoming
                    ? "text-neutral-300 dark:text-zinc-600"
                    : "text-neutral-400 dark:text-zinc-500"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
