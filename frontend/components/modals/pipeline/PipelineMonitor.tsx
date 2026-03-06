import { useState, useEffect, useCallback } from "react";
import { Square, ShieldAlert, CheckCircle2, AlertTriangle, XCircle } from "lucide-react";
import { usePolling } from "../../../hooks/usePolling";
import { pipelineApi, type PipelineConfig, type PipelineStatus, type PipelineState } from "../../../lib/api/pipeline";
import { camerasApi } from "../../../lib/api";
import type { CameraConfig } from "../../../lib/api/types";
import CameraFeed from "../../ui/CameraFeed";
import StepProgressBar from "./StepProgressBar";

const TERMINAL: PipelineState[] = ["completed", "estop", "error"];

interface PipelineMonitorProps {
  config: PipelineConfig;
  onDone: () => void;
}

export default function PipelineMonitor({ config, onDone }: PipelineMonitorProps) {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [cameras, setCameras] = useState<CameraConfig[]>([]);

  useEffect(() => {
    camerasApi.config().then(setCameras).catch(() => {});
  }, []);

  const polling = !status || !TERMINAL.includes(status.state);

  usePolling(
    useCallback(async () => {
      try {
        const s = await pipelineApi.status();
        setStatus(s);
      } catch { /* ignore transient */ }
    }, []),
    200,
    polling,
  );

  const handleStop = async () => {
    try { await pipelineApi.stop(); } catch { /* */ }
    onDone();
  };

  const handleEstop = async () => {
    try { await pipelineApi.estop(); } catch { /* */ }
  };

  const handleReset = async () => {
    try { await pipelineApi.stop(); } catch { /* */ }
    onDone();
  };

  const st = status?.state ?? "idle";
  const tp = status?.transition_progress;
  const isLastStep = status ? status.current_step_index >= status.total_steps - 1 : true;
  const showTransition = !isLastStep && tp;

  const gridCols = cameras.length <= 1 ? "grid-cols-1" : cameras.length <= 4 ? "grid-cols-2" : "grid-cols-3";

  return (
    <div className="flex flex-col h-full relative">
      {/* 1. Step Progress Bar */}
      <div className="flex-shrink-0 mb-4">
        <StepProgressBar
          steps={config.steps}
          currentIndex={status?.current_step_index ?? 0}
          state={st}
          stepElapsed={status?.step_elapsed_seconds}
        />
      </div>

      {/* 2. Transition Panel */}
      {showTransition && (
        <div className="flex-shrink-0 mb-4 bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700 p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-zinc-400">
              {tp.label}
            </span>
            {tp.debounce_required > 0 && (
              <div className="flex items-center gap-0.5">
                {Array.from({ length: tp.debounce_required }).map((_, i) => (
                  <div
                    key={i}
                    className={`w-1.5 h-1.5 rounded-full ${
                      i < tp.debounce_current
                        ? "bg-black dark:bg-white"
                        : "bg-neutral-300 dark:bg-zinc-600"
                    }`}
                  />
                ))}
              </div>
            )}
          </div>
          {tp.threshold_value > 0 && (
            <div className="w-full bg-neutral-200 dark:bg-zinc-700 rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-black dark:bg-white rounded-full transition-all"
                style={{ width: `${Math.min(100, (tp.current_value / tp.threshold_value) * 100)}%` }}
              />
            </div>
          )}
          {tp.label === "manual" && (
            <button
              onClick={() => pipelineApi.trigger().catch(() => {})}
              className="w-full py-2.5 bg-black dark:bg-white text-white dark:text-black rounded-lg text-sm font-medium hover:opacity-90 transition-opacity"
            >
              Trigger &rarr;
            </button>
          )}
        </div>
      )}

      {/* 3. Camera Grid */}
      <div className="flex-1 min-h-0 overflow-hidden rounded-xl bg-black">
        {cameras.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-zinc-500 text-sm">No cameras connected</p>
          </div>
        ) : (
          <div className={`grid ${gridCols} gap-1 h-full p-1`}>
            {cameras.map((cam) => (
              <div key={cam.id} className="aspect-video rounded-lg overflow-hidden bg-zinc-900">
                <CameraFeed cameraId={cam.id} showOverlay mode="contain" />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 4. Footer */}
      <div className="flex-shrink-0 mt-4 flex items-center justify-between">
        <span className="text-xs text-neutral-500 dark:text-zinc-400 truncate">
          {config.name} &middot; {status?.current_step_name || "—"}
          {status?.total_elapsed_seconds ? ` · ${Math.floor(status.total_elapsed_seconds)}s` : ""}
        </span>
        <div className="flex items-center gap-2">
          <button onClick={handleStop} className="px-4 py-2 rounded-lg text-xs font-medium bg-neutral-200 dark:bg-zinc-700 text-neutral-700 dark:text-zinc-300 hover:bg-neutral-300 dark:hover:bg-zinc-600 transition-colors flex items-center gap-1.5">
            <Square className="w-3 h-3" /> Stop
          </button>
          <button onClick={handleEstop} className="px-4 py-2 rounded-lg text-xs font-bold bg-red-600 text-white hover:bg-red-700 transition-colors flex items-center gap-1.5 shadow-md">
            <ShieldAlert className="w-3.5 h-3.5" /> E-STOP
          </button>
        </div>
      </div>

      {/* State Overlays */}
      {st === "completed" && (
        <div className="absolute inset-0 bg-white/90 dark:bg-zinc-900/90 backdrop-blur-sm rounded-xl flex flex-col items-center justify-center gap-4 z-10">
          <CheckCircle2 className="w-12 h-12 text-emerald-500" />
          <p className="text-lg font-semibold text-neutral-900 dark:text-zinc-100">Pipeline complete</p>
          <p className="text-sm text-neutral-500 dark:text-zinc-400">{Math.floor(status?.total_elapsed_seconds ?? 0)}s elapsed</p>
          <button onClick={onDone} className="px-6 py-2.5 rounded-xl bg-black dark:bg-white text-white dark:text-black text-sm font-medium hover:opacity-90">Done</button>
        </div>
      )}
      {st === "estop" && (
        <div className="absolute inset-0 bg-red-50/95 dark:bg-red-950/95 backdrop-blur-sm rounded-xl flex flex-col items-center justify-center gap-4 z-10">
          <AlertTriangle className="w-12 h-12 text-red-600" />
          <p className="text-lg font-bold text-red-700 dark:text-red-300">Emergency Stop</p>
          <button onClick={handleReset} className="px-6 py-2.5 rounded-xl bg-red-600 text-white text-sm font-medium hover:bg-red-700">Reset</button>
        </div>
      )}
      {st === "error" && (
        <div className="absolute inset-0 bg-white/90 dark:bg-zinc-900/90 backdrop-blur-sm rounded-xl flex flex-col items-center justify-center gap-4 z-10">
          <XCircle className="w-12 h-12 text-red-500" />
          <p className="text-lg font-semibold text-neutral-900 dark:text-zinc-100">Pipeline Error</p>
          <p className="text-sm text-neutral-500 dark:text-zinc-400 max-w-sm text-center">{status?.error_message}</p>
          <button onClick={onDone} className="px-6 py-2.5 rounded-xl bg-neutral-200 dark:bg-zinc-700 text-neutral-700 dark:text-zinc-300 text-sm font-medium hover:opacity-90">Back</button>
        </div>
      )}
    </div>
  );
}
