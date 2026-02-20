import { useRef, useEffect, useState } from "react";
import { PanelRight } from "lucide-react";

interface RecordingBottomBarProps {
  episodeActive: boolean;
  episodeCount: number;
  episodeStartTime: number | null;
  onToggleSidePanel: () => void;
  recordingFps?: number;
}

function formatDuration(ms: number): string {
  const totalSec = ms / 1000;
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  const mm = String(min).padStart(2, "0");
  const ss = sec.toFixed(1).padStart(4, "0");
  return `${mm}:${ss}`;
}

export default function RecordingBottomBar({
  episodeActive,
  episodeCount,
  episodeStartTime,
  onToggleSidePanel,
  recordingFps = 30,
}: RecordingBottomBarProps) {
  const [timerState, setTimerState] = useState({ elapsed: "00:00.0", frameCount: 0 });
  const rafRef = useRef<number>(0);

  useEffect(() => {
    if (episodeStartTime == null) return;
    const start = episodeStartTime;
    const tick = () => {
      const ms = Date.now() - start;
      setTimerState({
        elapsed: formatDuration(ms),
        frameCount: Math.floor((ms / 1000) * recordingFps),
      });
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [episodeStartTime, recordingFps]);

  const episodeLabel = episodeActive
    ? `Episode ${episodeCount + 1}`
    : episodeCount > 0
      ? `${episodeCount} saved`
      : "Ready";

  return (
    <div className="fixed bottom-0 left-0 right-0 z-40 h-12 bg-zinc-900 text-white border-t border-zinc-800 flex items-center px-4 gap-5 font-mono text-sm select-none">
      {/* REC / READY indicator */}
      <div className="flex items-center gap-2 min-w-[80px]">
        {episodeActive ? (
          <>
            <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
            <span className="text-red-400 font-bold tracking-wider text-xs">
              REC
            </span>
          </>
        ) : (
          <>
            <div className="w-2.5 h-2.5 rounded-full bg-zinc-500" />
            <span className="text-zinc-400 font-medium text-xs">READY</span>
          </>
        )}
      </div>

      {/* Divider */}
      <div className="w-px h-5 bg-zinc-700" />

      {/* Episode label */}
      <span className="text-zinc-300 text-xs">{episodeLabel}</span>

      {/* Divider */}
      <div className="w-px h-5 bg-zinc-700" />

      {/* Duration */}
      <span className="text-zinc-200 tabular-nums text-xs font-semibold min-w-[70px]">
        {episodeActive ? timerState.elapsed : "--:--.-"}
      </span>

      {/* Divider */}
      <div className="w-px h-5 bg-zinc-700" />

      {/* Frame count */}
      <span className="text-zinc-400 text-xs tabular-nums">
        {episodeActive ? `${timerState.frameCount} frames` : "— frames"}
      </span>

      {/* Divider */}
      <div className="w-px h-5 bg-zinc-700" />

      {/* FPS */}
      <span className="text-zinc-500 text-xs">{recordingFps} fps</span>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Shortcut hints */}
      <span className="text-zinc-600 text-[10px]">
        SPACE start/save &middot; D discard
      </span>
      <span className="text-zinc-600 text-[10px] flex items-center gap-1">
        <kbd className="px-1.5 py-0.5 rounded bg-zinc-800 border border-zinc-700 text-zinc-400 text-[10px] font-mono">
          ESC
        </kbd>
        <span>e-stop</span>
      </span>

      {/* Side panel toggle */}
      <button
        onClick={onToggleSidePanel}
        className="p-1.5 rounded-md hover:bg-zinc-800 transition-colors text-zinc-400 hover:text-white"
        title="Toggle panel (])"
      >
        <PanelRight className="w-4 h-4" />
      </button>
    </div>
  );
}
