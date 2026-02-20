import { useRef, useEffect, useState } from "react";
import useSWR from "swr";
import { Loader2 } from "lucide-react";
import EpisodeList from "./EpisodeList";
import CameraFeedGrid from "./CameraFeedGrid";
import { armsApi, toolPairingsApi } from "../../../lib/api";
import type { Arm, EpisodeRecord, ListenerStatus } from "../../../lib/api/types";

interface RecordingSidePanelProps {
  isOpen: boolean;
  episodeActive: boolean;
  episodeCount: number;
  episodeStartTime: number | null;
  datasetName: string;
  episodeHistory: EpisodeRecord[];
  onSaveEpisode: () => void;
  onDiscardEpisode: () => void;
  onEndSession: () => void;
  isBusy: boolean;
}

function formatDuration(ms: number): string {
  const totalSec = ms / 1000;
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${String(min).padStart(2, "0")}:${sec.toFixed(1).padStart(4, "0")}`;
}

export default function RecordingSidePanel({
  isOpen,
  episodeActive,
  episodeCount,
  episodeStartTime,
  datasetName,
  episodeHistory,
  onSaveEpisode,
  onDiscardEpisode,
  onEndSession,
  isBusy,
}: RecordingSidePanelProps) {
  const [elapsed, setElapsed] = useState("00:00.0");
  const rafRef = useRef<number>(0);

  useEffect(() => {
    if (episodeStartTime == null) return;
    const start = episodeStartTime;
    const tick = () => {
      setElapsed(formatDuration(Date.now() - start));
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [episodeStartTime]);

  // Arm health polling (only when panel is open)
  const { data: armsData } = useSWR<{ arms: Arm[]; summary: { total_arms: number; connected: number } }>(
    isOpen ? "/recording/arms-health" : null,
    () => armsApi.list(),
    { refreshInterval: 3000 }
  );
  const { data: listenerStatus } = useSWR<ListenerStatus>(
    isOpen ? "/recording/listener-health" : null,
    () => toolPairingsApi.listenerStatus(),
    { refreshInterval: 3000 }
  );

  return (
    <div
      className={`fixed right-0 top-0 h-[calc(100vh-48px)] w-[300px] z-40 bg-zinc-900 border-l border-zinc-800 text-white transition-transform duration-200 ${
        isOpen ? "translate-x-0" : "translate-x-full"
      }`}
    >
      <div className="flex flex-col h-full overflow-y-auto p-4 gap-4">
        {/* Header */}
        <div>
          <h2 className="text-sm font-semibold text-zinc-100">
            Recording Session
          </h2>
          <p className="text-xs text-zinc-500 mt-0.5 truncate">
            {datasetName || "Untitled dataset"}
          </p>
        </div>

        {/* Current episode */}
        <div className="bg-zinc-800/60 rounded-xl p-3 space-y-2">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                episodeActive ? "bg-red-500 animate-pulse" : "bg-zinc-500"
              }`}
            />
            <span className="text-xs font-medium text-zinc-200">
              {episodeActive
                ? `Episode ${episodeCount + 1} — Recording`
                : "Ready"}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <div>
              <span className="text-[10px] text-zinc-500 uppercase tracking-wider">
                Duration
              </span>
              <p className="text-sm font-mono text-zinc-200 tabular-nums">
                {episodeActive ? elapsed : "--:--.-"}
              </p>
            </div>
            <div>
              <span className="text-[10px] text-zinc-500 uppercase tracking-wider">
                Episodes
              </span>
              <p className="text-sm font-mono text-zinc-200">{episodeCount}</p>
            </div>
          </div>
        </div>

        {/* Arm health */}
        {armsData && armsData.arms.length > 0 && (
          <div>
            <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold mb-2">
              Arms
            </h3>
            <div className="space-y-1.5">
              {armsData.arms.map((arm: Arm) => (
                <div key={arm.id} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${
                      arm.status === "connected" ? "bg-emerald-500"
                        : arm.status === "error" ? "bg-red-500"
                        : "bg-zinc-500"
                    }`} />
                    <span className="text-xs text-zinc-300">{arm.name}</span>
                  </div>
                  <span className={`text-xs ${
                    arm.status === "connected" ? "text-emerald-400"
                      : arm.status === "error" ? "text-red-400"
                      : "text-zinc-500"
                  }`}>
                    {arm.status === "connected" ? "OK"
                      : arm.status === "error" ? "Error"
                      : arm.status === "connecting" ? "Connecting"
                      : "Offline"}
                  </span>
                </div>
              ))}
              {listenerStatus?.running && (
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-emerald-500" />
                    <span className="text-xs text-zinc-300">Tool</span>
                  </div>
                  <span className="text-xs text-zinc-500">
                    {Object.values(listenerStatus.tool_states).some(Boolean) ? "Active" : "Idle"}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Camera feeds */}
        <div>
          <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold mb-2">
            Camera Feeds
          </h3>
          <CameraFeedGrid compact />
        </div>

        {/* Episode history */}
        <div className="flex-1 min-h-0">
          <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold mb-2">
            Episodes
          </h3>
          <EpisodeList episodes={episodeHistory} />
        </div>

        {/* Action buttons */}
        <div className="flex gap-2">
          <button
            onClick={onSaveEpisode}
            disabled={!episodeActive || isBusy}
            className="flex-1 py-2 rounded-lg text-xs font-semibold transition-colors bg-emerald-600 hover:bg-emerald-700 text-white disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center gap-1.5"
          >
            {isBusy ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : (
              "Save Episode"
            )}
          </button>
          <button
            onClick={onDiscardEpisode}
            disabled={!episodeActive || isBusy}
            className="flex-1 py-2 rounded-lg text-xs font-semibold transition-colors bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-600/30 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Discard
          </button>
        </div>

        {/* End session */}
        <button
          onClick={onEndSession}
          className="text-neutral-400 hover:text-white underline text-sm transition-colors text-center"
        >
          End Recording Session
        </button>
      </div>
    </div>
  );
}
