import { useState, useEffect, useRef, useCallback } from "react";
import RecordingBottomBar from "./RecordingBottomBar";
import RecordingSidePanel from "./RecordingSidePanel";
import { recordingApi, systemApi } from "../../../lib/api";
import { usePolling } from "../../../hooks/usePolling";
import type { RecordingStatus, EpisodeRecord } from "../../../lib/api/types";

interface RecordingActiveViewProps {
  onSessionEnded: () => void;
  datasetName?: string;
}

export default function RecordingActiveView({
  onSessionEnded,
  datasetName = "",
}: RecordingActiveViewProps) {
  const [status, setStatus] = useState<RecordingStatus>({
    session_active: true,
    episode_active: false,
    episode_count: 0,
  });
  const [sidePanelOpen, setSidePanelOpen] = useState(true);
  const [episodeStartTime, setEpisodeStartTime] = useState<number | null>(null);
  const [episodeHistory, setEpisodeHistory] = useState<EpisodeRecord[]>([]);
  const [isBusy, setIsBusy] = useState(false);

  // Track previous episode_active to detect transitions
  const prevEpisodeActive = useRef(false);

  // --- State polling (5Hz) ---
  usePolling(
    useCallback(async () => {
      try {
        const s = await recordingApi.status();
        setStatus(s);
      } catch {
        // ignore transient errors
      }
    }, []),
    200,
    true
  );

  // Detect episode_active transitions to set/clear start time
  useEffect(() => {
    if (status.episode_active && !prevEpisodeActive.current) {
      // Episode just started
      setEpisodeStartTime(Date.now());
    } else if (!status.episode_active && prevEpisodeActive.current) {
      // Episode just ended (externally or via our handlers)
      setEpisodeStartTime(null);
    }
    prevEpisodeActive.current = status.episode_active;
  }, [status.episode_active]);

  // Detect session ended externally
  useEffect(() => {
    if (!status.session_active) {
      onSessionEnded();
    }
  }, [status.session_active, onSessionEnded]);

  // --- Handlers ---
  const handleStartEpisode = useCallback(async () => {
    if (isBusy || status.episode_active) return;
    setIsBusy(true);
    try {
      await recordingApi.startEpisode();
    } catch (e) {
      console.error("Failed to start episode:", e);
    } finally {
      setIsBusy(false);
    }
  }, [isBusy, status.episode_active]);

  const handleSaveEpisode = useCallback(async () => {
    if (isBusy || !status.episode_active) return;
    setIsBusy(true);
    const duration =
      episodeStartTime != null ? (Date.now() - episodeStartTime) / 1000 : 0;
    try {
      await recordingApi.stopEpisode();
      setEpisodeHistory((prev) => [
        ...prev,
        { index: status.episode_count + 1, duration, status: "saved" },
      ]);
    } catch (e) {
      console.error("Failed to save episode:", e);
    } finally {
      setIsBusy(false);
    }
  }, [isBusy, status.episode_active, status.episode_count, episodeStartTime]);

  const handleDiscardEpisode = useCallback(async () => {
    if (isBusy || !status.episode_active) return;
    setIsBusy(true);
    const duration =
      episodeStartTime != null ? (Date.now() - episodeStartTime) / 1000 : 0;
    try {
      await recordingApi.stopEpisode();
      await recordingApi.deleteLastEpisode();
      setEpisodeHistory((prev) => [
        ...prev,
        { index: status.episode_count + 1, duration, status: "discarded" },
      ]);
    } catch (e) {
      console.error("Failed to discard episode:", e);
    } finally {
      setIsBusy(false);
    }
  }, [isBusy, status.episode_active, status.episode_count, episodeStartTime]);

  const handleEndSession = useCallback(async () => {
    if (status.episode_active) {
      if (!confirm("An episode is still recording. Discard and end session?"))
        return;
      try {
        await recordingApi.stopEpisode();
        await recordingApi.deleteLastEpisode();
      } catch {
        // proceed anyway
      }
    }
    try {
      await recordingApi.stopSession();
    } catch (e) {
      console.error("Failed to stop session:", e);
    }
    onSessionEnded();
  }, [status.episode_active, onSessionEnded]);

  // --- Keyboard shortcuts ---
  // Use refs so the keydown handler always sees the latest values
  const statusRef = useRef(status);
  const isBusyRef = useRef(isBusy);
  statusRef.current = status;
  isBusyRef.current = isBusy;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement
      )
        return;

      switch (e.key) {
        case " ":
          e.preventDefault();
          if (statusRef.current.episode_active) {
            handleSaveEpisode();
          } else {
            handleStartEpisode();
          }
          break;
        case "d":
        case "D":
          if (statusRef.current.episode_active) {
            handleDiscardEpisode();
          }
          break;
        case "Escape":
          systemApi.emergencyStop().catch(() => {});
          break;
        case "]":
          setSidePanelOpen((prev) => !prev);
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleStartEpisode, handleSaveEpisode, handleDiscardEpisode]);

  return (
    <>
      <RecordingBottomBar
        episodeActive={status.episode_active}
        episodeCount={status.episode_count}
        episodeStartTime={episodeStartTime}
        onToggleSidePanel={() => setSidePanelOpen((prev) => !prev)}
        recordingFps={30}
      />
      <RecordingSidePanel
        isOpen={sidePanelOpen}
        episodeActive={status.episode_active}
        episodeCount={status.episode_count}
        episodeStartTime={episodeStartTime}
        datasetName={datasetName}
        episodeHistory={episodeHistory}
        onSaveEpisode={handleSaveEpisode}
        onDiscardEpisode={handleDiscardEpisode}
        onEndSession={handleEndSession}
        isBusy={isBusy}
      />
    </>
  );
}
