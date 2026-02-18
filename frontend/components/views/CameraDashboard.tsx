"use client";

import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  Settings,
  Power,
  PowerOff,
  Camera,
  Loader2,
  ChevronLeft,
  ChevronRight,
  PictureInPicture2,
} from "lucide-react";
import { camerasApi, recordingApi } from "../../lib/api";
import { usePolling } from "../../hooks/usePolling";
import CameraFeed from "../ui/CameraFeed";
import CameraLayoutSelector from "../ui/CameraLayoutSelector";
import CameraModal from "../modals/cameras";
import type { LayoutMode } from "../ui/CameraLayoutSelector";
import type { CameraConfig, CameraStatusEntry } from "../../lib/api/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface RecordingStatus {
  session_active?: boolean;
  episode_active?: boolean;
  episode_count?: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function connectedCount(
  configs: CameraConfig[],
  status: Record<string, CameraStatusEntry>
): number {
  return configs.filter((c) => status[c.id]?.status === "connected").length;
}

function fpsColor(actual: number | null | undefined, configured: number): string {
  if (actual == null) return "text-white/40";
  const ratio = actual / configured;
  if (ratio < 0.25) return "text-red-400";
  if (ratio < 0.5) return "text-orange-400";
  return "text-emerald-400";
}

function statusDotColor(status?: CameraStatusEntry): string {
  if (!status) return "bg-white/20";
  switch (status.status) {
    case "connected":
      return "bg-emerald-500";
    case "connecting":
    case "disconnecting":
      return "bg-yellow-500 animate-pulse";
    case "error":
      return "bg-red-500";
    default:
      return "bg-white/20";
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CameraDashboard() {
  const router = useRouter();
  const mountedRef = useRef(true);

  // Layout
  const [layoutMode, setLayoutMode] = useState<LayoutMode>("grid");
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);

  // Data
  const [cameraConfigs, setCameraConfigs] = useState<CameraConfig[]>([]);
  const [cameraStatus, setCameraStatus] = useState<Record<string, CameraStatusEntry>>({});
  const [recordingStatus, setRecordingStatus] = useState<RecordingStatus>({});

  // UI state
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isConnectingAll, setIsConnectingAll] = useState(false);
  const [isDisconnectingAll, setIsDisconnectingAll] = useState(false);
  const [pipCamera, setPipCamera] = useState<string | null>(null);
  const [showHealthStrip, setShowHealthStrip] = useState(true);

  // PiP refs
  const pipIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pipVideoRef = useRef<HTMLVideoElement | null>(null);

  // Derived
  const connectedCameras = useMemo(
    () => cameraConfigs.filter((c) => cameraStatus[c.id]?.status === "connected"),
    [cameraConfigs, cameraStatus]
  );

  const isRecording = recordingStatus.episode_active === true;

  // Ensure selectedCamera is valid
  const effectiveSelected = useMemo(() => {
    if (selectedCamera && cameraConfigs.some((c) => c.id === selectedCamera)) {
      return selectedCamera;
    }
    return connectedCameras[0]?.id ?? cameraConfigs[0]?.id ?? null;
  }, [selectedCamera, cameraConfigs, connectedCameras]);

  // Index for fullscreen cycling
  const selectedIndex = useMemo(() => {
    if (!effectiveSelected) return 0;
    const idx = cameraConfigs.findIndex((c) => c.id === effectiveSelected);
    return idx >= 0 ? idx : 0;
  }, [effectiveSelected, cameraConfigs]);

  // ---------------------------------------------------------------------------
  // Data fetching
  // ---------------------------------------------------------------------------

  // Initial config load
  useEffect(() => {
    mountedRef.current = true;
    camerasApi.config().then((data) => {
      if (mountedRef.current && Array.isArray(data)) setCameraConfigs(data);
    }).catch(() => {});
    return () => { mountedRef.current = false; };
  }, []);

  // Poll camera status
  usePolling(
    useCallback(() => {
      camerasApi.status().then((s) => {
        if (mountedRef.current) setCameraStatus(s);
      }).catch(() => {});
    }, []),
    3000,
    true
  );

  // Poll recording status
  usePolling(
    useCallback(() => {
      recordingApi.status().then((s: any) => {
        if (mountedRef.current) setRecordingStatus(s ?? {});
      }).catch(() => {});
    }, []),
    2000,
    true
  );

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  const handleConnectAll = useCallback(async () => {
    setIsConnectingAll(true);
    try {
      for (const cfg of cameraConfigs) {
        if (cameraStatus[cfg.id]?.status === "connected") continue;
        try { await camerasApi.connect(cfg.id); } catch {}
      }
    } finally {
      if (mountedRef.current) setIsConnectingAll(false);
    }
  }, [cameraConfigs, cameraStatus]);

  const handleDisconnectAll = useCallback(async () => {
    setIsDisconnectingAll(true);
    try {
      for (const cfg of cameraConfigs) {
        if (cameraStatus[cfg.id]?.status !== "connected") continue;
        try { await camerasApi.disconnect(cfg.id); } catch {}
      }
    } finally {
      if (mountedRef.current) setIsDisconnectingAll(false);
    }
  }, [cameraConfigs, cameraStatus]);

  const handleReconnectAll = useCallback(async () => {
    setIsConnectingAll(true);
    try {
      for (const cfg of cameraConfigs) {
        try { await camerasApi.disconnect(cfg.id); } catch {}
      }
      // Short pause before reconnecting
      await new Promise((r) => setTimeout(r, 500));
      for (const cfg of cameraConfigs) {
        try { await camerasApi.connect(cfg.id); } catch {}
      }
    } finally {
      if (mountedRef.current) setIsConnectingAll(false);
    }
  }, [cameraConfigs]);

  const handleSettingsClose = useCallback(() => {
    setIsSettingsOpen(false);
    camerasApi.config().then((data) => {
      if (mountedRef.current && Array.isArray(data)) setCameraConfigs(data);
    }).catch(() => {});
  }, []);

  const cycleCamera = useCallback(
    (direction: 1 | -1) => {
      if (cameraConfigs.length === 0) return;
      const next = (selectedIndex + direction + cameraConfigs.length) % cameraConfigs.length;
      setSelectedCamera(cameraConfigs[next].id);
    },
    [cameraConfigs, selectedIndex]
  );

  // ---------------------------------------------------------------------------
  // PiP
  // ---------------------------------------------------------------------------

  const startPip = useCallback(
    (cameraId: string) => {
      // Clean up existing PiP
      stopPip();

      const imgEl = document.querySelector(
        `img[data-camera-id="${cameraId}"]`
      ) as HTMLImageElement | null;
      if (!imgEl) return;

      const canvas = document.createElement("canvas");
      canvas.width = imgEl.naturalWidth || 640;
      canvas.height = imgEl.naturalHeight || 480;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Draw frames periodically
      pipIntervalRef.current = setInterval(() => {
        if (imgEl.complete && imgEl.naturalWidth > 0) {
          canvas.width = imgEl.naturalWidth;
          canvas.height = imgEl.naturalHeight;
          ctx.drawImage(imgEl, 0, 0);
        }
      }, 33);

      const stream = canvas.captureStream(30);
      const video = document.createElement("video");
      video.srcObject = stream;
      video.muted = true;
      video.style.display = "none";
      document.body.appendChild(video);
      pipVideoRef.current = video;

      video.play().then(() => {
        video
          .requestPictureInPicture()
          .then(() => {
            if (mountedRef.current) setPipCamera(cameraId);
          })
          .catch(() => {
            stopPip();
          });
      });

      video.addEventListener("leavepictureinpicture", () => {
        stopPip();
      });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  const stopPip = useCallback(() => {
    if (pipIntervalRef.current) {
      clearInterval(pipIntervalRef.current);
      pipIntervalRef.current = null;
    }
    if (pipVideoRef.current) {
      pipVideoRef.current.srcObject = null;
      pipVideoRef.current.remove();
      pipVideoRef.current = null;
    }
    if (document.pictureInPictureElement) {
      document.exitPictureInPicture().catch(() => {});
    }
    if (mountedRef.current) setPipCamera(null);
  }, []);

  // Cleanup PiP on unmount
  useEffect(() => {
    return () => { stopPip(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------------------------------------------------------------------------
  // Keyboard shortcuts
  // ---------------------------------------------------------------------------

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (isSettingsOpen) return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      switch (e.key) {
        case "g":
        case "G":
          e.preventDefault();
          setLayoutMode("grid");
          break;
        case "f":
        case "F":
          e.preventDefault();
          setLayoutMode("focus");
          break;
        case " ":
          e.preventDefault();
          setLayoutMode((prev) => (prev === "fullscreen" ? "grid" : "fullscreen"));
          break;
        case "Escape":
          e.preventDefault();
          setLayoutMode("grid");
          break;
        case "ArrowLeft":
          if (layoutMode === "fullscreen") {
            e.preventDefault();
            cycleCamera(-1);
          }
          break;
        case "ArrowRight":
          if (layoutMode === "fullscreen") {
            e.preventDefault();
            cycleCamera(1);
          }
          break;
        case "r":
        case "R":
          e.preventDefault();
          handleReconnectAll();
          break;
        default: {
          const num = parseInt(e.key, 10);
          if (num >= 1 && num <= 9 && num <= cameraConfigs.length) {
            e.preventDefault();
            setSelectedCamera(cameraConfigs[num - 1].id);
            setLayoutMode("fullscreen");
          }
        }
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isSettingsOpen, layoutMode, cameraConfigs, cycleCamera, handleReconnectAll]);

  // ---------------------------------------------------------------------------
  // Render: Empty state
  // ---------------------------------------------------------------------------

  const renderEmptyState = () => (
    <div className="flex-1 flex items-center justify-center">
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center justify-center">
          <Camera className="w-8 h-8 text-white/20" />
        </div>
        <div>
          <h3 className="text-white font-medium text-lg">No cameras configured</h3>
          <p className="text-white/40 text-sm mt-1">
            Set up your cameras to start monitoring
          </p>
        </div>
        <button
          onClick={() => setIsSettingsOpen(true)}
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded-xl transition-colors duration-200"
        >
          Configure Cameras
        </button>
      </div>
    </div>
  );

  // ---------------------------------------------------------------------------
  // Render: Camera card wrapper (used in Grid & Focus)
  // ---------------------------------------------------------------------------

  const renderCameraCard = (
    cfg: CameraConfig,
    opts?: { large?: boolean; onClick?: () => void }
  ) => {
    const status = cameraStatus[cfg.id];
    const isConnected = status?.status === "connected";

    return (
      <div
        key={cfg.id}
        className={`bg-neutral-900 rounded-2xl border border-white/5 overflow-hidden relative group transition-all duration-200 ease-out ${
          opts?.large ? "" : "hover:scale-[1.01] hover:shadow-xl"
        } ${opts?.onClick ? "cursor-pointer" : ""}`}
        onClick={opts?.onClick}
      >
        {isConnected ? (
          <CameraFeed
            cameraId={cfg.id}
            mode="contain"
            maxStreamWidth={opts?.large ? 1280 : 800}
            quality={opts?.large ? 90 : 80}
            showOverlay={true}
            showFullscreenButton={false}
            label={cfg.id}
          />
        ) : (
          <div className="w-full h-full min-h-[200px] flex items-center justify-center">
            <div className="flex flex-col items-center gap-2">
              <Camera className="w-6 h-6 text-white/15" />
              <span className="text-white/30 text-xs">{cfg.id}</span>
              <span className="text-white/20 text-[10px]">
                {status?.status ?? "disconnected"}
              </span>
            </div>
          </div>
        )}

        {/* Recording overlay */}
        {isRecording && isConnected && (
          <div className="absolute top-3 right-3 z-30 flex items-center gap-1.5 bg-red-600/80 backdrop-blur-sm rounded-full px-2.5 py-1">
            <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
            <span className="text-white text-[10px] font-bold tracking-wider">
              REC
            </span>
          </div>
        )}

        {/* PiP button */}
        {isConnected && (
          <div className="absolute bottom-3 left-3 z-30 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            <button
              onClick={(e) => {
                e.stopPropagation();
                if (pipCamera === cfg.id) {
                  stopPip();
                } else {
                  startPip(cfg.id);
                }
              }}
              className={`p-1.5 rounded-full border transition-colors duration-200 ${
                pipCamera === cfg.id
                  ? "bg-emerald-600/80 border-emerald-500/50 text-white"
                  : "bg-black/50 backdrop-blur-md border-white/10 text-white/70 hover:text-white"
              }`}
              title={pipCamera === cfg.id ? "Exit PiP" : "Picture-in-Picture"}
            >
              <PictureInPicture2 className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>
    );
  };

  // ---------------------------------------------------------------------------
  // Render: Grid layout
  // ---------------------------------------------------------------------------

  const renderGrid = () => (
    <div
      className="grid gap-4 h-full auto-rows-fr"
      style={{
        gridTemplateColumns: "repeat(auto-fill, minmax(min(400px, 100%), 1fr))",
      }}
    >
      {cameraConfigs.map((cfg) => renderCameraCard(cfg))}
    </div>
  );

  // ---------------------------------------------------------------------------
  // Render: Focus layout
  // ---------------------------------------------------------------------------

  const renderFocus = () => {
    const sidebarCameras = cameraConfigs.filter(
      (c) => c.id !== effectiveSelected
    );

    return (
      <div className="flex gap-4 h-full">
        {/* Primary (70%) */}
        <div className="flex-[7] min-w-0">
          {effectiveSelected &&
            renderCameraCard(
              cameraConfigs.find((c) => c.id === effectiveSelected) ??
                cameraConfigs[0],
              { large: true }
            )}
        </div>

        {/* Sidebar (30%) */}
        {sidebarCameras.length > 0 && (
          <div className="flex-[3] min-w-0 flex flex-col gap-3 overflow-y-auto">
            {sidebarCameras.map((cfg) =>
              renderCameraCard(cfg, {
                onClick: () => setSelectedCamera(cfg.id),
              })
            )}
          </div>
        )}
      </div>
    );
  };

  // ---------------------------------------------------------------------------
  // Render: Fullscreen layout
  // ---------------------------------------------------------------------------

  const renderFullscreen = () => {
    const cfg = cameraConfigs[selectedIndex];
    if (!cfg) return renderEmptyState();

    const status = cameraStatus[cfg.id];
    const isConnected = status?.status === "connected";

    return (
      <div className="relative h-full group">
        {/* Camera feed */}
        <div className="h-full">
          {isConnected ? (
            <CameraFeed
              cameraId={cfg.id}
              mode="contain"
              maxStreamWidth={1280}
              quality={90}
              showOverlay={false}
              showFullscreenButton={false}
            />
          ) : (
            <div className="h-full flex items-center justify-center bg-neutral-900 rounded-2xl">
              <div className="flex flex-col items-center gap-2">
                <Camera className="w-8 h-8 text-white/15" />
                <span className="text-white/30 text-sm">{cfg.id} - Disconnected</span>
              </div>
            </div>
          )}
        </div>

        {/* Recording overlay */}
        {isRecording && isConnected && (
          <div className="absolute top-4 right-4 z-30 flex items-center gap-1.5 bg-red-600/80 backdrop-blur-sm rounded-full px-3 py-1.5">
            <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
            <span className="text-white text-xs font-bold tracking-wider">REC</span>
          </div>
        )}

        {/* Bottom overlay: camera name + resolution */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
          <div className="bg-black/60 backdrop-blur-md rounded-full px-4 py-2 flex items-center gap-3 border border-white/10">
            <span className="text-white text-sm font-medium">{cfg.id}</span>
            {status?.actual_width && status?.actual_height && (
              <span className="text-white/50 font-mono text-xs">
                {status.actual_width}x{status.actual_height}
              </span>
            )}
            <span className="text-white/30 text-xs">
              {selectedIndex + 1} / {cameraConfigs.length}
            </span>
          </div>
        </div>

        {/* Navigation arrows */}
        {cameraConfigs.length > 1 && (
          <>
            <button
              onClick={() => cycleCamera(-1)}
              className="absolute left-4 top-1/2 -translate-y-1/2 z-30 opacity-0 group-hover:opacity-100 transition-opacity duration-200 p-3 bg-black/40 hover:bg-black/60 backdrop-blur-md rounded-full border border-white/10 text-white/70 hover:text-white"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>
            <button
              onClick={() => cycleCamera(1)}
              className="absolute right-4 top-1/2 -translate-y-1/2 z-30 opacity-0 group-hover:opacity-100 transition-opacity duration-200 p-3 bg-black/40 hover:bg-black/60 backdrop-blur-md rounded-full border border-white/10 text-white/70 hover:text-white"
            >
              <ChevronRight className="w-5 h-5" />
            </button>
          </>
        )}

        {/* PiP button */}
        {isConnected && (
          <div className="absolute top-4 left-4 z-30 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            <button
              onClick={() => {
                if (pipCamera === cfg.id) stopPip();
                else startPip(cfg.id);
              }}
              className={`p-2 rounded-full border transition-colors duration-200 ${
                pipCamera === cfg.id
                  ? "bg-emerald-600/80 border-emerald-500/50 text-white"
                  : "bg-black/40 hover:bg-black/60 backdrop-blur-md border-white/10 text-white/70 hover:text-white"
              }`}
              title={pipCamera === cfg.id ? "Exit PiP" : "Picture-in-Picture"}
            >
              <PictureInPicture2 className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
    );
  };

  // ---------------------------------------------------------------------------
  // Render: Health strip
  // ---------------------------------------------------------------------------

  const renderHealthStrip = () => (
    <div
      className={`h-8 border-t border-white/5 bg-neutral-900/50 px-4 flex items-center gap-3 overflow-x-auto ${
        showHealthStrip ? "flex" : "hidden"
      } sm:flex`}
    >
      {/* Toggle for mobile */}
      <button
        onClick={() => setShowHealthStrip((p) => !p)}
        className="sm:hidden text-white/30 text-[10px] shrink-0"
      >
        {showHealthStrip ? "Hide" : "Status"}
      </button>

      {cameraConfigs.map((cfg) => {
        const status = cameraStatus[cfg.id];
        const actualFps = status?.actual_fps;
        const configuredFps = cfg.fps || 30;

        return (
          <div
            key={cfg.id}
            className="flex items-center gap-2 shrink-0 px-2.5 py-0.5 rounded-full bg-white/5 border border-white/5"
          >
            <div className={`w-1.5 h-1.5 rounded-full ${statusDotColor(status)}`} />
            <span className="text-white/60 text-[10px]">{cfg.id}</span>
            {status?.actual_width && status?.actual_height && (
              <span className="text-white/30 font-mono text-[10px]">
                {status.actual_width}x{status.actual_height}
              </span>
            )}
            <span
              className={`font-mono text-[10px] font-medium ${fpsColor(
                actualFps,
                configuredFps
              )}`}
            >
              {actualFps != null ? `${Math.round(actualFps)}fps` : "--fps"}
            </span>
          </div>
        );
      })}
    </div>
  );

  // ---------------------------------------------------------------------------
  // Main render
  // ---------------------------------------------------------------------------

  const connected = connectedCount(cameraConfigs, cameraStatus);

  return (
    <div className="h-screen w-screen bg-neutral-950 flex flex-col text-white">
      {/* Header */}
      <header className="h-14 shrink-0 border-b border-white/5 px-4 flex items-center justify-between gap-4">
        {/* Left: Back + Title + Badge */}
        <div className="flex items-center gap-3 min-w-0">
          <button
            onClick={() => router.push("/")}
            className="p-2 rounded-xl hover:bg-white/5 transition-colors duration-200 text-white/60 hover:text-white"
            title="Back to dashboard"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <h1 className="text-sm font-semibold tracking-tight truncate">
            Camera Dashboard
          </h1>
          {cameraConfigs.length > 0 && (
            <span className="px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-[10px] text-white/50 font-mono shrink-0">
              {connected}/{cameraConfigs.length} connected
            </span>
          )}
        </div>

        {/* Center: Layout selector */}
        <div className="hidden md:flex">
          <CameraLayoutSelector mode={layoutMode} onChange={setLayoutMode} />
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={handleConnectAll}
            disabled={isConnectingAll || cameraConfigs.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-600/20 hover:bg-emerald-600/30 border border-emerald-500/20 text-emerald-400 text-xs font-medium transition-colors duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isConnectingAll ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : (
              <Power className="w-3 h-3" />
            )}
            <span className="hidden sm:inline">Connect All</span>
          </button>
          <button
            onClick={handleDisconnectAll}
            disabled={isDisconnectingAll || connected === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white/60 hover:text-white text-xs font-medium transition-colors duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isDisconnectingAll ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : (
              <PowerOff className="w-3 h-3" />
            )}
            <span className="hidden sm:inline">Disconnect All</span>
          </button>
          <button
            onClick={() => setIsSettingsOpen(true)}
            className="p-2 rounded-xl hover:bg-white/5 transition-colors duration-200 text-white/40 hover:text-white"
            title="Camera settings"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Mobile layout selector */}
      <div className="md:hidden flex justify-center py-2 border-b border-white/5">
        <CameraLayoutSelector mode={layoutMode} onChange={setLayoutMode} />
      </div>

      {/* Main area */}
      <main className="flex-1 min-h-0 p-4 overflow-auto">
        {cameraConfigs.length === 0
          ? renderEmptyState()
          : layoutMode === "grid"
          ? renderGrid()
          : layoutMode === "focus"
          ? renderFocus()
          : renderFullscreen()}
      </main>

      {/* Health strip */}
      {cameraConfigs.length > 0 && renderHealthStrip()}

      {/* Mobile health strip toggle */}
      <div className="sm:hidden border-t border-white/5">
        <button
          onClick={() => setShowHealthStrip((p) => !p)}
          className="w-full py-1 text-center text-white/30 text-[10px]"
        >
          {showHealthStrip ? "Hide Status" : "Show Status"}
        </button>
      </div>

      {/* Camera settings modal */}
      <CameraModal isOpen={isSettingsOpen} onClose={handleSettingsClose} />
    </div>
  );
}
