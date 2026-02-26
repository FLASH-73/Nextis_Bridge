import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { Rocket, Volume2, VolumeX, Bot, User, Pause, ShieldAlert } from "lucide-react";
import { motion, useDragControls } from "framer-motion";
import { useResizable } from "../../../hooks/useResizable";
import { usePolling } from "../../../hooks/usePolling";
import { EmergencyStop } from "../../EmergencyStop";
import DeploySetup from "./DeploySetup";
import DeployRuntime from "./DeployRuntime";
import type { DeployModalProps } from "./types";
import type { DeploymentStatus, PolicyInfo, Arm, CameraConfig } from "../../../lib/api/types";
import { deployApi, policiesApi, armsApi, camerasApi } from "../../../lib/api";

const STATE_LABELS: Record<string, string> = {
  idle: "IDLE",
  starting: "STARTING",
  running: "AUTONOMOUS",
  human_active: "HUMAN ACTIVE",
  paused: "PAUSED",
  stopping: "STOPPING",
  estop: "E-STOP",
  error: "ERROR",
};

const STATE_COLORS: Record<string, string> = {
  idle: "bg-neutral-100 dark:bg-zinc-800 text-neutral-500 dark:text-zinc-400 border border-neutral-200 dark:border-zinc-700",
  starting: "bg-blue-100 dark:bg-blue-950/50 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800",
  running: "bg-emerald-100 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800",
  human_active: "bg-blue-100 dark:bg-blue-950/50 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800",
  paused: "bg-amber-100 dark:bg-amber-950/50 text-amber-700 dark:text-amber-300 border border-amber-200 dark:border-amber-800",
  stopping: "bg-neutral-100 dark:bg-zinc-800 text-neutral-500 dark:text-zinc-400 border border-neutral-200 dark:border-zinc-700",
  estop: "bg-red-100 dark:bg-red-950/50 text-red-700 dark:text-red-300 border border-red-200 dark:border-red-800",
  error: "bg-red-100 dark:bg-red-950/50 text-red-700 dark:text-red-300 border border-red-200 dark:border-red-800",
};

export default function DeployModal({ isOpen, onClose, maximizedWindow, setMaximizedWindow }: DeployModalProps) {
  const dragControls = useDragControls();
  const { size, handleResizeMouseDown } = useResizable({ initialSize: { width: 950, height: 700 } });
  const [isMinimized, setIsMinimized] = useState(false);
  const isMaximized = maximizedWindow === "deploy";

  // Phase state
  const [phase, setPhase] = useState<"setup" | "running">("setup");

  // Data shared between phases
  const [policies, setPolicies] = useState<PolicyInfo[]>([]);
  const [arms, setArms] = useState<Arm[]>([]);
  const [cameraConfigs, setCameraConfigs] = useState<CameraConfig[]>([]);
  const [status, setStatus] = useState<DeploymentStatus | null>(null);

  // Voice feedback (persisted in localStorage)
  const [voiceMuted, setVoiceMuted] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("deploy-voice-muted") === "true";
    }
    return false;
  });
  const prevStateRef = useRef<string | null>(null);

  const speak = useCallback((text: string) => {
    if (voiceMuted) return;
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.1;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  }, [voiceMuted]);

  const toggleVoice = () => {
    setVoiceMuted((prev) => {
      const next = !prev;
      if (typeof window !== "undefined") {
        localStorage.setItem("deploy-voice-muted", String(next));
      }
      return next;
    });
  };

  // Voice announcements on state transitions
  useEffect(() => {
    if (status?.state && status.state !== prevStateRef.current) {
      const announcements: Record<string, string> = {
        running: "Autonomous mode",
        human_active: "Intervention detected",
        paused: "Paused. Waiting for your decision.",
        estop: "Emergency stop activated",
        idle: prevStateRef.current ? "Deployment stopped" : "",
      };
      if (announcements[status.state]) {
        speak(announcements[status.state]);
      }
      prevStateRef.current = status.state;
    }
  }, [status?.state, speak]);

  // Filter cameras to those used by the policy
  const activeCameras = useMemo(() => {
    if (!status || status.state === "idle" || !status.policy_config?.cameras?.length) {
      return cameraConfigs;
    }
    const policyCameras = status.policy_config.cameras;
    return cameraConfigs.filter((cam) => {
      const camId = cam.id?.toLowerCase() || "";
      return policyCameras.some((pc) => {
        const pcLower = pc.toLowerCase();
        return camId === pcLower || camId === pcLower.replace("_", "") || camId.replace("_", "") === pcLower;
      });
    });
  }, [status, cameraConfigs]);

  // Load data on open
  useEffect(() => {
    if (isOpen) {
      fetchPolicies();
      fetchArms();
      fetchCameras();
      checkStatus();
    }
  }, [isOpen]);

  const fetchPolicies = async () => {
    try {
      const data = await policiesApi.list();
      setPolicies(data.filter((p) => p.status === "completed" && p.checkpoint_path));
    } catch (e) {
      console.error("Failed to fetch policies:", e);
    }
  };

  const fetchArms = async () => {
    try {
      const data = await armsApi.list();
      setArms(data.arms || []);
    } catch (e) {
      console.error("Failed to fetch arms:", e);
    }
  };

  const fetchCameras = async () => {
    try {
      const data = await camerasApi.config();
      if (Array.isArray(data)) setCameraConfigs(data);
    } catch (e) {
      console.error(e);
    }
  };

  const checkStatus = async () => {
    try {
      const data = await deployApi.status();
      setStatus(data);
      if (data.state && data.state !== "idle") {
        setPhase("running");
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Poll status while running
  usePolling(async () => {
    try {
      const data = await deployApi.status();
      setStatus(data);
      // Auto-return to setup if deployment stopped externally
      if (data.state === "idle" && phase === "running") {
        setPhase("setup");
      }
    } catch (e) {
      console.error(e);
    }
  }, 300, isOpen && phase === "running");

  const handleDeployStarted = () => {
    setPhase("running");
  };

  const handleDeployStopped = () => {
    setPhase("setup");
    setStatus(null);
  };

  if (!isOpen) return null;

  return (
    <motion.div
      drag
      dragControls={dragControls}
      dragListener={false}
      dragMomentum={false}
      initial={{ opacity: 0, scale: 0.95, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      style={{
        width: isMaximized ? "calc(100vw - 40px)" : isMinimized ? "auto" : `${size.width}px`,
        height: isMaximized ? "calc(100vh - 100px)" : isMinimized ? "auto" : `${size.height}px`,
        zIndex: isMaximized ? 100 : 50,
      }}
      className={`fixed flex flex-col overflow-hidden font-sans glass-panel shadow-2xl rounded-3xl bg-white/90 dark:bg-zinc-900/90 border border-white/50 dark:border-zinc-700/50 backdrop-blur-3xl transition-all duration-300 ${isMaximized ? "top-20 left-5" : "top-20 left-1/2 -translate-x-1/2"}`}
    >
      {/* Header */}
      <div
        onPointerDown={(e) => dragControls.start(e)}
        className="h-16 bg-white/30 dark:bg-zinc-800/30 border-b border-black/5 dark:border-white/5 flex items-center justify-between px-6 cursor-grab active:cursor-grabbing select-none flex-none"
      >
        <div className="flex items-center gap-4">
          <div className="flex gap-2 mr-2">
            <button onClick={onClose} className="w-3.5 h-3.5 rounded-full bg-[#FF5F57] hover:brightness-90 transition-all" />
            <button onClick={() => setIsMinimized(!isMinimized)} className="w-3.5 h-3.5 rounded-full bg-[#FEBC2E] hover:brightness-90 transition-all" />
            <button onClick={() => setMaximizedWindow(isMaximized ? null : "deploy")} className="w-3.5 h-3.5 rounded-full bg-[#28C840] hover:brightness-90 transition-all" />
          </div>
          <span className="font-semibold text-neutral-800 dark:text-zinc-200 text-lg tracking-tight flex items-center gap-2">
            <Rocket className="w-5 h-5 text-emerald-500" /> Deploy
          </span>
        </div>

        <div className="flex items-center gap-4">
          {phase === "running" && (
            <button
              onClick={toggleVoice}
              className="p-1.5 rounded-lg hover:bg-neutral-100 dark:hover:bg-zinc-800 transition-colors"
              title={voiceMuted ? "Unmute voice feedback" : "Mute voice feedback"}
            >
              {voiceMuted ? <VolumeX className="w-4 h-4 text-neutral-400 dark:text-zinc-500" /> : <Volume2 className="w-4 h-4 text-emerald-500" />}
            </button>
          )}
          {status && status.state !== "idle" && (
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold transition-all ${STATE_COLORS[status.state] || STATE_COLORS.idle}`}>
              {status.state === "human_active" ? <User className="w-3.5 h-3.5" /> :
               status.state === "paused" ? <Pause className="w-3.5 h-3.5" /> :
               status.state === "estop" ? <ShieldAlert className="w-3.5 h-3.5" /> :
               <Bot className="w-3.5 h-3.5" />}
              {STATE_LABELS[status.state] || status.state.toUpperCase()}
            </div>
          )}
          <EmergencyStop />
        </div>
      </div>

      {!isMinimized && (
        <div className="flex-1 flex flex-col p-6 overflow-hidden">
          {phase === "setup" ? (
            <DeploySetup
              policies={policies}
              arms={arms}
              onDeployStarted={handleDeployStarted}
              speak={speak}
            />
          ) : (
            <DeployRuntime
              status={status}
              activeCameras={activeCameras}
              speak={speak}
              onStopped={handleDeployStopped}
            />
          )}
        </div>
      )}

      {/* Resize Handle */}
      {!isMaximized && !isMinimized && (
        <div
          onMouseDown={handleResizeMouseDown}
          className="absolute bottom-2 right-2 w-4 h-4 cursor-se-resize opacity-30 hover:opacity-60 transition-opacity"
        >
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M22 22H20V20H22V22ZM22 18H20V16H22V18ZM18 22H16V20H18V22ZM22 14H20V12H22V14ZM18 18H16V16H18V18ZM14 22H12V20H14V22Z" />
          </svg>
        </div>
      )}
    </motion.div>
  );
}
