"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { Play, Mic, Search, Layers, Command, GripHorizontal, Minimize2, Maximize2, X, Plus, Terminal, RefreshCw, Power, Loader2, Activity, Database, Upload, User, LogOut, Cloud, Sparkles, Settings } from 'lucide-react';
import TaskGraph from '../components/TaskGraph';
import CalibrationModal from '../components/CalibrationModal';
import CameraModal from '../components/CameraModal';
import ArmManagerModal from '../components/ArmManagerModal';
import { useDraggable } from '../hooks/useDraggable';
import TeleopModal from '../components/TeleopModal';
import RecordingModal from '../components/RecordingModal';
import DatasetViewerModal from '../components/DatasetViewerModal';
import UploadModal from '../components/UploadModal';
import AuthModal from '../components/AuthModal';
import TrainModal from '../components/TrainModal';
import HILModal from '../components/HILModal';
import RLTrainingModal from '../components/RLTrainingModal';
import { useAuth } from '../lib/AuthContext';
import { useTheme } from '../lib/ThemeContext';
import { ThemeToggle } from '../components/ThemeToggle';

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function Dashboard() {
  const router = useRouter();
  const { user, loading: authLoading, signOut } = useAuth();
  const { resolvedTheme } = useTheme();
  const [isAuthOpen, setIsAuthOpen] = useState(false);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [currentPlan, setCurrentPlan] = useState<any>(null);
  // Status State
  const [statusText, setStatusText] = useState("OFFLINE");
  const [connectionState, setConnectionState] = useState("DISCONNECTED");
  const [executionState, setExecutionState] = useState("IDLE");
  const [isRestarting, setIsRestarting] = useState(false);
  const [isThinking, setIsThinking] = useState(false);

  // UI States
  const [isStatusMenuOpen, setIsStatusMenuOpen] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [restartConfirm, setRestartConfirm] = useState(false);

  // Damiao Velocity Limiter State
  const [hasDamiao, setHasDamiao] = useState(false);
  const [velocityLimit, setVelocityLimit] = useState(0.1);
  const [isUpdatingVelocity, setIsUpdatingVelocity] = useState(false);

  // Modal States
  const [isCalibrationOpen, setIsCalibrationOpen] = useState(false);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isTeleopOpen, setIsTeleopOpen] = useState(false);
  const [isRecordingOpen, setIsRecordingOpen] = useState(false);
  const [isDatasetViewerOpen, setIsDatasetViewerOpen] = useState(false);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isTrainOpen, setIsTrainOpen] = useState(false);
  const [isHILOpen, setIsHILOpen] = useState(false);
  const [isRLTrainingOpen, setIsRLTrainingOpen] = useState(false);
  const [isArmManagerOpen, setIsArmManagerOpen] = useState(false);

  // Arm status for status menu quick view
  const [armsSummary, setArmsSummary] = useState<{total_arms: number, connected: number}>({ total_arms: 0, connected: 0 });

  // Global maximized state - only one window can be maximized at a time
  const [maximizedWindow, setMaximizedWindow] = useState<string | null>(null);

  // Chat Window State
  const chatDrag = useDraggable({ x: 20, y: 80 });
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isChatMinimized, setIsChatMinimized] = useState(false);

  // Initial Status Check
  useEffect(() => {
    const fetchStatus = () => {
      fetch("http://127.0.0.1:8000/status")
        .then(res => res.json())
        .then(data => {
          // If we were manually restarting/reconnecting, check if we can clear flag
          if (data.connection === "CONNECTED") {
            if (isRestarting) setIsRestarting(false);
            if (isReconnecting) setIsReconnecting(false);
          }

          // Should we stop reconnecting if there is an error? 
          // Only if it's a "fresh" error, but hard to tell. 
          // For now, let user cancel or retry.

          setStatusText(data.status);
          setConnectionState(data.connection);
          setExecutionState(data.execution);
        })
        .catch(() => {
          setConnectionState("DISCONNECTED");
          setStatusText("OFFLINE");
        });
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 1000); // Faster polling for responsiveness
    return () => clearInterval(interval);
  }, [isRestarting, isReconnecting]);

  // Damiao velocity limit polling
  useEffect(() => {
    const fetchDamiaoStatus = () => {
      fetch("http://127.0.0.1:8000/robot/velocity-limit")
        .then(res => res.json())
        .then(data => {
          setHasDamiao(data.has_velocity_limit || false);
          if (data.has_velocity_limit && typeof data.velocity_limit === 'number' && !isNaN(data.velocity_limit)) {
            setVelocityLimit(data.velocity_limit);
          }
        })
        .catch(() => {
          setHasDamiao(false);
        });
    };

    fetchDamiaoStatus();
    const interval = setInterval(fetchDamiaoStatus, 5000); // Poll every 5s
    return () => clearInterval(interval);
  }, []);

  // Arm registry summary polling
  useEffect(() => {
    const fetchArmsSummary = () => {
      fetch("http://127.0.0.1:8000/arms")
        .then(res => res.json())
        .then(data => {
          if (data.summary) {
            setArmsSummary(data.summary);
          }
        })
        .catch(() => {});
    };

    fetchArmsSummary();
    const interval = setInterval(fetchArmsSummary, 5000);
    return () => clearInterval(interval);
  }, []);

  // Update velocity limit on the robot
  const updateVelocityLimit = async (newLimit: number) => {
    setIsUpdatingVelocity(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/robot/velocity-limit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit: newLimit })
      });
      if (res.ok) {
        setVelocityLimit(newLimit);
      }
    } catch (e) {
      console.error("Failed to update velocity limit:", e);
    } finally {
      setIsUpdatingVelocity(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg = { role: "user" as const, content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsThinking(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: input,
          messages: [...messages, userMsg]
        }),
      });

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.reply },
      ]);

      if (data.plan) {
        setCurrentPlan(data.plan);
      }
      setIsThinking(false);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error connecting to server." },
      ]);
      setIsThinking(false);
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setCurrentPlan(null);
    setInput("");
  };

  // Helper to determine LED Color
  const getStatusColor = () => {
    if (isRestarting) return "bg-orange-500 shadow-[0_0_8px_rgba(249,115,22,0.6)]";

    switch (connectionState) {
      case "DISCONNECTED":
      case "ERROR":
        return "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]"; // Red
      case "INITIALIZING":
        return "bg-yellow-400 shadow-[0_0_8px_rgba(250,204,21,0.6)]"; // Yellow
      case "MOCK":
        return "bg-purple-500 shadow-[0_0_8px_rgba(168,85,247,0.6)]"; // Purple
      case "CONNECTED":
        // If connected, check execution
        if (executionState !== "IDLE") return "bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)]"; // Blue
        return "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]"; // Green (Ready)
      default:
        return "bg-gray-400";
    }
  };

  const currentStatusText = isRestarting ? "RESTARTING..." : statusText;

  return (
    <div className="h-screen w-screen overflow-hidden bg-neutral-50 dark:bg-zinc-950 font-sans selection:bg-black selection:text-white dark:selection:bg-white dark:selection:text-black">

      {/* Modals */}
      <CalibrationModal
        isOpen={isCalibrationOpen}
        onClose={() => setIsCalibrationOpen(false)}
        language="en"
      />
      <CameraModal
        isOpen={isCameraOpen}
        onClose={() => setIsCameraOpen(false)}
      />
      <TeleopModal
        isOpen={isTeleopOpen}
        onClose={() => { setIsTeleopOpen(false); if (maximizedWindow === 'teleop') setMaximizedWindow(null); }}
        maximizedWindow={maximizedWindow}
        setMaximizedWindow={setMaximizedWindow}
      />
      <RecordingModal
        isOpen={isRecordingOpen}
        onClose={() => { setIsRecordingOpen(false); if (maximizedWindow === 'recording') setMaximizedWindow(null); }}
        maximizedWindow={maximizedWindow}
        setMaximizedWindow={setMaximizedWindow}
      />
      <DatasetViewerModal
        isOpen={isDatasetViewerOpen}
        onClose={() => { setIsDatasetViewerOpen(false); if (maximizedWindow === 'datasetViewer') setMaximizedWindow(null); }}
        maximizedWindow={maximizedWindow}
        setMaximizedWindow={setMaximizedWindow}
      />
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploadComplete={(dataset) => {
          console.log("Upload complete:", dataset);
          setIsUploadOpen(false);
        }}
      />
      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
      />
      <TrainModal
        isOpen={isTrainOpen}
        onClose={() => { setIsTrainOpen(false); if (maximizedWindow === 'train') setMaximizedWindow(null); }}
        maximizedWindow={maximizedWindow}
        setMaximizedWindow={setMaximizedWindow}
      />
      <HILModal
        isOpen={isHILOpen}
        onClose={() => { setIsHILOpen(false); if (maximizedWindow === 'hil') setMaximizedWindow(null); }}
        maximizedWindow={maximizedWindow}
        setMaximizedWindow={setMaximizedWindow}
      />
      <RLTrainingModal
        isOpen={isRLTrainingOpen}
        onClose={() => { setIsRLTrainingOpen(false); if (maximizedWindow === 'rl-training') setMaximizedWindow(null); }}
        maximizedWindow={maximizedWindow}
        setMaximizedWindow={setMaximizedWindow}
      />
      <ArmManagerModal
        isOpen={isArmManagerOpen}
        onClose={() => setIsArmManagerOpen(false)}
      />

      {/* 1. LAYER: BACKGROUND (Task Graph) */}
      <div className="absolute inset-0 z-0">
        <TaskGraph plan={currentPlan} darkMode={resolvedTheme === 'dark'} />
      </div>

      {/* 2. LAYER: UI OVERLAY (Header & Dock) */}
      <div className="absolute top-0 left-0 right-0 z-30 p-6 pointer-events-none flex justify-between items-start">
        {/* Branding + User */}
        <div className="pointer-events-auto flex items-center gap-2">
          <div className="flex items-center gap-3 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-xl border border-white/50 dark:border-zinc-700/50 px-4 py-2 rounded-2xl shadow-sm hover:bg-white dark:hover:bg-zinc-900 transition-colors cursor-default">
            <div className="w-8 h-8 bg-black dark:bg-white rounded-lg flex items-center justify-center shadow-lg shadow-black/20 dark:shadow-white/10">
              <span className="text-white dark:text-black font-bold tracking-tighter">N</span>
            </div>
            <span className="font-semibold tracking-tight text-neutral-900 dark:text-zinc-100">Nextis</span>
          </div>

          {/* Theme Toggle */}
          <div className="bg-white/80 dark:bg-zinc-900/80 backdrop-blur-xl border border-white/50 dark:border-zinc-700/50 rounded-2xl shadow-sm">
            <ThemeToggle />
          </div>

          {/* User Button */}
          {!authLoading && (
            user ? (
              <div className="flex items-center gap-2 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-xl border border-white/50 dark:border-zinc-700/50 pl-3 pr-2 py-1.5 rounded-2xl shadow-sm">
                <span className="text-xs text-neutral-600 dark:text-zinc-400 max-w-[120px] truncate">{user.email}</span>
                <button
                  onClick={() => signOut()}
                  className="p-1.5 hover:bg-neutral-100 dark:hover:bg-zinc-800 rounded-lg transition-colors"
                  title="Sign out"
                >
                  <LogOut className="w-4 h-4 text-neutral-500 dark:text-zinc-400" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => setIsAuthOpen(true)}
                className="flex items-center gap-2 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-xl border border-white/50 dark:border-zinc-700/50 px-4 py-2 rounded-2xl shadow-sm hover:bg-white dark:hover:bg-zinc-900 transition-colors"
              >
                <User className="w-4 h-4 text-neutral-600 dark:text-zinc-400" />
                <span className="text-xs font-medium text-neutral-700 dark:text-zinc-300">Sign In</span>
              </button>
            )
          )}
        </div>

        {/* Center Control Dock */}
        <div className="pointer-events-auto flex items-center gap-1 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-xl border border-white/50 dark:border-zinc-700/50 px-2 py-1.5 rounded-full shadow-lg transition-transform hover:scale-105">
          <div className="flex gap-1 pr-4 border-r border-neutral-200/50 dark:border-zinc-700/50 mr-2">
            <button
              onClick={() => setIsRecordingOpen(true)}
              className={`px-4 py-2 rounded-full text-xs font-bold transition-all flex items-center gap-1.5 ${isRecordingOpen ? 'bg-red-600 text-white shadow-md' : 'hover:bg-neutral-100/50 dark:hover:bg-zinc-800/50 text-red-600 hover:text-red-700'}`}
            >
              <div className={`w-2 h-2 rounded-full animate-pulse ${isRecordingOpen ? 'bg-white' : 'bg-red-500'}`} /> Studio
            </button>
            <button
              onClick={() => setIsDatasetViewerOpen(true)}
              className={`px-4 py-2 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 ${isDatasetViewerOpen ? 'bg-black dark:bg-white text-white dark:text-black shadow-md' : 'hover:bg-neutral-100/50 dark:hover:bg-zinc-800/50 text-neutral-600 dark:text-zinc-400 hover:text-black dark:hover:text-white'}`}
            >
              <Database className="w-3 h-3" /> Data
            </button>
            <button
              onClick={() => setIsUploadOpen(true)}
              className={`px-4 py-2 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 ${isUploadOpen ? 'bg-black dark:bg-white text-white dark:text-black shadow-md' : 'hover:bg-neutral-100/50 dark:hover:bg-zinc-800/50 text-neutral-600 dark:text-zinc-400 hover:text-black dark:hover:text-white'}`}
            >
              <Upload className="w-3 h-3" /> Upload
            </button>
            <button
              onClick={() => setIsTrainOpen(true)}
              className={`px-4 py-2 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 ${isTrainOpen ? 'bg-purple-600 text-white shadow-md' : 'hover:bg-purple-50 dark:hover:bg-purple-950 text-purple-600 dark:text-purple-400 hover:text-purple-700'}`}
            >
              <Sparkles className="w-3 h-3" /> Train
            </button>
            <button
              onClick={() => setIsHILOpen(true)}
              className={`px-4 py-2 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 ${isHILOpen ? 'bg-blue-600 text-white shadow-md' : 'hover:bg-blue-50 dark:hover:bg-blue-950 text-blue-600 dark:text-blue-400 hover:text-blue-700'}`}
            >
              <Play className="w-3 h-3" /> HIL
            </button>
            <button
              onClick={() => setIsRLTrainingOpen(true)}
              className={`px-4 py-2 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 ${isRLTrainingOpen ? 'bg-orange-600 text-white shadow-md' : 'hover:bg-orange-50 dark:hover:bg-orange-950 text-orange-600 dark:text-orange-400 hover:text-orange-700'}`}
            >
              <Activity className="w-3 h-3" /> RL
            </button>
            <button
              onClick={() => router.push('/dashboard')}
              className="px-4 py-2 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 hover:bg-neutral-100/50 dark:hover:bg-zinc-800/50 text-neutral-600 dark:text-zinc-400 hover:text-black dark:hover:text-white"
            >
              <Cloud className="w-3 h-3" /> Cloud
            </button>
            <button
              onClick={() => setIsChatOpen(true)}
              className={`px-4 py-2 rounded-full text-xs font-medium transition-all ${isChatOpen ? 'bg-black dark:bg-white text-white dark:text-black shadow-md' : 'hover:bg-neutral-100/50 dark:hover:bg-zinc-800/50 text-neutral-600 dark:text-zinc-400 hover:text-black dark:hover:text-white'}`}
            >
              Assistant
            </button>
            <button
              onClick={() => setIsCameraOpen(true)}
              className={`px-4 py-2 rounded-full text-xs font-medium transition-all ${isCameraOpen ? 'bg-black dark:bg-white text-white dark:text-black shadow-md' : 'hover:bg-neutral-100/50 dark:hover:bg-zinc-800/50 text-neutral-600 dark:text-zinc-400 hover:text-black dark:hover:text-white'}`}
            >
              Cameras
            </button>
            <button
              onClick={() => setIsArmManagerOpen(true)}
              className={`px-4 py-2 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 ${isArmManagerOpen ? 'bg-black dark:bg-white text-white dark:text-black shadow-md' : 'hover:bg-neutral-100/50 dark:hover:bg-zinc-800/50 text-neutral-600 dark:text-zinc-400 hover:text-black dark:hover:text-white'}`}
            >
              <Settings className="w-3 h-3" /> Arms
            </button>
            <button
              onClick={() => setIsCalibrationOpen(true)}
              className={`px-4 py-2 rounded-full text-xs font-medium transition-all ${isCalibrationOpen ? 'bg-black dark:bg-white text-white dark:text-black shadow-md' : 'hover:bg-neutral-100/50 dark:hover:bg-zinc-800/50 text-neutral-600 dark:text-zinc-400 hover:text-black dark:hover:text-white'}`}
            >
              Calibration
            </button>
            <button
              onClick={() => setIsTeleopOpen(true)}
              className={`px-4 py-2 rounded-full text-xs font-medium transition-all ${isTeleopOpen ? 'bg-black dark:bg-white text-white dark:text-black shadow-md' : 'hover:bg-neutral-100/50 dark:hover:bg-zinc-800/50 text-neutral-600 dark:text-zinc-400 hover:text-black dark:hover:text-white'}`}
            >
              Teleoperate
            </button>
          </div>

          <div className="h-6 w-[1px] bg-neutral-300/50 dark:bg-zinc-600/50 mx-1" />

          {/* STATUS CHIP (Clickable for Menu) */}
          <div className="relative">
            <button
              onClick={() => { setIsStatusMenuOpen(!isStatusMenuOpen); setRestartConfirm(false); }}
              className={`px-4 py-1.5 flex items-center gap-2 rounded-full transition-all border border-transparent ${isStatusMenuOpen ? 'bg-white dark:bg-zinc-800 border-black/5 dark:border-zinc-700 shadow-md' : 'hover:bg-black/5 dark:hover:bg-white/5'}`}
            >
              <div className={`w-2.5 h-2.5 rounded-full ${getStatusColor()} transition-colors duration-500`} />
              <span className="text-xs font-medium text-neutral-500 dark:text-zinc-400 uppercase tracking-wide min-w-[50px] text-left">{currentStatusText}</span>
            </button>

            {/* STATUS MENU POPOVER */}
            {isStatusMenuOpen && (
              <>
                {/* Backdrop */}
                <div className="fixed inset-0 z-40" onClick={() => setIsStatusMenuOpen(false)} />

                {/* Menu */}
                <div className="absolute top-12 right-0 w-64 bg-white/90 dark:bg-zinc-900/90 backdrop-blur-xl border border-white/50 dark:border-zinc-700/50 shadow-2xl rounded-2xl p-4 z-50 animate-in fade-in slide-in-from-top-2 flex flex-col gap-2">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-widest">System Control</span>
                    <div className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
                  </div>

                  {/* Connection Details */}
                  <div className="bg-neutral-50 dark:bg-zinc-800 rounded-lg p-3 border border-neutral-100 dark:border-zinc-700 mb-2">
                    <div className="flex justify-between items-center text-xs mb-1">
                      <span className="text-neutral-500 dark:text-zinc-400">Connection</span>
                      <span className="font-medium text-neutral-800 dark:text-zinc-200">{connectionState}</span>
                    </div>
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-neutral-500 dark:text-zinc-400">Execution</span>
                      <span className="font-medium text-neutral-800 dark:text-zinc-200">{executionState}</span>
                    </div>
                  </div>

                  {/* Damiao Velocity Limiter */}
                  {hasDamiao && (
                    <div className="bg-orange-50 dark:bg-orange-950/30 rounded-lg p-3 border border-orange-200 dark:border-orange-900/50 mb-2">
                      <div className="flex justify-between items-center mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-medium text-orange-800 dark:text-orange-300">Damiao Velocity</span>
                          <span className="px-1.5 py-0.5 text-[9px] font-bold bg-orange-200 dark:bg-orange-900/50 text-orange-700 dark:text-orange-400 rounded uppercase">
                            Safety
                          </span>
                        </div>
                        <span className="text-sm font-bold text-orange-600 dark:text-orange-400">
                          {Math.round((isNaN(velocityLimit) ? 0.1 : velocityLimit) * 100)}%
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={isNaN(velocityLimit) ? 10 : velocityLimit * 100}
                        onChange={(e) => {
                          const newVal = Number(e.target.value) / 100;
                          setVelocityLimit(newVal);
                        }}
                        onMouseUp={(e) => {
                          updateVelocityLimit(velocityLimit);
                        }}
                        onTouchEnd={() => {
                          updateVelocityLimit(velocityLimit);
                        }}
                        disabled={isUpdatingVelocity}
                        className="w-full h-2 bg-orange-200 dark:bg-orange-900/50 rounded-full appearance-none cursor-pointer accent-orange-500 disabled:opacity-50"
                      />
                      <p className="text-[10px] text-orange-600/70 dark:text-orange-400/70 mt-1.5">
                        Default 10% for safety. Increase gradually. High torque motors.
                      </p>
                    </div>
                  )}

                  {/* Arm Status Quick View */}
                  {armsSummary.total_arms > 0 && (
                    <div className="bg-neutral-50 dark:bg-zinc-800 rounded-lg p-3 border border-neutral-100 dark:border-zinc-700 mb-2">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-[10px] font-bold text-neutral-400 dark:text-zinc-500 uppercase">Arms</span>
                        <span className="text-xs text-neutral-600 dark:text-zinc-300">
                          {armsSummary.connected}/{armsSummary.total_arms} connected
                        </span>
                      </div>
                      <button
                        onClick={() => { setIsArmManagerOpen(true); setIsStatusMenuOpen(false); }}
                        className="w-full py-1.5 text-xs text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-950 rounded-lg transition-colors"
                      >
                        Manage Arms →
                      </button>
                    </div>
                  )}

                  {/* Actions */}
                  <button
                    disabled={isReconnecting || isRestarting}
                    onClick={async () => {
                      setIsReconnecting(true);
                      try {
                        const res = await fetch("http://127.0.0.1:8000/system/reconnect", { method: "POST" });
                        const data = await res.json();
                        if (data.status === 'initializing') {
                          setConnectionState("INITIALIZING");
                          setStatusText("CONNECTING...");
                        }
                      } catch (e) {
                        setStatusText("FAILED");
                        setIsReconnecting(false); // Only reset on immediate network error
                      }
                    }}
                    className="w-full py-2.5 rounded-xl bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 hover:border-blue-300 dark:hover:border-blue-600 hover:bg-blue-50 dark:hover:bg-blue-950 text-neutral-700 dark:text-zinc-300 font-medium text-xs flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                  >
                    {isReconnecting ? <Loader2 className="w-4 h-4 animate-spin text-blue-500" /> : <RefreshCw className="w-4 h-4 text-blue-500" />}
                    {isReconnecting ? "Reconnecting..." : "Reconnect Hardware"}
                  </button>

                  {/* Restart with Confirmation */}
                  {!restartConfirm ? (
                    <button
                      disabled={isRestarting}
                      onClick={() => setRestartConfirm(true)}
                      className="w-full py-2.5 rounded-xl bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 hover:border-red-300 dark:hover:border-red-600 hover:bg-red-50 dark:hover:bg-red-950 text-neutral-700 dark:text-zinc-300 font-medium text-xs flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                    >
                      <Power className="w-4 h-4 text-red-500" />
                      Restart System
                    </button>
                  ) : (
                    <div className="flex gap-2 animate-in fade-in slide-in-from-right-2">
                      <button
                        onClick={() => setRestartConfirm(false)}
                        className="flex-1 py-2.5 rounded-xl bg-neutral-100 dark:bg-zinc-800 hover:bg-neutral-200 dark:hover:bg-zinc-700 text-neutral-600 dark:text-zinc-400 font-medium text-xs transition-colors"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={async () => {
                          setIsRestarting(true);
                          setRestartConfirm(false);
                          setIsStatusMenuOpen(false); // Close menu
                          try {
                            const res = await fetch("http://127.0.0.1:8000/system/restart", { method: "POST" });
                            const data = await res.json();
                            if (data.status === 'restarting') {
                              setStatusText("RESTARTING...");
                              setConnectionState("INITIALIZING");
                            }
                          } catch (e) {
                            setStatusText("RESTARTING...");
                          }
                        }}
                        className="flex-1 py-2.5 rounded-xl bg-red-500 hover:bg-red-600 text-white font-medium text-xs flex items-center justify-center gap-2 shadow-lg shadow-red-200 dark:shadow-red-900/30 transition-all"
                      >
                        Confirm
                      </button>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* 3. LAYER: FLOATING CHAT WINDOW (Draggable) */}
      <div
        ref={maximizedWindow === 'chat' ? null : chatDrag.ref}
        style={{
          // Transform handled by useDraggable Ref when not maximized
          width: maximizedWindow === 'chat' ? 'calc(100vw - 40px)' : (isChatMinimized ? 'auto' : '380px'),
          height: maximizedWindow === 'chat' ? 'calc(100vh - 120px)' : (isChatMinimized ? 'auto' : '550px'),
          top: maximizedWindow === 'chat' ? '100px' : 'auto', // Push down below header
          left: maximizedWindow === 'chat' ? '20px' : 'auto',
          opacity: isChatOpen ? 1 : 0,
          pointerEvents: isChatOpen ? 'auto' : 'none',
          zIndex: maximizedWindow === 'chat' ? 100 : 40,
        }}
        className={`absolute flex flex-col shadow-2xl rounded-2xl overflow-hidden glass border border-white/50 dark:border-zinc-700/50 backdrop-blur-3xl transition-all duration-300 ease-[cubic-bezier(0.25,1,0.5,1)] ${!isChatOpen && 'scale-90'}`}
      >
        {/* Window Handle */}
        <div
          onMouseDown={maximizedWindow !== 'chat' ? chatDrag.handleMouseDown : undefined}
          onDoubleClick={() => setMaximizedWindow(maximizedWindow === 'chat' ? null : 'chat')}
          className={`h-10 bg-white/40 dark:bg-zinc-800/40 border-b border-black/5 dark:border-white/5 flex items-center justify-between px-4 select-none ${maximizedWindow !== 'chat' ? 'cursor-grab active:cursor-grabbing' : 'cursor-default'}`}
        >
          <div className="flex items-center gap-2">
            <div className="flex gap-1.5 mr-2">
              {/* Close */}
              <button
                onClick={() => { setIsChatOpen(false); if (maximizedWindow === 'chat') setMaximizedWindow(null); }}
                className="w-3 h-3 rounded-full bg-[#FF5F57] border border-black/10 hover:brightness-90 transition-all flex items-center justify-center group"
              >
                <X className="w-2 h-2 text-black/50 opacity-0 group-hover:opacity-100" />
              </button>
              {/* Minimize */}
              <button
                onClick={() => {
                  setIsChatMinimized(!isChatMinimized);
                  if (maximizedWindow === 'chat') setMaximizedWindow(null);
                }}
                className="w-3 h-3 rounded-full bg-[#FEBC2E] border border-black/10 hover:brightness-90 transition-all flex items-center justify-center group"
              >
                <Minimize2 className="w-2 h-2 text-black/50 opacity-0 group-hover:opacity-100" />
              </button>
              {/* Maximize */}
              <button
                onClick={() => {
                  setMaximizedWindow(maximizedWindow === 'chat' ? null : 'chat');
                  if (isChatMinimized) setIsChatMinimized(false);
                }}
                className="w-3 h-3 rounded-full bg-[#28C840] border border-black/10 hover:brightness-90 transition-all flex items-center justify-center group"
              >
                <Maximize2 className="w-2 h-2 text-black/50 opacity-0 group-hover:opacity-100" />
              </button>
            </div>
            <span className="ml-2 text-xs font-semibold text-neutral-600 dark:text-zinc-400 tracking-wide">Assistant</span>
          </div>

          <div className="flex items-center gap-2">
            {/* New Chat Button */}
            <button
              onClick={startNewChat}
              title="New Chat"
              className="p-1 hover:bg-black/5 dark:hover:bg-white/5 rounded-md transition-colors text-neutral-400 dark:text-zinc-500 hover:text-black dark:hover:text-white"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
        </div>

        {!isChatMinimized && (
          <>
            {/* Chat Content */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-white/30 dark:bg-zinc-900/30 custom-scrollbar">
              {messages.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center opacity-40 gap-3 pb-10">
                  <div className="w-12 h-12 bg-black/5 dark:bg-white/5 rounded-full flex items-center justify-center">
                    <Terminal className="w-6 h-6 text-black dark:text-white" />
                  </div>
                  <p className="text-sm font-medium text-black dark:text-white">How can I help you?</p>
                </div>
              )}
              {messages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] py-2.5 px-4 text-sm rounded-2xl shadow-sm backdrop-blur-sm leading-relaxed ${msg.role === 'user'
                    ? 'bg-black dark:bg-white text-white dark:text-black'
                    : 'bg-white/80 dark:bg-zinc-800/80 text-neutral-800 dark:text-zinc-200 border border-white/50 dark:border-zinc-700/50'
                    }`}>
                    {msg.content}
                  </div>
                </div>
              ))}
              {isThinking && (
                <div className="flex justify-start animate-pulse">
                  <div className="bg-white/50 dark:bg-zinc-800/50 px-4 py-2 rounded-full text-xs text-neutral-500 dark:text-zinc-400">
                    Thinking...
                  </div>
                </div>
              )}
            </div>

            {/* Input Footer */}
            <div className="p-3 bg-white/60 dark:bg-zinc-900/60 border-t border-white/50 dark:border-zinc-700/50">
              <div className="relative">
                <input
                  className="w-full bg-white/50 dark:bg-zinc-800/50 hover:bg-white dark:hover:bg-zinc-800 focus:bg-white dark:focus:bg-zinc-800 border border-transparent focus:border-black/5 dark:focus:border-white/5 rounded-xl py-3 pl-4 pr-10 text-sm text-neutral-900 dark:text-zinc-100 outline-none transition-all placeholder:text-neutral-400 dark:placeholder:text-zinc-500 shadow-sm"
                  placeholder="Type a message..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim()}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 bg-black dark:bg-white text-white dark:text-black rounded-lg hover:scale-110 active:scale-95 transition-all shadow-md disabled:opacity-20 disabled:hover:scale-100"
                >
                  <div className="w-2.5 h-2.5 bg-white dark:bg-black rounded-sm" />
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      {/* 4. EXECUTE BUTTON (Floating Bottom Right) */}
      {currentPlan && (
        <div className="absolute bottom-10 right-10 z-50 animate-in slide-in-from-bottom-10 fade-in duration-500">
          <button
            onClick={async () => {
              await fetch("http://127.0.0.1:8000/execute", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ plan: currentPlan })
              });
            }}
            className="px-8 py-4 bg-black/90 dark:bg-white/90 hover:bg-black dark:hover:bg-white text-white dark:text-black rounded-full shadow-2xl backdrop-blur-xl border border-white/10 dark:border-black/10 flex items-center gap-3 font-semibold tracking-wide hover:scale-105 transition-all active:scale-95 group"
          >
            <Play className="w-5 h-5 fill-white dark:fill-black group-hover:scale-110 transition-transform" />
            START SEQUENCE
          </button>
        </div>
      )}

    </div>
  );
}
