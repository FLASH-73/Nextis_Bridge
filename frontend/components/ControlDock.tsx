"use client";

import { useRouter } from "next/navigation";
import {
  Play,
  Activity,
  Database,
  Upload,
  Sparkles,
  Settings,
  Cpu,
  Cloud,
  Camera,
  Rocket,
} from "lucide-react";
import StatusMenu from "./StatusMenu";

interface ModalStates {
  isRecordingOpen: boolean;
  isDatasetViewerOpen: boolean;
  isUploadOpen: boolean;
  isTrainOpen: boolean;
  isHILOpen: boolean;
  isRLTrainingOpen: boolean;
  isDeployOpen: boolean;
  isChatOpen: boolean;
  isArmManagerOpen: boolean;
  isMotorMonitorOpen: boolean;
  isCalibrationOpen: boolean;
  isTeleopOpen: boolean;
}

interface ControlDockProps {
  modalStates: ModalStates;
  setIsRecordingOpen: (open: boolean) => void;
  setIsDatasetViewerOpen: (open: boolean) => void;
  setIsUploadOpen: (open: boolean) => void;
  setIsTrainOpen: (open: boolean) => void;
  setIsHILOpen: (open: boolean) => void;
  setIsRLTrainingOpen: (open: boolean) => void;
  setIsDeployOpen: (open: boolean) => void;
  setIsChatOpen: (open: boolean) => void;
  setIsArmManagerOpen: (open: boolean) => void;
  setIsMotorMonitorOpen: (open: boolean) => void;
  setIsCalibrationOpen: (open: boolean) => void;
  setIsTeleopOpen: (open: boolean) => void;
}

export default function ControlDock({
  modalStates,
  setIsRecordingOpen,
  setIsDatasetViewerOpen,
  setIsUploadOpen,
  setIsTrainOpen,
  setIsHILOpen,
  setIsRLTrainingOpen,
  setIsDeployOpen,
  setIsChatOpen,
  setIsArmManagerOpen,
  setIsMotorMonitorOpen,
  setIsCalibrationOpen,
  setIsTeleopOpen,
}: ControlDockProps) {
  const router = useRouter();

  const dockBtn = (
    isActive: boolean,
    onClick: () => void,
    label: string,
    icon?: React.ReactNode,
    activeClass?: string
  ) => {
    const defaultActive =
      "bg-black dark:bg-white text-white dark:text-black shadow-md";
    const defaultInactive =
      "hover:bg-neutral-100/50 dark:hover:bg-zinc-800/50 text-neutral-600 dark:text-zinc-400 hover:text-black dark:hover:text-white";
    return (
      <button
        onClick={onClick}
        className={`px-4 py-2 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 ${isActive ? (activeClass || defaultActive) : (activeClass ? defaultInactive : defaultInactive)}`}
      >
        {icon}
        {label}
      </button>
    );
  };

  return (
    <div className="pointer-events-auto flex items-center gap-1 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-xl border border-white/50 dark:border-zinc-700/50 px-2 py-1.5 rounded-full shadow-lg transition-transform hover:scale-105">
      <div className="flex gap-1 pr-4 border-r border-neutral-200/50 dark:border-zinc-700/50 mr-2">
        <button
          onClick={() => setIsRecordingOpen(true)}
          className={`px-4 py-2 rounded-full text-xs font-bold transition-all flex items-center gap-1.5 ${modalStates.isRecordingOpen ? "bg-red-600 text-white shadow-md" : "hover:bg-neutral-100/50 dark:hover:bg-zinc-800/50 text-red-600 hover:text-red-700"}`}
        >
          <div
            className={`w-2 h-2 rounded-full animate-pulse ${modalStates.isRecordingOpen ? "bg-white" : "bg-red-500"}`}
          />{" "}
          Studio
        </button>
        {dockBtn(
          modalStates.isDatasetViewerOpen,
          () => setIsDatasetViewerOpen(true),
          "Data",
          <Database className="w-3 h-3" />
        )}
        {dockBtn(
          modalStates.isUploadOpen,
          () => setIsUploadOpen(true),
          "Upload",
          <Upload className="w-3 h-3" />
        )}
        <button
          onClick={() => setIsTrainOpen(true)}
          className={`px-4 py-2 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 ${modalStates.isTrainOpen ? "bg-purple-600 text-white shadow-md" : "hover:bg-purple-50 dark:hover:bg-purple-950 text-purple-600 dark:text-purple-400 hover:text-purple-700"}`}
        >
          <Sparkles className="w-3 h-3" /> Train
        </button>
        <button
          onClick={() => setIsHILOpen(true)}
          className={`px-4 py-2 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 ${modalStates.isHILOpen ? "bg-blue-600 text-white shadow-md" : "hover:bg-blue-50 dark:hover:bg-blue-950 text-blue-600 dark:text-blue-400 hover:text-blue-700"}`}
        >
          <Play className="w-3 h-3" /> HIL
        </button>
        <button
          onClick={() => setIsRLTrainingOpen(true)}
          className={`px-4 py-2 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 ${modalStates.isRLTrainingOpen ? "bg-orange-600 text-white shadow-md" : "hover:bg-orange-50 dark:hover:bg-orange-950 text-orange-600 dark:text-orange-400 hover:text-orange-700"}`}
        >
          <Activity className="w-3 h-3" /> RL
        </button>
        <button
          onClick={() => setIsDeployOpen(true)}
          className={`px-4 py-2 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 ${modalStates.isDeployOpen ? "bg-emerald-600 text-white shadow-md" : "hover:bg-emerald-50 dark:hover:bg-emerald-950 text-emerald-600 dark:text-emerald-400 hover:text-emerald-700"}`}
        >
          <Rocket className="w-3 h-3" /> Deploy
        </button>
        <button
          onClick={() => router.push("/dashboard")}
          className="px-4 py-2 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 hover:bg-neutral-100/50 dark:hover:bg-zinc-800/50 text-neutral-600 dark:text-zinc-400 hover:text-black dark:hover:text-white"
        >
          <Cloud className="w-3 h-3" /> Cloud
        </button>
        {dockBtn(
          modalStates.isChatOpen,
          () => setIsChatOpen(true),
          "Assistant"
        )}
        <button
          onClick={() => router.push("/cameras")}
          className="px-4 py-2 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 hover:bg-neutral-100/50 dark:hover:bg-zinc-800/50 text-neutral-600 dark:text-zinc-400 hover:text-black dark:hover:text-white"
        >
          <Camera className="w-3 h-3" /> Cameras
        </button>
        {dockBtn(
          modalStates.isArmManagerOpen,
          () => setIsArmManagerOpen(true),
          "Arms",
          <Settings className="w-3 h-3" />
        )}
        {dockBtn(
          modalStates.isMotorMonitorOpen,
          () => setIsMotorMonitorOpen(true),
          "Motors",
          <Cpu className="w-3 h-3" />
        )}
        {dockBtn(
          modalStates.isCalibrationOpen,
          () => setIsCalibrationOpen(true),
          "Calibration"
        )}
        {dockBtn(
          modalStates.isTeleopOpen,
          () => setIsTeleopOpen(true),
          "Teleoperate"
        )}
      </div>

      <div className="h-6 w-[1px] bg-neutral-300/50 dark:bg-zinc-600/50 mx-1" />

      <StatusMenu onOpenArmManager={() => setIsArmManagerOpen(true)} />
    </div>
  );
}
