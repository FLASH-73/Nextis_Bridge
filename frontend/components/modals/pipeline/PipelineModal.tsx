import { useState, useRef } from "react";
import { GitBranch, AlertTriangle, ArrowLeft } from "lucide-react";
import { motion, useDragControls } from "framer-motion";
import { useResizable } from "../../../hooks/useResizable";
import { pipelineApi, type AlignmentWarning, type PipelineConfig } from "../../../lib/api/pipeline";
import PipelineBuilder from "./PipelineBuilder";
import PipelineMonitor from "./PipelineMonitor";

interface PipelineModalProps {
  isOpen: boolean;
  onClose: () => void;
  maximizedWindow: string | null;
  setMaximizedWindow: (w: string | null) => void;
}

type Phase = "builder" | "warnings" | "monitor";

export default function PipelineModal({ isOpen, onClose, maximizedWindow, setMaximizedWindow }: PipelineModalProps) {
  const dragControls = useDragControls();
  const { size, handleResizeMouseDown } = useResizable({ initialSize: { width: 950, height: 700 } });
  const [isMinimized, setIsMinimized] = useState(false);
  const isMaximized = maximizedWindow === "pipeline";

  const [phase, setPhase] = useState<Phase>("builder");
  const [warnings, setWarnings] = useState<AlignmentWarning[]>([]);
  const configRef = useRef<PipelineConfig | null>(null);

  const handleLoad = async (w: AlignmentWarning[], config: PipelineConfig) => {
    configRef.current = config;
    if (w.length > 0) {
      setWarnings(w);
      setPhase("warnings");
    } else {
      try {
        await pipelineApi.start();
        setPhase("monitor");
      } catch (e) {
        console.error("Failed to start pipeline:", e);
      }
    }
  };

  const handleProceed = async () => {
    try {
      await pipelineApi.start();
      setPhase("monitor");
    } catch (e) {
      console.error("Failed to start pipeline:", e);
    }
  };

  const handleDone = () => {
    setPhase("builder");
    setWarnings([]);
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
        className="h-14 bg-white/30 dark:bg-zinc-800/30 border-b border-black/5 dark:border-white/5 flex items-center justify-between px-5 cursor-grab active:cursor-grabbing select-none flex-none"
      >
        <div className="flex items-center gap-3">
          <div className="flex gap-2 mr-2">
            <button onClick={onClose} className="w-3.5 h-3.5 rounded-full bg-[#FF5F57] hover:brightness-90 transition-all" />
            <button onClick={() => setIsMinimized(!isMinimized)} className="w-3.5 h-3.5 rounded-full bg-[#FEBC2E] hover:brightness-90 transition-all" />
            <button onClick={() => setMaximizedWindow(isMaximized ? null : "pipeline")} className="w-3.5 h-3.5 rounded-full bg-[#28C840] hover:brightness-90 transition-all" />
          </div>
          <span className="font-semibold text-neutral-800 dark:text-zinc-200 text-lg tracking-tight flex items-center gap-2">
            <GitBranch className="w-5 h-5 text-teal-500" /> Pipeline
          </span>
        </div>
      </div>

      {!isMinimized && (
        <div className="flex-1 flex flex-col overflow-hidden">
          {phase === "builder" && (
            <PipelineBuilder onLoad={handleLoad} />
          )}

          {phase === "warnings" && (
            <div className="flex-1 flex flex-col p-6 overflow-auto">
              <div className="flex items-center gap-2 mb-4">
                <AlertTriangle className="w-5 h-5 text-amber-500" />
                <h2 className="text-lg font-semibold text-neutral-900 dark:text-zinc-100">
                  Alignment Warnings
                </h2>
              </div>
              <p className="text-sm text-neutral-500 dark:text-zinc-400 mb-4">
                Consecutive steps have misaligned action/observation distributions. This may cause jerky transitions.
              </p>
              <div className="space-y-2 flex-1 overflow-auto">
                {warnings.map((w, i) => (
                  <div key={i} className="bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded-xl p-3">
                    <div className="text-xs font-semibold text-amber-700 dark:text-amber-300">
                      {w.step_from} &rarr; {w.step_to} &middot; {w.joint_name}
                    </div>
                    <div className="text-xs text-amber-600 dark:text-amber-400 mt-1">{w.message}</div>
                  </div>
                ))}
              </div>
              <div className="flex items-center gap-3 mt-4 pt-4 border-t border-neutral-100 dark:border-zinc-800">
                <button onClick={() => setPhase("builder")} className="px-4 py-2.5 rounded-xl text-sm font-medium bg-neutral-200 dark:bg-zinc-700 text-neutral-700 dark:text-zinc-300 hover:bg-neutral-300 dark:hover:bg-zinc-600 flex items-center gap-1.5">
                  <ArrowLeft className="w-3.5 h-3.5" /> Back
                </button>
                <button onClick={handleProceed} className="flex-1 py-2.5 rounded-xl text-sm font-medium bg-amber-500 text-white hover:bg-amber-600 transition-colors">
                  Proceed Anyway
                </button>
              </div>
            </div>
          )}

          {phase === "monitor" && (
            <div className="flex-1 p-6 overflow-hidden flex flex-col">
              <PipelineMonitor config={configRef.current!} onDone={handleDone} />
            </div>
          )}
        </div>
      )}

      {!isMaximized && !isMinimized && (
        <div onMouseDown={handleResizeMouseDown} className="absolute bottom-2 right-2 w-4 h-4 cursor-se-resize opacity-30 hover:opacity-60 transition-opacity">
          <svg viewBox="0 0 24 24" fill="currentColor"><path d="M22 22H20V20H22V22ZM22 18H20V16H22V18ZM18 22H16V20H18V22ZM22 14H20V12H22V14ZM18 18H16V16H18V18ZM14 22H12V20H14V22Z" /></svg>
        </div>
      )}
    </motion.div>
  );
}
