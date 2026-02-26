import { useEffect, useCallback, useMemo } from "react";
import { X, Activity, AlertCircle, Loader2 } from "lucide-react";
import useSWR from "swr";
import {
  armsApi,
  camerasApi,
  toolsApi,
  triggersApi,
  toolPairingsApi,
  recordingApi,
} from "../../../lib/api";
import type {
  Arm,
  Pairing,
  CameraConfig,
  CameraStatusEntry,
  Tool,
  Trigger,
  ListenerStatus,
  RecordingOptions,
} from "../../../lib/api/types";
import ArmPairCard from "./ArmPairCard";
import CameraGrid from "./CameraGrid";
import ToolsList from "./ToolsList";
import DatasetConfig from "./DatasetConfig";

interface RecordingSetupPanelProps {
  isOpen: boolean;
  onClose: () => void;
  onStartSession: () => void;
  datasetConfig: { repo_id: string; task: string };
  setDatasetConfig: (config: { repo_id: string; task: string }) => void;
  selectedPairings: string[];
  setSelectedPairings: (names: string[]) => void;
  selectedCameras: string[];
  setSelectedCameras: (ids: string[]) => void;
  error: string;
  isStarting: boolean;
  recordExtendedState: boolean;
  setRecordExtendedState: (value: boolean) => void;
}

export default function RecordingSetupPanel({
  isOpen,
  onClose,
  onStartSession,
  datasetConfig,
  setDatasetConfig,
  selectedPairings,
  setSelectedPairings,
  selectedCameras,
  setSelectedCameras,
  error,
  isStarting,
  recordExtendedState,
  setRecordExtendedState,
}: RecordingSetupPanelProps) {
  // ── Data Fetching (SWR) ─────────────────────────────────────────────────
  const { data: armsData } = useSWR<{
    arms: Arm[];
    summary: { total_arms: number; connected: number };
  }>(isOpen ? "/arms" : null, () => armsApi.list());

  const { data: pairingsData } = useSWR<{ pairings: Pairing[] }>(
    isOpen ? "/arms/pairings" : null,
    () => armsApi.listPairings()
  );

  const { data: cameraConfigs } = useSWR<CameraConfig[]>(
    isOpen ? "/cameras/config" : null,
    () => camerasApi.config()
  );

  const { data: cameraStatuses } = useSWR<Record<string, CameraStatusEntry>>(
    isOpen ? "/cameras/status" : null,
    () => camerasApi.status(),
    { refreshInterval: 5000 }
  );

  const { data: tools } = useSWR<Tool[]>(isOpen ? "/tools" : null, () =>
    toolsApi.list()
  );

  const { data: triggers } = useSWR<Trigger[]>(
    isOpen ? "/triggers" : null,
    () => triggersApi.list()
  );

  const { data: listenerStatus } = useSWR<ListenerStatus>(
    isOpen ? "/tool-pairings/listener/status" : null,
    () => toolPairingsApi.listenerStatus(),
    { refreshInterval: 5000 }
  );

  const { data: recordingOptions } = useSWR<RecordingOptions>(
    isOpen ? "/recording/options" : null,
    () => recordingApi.options()
  );

  const arms = armsData?.arms ?? [];
  const pairings = useMemo(() => pairingsData?.pairings ?? [], [pairingsData]);
  const cameras = useMemo(() => cameraConfigs ?? [], [cameraConfigs]);

  // ── Auto-select all on first load ───────────────────────────────────────
  useEffect(() => {
    if (pairings.length > 0 && selectedPairings.length === 0) {
      setSelectedPairings(pairings.map((p) => p.name));
    }
  }, [pairings, selectedPairings.length, setSelectedPairings]);

  useEffect(() => {
    if (cameras.length > 0 && selectedCameras.length === 0) {
      setSelectedCameras(cameras.map((c) => c.id));
    }
  }, [cameras, selectedCameras.length, setSelectedCameras]);

  // ── Keyboard ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isOpen, onClose]);

  // ── Helpers ─────────────────────────────────────────────────────────────
  const togglePairing = useCallback(
    (name: string) => {
      setSelectedPairings(
        selectedPairings.includes(name)
          ? selectedPairings.filter((n) => n !== name)
          : [...selectedPairings, name]
      );
    },
    [selectedPairings, setSelectedPairings]
  );

  const toggleCamera = useCallback(
    (id: string) => {
      setSelectedCameras(
        selectedCameras.includes(id)
          ? selectedCameras.filter((c) => c !== id)
          : [...selectedCameras, id]
      );
    },
    [selectedCameras, setSelectedCameras]
  );

  // ── Start validation ────────────────────────────────────────────────────
  const canStart =
    selectedPairings.length > 0 &&
    selectedCameras.length > 0 &&
    datasetConfig.repo_id.trim().length > 0 &&
    datasetConfig.task.trim().length > 0 &&
    !isStarting;

  let disabledReason = "";
  if (selectedPairings.length === 0) disabledReason = "Select at least one arm pair";
  else if (selectedCameras.length === 0) disabledReason = "Select at least one camera";
  else if (!datasetConfig.repo_id.trim()) disabledReason = "Enter a dataset name";
  else if (!datasetConfig.task.trim()) disabledReason = "Enter a task description";

  const isLoading = !armsData || !pairingsData || !cameraConfigs;

  // ── Render ──────────────────────────────────────────────────────────────
  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm"
          onClick={onClose}
        />
      )}

      {/* Panel */}
      <div
        className={`fixed right-0 top-0 h-full z-50 transform transition-transform duration-300 ease-out ${
          isOpen ? "translate-x-0" : "translate-x-full"
        } w-[clamp(500px,65vw,900px)] max-sm:w-full bg-white dark:bg-zinc-900 border-l border-neutral-200 dark:border-zinc-700 shadow-2xl flex flex-col`}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-100 dark:border-zinc-800 flex-shrink-0">
          <div className="flex items-center gap-3">
            <Activity className="w-5 h-5 text-red-500" />
            <h2 className="text-lg font-semibold text-neutral-900 dark:text-zinc-100">
              Recording Setup
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-neutral-100 dark:hover:bg-zinc-800 rounded-lg transition-colors"
          >
            <X className="w-4 h-4 text-neutral-500 dark:text-zinc-400" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          {isLoading ? (
            <div className="flex items-center justify-center py-16 text-neutral-400 dark:text-zinc-500">
              <Loader2 className="w-5 h-5 animate-spin mr-2" />
              Loading devices...
            </div>
          ) : (
            <>
              {/* Arm Pairs */}
              <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700 p-4">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-zinc-400 mb-3">
                  Arm Pairs
                </h3>
                {pairings.length === 0 ? (
                  <p className="text-xs text-neutral-400 dark:text-zinc-500 text-center py-4">
                    No pairings configured
                  </p>
                ) : (
                  <div className="space-y-2">
                    {pairings.map((p) => (
                      <ArmPairCard
                        key={p.name}
                        pairing={p}
                        leader={arms.find((a) => a.id === p.leader_id)}
                        follower={arms.find((a) => a.id === p.follower_id)}
                        selected={selectedPairings.includes(p.name)}
                        onToggle={() => togglePairing(p.name)}
                      />
                    ))}
                  </div>
                )}
              </div>

              {/* Cameras */}
              <CameraGrid
                cameras={cameras}
                statuses={cameraStatuses ?? {}}
                selectedCameras={selectedCameras}
                onToggleCamera={toggleCamera}
              />

              {/* Extended State */}
              {recordingOptions?.supports_extended_state && (
                <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700 p-4">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-zinc-400 mb-3">
                    Motor State
                  </h3>
                  <label className="flex items-center justify-between cursor-pointer group">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-neutral-800 dark:text-zinc-200">
                        Record velocity &amp; torque
                      </p>
                      <p className="text-xs text-neutral-500 dark:text-zinc-400 mt-0.5">
                        Adds motor velocity (rad/s) and torque (Nm) to observation state.
                        Required for contact-rich tasks like insertion.
                        Zero CAN overhead — uses MIT response cache.
                      </p>
                    </div>
                    <div className="ml-4 flex-shrink-0">
                      <button
                        type="button"
                        role="switch"
                        aria-checked={recordExtendedState}
                        onClick={() => setRecordExtendedState(!recordExtendedState)}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                          recordExtendedState
                            ? 'bg-black dark:bg-white'
                            : 'bg-neutral-300 dark:bg-zinc-600'
                        }`}
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-white dark:bg-zinc-900 transition-transform ${
                            recordExtendedState ? 'translate-x-6' : 'translate-x-1'
                          }`}
                        />
                      </button>
                    </div>
                  </label>
                  {recordExtendedState && (
                    <div className="mt-3 text-xs text-neutral-500 dark:text-zinc-400 bg-neutral-100 dark:bg-zinc-800 rounded-lg px-3 py-2">
                      <span className="font-medium text-neutral-700 dark:text-zinc-300">State vector: </span>
                      7 positions + 7 velocities + 7 torques = 21 floats per timestep
                    </div>
                  )}
                </div>
              )}

              {/* Tools */}
              <ToolsList
                tools={tools ?? []}
                triggers={triggers ?? []}
                listenerStatus={listenerStatus ?? null}
              />

              {/* Dataset Config + Safety */}
              <DatasetConfig
                datasetConfig={datasetConfig}
                setDatasetConfig={setDatasetConfig}
              />
            </>
          )}
        </div>

        {/* Footer */}
        <div className="flex-shrink-0 px-6 py-4 border-t border-neutral-100 dark:border-zinc-800 space-y-2">
          {error && (
            <div className="bg-red-50 dark:bg-red-950/50 text-red-600 dark:text-red-400 px-3 py-2 rounded-lg text-xs flex items-center gap-2 border border-red-100 dark:border-red-900">
              <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
              {error}
            </div>
          )}

          <button
            onClick={onStartSession}
            disabled={!canStart}
            className={`w-full py-3 rounded-xl font-medium text-sm transition-all ${
              canStart
                ? "bg-black dark:bg-white text-white dark:text-black hover:bg-neutral-800 dark:hover:bg-zinc-200 hover:scale-[1.01] active:scale-[0.99] shadow-lg"
                : "bg-neutral-200 dark:bg-zinc-700 text-neutral-400 dark:text-zinc-500 cursor-not-allowed"
            }`}
          >
            {isStarting ? (
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                Starting...
              </span>
            ) : canStart ? (
              "Start Recording Session"
            ) : (
              disabledReason || "Start Recording Session"
            )}
          </button>
        </div>
      </div>
    </>
  );
}
