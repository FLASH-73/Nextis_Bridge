import React, { useState, useEffect } from 'react';
import { X, Play, StopCircle, Activity, Maximize2, Minimize2, AlertCircle, RotateCcw } from 'lucide-react';
import { motion, useDragControls } from 'framer-motion';
import { useResizable } from '../../../hooks/useResizable';
import { usePolling } from '../../../hooks/usePolling';
import { EmergencyStop } from '../../EmergencyStop';
import ControlGroupSelector from './ControlGroupSelector';
import TelemetryGraph from './TelemetryGraph';
import SettingsOverlay from './SettingsOverlay';
import CameraFeed from '../../ui/CameraFeed';
import { teleopApi, armsApi, calibrationApi, camerasApi, systemApi } from '../../../lib/api';

interface Pairing {
    leader_id: string;
    follower_id: string;
    name: string;
}

interface TeleopModalProps {
    isOpen: boolean;
    onClose: () => void;
    maximizedWindow: string | null;
    setMaximizedWindow: (window: string | null) => void;
}

export default function TeleopModal({ isOpen, onClose, maximizedWindow, setMaximizedWindow }: TeleopModalProps) {
    const dragControls = useDragControls();
    const { size, handleResizeMouseDown } = useResizable({ initialSize: { width: 950, height: 900 } });

    const [isRunning, setIsRunning] = useState(false);
    const [error, setError] = useState('');
    const [graphData, setGraphData] = useState<any[]>([]);
    const [torqueData, setTorqueData] = useState<any>({});
    const [isMinimized, setIsMinimized] = useState(false);
    const isMaximized = maximizedWindow === 'teleop';

    const [arms, setArms] = useState<any[]>([]);
    const [selectedArms, setSelectedArms] = useState<string[]>([]);
    const [showForceWarning, setShowForceWarning] = useState(false);
    const [cameraConfigs, setCameraConfigs] = useState<any[]>([]);
    const [pairings, setPairings] = useState<Pairing[]>([]);

    // Settings state
    const [showSettings, setShowSettings] = useState(false);
    const [wizardArm, setWizardArm] = useState<string | null>(null);
    const [activeArmId, setActiveArmId] = useState<string | null>(null);
    const [inversions, setInversions] = useState<{ [key: string]: boolean }>({});
    const [motors, setMotors] = useState<string[]>([]);
    const [assistEnabled, setAssistEnabled] = useState(false); // Default False

    // Haptic Tuning State
    const [tuning, setTuning] = useState({
        k_gravity: 1.0,
        k_assist: 0.5,
        k_haptic: 0.0,
        v_threshold: 2.0,
        k_damping: 5.0
    });

    // Initial Status Check
    useEffect(() => {
        if (isOpen) {
            checkStatus();
            fetchArms();
            fetchPairings();
            // Fetch Camera Configs
            camerasApi.config()
                .then(data => {
                    if (Array.isArray(data)) setCameraConfigs(data);
                })
                .catch(console.error);
        }
    }, [isOpen]);

    const checkStatus = async () => {
        try {
            const data = await teleopApi.status();
            setIsRunning(data.running);
        } catch (e) {
            console.error(e);
        }
    };

    const fetchArms = async () => {
        try {
            const data = await calibrationApi.arms();
            const armList = data.arms || [];
            setArms(armList);
            // Default: Select All
            if (selectedArms.length === 0) {
                setSelectedArms(armList.map((a: any) => a.id));
            }
        } catch (e) {
            console.error("Failed to fetch arms", e);
        }
    };

    const fetchPairings = async () => {
        try {
            const data = await armsApi.listPairings();
            setPairings(data.pairings || []);
        } catch (e) {
            console.error("Failed to fetch pairings", e);
        }
    };

    // Poll for telemetry data while running
    usePolling(async () => {
        try {
            const data = await teleopApi.data() as any;
            if (data.data) {
                if (Array.isArray(data.data)) {
                    setGraphData(data.data);
                } else {
                    if (data.data.history) setGraphData(data.data.history);
                    if (data.data.torque) setTorqueData(data.data.torque);
                }
            }
        } catch (e) {
            console.error(e);
        }
    }, 100, isOpen && isRunning);

    const handleStartClick = () => {
        if (isRunning) {
            toggleTeleop();
        } else {
            // Check Selection Validation (use role from API, fallback to substring)
            const leaders = selectedArms.filter(id => {
                const arm = arms.find(a => a.id === id);
                return arm?.role === 'leader' || arm?.type === 'leader' || id.includes('leader');
            });
            const followers = selectedArms.filter(id => {
                const arm = arms.find(a => a.id === id);
                return arm?.role === 'follower' || arm?.type === 'follower' || id.includes('follower');
            });

            if (leaders.length === 0 || followers.length === 0) {
                setError("Please select at least one Leader and one Follower.");
                return;
            }

            // Check calibration of SELECTED arms
            // We only care if the *active* arms are calibrated.
            const activeUncalibrated = arms.filter(a => selectedArms.includes(a.id) && !a.calibrated);

            if (activeUncalibrated.length > 0) {
                setShowForceWarning(true);
            } else {
                toggleTeleop();
            }
        }
    }

    const toggleTeleop = async (force = false) => {
        setError('');
        setShowForceWarning(false);
        try {
            if (isRunning) {
                await teleopApi.stop();
                setIsRunning(false);
            } else {
                const data = await teleopApi.start({ force, active_arms: selectedArms }) as any;
                if (data.status === 'error') {
                    setError(data.message);
                } else {
                    setIsRunning(true);
                }
            }
        } catch (e: any) {
            setError(e.message || "Connection Error");
        }
    };

    const handleReset = async () => {
        if (!confirm("Reconnect Hardware? This will briefly stop the system.")) return;
        try {
            setError('Reconnecting...');
            const data = await systemApi.reconnect() as any;
            if (data.status === 'busy') {
                setError('System Busy (Initializing...)');
            } else {
                setError('Reconnecting Hardware...');
                // Refresh page after delay? No, stay here?
                // User wants to stay in modal usually.
                // But reconnect might change successful arms.
                setTimeout(() => checkStatus(), 3000);
            }
        } catch (e) {
            setError('Reconnect Failed.');
        }
    };

    const handleSettingsOpen = async (armId: string) => {
        setActiveArmId(armId);
        setShowSettings(true);
        // Fetch current inversions and profiles
        try {
            const [invData, profData] = await Promise.all([
                calibrationApi.inversions(armId),
                calibrationApi.files(armId)
            ]);

            setInversions((invData as any).inversions || {});
            setMotors((invData as any).motors || []);
        } catch (e) {
            console.error(e);
        }
    }

    if (!isOpen) return null;

    return (
        <motion.div
            drag
            dragControls={dragControls}
            dragListener={false} // Only drag header
            dragMomentum={false} // Disable INERTIA (Stop exactly where released)
            initial={{ x: 500, y: 80, opacity: 0, scale: 0.95 }}
            animate={{
                opacity: 1,
                scale: 1,
                x: isMaximized ? 20 : undefined,
                y: isMaximized ? 80 : undefined,
            }}
            style={{
                width: isMaximized ? 'calc(100vw - 40px)' : isMinimized ? 'auto' : `${size.width}px`,
                height: isMaximized ? 'calc(100vh - 100px)' : isMinimized ? 'auto' : `${size.height}px`,
                zIndex: isMaximized ? 100 : 50,
            }}
            className="fixed flex flex-col overflow-hidden font-sans glass-panel shadow-2xl rounded-2xl bg-white/95 dark:bg-zinc-900/95 border border-white/50 dark:border-zinc-700/50 backdrop-blur-3xl transition-all duration-300"
        >
            {/* Header */}
            <div
                onPointerDown={(e) => dragControls.start(e)}
                className="h-14 bg-white/40 dark:bg-zinc-800/40 border-b border-black/5 dark:border-white/5 flex items-center justify-between px-5 cursor-grab active:cursor-grabbing select-none"
            >
                <div className="flex items-center gap-2">
                    <div className="flex gap-1.5 mr-3">
                        <button onClick={onClose} className="w-3 h-3 rounded-full bg-[#FF5F57] hover:brightness-90 btn-control transition-all"><X className="w-2 h-2 text-black/50 opacity-0 hover:opacity-100" /></button>
                        <button onClick={() => { setIsMinimized(!isMinimized); if (isMaximized) setMaximizedWindow(null); }} className="w-3 h-3 rounded-full bg-[#FEBC2E] hover:brightness-90 btn-control transition-all"><Minimize2 className="w-2 h-2 text-black/50 opacity-0 hover:opacity-100" /></button>
                        <button onClick={() => { setMaximizedWindow(isMaximized ? null : 'teleop'); if (isMinimized) setIsMinimized(false); }} className="w-3 h-3 rounded-full bg-[#28C840] hover:brightness-90 btn-control transition-all"><Maximize2 className="w-2 h-2 text-black/50 opacity-0 hover:opacity-100" /></button>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    <EmergencyStop />
                    <span className="font-semibold text-neutral-800 dark:text-zinc-200 text-sm tracking-wide flex items-center gap-2">
                        <Activity className="w-4 h-4 text-blue-500" /> Teleoperation

                    </span>

                    <button
                        onClick={handleReset}
                        title="Reconnect Hardware"
                        className="p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-md transition-colors text-neutral-500 dark:text-zinc-400 hover:text-neutral-800 dark:hover:text-zinc-200"
                    >
                        <RotateCcw className="w-3.5 h-3.5" />
                    </button>

                    {isRunning && <span className="text-[10px] font-bold px-2 py-0.5 bg-green-100 dark:bg-green-900/50 text-green-600 dark:text-green-400 rounded-full animate-pulse border border-green-200 dark:border-green-700 ml-2 tracking-wider">LIVE</span>}
                </div>
            </div>

            {!isMinimized && (
                <div className="flex-1 flex flex-col p-6 gap-6 overflow-y-auto relative">

                    {/* Settings Overlay */}
                    {showSettings && (
                        <SettingsOverlay
                            activeArmId={activeArmId}
                            wizardArm={wizardArm}
                            setWizardArm={setWizardArm}
                            inversions={inversions}
                            setInversions={setInversions}
                            motors={motors}
                            setMotors={setMotors}
                            assistEnabled={assistEnabled}
                            setAssistEnabled={setAssistEnabled}
                            tuning={tuning}
                            setTuning={setTuning}
                            onClose={() => setShowSettings(false)}
                            onRefresh={handleSettingsOpen}
                        />
                    )}

                    {/* Error Banner */}
                    {error && (
                        <div className="bg-red-50 dark:bg-red-950/50 border border-red-100 dark:border-red-900 text-red-600 dark:text-red-400 p-3 rounded-xl text-sm flex items-center justify-between animate-in fade-in slide-in-from-top-2">
                            <div className="flex items-center gap-2">
                                <AlertCircle className="w-4 h-4" /> {error}
                            </div>
                            <button
                                onClick={async () => {
                                    try {
                                        setError('Resetting System...');
                                        await systemApi.restart();
                                        setError('');
                                        // Force UI refresh after brief delay
                                        setTimeout(() => window.location.reload(), 1500);
                                    } catch (e) { setError('Reset Failed.'); }
                                }}
                                className="px-3 py-1 bg-red-100 dark:bg-red-900/50 hover:bg-red-200 dark:hover:bg-red-900 text-red-700 dark:text-red-300 rounded-lg text-xs font-bold transition-colors"
                            >
                                System Reset
                            </button>
                        </div>
                    )}

                    {/* Force Warning Modal Overlay */}
                    {showForceWarning && (
                        <div className="absolute inset-0 bg-white/90 dark:bg-zinc-900/90 backdrop-blur-sm z-50 flex flex-col items-center justify-center p-8 text-center animate-in fade-in">
                            <div className="w-16 h-16 bg-red-50 dark:bg-red-950 rounded-full flex items-center justify-center mb-4">
                                <AlertCircle className="w-8 h-8 text-red-500" />
                            </div>
                            <h3 className="text-2xl font-bold text-neutral-900 dark:text-zinc-100 mb-2">Uncalibrated System</h3>
                            <p className="text-neutral-500 dark:text-zinc-400 mb-8 max-w-md leading-relaxed">
                                Detailed calibration data is missing. Proceeding manually might cause
                                <b className="text-red-600 dark:text-red-400 font-medium"> unexpected motion</b>.
                            </p>
                            <div className="flex gap-4">
                                <button
                                    onClick={() => setShowForceWarning(false)}
                                    className="px-8 py-3 rounded-full border border-neutral-200 dark:border-zinc-700 hover:bg-neutral-50 dark:hover:bg-zinc-800 font-medium text-neutral-600 dark:text-zinc-400 transition-all"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={() => toggleTeleop(true)}
                                    className="px-8 py-3 rounded-full bg-red-500 text-white hover:bg-red-600 font-bold shadow-xl shadow-red-200 dark:shadow-red-900/30 transition-all hover:scale-105 active:scale-95"
                                >
                                    Force Start
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Control Group Selection */}
                    <ControlGroupSelector
                        arms={arms}
                        selectedArms={selectedArms}
                        setSelectedArms={setSelectedArms}
                        pairings={pairings}
                        handleSettingsOpen={handleSettingsOpen}
                        handleReset={handleReset}
                    />

                    {/* Cameras Area */}
                    <div className="flex justify-center gap-4 flex-none flex-wrap">
                        {cameraConfigs.length === 0 && (
                            <div className="text-neutral-400 dark:text-zinc-500 text-sm p-4 border border-dashed border-neutral-300 dark:border-zinc-700 rounded-lg">
                                No cameras configured
                            </div>
                        )}
                        {cameraConfigs.map((cam) => (
                            <div key={cam.id} className="flex-1 min-w-[280px] max-w-[50%] aspect-video">
                                <CameraFeed
                                    cameraId={cam.id}
                                    maxStreamWidth={800}
                                    mode="contain"
                                    quality={85}
                                />
                            </div>
                        ))}
                    </div>

                    {/* Telemetry Graph + Torque Bars */}
                    <TelemetryGraph
                        graphData={graphData}
                        torqueData={torqueData}
                        isRunning={isRunning}
                    />

                    {/* Footer / Controls */}
                    <div className="flex justify-center pt-2">
                        <button
                            onClick={handleStartClick}
                            className={`px-10 py-3.5 rounded-full font-bold shadow-xl transition-all hover:scale-[1.02] active:scale-[0.98] flex items-center gap-3 text-sm tracking-wide ${isRunning
                                ? 'bg-red-500 text-white shadow-red-200 dark:shadow-red-900/30 hover:bg-red-600'
                                : 'bg-neutral-900 dark:bg-zinc-100 text-white dark:text-zinc-900 shadow-neutral-200 dark:shadow-zinc-800 hover:bg-black dark:hover:bg-white'
                                }`}
                        >
                            {isRunning ? (
                                <> <StopCircle className="w-5 h-5" /> Stop Session </>
                            ) : (
                                <> <Play className="w-5 h-5 fill-current" /> Start Teleoperation </>
                            )}
                        </button>
                    </div>
                </div>
            )}

            {/* Resize Handle */}
            {!isMinimized && (
                <div
                    onMouseDown={handleResizeMouseDown}
                    className="absolute bottom-0 right-0 w-6 h-6 cursor-nwse-resize flex items-end justify-end p-1 z-50 group pointer-events-auto"
                >
                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className="text-neutral-300 dark:text-zinc-600 group-hover:text-blue-500 transition-colors">
                        <path d="M11 1L11 11L1 11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                        <path d="M8 4L8 8L4 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity="0.5" />
                    </svg>
                </div>
            )}
        </motion.div>
    );
}
