import React, { useState, useEffect, useRef } from 'react';
import { X, Play, StopCircle, Activity, Maximize2, Minimize2, AlertCircle, Settings, RotateCcw, Link2 } from 'lucide-react';
import { motion, useDragControls } from 'framer-motion';
import { useResizable } from '../hooks/useResizable';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import GravityWizard from './GravityWizard';
import { EmergencyStop } from './EmergencyStop';

interface Pairing {
    leader_id: string;
    follower_id: string;
    name: string;
}

const API_BASE = typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : 'http://127.0.0.1:8000';

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

    // Polling Interval Ref
    const pollRef = useRef<NodeJS.Timeout | null>(null);

    const [arms, setArms] = useState<any[]>([]);
    const [selectedArms, setSelectedArms] = useState<string[]>([]);
    const [showForceWarning, setShowForceWarning] = useState(false);
    const [cameraConfigs, setCameraConfigs] = useState<any[]>([]);
    const [pairings, setPairings] = useState<Pairing[]>([]);

    // Initial Status Check
    useEffect(() => {
        if (isOpen) {
            checkStatus();
            fetchArms();
            fetchPairings();
            // Fetch Camera Configs
            fetch(`${API_BASE}/cameras/config`)
                .then(res => res.json())
                .then(data => {
                    if (Array.isArray(data)) setCameraConfigs(data);
                })
                .catch(console.error);
        } else {
            stopPolling();
        }
    }, [isOpen]);

    const checkStatus = async () => {
        try {
            const res = await fetch(`${API_BASE}/teleop/status`);
            const data = await res.json();
            setIsRunning(data.running);
            if (data.running) {
                startPolling();
                // If running, we should ideally fetch the active arms from backend to show correct selection?
                // But simplified: just keep current UI.
            }
        } catch (e) {
            console.error(e);
        }
    };

    const fetchArms = async () => {
        try {
            const res = await fetch(`${API_BASE}/calibration/arms`);
            const data = await res.json();
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
            const res = await fetch(`${API_BASE}/arms/pairings`);
            const data = await res.json();
            setPairings(data.pairings || []);
        } catch (e) {
            console.error("Failed to fetch pairings", e);
        }
    };

    // Get active pairings based on current selection
    const activePairings = pairings.filter(p =>
        selectedArms.includes(p.leader_id) && selectedArms.includes(p.follower_id)
    );

    const startPolling = () => {
        if (pollRef.current) return;
        pollRef.current = setInterval(async () => {
            try {
                const res = await fetch(`${API_BASE}/teleop/data`);
                const data = await res.json();
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
        }, 100);
    };

    const stopPolling = () => {
        if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
        }
    };

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
                await fetch(`${API_BASE}/teleop/stop`, { method: 'POST' });
                setIsRunning(false);
                stopPolling();
            } else {
                const res = await fetch(`${API_BASE}/teleop/start`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ force, active_arms: selectedArms })
                });
                const data = await res.json();
                if (data.status === 'error') {
                    setError(data.message);
                } else {
                    setIsRunning(true);
                    startPolling();
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
            const res = await fetch(`${API_BASE}/system/reconnect`, { method: 'POST' });
            const data = await res.json();
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

    const [showSettings, setShowSettings] = useState(false);
    const [wizardArm, setWizardArm] = useState<string | null>(null);
    const [activeArmId, setActiveArmId] = useState<string | null>(null);
    const [inversions, setInversions] = useState<{ [key: string]: boolean }>({});
    const [motors, setMotors] = useState<string[]>([]);
    const [assistEnabled, setAssistEnabled] = useState(false); // Default False

    const [profiles, setProfiles] = useState<any[]>([]);

    // Haptic Tuning State
    const [tuning, setTuning] = useState({
        k_gravity: 1.0,
        k_assist: 0.5,
        k_haptic: 0.0,
        v_threshold: 2.0,
        k_damping: 5.0
    });

    // Debounced Update
    useEffect(() => {
        const timer = setTimeout(() => {
            fetch(`${API_BASE}/teleop/tune`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(tuning)
            }).catch(console.error);
        }, 300);
        return () => clearTimeout(timer);
    }, [tuning]);

    const updateTuning = (newValues: any) => {
        setTuning(prev => ({ ...prev, ...newValues }));
    };

    const handleSettingsOpen = async (armId: string) => {
        setActiveArmId(armId);
        setShowSettings(true);
        // Fetch current inversions and profiles
        try {
            const [invRes, profRes] = await Promise.all([
                fetch(`${API_BASE}/calibration/${armId}/inversions`),
                fetch(`${API_BASE}/calibration/${armId}/files`)
            ]);

            const invData = await invRes.json();
            const profData = await profRes.json();

            setInversions(invData.inversions || {});
            setMotors(invData.motors || []);
            setProfiles(profData.files || []);
        } catch (e) {
            console.error(e);
        }
    }

    const loadProfile = async (filename: string) => {
        if (!activeArmId) return;
        try {
            const res = await fetch(`${API_BASE}/calibration/${activeArmId}/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename })
            });
            const data = await res.json();
            if (data.status === 'success') {
                // Refresh list to update 'active' tag
                handleSettingsOpen(activeArmId);
            }
        } catch (e) {
            console.error("Failed to load profile", e);
        }
    }

    const toggleInversion = async (motor: string) => {
        if (!activeArmId) return;
        const newState = !inversions[motor];
        setInversions(prev => ({ ...prev, [motor]: newState }));

        // Save immediately
        try {
            await fetch(`${API_BASE}/calibration/${activeArmId}/inversions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ motor, inverted: newState })
            });
        } catch (e) {
            console.error("Failed to save inversion", e);
        }
    }

    const toggleAssist = async () => {
        const newState = !assistEnabled;
        setAssistEnabled(newState);
        try {
            await fetch(`${API_BASE}/teleop/assist/set`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: newState })
            });
        } catch (e) {
            console.error(e);
        }
    }

    // Use a fixed list of colors for lines
    const colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#00C49F"];


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
                        <div className="absolute inset-0 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-md z-40 flex flex-col animate-in fade-in slide-in-from-bottom-4 p-8">
                            <div className="flex justify-between items-center mb-8">
                                <div>
                                    <h2 className="text-2xl font-bold text-neutral-900 dark:text-zinc-100">Motor Configuration</h2>
                                    <p className="text-neutral-500 dark:text-zinc-400 text-sm">Customize direciton logic for {activeArmId}</p>
                                </div>
                                <button onClick={() => setShowSettings(false)} className="p-2 hover:bg-neutral-100 dark:hover:bg-zinc-800 rounded-full transition-colors"><X className="w-6 h-6 text-neutral-500 dark:text-zinc-400" /></button>
                            </div>

                            <div className="flex-1 overflow-y-auto">
                                {/* Gravity Wizard Overlay */}
                                {wizardArm && (
                                    <GravityWizard
                                        armId={wizardArm}
                                        armKey={wizardArm.includes('left') ? 'left' : wizardArm.includes('right') ? 'right' : 'default'}
                                        onClose={() => setWizardArm(null)}
                                    />
                                )}

                                {/* Auto-Align Wizard */}
                                <div className="mb-6 p-4 bg-blue-50/50 dark:bg-blue-950/30 border border-blue-100 dark:border-blue-900 rounded-xl">
                                    <div className="flex justify-between items-start mb-2">
                                        <h4 className="text-sm font-semibold text-blue-900 dark:text-blue-300 flex items-center gap-2">
                                            <Activity className="w-4 h-4" /> Smart Auto-Alignment
                                        </h4>
                                        {activeArmId && (
                                            <button
                                                onClick={() => setWizardArm(activeArmId)}
                                                className="px-3 py-1 bg-white dark:bg-zinc-800 border border-blue-200 dark:border-blue-700 text-blue-700 dark:text-blue-400 rounded-lg text-xs font-bold hover:bg-blue-50 dark:hover:bg-blue-950 transition-colors shadow-sm"
                                            >
                                                Recalibrate Gravity
                                            </button>
                                        )}
                                    </div>
                                    <p className="text-xs text-blue-700 dark:text-blue-400 mb-3">
                                        Eliminate trial-and-error! Follow the steps to automatically detect inverted motors.
                                    </p>

                                    <div className="flex gap-2">
                                        <button
                                            onClick={async () => {
                                                if (!activeArmId) return;
                                                // 1. Set Zero for both Leader and Follower
                                                // Assume naming convention: "left_follower" -> "left_leader"
                                                const side = activeArmId.split('_')[0];
                                                const pairs = [`${side}_leader`, `${side}_follower`];

                                                try {
                                                    await Promise.all(pairs.map(id => fetch(`${API_BASE}/calibration/${id}/set-zero`, { method: 'POST' })));
                                                    alert("Step 1 Done: Zero Pose Captured. Now move both arms to a new position (e.g. 45° forward) and click 'Align'.");
                                                } catch (e) {
                                                    alert("Failed to capture zero pose.");
                                                }
                                            }}
                                            className="flex-1 py-2 bg-white dark:bg-zinc-800 border border-blue-200 dark:border-blue-700 text-blue-800 dark:text-blue-300 rounded-lg text-xs font-medium hover:bg-blue-50 dark:hover:bg-blue-950 transition-colors"
                                        >
                                            1. Set Zero (Straight)
                                        </button>

                                        <button
                                            onClick={async () => {
                                                if (!activeArmId) return;
                                                // 2. Compute
                                                try {
                                                    const res = await fetch(`${API_BASE}/calibration/${activeArmId}/auto-align`, { method: 'POST' });
                                                    const data = await res.json();
                                                    if (data.status === 'success') {
                                                        alert(`Success! Inverted ${data.inverted_count} motors.\nChanges: ${JSON.stringify(data.changes, null, 2)}`);
                                                        handleSettingsOpen(activeArmId); // Refresh
                                                    } else {
                                                        alert(`Error: ${data.message}`);
                                                    }
                                                } catch (e) {
                                                    alert("Alignment failed.");
                                                }
                                            }}
                                            className="flex-1 py-2 bg-blue-600 text-white rounded-lg text-xs font-medium hover:bg-blue-700 transition-colors shadow-sm"
                                        >
                                            2. Auto-Align (Moved)
                                        </button>
                                    </div>
                                </div>

                                {/* Profiles Header */}
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="text-lg font-semibold text-neutral-800 dark:text-zinc-200">Calibration Profile</h3>
                                </div>
                                <div className="space-y-2 mb-8 max-h-[160px] overflow-y-auto pr-2">
                                    {profiles.length === 0 && <p className="text-neutral-400 dark:text-zinc-500 text-sm">No profiles found.</p>}
                                    {profiles.map(p => (
                                        <div key={p.name} className={`flex items-center justify-between p-3 rounded-lg border transition-all ${p.active ? 'bg-blue-50 dark:bg-blue-950/50 border-blue-200 dark:border-blue-800 shadow-sm' : 'bg-white dark:bg-zinc-800 border-neutral-100 dark:border-zinc-700'}`}>
                                            <div className="flex flex-col">
                                                <span className={`font-medium text-sm ${p.active ? 'text-blue-700 dark:text-blue-400' : 'text-neutral-700 dark:text-zinc-300'}`}>{p.name}</span>
                                                <span className="text-xs text-neutral-400 dark:text-zinc-500">{p.created}</span>
                                            </div>
                                            {p.active ? (
                                                <span className="text-xs font-bold text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/50 px-2.5 py-1 rounded-full">ACTIVE</span>
                                            ) : (
                                                <button
                                                    onClick={() => loadProfile(p.name)}
                                                    className="px-3 py-1.5 text-xs font-medium text-neutral-600 dark:text-zinc-400 bg-neutral-100 dark:bg-zinc-700 hover:bg-neutral-200 dark:hover:bg-zinc-600 rounded-md transition-colors"
                                                >
                                                    Load
                                                </button>
                                            )}
                                        </div>
                                    ))}
                                </div>

                                <h3 className="text-lg font-bold text-neutral-800 dark:text-zinc-200 mb-4">Motor Direction</h3>
                                <div className="grid grid-cols-2 gap-4 content-start">
                                    {motors.map(motor => (
                                        <div key={motor} className="flex items-center justify-between p-4 bg-white dark:bg-zinc-800 border border-neutral-100 dark:border-zinc-700 shadow-sm rounded-xl">
                                            <span className="font-medium text-neutral-700 dark:text-zinc-300">{motor}</span>
                                            <button
                                                onClick={() => toggleInversion(motor)}
                                                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${inversions[motor] ? 'bg-blue-600' : 'bg-neutral-200 dark:bg-zinc-600'}`}
                                            >
                                                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${inversions[motor] ? 'translate-x-6' : 'translate-x-1'}`} />
                                            </button>
                                        </div>
                                    ))}
                                </div>

                                {activeArmId?.includes('leader') && (
                                    <>
                                        <h3 className="text-lg font-bold text-neutral-800 dark:text-zinc-200 mb-4 mt-6">Active Assistance</h3>
                                        <div className="bg-white/50 dark:bg-zinc-800/50 border border-white/60 dark:border-zinc-700/60 p-5 rounded-2xl shadow-sm mb-6 space-y-5">
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <span className="font-semibold text-neutral-900 dark:text-zinc-100 block">Status</span>
                                                    <span className="text-xs text-neutral-500 dark:text-zinc-400">Enable/Disable all assistance</span>
                                                </div>
                                                <button
                                                    onClick={toggleAssist}
                                                    className={`relative inline-flex h-7 w-12 items-center rounded-full transition-colors focus:outline-none ${assistEnabled ? 'bg-black dark:bg-white' : 'bg-neutral-200 dark:bg-zinc-600'}`}
                                                >
                                                    <span className={`inline-block h-5 w-5 transform rounded-full bg-white dark:bg-zinc-900 shadow-sm transition-transform ${assistEnabled ? 'translate-x-6' : 'translate-x-1'}`} />
                                                </button>
                                            </div>

                                            {/* Slider: Gravity Compensation */}
                                            <div>
                                                <div className="flex justify-between text-xs font-medium mb-2">
                                                    <span className="text-neutral-600 dark:text-zinc-400">Gravity Compensation</span>
                                                    <span className="text-neutral-900 dark:text-zinc-100">{tuning.k_gravity.toFixed(2)}x</span>
                                                </div>
                                                <input
                                                    type="range"
                                                    min="0" max="2" step="0.05"
                                                    value={tuning.k_gravity}
                                                    onChange={(e) => updateTuning({ k_gravity: parseFloat(e.target.value) })}
                                                    className="w-full h-1.5 bg-neutral-200 dark:bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-black dark:accent-white"
                                                />
                                            </div>

                                            {/* Slider: Friction Transparency */}
                                            <div>
                                                <div className="flex justify-between text-xs font-medium mb-2">
                                                    <span className="text-neutral-600 dark:text-zinc-400">Transparency (Anti-Friction)</span>
                                                    <span className="text-neutral-900 dark:text-zinc-100">{tuning.k_assist.toFixed(1)}x</span>
                                                </div>
                                                <input
                                                    type="range"
                                                    min="0" max="5" step="0.1"
                                                    value={tuning.k_assist}
                                                    onChange={(e) => updateTuning({ k_assist: parseFloat(e.target.value) })}
                                                    className="w-full h-1.5 bg-neutral-200 dark:bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-black dark:accent-white"
                                                />
                                            </div>

                                            {/* Slider: Haptic Feedback */}
                                            <div>
                                                <div className="flex justify-between text-xs font-medium mb-2">
                                                    <span className="text-neutral-600 dark:text-zinc-400">Haptic Feedback (Force)</span>
                                                    <span className="text-neutral-900 dark:text-zinc-100">{(tuning.k_haptic * 100).toFixed(0)}%</span>
                                                </div>
                                                <input
                                                    type="range"
                                                    min="0" max="1" step="0.05"
                                                    value={tuning.k_haptic}
                                                    onChange={(e) => updateTuning({ k_haptic: parseFloat(e.target.value) })}
                                                    className="w-full h-1.5 bg-neutral-200 dark:bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-black dark:accent-white"
                                                />
                                            </div>

                                            {/* Slider: Velocity Threshold */}
                                            <div>
                                                <div className="flex justify-between text-xs font-medium mb-2">
                                                    <span className="text-neutral-600 dark:text-zinc-400">Assist Sensitivity (v-threshold)</span>
                                                    <span className="text-neutral-900 dark:text-zinc-100">{tuning.v_threshold.toFixed(1)}</span>
                                                </div>
                                                <input
                                                    type="range"
                                                    min="0.5" max="10" step="0.5"
                                                    value={tuning.v_threshold}
                                                    onChange={(e) => updateTuning({ v_threshold: parseFloat(e.target.value) })}
                                                    className="w-full h-1.5 bg-neutral-200 dark:bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-black dark:accent-white"
                                                />
                                                <p className="text-[10px] text-neutral-400 dark:text-zinc-500 mt-1">Lower = More sensitive assist</p>
                                            </div>

                                            {/* Slider: Damping (Stability) */}
                                            <div>
                                                <div className="flex justify-between text-xs font-medium mb-2">
                                                    <span className="text-neutral-600 dark:text-zinc-400">Motion Damping (Stability)</span>
                                                    <span className="text-neutral-900 dark:text-zinc-100">{tuning.k_damping?.toFixed(1) || "5.0"}</span>
                                                </div>
                                                <input
                                                    type="range"
                                                    min="0" max="20" step="1"
                                                    value={tuning.k_damping || 5.0}
                                                    onChange={(e) => updateTuning({ k_damping: parseFloat(e.target.value) })}
                                                    className="w-full h-1.5 bg-neutral-200 dark:bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-black dark:accent-white"
                                                />
                                                <p className="text-[10px] text-neutral-400 dark:text-zinc-500 mt-1">Increase to stop drift/rotation</p>
                                            </div>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>
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
                                        await fetch(`${API_BASE}/system/reset`, { method: 'POST' });
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
                    <div className="flex flex-col gap-3">
                        <div className="flex justify-between items-end px-1">
                            <span className="text-xs font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-widest">Active Control Group</span>
                            <div className="flex gap-3">
                                <button
                                    onClick={() => setSelectedArms(arms.map(a => a.id))}
                                    className="text-[10px] uppercase font-bold text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
                                >
                                    Select All
                                </button>
                                <button
                                    onClick={() => setSelectedArms([])}
                                    className="text-[10px] uppercase font-bold text-neutral-400 dark:text-zinc-500 hover:text-neutral-600 dark:hover:text-zinc-400 transition-colors"
                                >
                                    Clear
                                </button>
                            </div>
                        </div>

                        <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-hide justify-center">
                            {arms.length > 0 ? (
                                arms.map(arm => {
                                    const isSelected = selectedArms.includes(arm.id);
                                    return (
                                        <div
                                            key={arm.id}
                                            onClick={() => {
                                                if (isSelected) {
                                                    setSelectedArms(prev => prev.filter(id => id !== arm.id));
                                                } else {
                                                    setSelectedArms(prev => [...prev, arm.id]);
                                                }
                                            }}
                                            className={`group relative flex items-center gap-3 px-4 py-3 rounded-xl border transition-all min-w-[180px] cursor-pointer select-none
                                                ${isSelected
                                                    ? 'bg-neutral-900 dark:bg-zinc-100 border-neutral-900 dark:border-zinc-100 shadow-lg shadow-neutral-200 dark:shadow-zinc-800'
                                                    : 'bg-white dark:bg-zinc-800 border-neutral-200 dark:border-zinc-700 hover:border-neutral-300 dark:hover:border-zinc-600 hover:shadow-sm'}`}
                                        >
                                            <div className={`w-2.5 h-2.5 rounded-full ring-2 transition-all 
                                                ${arm.calibrated
                                                    ? 'bg-green-500 ring-green-500/20'
                                                    : 'bg-amber-500 ring-amber-500/20'}`}
                                            />

                                            <div className="flex flex-col">
                                                <span className={`font-semibold text-sm transition-colors ${isSelected ? 'text-white dark:text-zinc-900' : 'text-neutral-700 dark:text-zinc-300'}`}>
                                                    {arm.name}
                                                </span>
                                                <span className={`text-[10px] uppercase tracking-wider font-medium transition-colors ${isSelected ? 'text-neutral-400 dark:text-zinc-500' : 'text-neutral-400 dark:text-zinc-500'}`}>
                                                    {arm.calibrated ? 'Ready' : 'Uncalibrated'}
                                                </span>
                                            </div>

                                            {/* Config Button (stopPropagation) */}
                                            <button
                                                onClick={(e) => { e.stopPropagation(); handleSettingsOpen(arm.id); }}
                                                className={`ml-auto p-1.5 rounded-lg transition-all
                                                    ${isSelected ? 'text-neutral-500 dark:text-zinc-600 hover:text-white dark:hover:text-zinc-900 hover:bg-white/10 dark:hover:bg-zinc-900/10' : 'text-neutral-300 dark:text-zinc-600 hover:text-blue-500 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-950'}`}
                                                title="Configure Motors"
                                            >
                                                <Settings className="w-4 h-4" />
                                            </button>

                                            {/* Selection Checkmark Indicator */}
                                            {isSelected && (
                                                <div className="absolute top-2 right-2 w-2 h-2 bg-blue-500 rounded-full animate-in zoom-in" />
                                            )}
                                        </div>
                                    );
                                })
                            ) : (
                                <div className="flex-1 flex items-center justify-between bg-amber-50 dark:bg-amber-950/50 border border-amber-100 dark:border-amber-900 rounded-xl px-4 py-2">
                                    <span className="text-amber-700 dark:text-amber-400 text-sm font-medium flex items-center gap-2">
                                        <AlertCircle className="w-4 h-4" /> No Robots Found
                                    </span>
                                    <button
                                        onClick={handleReset}
                                        className="px-3 py-1 bg-white dark:bg-zinc-800 border border-amber-200 dark:border-amber-700 text-amber-800 dark:text-amber-400 rounded-lg text-xs font-bold hover:bg-amber-100 dark:hover:bg-amber-950 transition-colors shadow-sm"
                                    >
                                        Reconnect Hardware
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* Active Pairings Display */}
                        {activePairings.length > 0 && (
                            <div className="mt-3 flex items-center justify-center gap-3 flex-wrap">
                                <span className="text-[10px] font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-widest">Active Pairings:</span>
                                {activePairings.map((p, i) => (
                                    <div key={i} className="flex items-center gap-1.5 px-2.5 py-1 bg-blue-50 dark:bg-blue-950/50 border border-blue-100 dark:border-blue-900 rounded-full">
                                        <span className="text-xs font-medium text-purple-600 dark:text-purple-400">
                                            {arms.find(a => a.id === p.leader_id)?.name || p.leader_id}
                                        </span>
                                        <Link2 className="w-3 h-3 text-blue-400" />
                                        <span className="text-xs font-medium text-blue-600 dark:text-blue-400">
                                            {arms.find(a => a.id === p.follower_id)?.name || p.follower_id}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Cameras Area */}
                    <div className="flex justify-center gap-4 flex-none flex-wrap">
                        {cameraConfigs.length === 0 && (
                            <div className="text-neutral-400 dark:text-zinc-500 text-sm p-4 border border-dashed border-neutral-300 dark:border-zinc-700 rounded-lg">
                                No cameras configured
                            </div>
                        )}
                        {cameraConfigs.map((cam) => (
                            <div key={cam.id} className="w-80 h-60 bg-neutral-100/50 dark:bg-zinc-800/50 rounded-2xl overflow-hidden relative border border-white/50 dark:border-zinc-700/50 shadow-inner group flex-none">
                                <img
                                    src={`${API_BASE}/video_feed/${cam.id}`}
                                    alt={cam.id}
                                    className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity"
                                    onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                                />
                                <div className="absolute bottom-3 left-3 text-white text-[10px] font-medium bg-black/40 backdrop-blur-md px-2.5 py-1 rounded-full border border-white/10">
                                    {cam.id}
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Graph Area */}
                    <div className="flex-1 min-h-[11rem] bg-white dark:bg-zinc-800 rounded-2xl border border-neutral-100 dark:border-zinc-700 p-4 shadow-sm relative">
                        <div className="absolute top-3 left-4 text-[10px] font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-widest z-10">Real-time Telemetry</div>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={graphData}>
                                <XAxis dataKey="time" hide />
                                <YAxis domain={['auto', 'auto']} hide />
                                <Tooltip
                                    contentStyle={{ backgroundColor: 'var(--glass-bg)', borderRadius: '16px', border: '1px solid var(--glass-border)', boxShadow: '0 10px 40px -10px rgba(0,0,0,0.1)', padding: '12px' }}
                                    itemStyle={{ fontSize: '11px', color: 'var(--foreground)', fontWeight: 500, opacity: 0.7 }}
                                    labelStyle={{ display: 'none' }}
                                />
                                {graphData.length > 0 && Object.keys(graphData[0]).filter(k => k !== 'time').map((key, index) => (
                                    <Line
                                        key={key}
                                        type="monotone"
                                        dataKey={key}
                                        stroke={colors[index % colors.length]}
                                        strokeWidth={2}
                                        dot={false}
                                        isAnimationActive={false}
                                        strokeOpacity={0.8}
                                    />
                                ))}
                            </LineChart>
                        </ResponsiveContainer>
                        {!isRunning && graphData.length === 0 && (
                            <div className="absolute inset-0 flex items-center justify-center">
                                <span className="text-neutral-300 dark:text-zinc-600 text-sm font-medium bg-neutral-50 dark:bg-zinc-900 px-4 py-2 rounded-full">Waiting for stream...</span>
                            </div>
                        )}
                    </div>

                    {/* Torque / Motor Status */}
                    {Object.keys(torqueData).length > 0 && (
                        <div className="flex flex-col gap-2">
                            <span className="text-[10px] font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-widest px-1">Motor Load</span>
                            <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
                                {Object.entries(torqueData).map(([key, val]: [string, any]) => {
                                    // Scale 0-1000
                                    // < 500: Green, 500-800: Yellow, > 800: Red
                                    const value = typeof val === 'number' ? val : 0;
                                    const percentage = Math.min((value / 1000) * 100, 100);

                                    let colorClass = "bg-green-500";
                                    if (value > 800) colorClass = "bg-red-500";
                                    else if (value > 500) colorClass = "bg-yellow-500";

                                    // Shorten key name if needed
                                    const label = key.replace(/link/g, 'L').replace(/_follower/g, ' (F)');

                                    return (
                                        <div key={key} className="bg-neutral-50 dark:bg-zinc-800 border border-neutral-100 dark:border-zinc-700 p-2 rounded-xl flex flex-col gap-1.5 shadow-sm">
                                            <div className="flex justify-between items-end">
                                                <span className="text-[10px] font-semibold text-neutral-600 dark:text-zinc-400 truncate" title={key}>{label}</span>
                                                <span className={`text-[10px] font-mono ${value > 500 ? 'text-neutral-900 dark:text-zinc-100 font-bold' : 'text-neutral-400 dark:text-zinc-500'}`}>{value}</span>
                                            </div>
                                            <div className="h-1.5 w-full bg-neutral-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                                                <div
                                                    className={`h-full rounded-full transition-all duration-300 ${colorClass}`}
                                                    style={{ width: `${percentage}%` }}
                                                />
                                            </div>
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    )}

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
