import React, { useState, useEffect } from 'react';
import { X, Activity } from 'lucide-react';
import GravityWizard from '../../GravityWizard';
import { teleopApi, calibrationApi } from '../../../lib/api';

interface TuningState {
    k_gravity: number;
    k_assist: number;
    k_haptic: number;
    v_threshold: number;
    k_damping: number;
}

interface SettingsOverlayProps {
    activeArmId: string | null;
    wizardArm: string | null;
    setWizardArm: React.Dispatch<React.SetStateAction<string | null>>;
    inversions: { [key: string]: boolean };
    setInversions: React.Dispatch<React.SetStateAction<{ [key: string]: boolean }>>;
    motors: string[];
    setMotors: React.Dispatch<React.SetStateAction<string[]>>;
    assistEnabled: boolean;
    setAssistEnabled: React.Dispatch<React.SetStateAction<boolean>>;
    tuning: TuningState;
    setTuning: React.Dispatch<React.SetStateAction<TuningState>>;
    onClose: () => void;
    onRefresh: (armId: string) => void;
}

export default function SettingsOverlay({
    activeArmId,
    wizardArm,
    setWizardArm,
    inversions,
    setInversions,
    motors,
    setMotors,
    assistEnabled,
    setAssistEnabled,
    tuning,
    setTuning,
    onClose,
    onRefresh,
}: SettingsOverlayProps) {
    const [profiles, setProfiles] = useState<any[]>([]);

    // Fetch profiles when activeArmId changes
    useEffect(() => {
        if (activeArmId) {
            calibrationApi.files(activeArmId)
                .then(data => setProfiles(data.files || []))
                .catch(console.error);
        }
    }, [activeArmId]);

    // Debounced Update
    useEffect(() => {
        const timer = setTimeout(() => {
            teleopApi.tune({ ...tuning }).catch(console.error);
        }, 300);
        return () => clearTimeout(timer);
    }, [tuning]);

    const updateTuning = (newValues: any) => {
        setTuning(prev => ({ ...prev, ...newValues }));
    };

    const loadProfile = async (filename: string) => {
        if (!activeArmId) return;
        try {
            await calibrationApi.load(activeArmId, filename);
            // Refresh list to update 'active' tag
            onRefresh(activeArmId);
        } catch (e) {
            console.error("Failed to load profile", e);
        }
    };

    const toggleInversion = async (motor: string) => {
        if (!activeArmId) return;
        const newState = !inversions[motor];
        setInversions(prev => ({ ...prev, [motor]: newState }));

        // Save immediately
        try {
            await calibrationApi.setInversions(activeArmId, { motor, inverted: newState } as any);
        } catch (e) {
            console.error("Failed to save inversion", e);
        }
    };

    const toggleAssist = async () => {
        const newState = !assistEnabled;
        setAssistEnabled(newState);
        try {
            await teleopApi.setAssist({ enabled: newState });
        } catch (e) {
            console.error(e);
        }
    };

    return (
        <div className="absolute inset-0 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-md z-40 flex flex-col animate-in fade-in slide-in-from-bottom-4 p-8">
            <div className="flex justify-between items-center mb-8">
                <div>
                    <h2 className="text-2xl font-bold text-neutral-900 dark:text-zinc-100">Motor Configuration</h2>
                    <p className="text-neutral-500 dark:text-zinc-400 text-sm">Customize direciton logic for {activeArmId}</p>
                </div>
                <button onClick={onClose} className="p-2 hover:bg-neutral-100 dark:hover:bg-zinc-800 rounded-full transition-colors"><X className="w-6 h-6 text-neutral-500 dark:text-zinc-400" /></button>
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
                                    await Promise.all(pairs.map(id => calibrationApi.setZero(id)));
                                    alert("Step 1 Done: Zero Pose Captured. Now move both arms to a new position (e.g. 45\u00b0 forward) and click 'Align'.");
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
                                    const data = await calibrationApi.autoAlign(activeArmId) as any;
                                    if (data.status === 'success') {
                                        alert(`Success! Inverted ${data.inverted_count} motors.\nChanges: ${JSON.stringify(data.changes, null, 2)}`);
                                        onRefresh(activeArmId); // Refresh
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
    );
}
