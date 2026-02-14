import React from 'react';
import { X, Save, RotateCcw, Check, ArrowRight, AlertTriangle, Play, Pause } from 'lucide-react';

type Step = 'SELECT_ARM' | 'PREPARE' | 'HOMING' | 'RANGES' | 'SAVE';

interface ArmInfo {
    id: string;
    name: string;
    calibrated: boolean;
    type: 'leader' | 'follower';
    motor_type?: 'sts3215' | 'damiao';
}

interface MotorState {
    name: string;
    id: number;
    min: number;
    max: number;
    pos: number;
    visited_min?: number;
    visited_max?: number;
}

interface CalibrationPanelProps {
    activeStep: Step;
    setActiveStep: (step: Step) => void;
    selectedArm: string;
    arms: ArmInfo[];
    motors: MotorState[];
    isTorqueEnabled: boolean;
    statusMessage: string;
    isRecording: boolean;
    inversions: Record<string, boolean>;
    showProfiles: boolean;
    setShowProfiles: (show: boolean) => void;
    profiles: any[];
    saveName: string;
    setSaveName: (name: string) => void;
    toggleTorque: () => Promise<void>;
    performHoming: () => Promise<void>;
    startRecording: () => Promise<void>;
    stopRecording: () => Promise<void>;
    saveCalibration: () => Promise<void>;
    toggleInversion: (motorName: string) => Promise<void>;
    loadProfile: (filename: string) => Promise<void>;
    deleteProfile: (filename: string) => Promise<void>;
    t: Record<string, any>;
}

export default function CalibrationPanel({
    activeStep,
    setActiveStep,
    selectedArm,
    arms,
    motors,
    isTorqueEnabled,
    statusMessage,
    isRecording,
    inversions,
    showProfiles,
    setShowProfiles,
    profiles,
    saveName,
    setSaveName,
    toggleTorque,
    performHoming,
    startRecording,
    stopRecording,
    saveCalibration,
    toggleInversion,
    loadProfile,
    deleteProfile,
    t,
}: CalibrationPanelProps) {
    return (
        <>
            {/* Profile Manager Overlay */}
            {showProfiles && (
                <div className="absolute inset-0 z-50 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-md flex flex-col p-8 animate-in fade-in duration-200">
                    <div className="flex items-center justify-between mb-8">
                        <div>
                            <h3 className="text-2xl font-light text-black dark:text-white">Manage Profiles</h3>
                            <p className="text-neutral-500 dark:text-zinc-400 text-sm mt-1">Saved calibrations for {arms.find(a => a.id === selectedArm)?.name}</p>
                        </div>
                        <button onClick={() => setShowProfiles(false)} className="px-4 py-2 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-full text-sm text-black dark:text-white hover:bg-neutral-50 dark:hover:bg-zinc-700 shadow-sm">
                            Close
                        </button>
                    </div>

                    <div className="flex-1 overflow-y-auto space-y-3 pr-2">
                        {profiles.length === 0 ? (
                            <div className="text-center py-12 text-neutral-400 dark:text-zinc-500">No saved profiles found.</div>
                        ) : (
                            profiles.map(p => (
                                <div
                                    key={p.name}
                                    className={`flex items-center justify-between p-4 rounded-xl border shadow-sm transition-all group relative overflow-hidden
                                        ${p.active
                                            ? 'bg-green-50/60 dark:bg-green-950/60 border-green-500/50 shadow-green-100 dark:shadow-green-900/20 ring-1 ring-green-500/50'
                                            : 'bg-white dark:bg-zinc-800 border-neutral-100 dark:border-zinc-700 hover:shadow-md'
                                        }`}
                                >
                                    <div>
                                        <div className="flex items-center gap-2">
                                            <div className="font-medium text-neutral-800 dark:text-zinc-200">{p.name}</div>
                                            {p.active && (
                                                <span className="px-2 py-0.5 rounded-full bg-green-200 dark:bg-green-900 text-green-700 dark:text-green-300 text-[10px] uppercase font-bold tracking-wider">
                                                    Active
                                                </span>
                                            )}
                                        </div>
                                        <div className="text-xs text-neutral-400 dark:text-zinc-500 mt-0.5">{p.created}</div>
                                    </div>
                                    <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity z-10">
                                        <button
                                            onClick={() => loadProfile(p.name)}
                                            className="px-3 py-1.5 bg-black dark:bg-white text-white dark:text-black text-xs rounded-lg hover:bg-neutral-800 dark:hover:bg-zinc-200"
                                            disabled={p.active}
                                        >
                                            {p.active ? 'Loaded' : 'Load'}
                                        </button>
                                        <button
                                            onClick={() => deleteProfile(p.name)}
                                            className="p-2 text-red-500 hover:bg-red-50 dark:hover:bg-red-950 rounded-lg"
                                            title="Delete Profile"
                                        >
                                            <X className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>

                    <div className="pt-6 border-t border-black/5 dark:border-white/5 mt-auto flex justify-center">
                        <button
                            onClick={() => { setShowProfiles(false); setActiveStep('PREPARE'); }}
                            className="px-6 py-2.5 bg-blue-600 text-white rounded-full font-medium shadow-lg shadow-blue-200 dark:shadow-blue-900/30 hover:bg-blue-700 hover:scale-105 transition-all text-sm flex items-center gap-2"
                        >
                            <Play className="w-4 h-4 fill-current" /> Start New Calibration
                        </button>
                    </div>
                </div>
            )}

            {/* PREPARE Step */}
            {activeStep === 'PREPARE' && !showProfiles && (
                <div className="space-y-8 my-auto text-center">
                    <h3 className="text-2xl font-light text-black dark:text-white">{t.prepare_title}</h3>
                    <div className="p-4 bg-orange-50 dark:bg-orange-950/50 border border-orange-100 dark:border-orange-900/50 rounded-2xl text-left inline-block max-w-md">
                        <div className="flex gap-3 text-orange-600 dark:text-orange-400 font-medium text-sm items-center mb-2">
                            <AlertTriangle className="w-4 h-4" /> {t.warning}
                        </div>
                        <p className="text-orange-900/60 dark:text-orange-300/60 text-xs leading-relaxed">{t.warning_text}</p>
                    </div>
                    {/* Damiao-specific high-torque warning */}
                    {arms.find(a => a.id === selectedArm)?.motor_type === 'damiao' && (
                        <div className="p-4 bg-red-50 dark:bg-red-950/50 border border-red-200 dark:border-red-900/50 rounded-2xl text-left inline-block max-w-md">
                            <div className="flex gap-3 text-red-600 dark:text-red-400 font-medium text-sm items-center mb-2">
                                <AlertTriangle className="w-4 h-4" /> High-Torque Motor Warning
                            </div>
                            <p className="text-red-900/70 dark:text-red-300/70 text-xs leading-relaxed">
                                This is a Damiao high-torque arm. <strong>Velocity is limited to 10% during calibration</strong> for safety.
                                Support the arm firmly before disabling torque - these motors are powerful and the arm is heavy.
                            </p>
                        </div>
                    )}
                    <div className="space-y-4 max-w-xs mx-auto">
                        <button
                            onClick={toggleTorque}
                            className={`w-full py-4 rounded-xl font-medium transition-all border flex items-center justify-center gap-2 ${!isTorqueEnabled
                                ? 'bg-red-50 dark:bg-red-950/50 border-red-100 dark:border-red-900/50 text-red-600 dark:text-red-400'
                                : 'bg-white dark:bg-zinc-800 border-neutral-200 dark:border-zinc-700 text-neutral-600 dark:text-zinc-300 hover:border-black/20 dark:hover:border-white/20'
                                }`}
                        >
                            {isTorqueEnabled ? t.disable_torque : t.enable_torque}
                        </button>
                    </div>

                    <div className="flex justify-between max-w-md mx-auto w-full pt-4">
                        <button onClick={() => setActiveStep('SELECT_ARM')} className="text-black/40 dark:text-white/40 hover:text-black dark:hover:text-white">
                            {t.back}
                        </button>
                        <button onClick={() => setActiveStep('HOMING')} className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 font-medium">
                            {t.next}
                        </button>
                    </div>
                </div>
            )}

            {/* HOMING Step */}
            {activeStep === 'HOMING' && !showProfiles && (
                <div className="space-y-8 my-auto text-center">
                    <h3 className="text-2xl font-light text-black dark:text-white">{t.homing_title}</h3>
                    <p className="text-neutral-500 dark:text-zinc-400 max-w-sm mx-auto">{t.homing_desc}</p>

                    <button
                        onClick={performHoming}
                        className="px-10 py-5 bg-black dark:bg-white text-white dark:text-black rounded-2xl shadow-xl hover:scale-105 transition-all active:scale-95 flex items-center gap-3 mx-auto"
                    >
                        <RotateCcw className="w-5 h-5" />
                        {t.set_home}
                    </button>
                    {statusMessage && <p className="text-green-600 dark:text-green-400 font-medium animate-in fade-in">{statusMessage}</p>}

                    <div className="flex justify-between max-w-md mx-auto w-full pt-4">
                        <button onClick={() => setActiveStep('PREPARE')} className="text-black/40 dark:text-white/40 hover:text-black dark:hover:text-white">
                            {t.back}
                        </button>
                        <button onClick={() => setActiveStep('RANGES')} className="text-neutral-400 dark:text-zinc-500 hover:text-black dark:hover:text-white">
                            {t.skip_home}
                        </button>
                    </div>
                </div>
            )}

            {/* RANGES Step */}
            {activeStep === 'RANGES' && !showProfiles && (
                <div className="space-y-6 h-full flex flex-col">
                    <div className="flex justify-between items-center px-2">
                        <h3 className="text-xl font-light text-black dark:text-white">{t.ranges_title}</h3>
                        {!isRecording ? (
                            <button onClick={startRecording} className="px-6 py-2 bg-black dark:bg-white text-white dark:text-black rounded-full text-sm font-medium shadow-md hover:scale-105 transition-all flex items-center gap-2">
                                <Play className="w-4 h-4 fill-current" /> {t.start_recording}
                            </button>
                        ) : (
                            <button onClick={stopRecording} className="px-6 py-2 bg-red-500 text-white rounded-full text-sm font-medium shadow-md hover:bg-red-600 transition-colors flex items-center gap-2 animate-pulse">
                                <Pause className="w-4 h-4 fill-current" /> {t.stop_recording}
                            </button>
                        )}
                    </div>

                    <div className="flex-1 overflow-y-auto">
                        {/* Table Header */}
                        <div className="grid grid-cols-[1fr,1fr,1fr,1fr,auto] gap-4 mb-2 px-4 text-xs font-medium text-neutral-400 dark:text-zinc-500 uppercase tracking-wide">
                            <div className="text-left">Joint</div>
                            <div className="text-center">Min (Reached)</div>
                            <div className="text-center">Current</div>
                            <div className="text-center">Max (Reached)</div>
                            <div className="text-center">Reverse</div>
                        </div>

                        <div className="space-y-2">
                            {motors.map((motor) => (
                                <div key={motor.name} className="grid grid-cols-[1fr,1fr,1fr,1fr,auto] gap-4 items-center bg-white/60 dark:bg-zinc-800/60 p-4 rounded-xl border border-white/60 dark:border-zinc-700/60 shadow-sm transition-all hover:bg-white/80 dark:hover:bg-zinc-800/80">
                                    <div className="font-medium text-neutral-700 dark:text-zinc-300 text-sm truncate">{motor.name}</div>

                                    <div className="text-center font-mono text-sm text-neutral-500 dark:text-zinc-400">
                                        {motor.visited_min !== undefined ? motor.visited_min : '-'}
                                    </div>

                                    <div className="text-center font-mono text-base font-semibold text-black dark:text-white bg-black/5 dark:bg-white/5 rounded-md py-1">
                                        {motor.pos}
                                    </div>

                                    <div className="text-center font-mono text-sm text-neutral-500 dark:text-zinc-400">
                                        {motor.visited_max !== undefined ? motor.visited_max : '-'}
                                    </div>

                                    <div className="flex justify-center">
                                        <label className="relative inline-flex items-center cursor-pointer" title="Invert Motor Direction">
                                            <input
                                                type="checkbox"
                                                className="sr-only peer"
                                                checked={inversions[motor.name] || false}
                                                onChange={() => toggleInversion(motor.name)}
                                            />
                                            <div className="w-9 h-5 bg-gray-200 dark:bg-zinc-700 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-500 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 dark:after:border-zinc-600 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600"></div>
                                        </label>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="flex justify-between pt-4 mt-auto">
                        <button onClick={() => setActiveStep('HOMING')} className="text-black/40 dark:text-white/40 hover:text-black dark:hover:text-white">
                            {t.back}
                        </button>
                        <button
                            onClick={() => {
                                if (isRecording) stopRecording();
                                setActiveStep('SAVE');
                            }}
                            className="px-6 py-2 bg-neutral-900 dark:bg-zinc-100 text-white dark:text-black rounded-lg text-sm font-medium hover:bg-black dark:hover:bg-white transition-colors"
                        >
                            {t.finish}
                        </button>
                    </div>
                </div>
            )}

            {/* SAVE Step */}
            {activeStep === 'SAVE' && !showProfiles && (
                <div className="space-y-8 my-auto text-center">
                    <h3 className="text-2xl font-light text-black dark:text-white">{t.save_title}</h3>

                    <div className="p-8 bg-white dark:bg-zinc-800 border border-neutral-100 dark:border-zinc-700 rounded-3xl shadow-sm inline-block w-full max-w-sm">
                        <div className="w-16 h-16 bg-green-50 dark:bg-green-950/50 rounded-full flex items-center justify-center mx-auto mb-4 text-green-500 dark:text-green-400">
                            <Check className="w-8 h-8" />
                        </div>
                        <p className="text-neutral-500 dark:text-zinc-400 mb-6">{t.ready_save}</p>

                        <div className="mb-2 text-left">
                            <label className="text-xs font-semibold text-neutral-400 dark:text-zinc-500 uppercase tracking-wide ml-1">Profile Name (Optional)</label>
                            <input
                                type="text"
                                value={saveName}
                                onChange={(e) => setSaveName(e.target.value)}
                                placeholder="e.g. standard_calibration"
                                className="w-full mt-1 px-4 py-3 bg-neutral-50 dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-black/5 dark:focus:ring-white/5 transition-all placeholder:text-neutral-300 dark:placeholder:text-zinc-600 font-medium text-black dark:text-white"
                            />
                        </div>
                    </div>

                    <button
                        onClick={saveCalibration}
                        className="w-full max-w-xs mx-auto py-4 bg-black dark:bg-white text-white dark:text-black rounded-xl shadow-lg hover:scale-[1.02] transition-all flex items-center justify-center gap-2"
                    >
                        <Save className="w-4 h-4" /> {t.save_btn}
                    </button>
                    {statusMessage && <p className="text-neutral-500 dark:text-zinc-400 text-sm">{statusMessage}</p>}

                    <button onClick={() => setActiveStep('RANGES')} className="text-black/40 dark:text-white/40 hover:text-black dark:hover:text-white block mx-auto mt-4">
                        {t.back}
                    </button>
                </div>
            )}
        </>
    );
}
