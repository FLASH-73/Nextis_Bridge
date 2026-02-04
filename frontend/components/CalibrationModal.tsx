import React, { useState, useEffect, useRef } from 'react';
import { X, Save, RotateCcw, Check, ArrowRight, AlertTriangle, Play, Pause, Settings } from 'lucide-react';

interface CalibrationModalProps {
    isOpen: boolean;
    onClose: () => void;
    language: 'en' | 'de';
}

interface ArmInfo {
    id: string;
    name: string;
    calibrated: boolean;
    type: 'leader' | 'follower';
    motor_type?: 'sts3215' | 'damiao';  // Motor type for special handling
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

type Step = 'SELECT_ARM' | 'PREPARE' | 'HOMING' | 'RANGES' | 'SAVE';

const TRANSLATIONS = {
    en: {
        title: "Calibration",
        select_title: "Select Component",
        select_desc: "Choose a unit to calibrate.",
        start: "Start",
        prepare_title: "Preparation",
        warning: "Caution",
        warning_text: "Robot may fall when torque is disabled.",
        prepare_steps: [
            "Hold the arm securely.",
            "Disable torque to move manually.",
            "Move to neutral position."
        ],
        disable_torque: "Unlock Motors",
        enable_torque: "Lock Motors",
        next: "Next",
        back: "Back",
        homing_title: "Zero Position",
        homing_desc: "Move robot to mechanical zero (upright/centered). This sets reference 2048.",
        set_home: "Set Zero",
        skip_home: "Skip",
        homing_success: "Zero Set (2048)",
        homing_fail: "Homing Failed",
        ranges_title: "Range Discovery",
        ranges_desc: "Move robot through full motion range manually.",
        start_recording: "Start Learning",
        stop_recording: "Stop Learning",
        recording: "Learning...",
        finish: "Review",
        save_title: "Finalize",
        save_desc: "Save configuration to robot memory.",
        ready_save: "Ready to Save",
        save_btn: "Save to Disk",
        saved: "Saved",
        failed_save: "Error",
        calibrated: "Ready",
        uncalibrated: "Needs Setup",
        leader: "Leader",
        follower: "Follower"
    },
    de: {
        title: "Kalibrierung",
        select_title: "Komponente",
        select_desc: "Einheit wählen.",
        start: "Start",
        prepare_title: "Vorbereitung",
        warning: "Vorsicht",
        warning_text: "Roboter kann fallen ohne Drehmoment.",
        prepare_steps: [
            "Arm festhalten.",
            "Motoren entsperren.",
            "Neutrale Position einnehmen."
        ],
        disable_torque: "Motoren Entsperren",
        enable_torque: "Motoren Sperren",
        next: "Weiter",
        back: "Zurück",
        homing_title: "Nullposition",
        homing_desc: "Roboter in Nullstellung bringen (aufrecht). Referenz 2048.",
        set_home: "Null Setzen",
        skip_home: "Überspringen",
        homing_success: "Null Gesetzt (2048)",
        homing_fail: "Fehler",
        ranges_title: "Bereichslernen",
        ranges_desc: "Roboter durch vollen Bereich bewegen.",
        start_recording: "Lernen Starten",
        stop_recording: "Lernen Stoppen",
        recording: "Lerne...",
        finish: "Prüfen",
        save_title: "Abschließen",
        save_desc: "Konfiguration speichern.",
        ready_save: "Bereit",
        save_btn: "Speichern",
        saved: "Gespeichert",
        failed_save: "Fehler",
        calibrated: "Bereit",
        uncalibrated: "Einrichtung nötig",
        leader: "Leader",
        follower: "Follower"
    }
};

export default function CalibrationModal({ isOpen, onClose, language }: CalibrationModalProps) {
    const t = TRANSLATIONS[language];
    const [activeStep, setActiveStep] = useState<Step>('SELECT_ARM');
    const [arms, setArms] = useState<ArmInfo[]>([]);
    const [selectedArm, setSelectedArm] = useState<string>('');
    const [motors, setMotors] = useState<MotorState[]>([]);
    const [isTorqueEnabled, setIsTorqueEnabled] = useState(true);
    const [statusMessage, setStatusMessage] = useState('');

    // Auto-Range State
    const [isRecording, setIsRecording] = useState(false);

    // Profile Management
    const [profiles, setProfiles] = useState<any[]>([]);
    const [saveName, setSaveName] = useState('');
    const [showProfiles, setShowProfiles] = useState(false);

    // Inversions State
    const [inversions, setInversions] = useState<Record<string, boolean>>({});

    // Fetch available arms on open
    useEffect(() => {
        if (isOpen) {
            fetchArms();
        }
    }, [isOpen]);

    const fetchArms = () => {
        fetch('http://127.0.0.1:8000/calibration/arms')
            .then(res => res.json())
            .then(data => {
                setArms(data.arms);
            })
            .catch(err => console.error("Failed to fetch arms", err));
    };

    // Clear profiles when switching arms or closing profile view
    useEffect(() => {
        if (!showProfiles) {
            setProfiles([]);
        }
    }, [showProfiles, selectedArm]);

    const fetchProfiles = (armId: string) => {
        fetch(`http://127.0.0.1:8000/calibration/${armId}/files`)
            .then(res => res.json())
            .then(data => setProfiles(data.files))
            .catch(err => console.error(err));
    };

    const loadProfile = async (filename: string) => {
        try {
            const res = await fetch(`http://127.0.0.1:8000/calibration/${selectedArm}/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename })
            });
            if (res.ok) {
                // Refresh arms status
                fetchArms();
                setShowProfiles(false);
            }
        } catch (err) {
            console.error(err);
        }
    };

    const deleteProfile = async (filename: string) => {
        try {
            await fetch(`http://127.0.0.1:8000/calibration/${selectedArm}/delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename })
            });
            fetchProfiles(selectedArm);
        } catch (err) {
            console.error(err);
        }
    };

    const fetchInversions = (armId: string) => {
        fetch(`http://127.0.0.1:8000/calibration/${armId}/inversions`)
            .then(res => res.json())
            .then(data => setInversions(data.inversions))
            .catch(err => console.error("Failed to fetch inversions", err));
    };

    const toggleInversion = async (motorName: string) => {
        const current = inversions[motorName] || false;
        try {
            await fetch(`http://127.0.0.1:8000/calibration/${selectedArm}/inversions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ motor: motorName, inverted: !current })
            });
            // Optimistic update
            setInversions(prev => ({ ...prev, [motorName]: !current }));
        } catch (err) {
            console.error("Failed to set inversion", err);
        }
    };

    // Poll motor state AND fetch inversions when in RANGES step
    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isOpen && activeStep === 'RANGES' && selectedArm) {
            fetchInversions(selectedArm); // Fetch once on entry

            interval = setInterval(() => {
                fetch(`http://127.0.0.1:8000/calibration/${selectedArm}/state`)
                    .then(res => res.json())
                    .then(data => {
                        setMotors(data.state);
                    })
                    .catch(err => console.error("Failed to fetch state", err));
            }, 100);
        }
        return () => clearInterval(interval);
    }, [isOpen, activeStep, selectedArm]);

    const toggleTorque = async () => {
        const shouldEnable = !isTorqueEnabled;
        try {
            await fetch(`http://127.0.0.1:8000/calibration/${selectedArm}/torque`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enable: shouldEnable })
            });
            setIsTorqueEnabled(shouldEnable);
        } catch (err) {
            console.error("Failed to toggle torque", err);
        }
    };

    const performHoming = async () => {
        try {
            const res = await fetch(`http://127.0.0.1:8000/calibration/${selectedArm}/homing`, { method: 'POST' });
            const data = await res.json();

            if (res.ok && data.status === 'success') {
                setStatusMessage(t.homing_success);
                // Show warning if present
                if (data.message && data.message.includes("WARNING")) {
                    alert(data.message);
                }
                setTimeout(() => {
                    setActiveStep('RANGES');
                    setStatusMessage('');
                }, 1500);
            } else {
                setStatusMessage(t.homing_fail);
                alert(data.message || "Homing Failed");
            }
        } catch (err) {
            setStatusMessage("Error during homing.");
        }
    };

    const startRecording = async () => {
        try {
            await fetch(`http://127.0.0.1:8000/calibration/${selectedArm}/discovery/start`, { method: 'POST' });
            setIsRecording(true);
        } catch (e) {
            console.error(e);
        }
    };

    const stopRecording = async () => {
        try {
            const res = await fetch(`http://127.0.0.1:8000/calibration/${selectedArm}/discovery/stop`, { method: 'POST' });
            const data = await res.json();
            setIsRecording(false);

            if (data.warnings && data.warnings.length > 0) {
                alert(`Warning: ${data.message}\nReview calibration carefully.`);
            } else if (data.message) {
                // Optional success toast?
                console.log(data.message);
            }
        } catch (e) {
            console.error(e);
        }
    };

    const saveCalibration = async () => {
        try {
            // Use the named save endpoint if a name is provided
            const endpoint = saveName
                ? `http://127.0.0.1:8000/calibration/${selectedArm}/save_named`
                : `http://127.0.0.1:8000/calibration/${selectedArm}/save`;

            await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: saveName })
            });

            setStatusMessage(t.saved);
            fetchArms();
            setTimeout(() => {
                setActiveStep('SELECT_ARM');
                setSelectedArm('');
                setStatusMessage('');
                setSaveName('');
            }, 2000);
        } catch (err) {
            setStatusMessage(t.failed_save);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 flex items-center justify-center z-[60]">
            <div className="absolute inset-0 bg-white/30 dark:bg-black/30 backdrop-blur-sm" onClick={onClose} />
            <div className="bg-white/90 dark:bg-zinc-900/90 backdrop-blur-2xl border border-white/60 dark:border-zinc-700/60 rounded-3xl w-[900px] h-[700px] flex flex-col shadow-2xl overflow-hidden relative animate-in zoom-in-95 duration-200">

                {/* Glass Header */}
                <div className="flex items-center justify-between px-8 py-6 border-b border-black/5 dark:border-white/5 bg-white/40 dark:bg-zinc-800/40">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-black/5 dark:bg-white/5 rounded-lg">
                            <Settings className="w-5 h-5 text-black/70 dark:text-white/70" />
                        </div>
                        <h2 className="text-xl font-light tracking-tight text-black dark:text-white">{t.title}</h2>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-black/5 dark:hover:bg-white/5 rounded-full text-neutral-400 hover:text-black dark:hover:text-white transition-colors">
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Progress Indicators */}
                {activeStep !== 'SELECT_ARM' && !showProfiles && (
                    <div className="flex justify-center py-6 border-b border-black/5 dark:border-white/5 bg-white/20 dark:bg-zinc-800/20">
                        <div className="flex gap-2">
                            {['Select', 'Prepare', 'Home', 'Range', 'Save'].map((s, i) => {
                                const idx = ['SELECT_ARM', 'PREPARE', 'HOMING', 'RANGES', 'SAVE'].indexOf(activeStep);
                                return (
                                    <div key={s} className={`h-1 rounded-full transition-all duration-300 ${i <= idx ? 'w-8 bg-black dark:bg-white' : 'w-2 bg-black/10 dark:bg-white/10'}`} />
                                )
                            })}
                        </div>
                    </div>
                )}

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

                {/* Content */}
                <div className="flex-1 p-8 overflow-y-auto bg-gradient-to-b from-white/40 dark:from-zinc-800/40 to-white/10 dark:to-zinc-900/10">
                    <div className="max-w-3xl mx-auto h-full flex flex-col">

                        {activeStep === 'SELECT_ARM' && !showProfiles && (
                            <div className="space-y-6 my-auto">
                                <div className="text-center mb-8">
                                    <h3 className="text-2xl font-light text-black dark:text-white mb-1">{t.select_title}</h3>
                                    <p className="text-neutral-500 dark:text-zinc-400">{t.select_desc}</p>
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    {arms.map(arm => (
                                        <button
                                            key={arm.id}
                                            onClick={() => setSelectedArm(arm.id)}
                                            className={`p-6 rounded-2xl border transition-all flex flex-col gap-2 relative group text-left
                                                ${selectedArm === arm.id
                                                    ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-950/50 shadow-blue-100 dark:shadow-blue-900/20 ring-1 ring-blue-500'
                                                    : arm.motor_type === 'damiao'
                                                        ? 'border-orange-200 dark:border-orange-800/60 bg-orange-50/60 dark:bg-orange-950/30 hover:bg-orange-50 dark:hover:bg-orange-950/50 hover:scale-[1.02] shadow-sm'
                                                        : 'border-white/60 dark:border-zinc-700/60 bg-white/60 dark:bg-zinc-800/60 hover:bg-white dark:hover:bg-zinc-800 hover:scale-[1.02] shadow-sm'}`}
                                        >
                                            <div className="flex justify-between items-start w-full">
                                                <span className="text-lg font-medium text-black dark:text-white capitalize">{arm.name}</span>
                                                {arm.calibrated ? (
                                                    <div className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.4)]" />
                                                ) : (
                                                    <div className="w-2 h-2 rounded-full bg-orange-400" />
                                                )}
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <span className="text-xs text-neutral-400 dark:text-zinc-500">{arm.type === 'leader' ? t.leader : t.follower}</span>
                                                {arm.motor_type === 'damiao' && (
                                                    <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-orange-100 dark:bg-orange-900/50 text-orange-600 dark:text-orange-400 rounded uppercase">
                                                        High Torque
                                                    </span>
                                                )}
                                            </div>
                                        </button>
                                    ))}
                                </div>
                                <div className="flex justify-center pt-8 gap-3">
                                    <button
                                        onClick={() => {
                                            fetchProfiles(selectedArm);
                                            setShowProfiles(true);
                                        }}
                                        disabled={!selectedArm}
                                        className="px-6 py-3 bg-white dark:bg-zinc-800 text-black dark:text-white border border-neutral-200 dark:border-zinc-700 rounded-full font-medium shadow-sm hover:bg-neutral-50 dark:hover:bg-zinc-700 transition-all disabled:opacity-30"
                                    >
                                        Manage Profiles
                                    </button>

                                    <button
                                        onClick={() => setActiveStep('PREPARE')}
                                        disabled={!selectedArm}
                                        className="px-8 py-3 bg-black dark:bg-white text-white dark:text-black rounded-full font-medium shadow-lg hover:scale-105 active:scale-95 transition-all disabled:opacity-30 disabled:hover:scale-100 flex items-center gap-2"
                                    >
                                        {t.start} <ArrowRight className="w-4 h-4 ml-1" />
                                    </button>
                                </div>
                            </div>
                        )}

                        {/* ... PREPARE, HOMING, RANGES STEPS ... */}

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

                    </div>
                </div>
            </div>
        </div >
    );
}
