import React, { useState, useEffect } from 'react';
import { X, ArrowRight, Settings } from 'lucide-react';
import CalibrationPanel from './CalibrationPanel';
import { calibrationApi, api } from '../../../lib/api';

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
        calibrationApi.arms()
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
        calibrationApi.files(armId)
            .then(data => setProfiles(data.files))
            .catch(err => console.error(err));
    };

    const loadProfile = async (filename: string) => {
        try {
            await calibrationApi.load(selectedArm, filename);
            fetchArms();
            setShowProfiles(false);
        } catch (err) {
            console.error(err);
        }
    };

    const deleteProfile = async (filename: string) => {
        try {
            await calibrationApi.deleteFile(selectedArm, filename);
            fetchProfiles(selectedArm);
        } catch (err) {
            console.error(err);
        }
    };

    const fetchInversions = (armId: string) => {
        calibrationApi.inversions(armId)
            .then((data: any) => setInversions(data.inversions))
            .catch(err => console.error("Failed to fetch inversions", err));
    };

    const toggleInversion = async (motorName: string) => {
        const current = inversions[motorName] || false;
        try {
            await api.post(`/calibration/${selectedArm}/inversions`, { motor: motorName, inverted: !current });
            setInversions(prev => ({ ...prev, [motorName]: !current }));
        } catch (err) {
            console.error("Failed to set inversion", err);
        }
    };

    // Poll motor state AND fetch inversions when in RANGES step
    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isOpen && activeStep === 'RANGES' && selectedArm) {
            fetchInversions(selectedArm);

            interval = setInterval(() => {
                calibrationApi.state(selectedArm)
                    .then((data: any) => {
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
            await calibrationApi.torque(selectedArm, shouldEnable);
            setIsTorqueEnabled(shouldEnable);
        } catch (err) {
            console.error("Failed to toggle torque", err);
        }
    };

    const performHoming = async () => {
        try {
            const data: any = await calibrationApi.homing(selectedArm);

            if (data.status === 'success') {
                setStatusMessage(t.homing_success);
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
            await calibrationApi.startDiscovery(selectedArm);
            setIsRecording(true);
        } catch (e: any) {
            console.error("Failed to start discovery:", e);
            alert(`Failed to start recording: ${e.message || 'Unknown error'}`);
        }
    };

    const stopRecording = async () => {
        try {
            const data: any = await calibrationApi.stopDiscovery(selectedArm);
            setIsRecording(false);

            if (data?.warnings && data.warnings.length > 0) {
                alert(`Warning: ${data.message}\nReview calibration carefully.`);
            } else if (data?.message) {
                console.log(data.message);
            }
        } catch (e) {
            console.error("Failed to stop discovery:", e);
            setIsRecording(false);
        }
    };

    const saveCalibration = async () => {
        try {
            if (saveName) {
                await calibrationApi.saveNamed(selectedArm, saveName);
            } else {
                await calibrationApi.save(selectedArm);
            }

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

                        {/* PREPARE, HOMING, RANGES, SAVE steps + Profile Overlay */}
                        <CalibrationPanel
                            activeStep={activeStep}
                            setActiveStep={setActiveStep}
                            selectedArm={selectedArm}
                            arms={arms}
                            motors={motors}
                            isTorqueEnabled={isTorqueEnabled}
                            statusMessage={statusMessage}
                            isRecording={isRecording}
                            inversions={inversions}
                            showProfiles={showProfiles}
                            setShowProfiles={setShowProfiles}
                            profiles={profiles}
                            saveName={saveName}
                            setSaveName={setSaveName}
                            toggleTorque={toggleTorque}
                            performHoming={performHoming}
                            startRecording={startRecording}
                            stopRecording={stopRecording}
                            saveCalibration={saveCalibration}
                            toggleInversion={toggleInversion}
                            loadProfile={loadProfile}
                            deleteProfile={deleteProfile}
                            t={t}
                        />

                    </div>
                </div>
            </div>
        </div >
    );
}
