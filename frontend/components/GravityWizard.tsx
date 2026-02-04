import React, { useState } from 'react';
import { ArrowLeft, ArrowRight, CheckCircle, RotateCcw, Weight, Activity, Lock, Unlock, Play } from 'lucide-react';

interface GravityWizardProps {
    armKey: string;
    armId: string;
    onClose: () => void;
    apiBase: string;
}

export default function GravityWizard({ armKey, armId, onClose, apiBase }: GravityWizardProps) {
    const [step, setStep] = useState<'INTRO' | 'CAPTURE' | 'COMPUTING' | 'DONE'>('INTRO');
    const [samples, setSamples] = useState(0);
    const [isCapturing, setIsCapturing] = useState(false);
    const [error, setError] = useState('');
    const [isLocked, setIsLocked] = useState(true); // Torque State

    const POSES = [
        { title: "The Resting", desc: "Fold the arm completely compact (Link 2 on Link 1), as if it's sleeping." },
        { title: "The Soldier", desc: "Point the arm straight up vertically towards the ceiling." },
        { title: "The Forward Reach", desc: "Extend the arm horizontally straight forward (90°)." },
        { title: "The High Reach", desc: "Reach forward and up at a 45 degree angle." },
        { title: "The Low Reach", desc: "Reach forward and down towards the table." },
        { title: "The Left Twist", desc: "Rotate the base 90° LEFT and reach forward." },
        { title: "The Right Twist", desc: "Rotate the base 90° RIGHT and reach forward." },
        { title: "The Elbow Up", desc: "Reach forward but keep the elbow pointing skyward (90° bend)." },
        { title: "The Wrist Flex", desc: "Any position, but rotate the wrist joints significantly." },
        { title: "The Freestyle", desc: "Move to any random comfortable position you haven't used." },
    ];

    // Ensure we don't overflow poses if user does more samples
    const currentPose = POSES[samples] || { title: "Extra Sample", desc: "Any new pose." };

    // Toggle Torque Helper
    const setTorque = async (enable: boolean) => {
        try {
            await fetch(`${apiBase}/calibration/${armId}/torque`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enable })
            });
            setIsLocked(enable);
        } catch (e) {
            console.error("Torque toggle failed", e);
        }
    };

    const startCalibration = async () => {
        try {
            // Use armId (full unique ID) instead of armKey
            const res = await fetch(`${apiBase}/calibration/${armId}/gravity/start`, { method: 'POST' });
            const data = await res.json();
            if (data.status === 'success') {
                setStep('CAPTURE');
                setSamples(0);
                // Auto-unlock for first pose
                await setTorque(false);
            } else {
                setError(data.message);
            }
        } catch (e: any) {
            setError(e.message);
        }
    };

    const captureSample = async () => {
        if (!isLocked && !confirm("Torque is OFF. The arm might fall if you let go. Capture anyway?")) return;

        setIsCapturing(true);
        try {
            const res = await fetch(`${apiBase}/calibration/${armId}/gravity/sample`, { method: 'POST' });
            const data = await res.json();
            if (data.status === 'success') {
                setSamples(data.samples);
                // Auto-unlock for next pose
                await setTorque(false);
            } else {
                setError(data.message);
            }
        } catch (e: any) {
            setError(e.message);
        } finally {
            setIsCapturing(false);
        }
    };

    const finishCalibration = async () => {
        setStep('COMPUTING');
        try {
            const res = await fetch(`${apiBase}/calibration/${armId}/gravity/compute`, { method: 'POST' });
            const data = await res.json();
            if (data.status === 'success') {
                setStep('DONE');
                await setTorque(true); // Re-lock when done
            } else {
                setStep('CAPTURE');
                setError(data.message);
            }
        } catch (e: any) {
            setStep('CAPTURE');
            setError(e.message);
        }
    };

    return (
        <div className="absolute inset-0 bg-white dark:bg-zinc-900 z-50 flex flex-col animate-in fade-in zoom-in-95 duration-200">
            {/* Header */}
            <div className="flex items-center justify-between px-8 py-6 border-b border-neutral-100 dark:border-zinc-800">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-neutral-50 dark:bg-zinc-800 rounded-lg">
                        <Weight className="w-5 h-5 text-neutral-600 dark:text-zinc-400" />
                    </div>
                    <div>
                        <h2 className="text-xl font-bold tracking-tight text-neutral-900 dark:text-zinc-100">Gravity Compensator</h2>
                        <p className="text-xs text-neutral-400 dark:text-zinc-500 font-medium uppercase tracking-wider">
                            Calibrating {armId.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())}
                        </p>
                    </div>
                </div>
                {step !== 'COMPUTING' && (
                    <button onClick={async () => { await setTorque(true); onClose(); }} className="p-2 hover:bg-neutral-50 dark:hover:bg-zinc-800 rounded-full transition-colors">
                        <ArrowLeft className="w-5 h-5 text-neutral-400 dark:text-zinc-500" />
                    </button>
                )}
            </div>

            {/* Content */}
            <div className="flex-1 flex flex-col items-center justify-center p-8 text-center bg-neutral-50/30 dark:bg-zinc-800/30">

                {step === 'INTRO' && (
                    <div className="max-w-md space-y-8 animate-in slide-in-from-bottom-4">
                        <div className="w-20 h-20 bg-blue-50 dark:bg-blue-950/50 rounded-full flex items-center justify-center mx-auto text-blue-500 dark:text-blue-400 shadow-xl shadow-blue-100 dark:shadow-blue-900/30 ring-4 ring-white dark:ring-zinc-800">
                            <Activity className="w-10 h-10" />
                        </div>
                        <div className="space-y-4">
                            <h3 className="text-2xl font-semibold text-neutral-900 dark:text-zinc-100">Let's make it weightless.</h3>
                            <p className="text-neutral-500 dark:text-zinc-400 leading-relaxed">
                                We'll learn the exact gravity dynamics of your leader arm.
                                You'll step through <span className="text-neutral-900 dark:text-zinc-100 font-bold">10 specific poses</span> to map the physics.
                            </p>
                        </div>
                        <button
                            onClick={startCalibration}
                            className="bg-neutral-900 dark:bg-zinc-100 text-white dark:text-black px-8 py-4 rounded-full font-bold shadow-lg shadow-neutral-200 dark:shadow-zinc-900/30 hover:bg-black dark:hover:bg-white hover:scale-105 active:scale-95 transition-all w-full flex items-center justify-center gap-2"
                        >
                            Start Learning <ArrowRight className="w-4 h-4" />
                        </button>
                    </div>
                )}

                {step === 'CAPTURE' && (
                    <div className="max-w-md w-full space-y-8 animate-in fade-in">
                        <div className="space-y-2">
                            <div className="flex justify-between text-xs font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-widest px-1">
                                <span>Progress</span>
                                <span>{samples} / 10</span>
                            </div>
                            <div className="h-3 bg-neutral-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-blue-500 transition-all duration-500 ease-out"
                                    style={{ width: `${(samples / 10) * 100}%` }}
                                />
                            </div>
                        </div>

                        {/* Torque Control + Capture */}
                        <div className="py-6 flex flex-col items-center gap-6">

                            {/* Torque Toggle */}
                            <button
                                onClick={() => setTorque(!isLocked)}
                                className={`flex items-center gap-2 px-6 py-2 rounded-full font-bold text-xs uppercase tracking-wide transition-all ${isLocked
                                    ? 'bg-blue-100 dark:bg-blue-950 text-blue-700 dark:text-blue-300 hover:bg-blue-200 dark:hover:bg-blue-900 ring-2 ring-blue-500 ring-offset-2 dark:ring-offset-zinc-900'
                                    : 'bg-neutral-200 dark:bg-zinc-700 text-neutral-600 dark:text-zinc-300 hover:bg-neutral-300 dark:hover:bg-zinc-600'
                                    }`}
                            >
                                {isLocked ? <Lock className="w-4 h-4" /> : <Unlock className="w-4 h-4" />}
                                {isLocked ? "Arm Locked (Ready)" : "Arm Free (Move)"}
                            </button>

                            {isCapturing ? (
                                <div className="w-32 h-32 flex items-center justify-center">
                                    <div className="w-20 h-20 rounded-full border-4 border-blue-500/30 border-t-blue-500 animate-spin" />
                                </div>
                            ) : (
                                <div className={`w-32 h-32 rounded-full flex items-center justify-center mx-auto shadow-xl ring-4 ring-white dark:ring-zinc-800 border transition-all cursor-pointer group select-none
                                    ${isLocked
                                        ? 'bg-blue-500 shadow-blue-200 dark:shadow-blue-900/30 hover:scale-105 active:scale-95 text-white'
                                        : 'bg-white dark:bg-zinc-800 shadow-neutral-100 dark:shadow-zinc-900/30 text-neutral-300 dark:text-zinc-600 opacity-50 cursor-not-allowed'}`}
                                    onClick={() => isLocked && captureSample()}
                                >
                                    <div className="text-center">
                                        <div className="text-xs font-bold uppercase tracking-widest mb-1">
                                            {isLocked ? "Click to" : "Lock First"}
                                        </div>
                                        <span className="text-2xl font-bold">Capture</span>
                                    </div>
                                </div>
                            )}
                        </div>

                        <div className="space-y-2 min-h-[5rem]">
                            <h4 className="text-xl font-bold text-blue-600 dark:text-blue-400">Pose {samples + 1}: {currentPose.title}</h4>
                            <p className="text-neutral-500 dark:text-zinc-400">
                                {isCapturing
                                    ? "Hold still... Measuring torque..."
                                    : isLocked
                                        ? "Arm is locked. Click Capture when steady."
                                        : "Unlock arm, move to position, then LOCK it."}
                            </p>
                        </div>

                        {samples >= 10 && (
                            <button
                                onClick={finishCalibration}
                                className="bg-green-500 text-white px-8 py-4 rounded-full font-bold shadow-xl shadow-green-200 dark:shadow-green-900/30 hover:bg-green-600 hover:scale-105 active:scale-95 transition-all w-full flex items-center justify-center gap-2 animate-in slide-in-from-bottom-2"
                            >
                                <CheckCircle className="w-5 h-5" /> Finish Calibration
                            </button>
                        )}

                        {error && (
                            <div className="bg-red-50 dark:bg-red-950/50 text-red-600 dark:text-red-400 p-3 rounded-lg text-sm font-medium">
                                {error}
                            </div>
                        )}
                    </div>
                )}

                {step === 'COMPUTING' && (
                    <div className="space-y-6 animate-in fade-in text-center">
                        <div className="w-16 h-16 border-4 border-neutral-200 dark:border-zinc-700 border-t-neutral-900 dark:border-t-zinc-100 rounded-full animate-spin mx-auto" />
                        <h3 className="text-xl font-medium text-neutral-900 dark:text-zinc-100">Processing Physics Model...</h3>
                    </div>
                )}

                {step === 'DONE' && (
                    <div className="max-w-md space-y-8 animate-in scale-in-95 duration-300">
                        <div className="w-24 h-24 bg-green-50 dark:bg-green-950/50 rounded-full flex items-center justify-center mx-auto text-green-500 dark:text-green-400 shadow-xl shadow-green-100 dark:shadow-green-900/30 ring-4 ring-white dark:ring-zinc-800">
                            <CheckCircle className="w-12 h-12" />
                        </div>
                        <div className="space-y-2">
                            <h3 className="text-2xl font-bold text-neutral-900 dark:text-zinc-100">Calibration Complete!</h3>
                            <p className="text-neutral-500 dark:text-zinc-400">
                                The gravity model has been updated. The leader arm should now feel weightless.
                            </p>
                        </div>
                        <button
                            onClick={() => { setTorque(true); onClose(); }}
                            className="bg-neutral-900 dark:bg-zinc-100 text-white dark:text-black px-8 py-4 rounded-full font-bold shadow-lg shadow-neutral-200 dark:shadow-zinc-900/30 hover:bg-black dark:hover:bg-white transition-all w-full"
                        >
                            Done
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
