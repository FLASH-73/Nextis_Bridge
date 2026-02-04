import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { X, Play, StopCircle, User, Bot, Maximize2, Minimize2, AlertCircle, RefreshCw, Loader2, SkipForward, Volume2, VolumeX, Pause } from 'lucide-react';
import { motion, useDragControls } from 'framer-motion';
import { useResizable } from '../hooks/useResizable';
import { EmergencyStop } from './EmergencyStop';

const API_BASE = typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : 'http://127.0.0.1:8000';

interface HILModalProps {
    isOpen: boolean;
    onClose: () => void;
    maximizedWindow: string | null;
    setMaximizedWindow: (window: string | null) => void;
}

interface Policy {
    id: string;
    name: string;
    policy_type: string;
    status: string;
    checkpoint_path: string;
}

interface HILStatus {
    active: boolean;
    mode: 'idle' | 'autonomous' | 'human' | 'paused';
    policy_id: string;
    intervention_dataset: string;
    task: string;
    episode_active: boolean;
    episode_count: number;
    intervention_count: number;
    current_episode_interventions: number;
    autonomous_frames: number;
    human_frames: number;
    // Policy configuration - which cameras/arms the policy was trained on
    policy_config?: {
        cameras: string[];
        arms: string[];
        type: string;
    };
    // Safety settings
    movement_scale?: number;
}

export default function HILModal({ isOpen, onClose, maximizedWindow, setMaximizedWindow }: HILModalProps) {
    const dragControls = useDragControls();
    const { size, handleResizeMouseDown } = useResizable({ initialSize: { width: 900, height: 650 } });
    const [isMinimized, setIsMinimized] = useState(false);
    const isMaximized = maximizedWindow === 'hil';

    // Setup state
    const [step, setStep] = useState<'setup' | 'running'>('setup');
    const [policies, setPolicies] = useState<Policy[]>([]);
    const [selectedPolicy, setSelectedPolicy] = useState('');
    const [interventionDataset, setInterventionDataset] = useState('hil_interventions/v1');
    const [task, setTask] = useState('Intervention correction');
    const [movementScale, setMovementScale] = useState(1.0);
    const [error, setError] = useState('');
    const [isStarting, setIsStarting] = useState(false);

    // Running state
    const [status, setStatus] = useState<HILStatus | null>(null);
    const [cameraConfigs, setCameraConfigs] = useState<any[]>([]);
    const pollRef = useRef<NodeJS.Timeout | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);

    // Voice feedback state (persisted in localStorage)
    const [voiceMuted, setVoiceMuted] = useState(() => {
        if (typeof window !== 'undefined') {
            return localStorage.getItem('hil-voice-muted') === 'true';
        }
        return false;
    });
    const prevModeRef = useRef<string | null>(null);

    // Voice feedback using Web Speech API
    const speak = useCallback((text: string) => {
        if (voiceMuted) return;
        if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
            // Cancel any ongoing speech
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.1;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        }
    }, [voiceMuted]);

    // Toggle voice mute and persist
    const toggleVoice = () => {
        setVoiceMuted(prev => {
            const newVal = !prev;
            if (typeof window !== 'undefined') {
                localStorage.setItem('hil-voice-muted', String(newVal));
            }
            return newVal;
        });
    };

    // Track mode changes and announce via voice
    useEffect(() => {
        if (status?.mode && status.mode !== prevModeRef.current) {
            const announcements: Record<string, string> = {
                'autonomous': 'Autonomous mode',
                'human': 'Intervention detected',
                'paused': 'Paused. Waiting for your decision.',
                'idle': prevModeRef.current ? 'Episode saved' : '', // Don't announce on first load
            };
            if (announcements[status.mode]) {
                speak(announcements[status.mode]);
            }
            prevModeRef.current = status.mode;
        }
    }, [status?.mode, speak]);

    // Filter cameras based on policy configuration
    // Only show cameras the policy was trained on
    const activeCameras = useMemo(() => {
        if (!status?.active || !status?.policy_config?.cameras?.length) {
            // No active session or no config - show all cameras
            return cameraConfigs;
        }

        const policyCameras = status.policy_config.cameras;
        return cameraConfigs.filter(cam => {
            // Match camera_1 or camera1 format
            const camId = cam.id?.toLowerCase() || '';
            return policyCameras.some(pc => {
                const pcLower = pc.toLowerCase();
                return camId === pcLower ||
                       camId === pcLower.replace('_', '') ||
                       camId.replace('_', '') === pcLower;
            });
        });
    }, [status, cameraConfigs]);

    // Load policies on open
    useEffect(() => {
        if (isOpen) {
            fetchPolicies();
            fetchCameras();
            checkStatus();
        } else {
            stopPolling();
        }
    }, [isOpen]);

    const fetchPolicies = async () => {
        try {
            const res = await fetch(`${API_BASE}/policies`);
            const data = await res.json();
            // Filter to only completed policies with checkpoints
            const deployable = data.filter((p: Policy) => p.status === 'completed' && p.checkpoint_path);
            setPolicies(deployable);
        } catch (e) {
            console.error('Failed to fetch policies:', e);
        }
    };

    const fetchCameras = async () => {
        try {
            const res = await fetch(`${API_BASE}/cameras/config`);
            const data = await res.json();
            if (Array.isArray(data)) setCameraConfigs(data);
        } catch (e) {
            console.error(e);
        }
    };

    const checkStatus = async () => {
        try {
            const res = await fetch(`${API_BASE}/hil/status`);
            const data: HILStatus = await res.json();
            setStatus(data);
            if (data.active) {
                setStep('running');
                startPolling();
            }
        } catch (e) {
            console.error(e);
        }
    };

    const startPolling = () => {
        if (pollRef.current) return;
        pollRef.current = setInterval(async () => {
            try {
                const res = await fetch(`${API_BASE}/hil/status`);
                const data: HILStatus = await res.json();
                setStatus(data);
            } catch (e) {
                console.error(e);
            }
        }, 500);
    };

    const stopPolling = () => {
        if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
        }
    };

    const startSession = async () => {
        setError('');
        setIsStarting(true);

        if (!selectedPolicy) {
            setError('Please select a policy');
            setIsStarting(false);
            return;
        }

        try {
            const res = await fetch(`${API_BASE}/hil/session/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    policy_id: selectedPolicy,
                    intervention_dataset: interventionDataset,
                    task,
                    movement_scale: movementScale
                })
            });
            const data = await res.json();
            if (data.status === 'started') {
                setStep('running');
                startPolling();
            } else {
                setError(data.error || 'Failed to start HIL session');
            }
        } catch (e: any) {
            setError(e.message);
        } finally {
            setIsStarting(false);
        }
    };

    const stopSession = async () => {
        if (!confirm('Stop HIL session? This will save all recorded data.')) return;

        try {
            await fetch(`${API_BASE}/hil/session/stop`, { method: 'POST' });
            stopPolling();
            setStep('setup');
            setStatus(null);
        } catch (e) {
            console.error(e);
        }
    };

    const toggleEpisode = async () => {
        setIsProcessing(true);
        try {
            if (status?.episode_active) {
                await fetch(`${API_BASE}/hil/episode/stop`, { method: 'POST' });
            } else {
                await fetch(`${API_BASE}/hil/episode/start`, { method: 'POST' });
                speak(`Starting episode ${(status?.episode_count || 0) + 1}, autonomous mode`);
            }
        } finally {
            setIsProcessing(false);
        }
    };

    const nextEpisode = async () => {
        // Stop current episode and immediately start next one
        // Robot will try autonomously again
        setIsProcessing(true);
        try {
            await fetch(`${API_BASE}/hil/episode/next`, { method: 'POST' });
            speak(`Next episode, autonomous mode`);
        } catch (e) {
            console.error('Failed to start next episode:', e);
        } finally {
            setIsProcessing(false);
        }
    };

    const resumeAutonomous = async () => {
        // Explicitly resume autonomous mode after being paused
        setIsProcessing(true);
        try {
            await fetch(`${API_BASE}/hil/resume`, { method: 'POST' });
            speak('Resuming autonomous mode');
        } catch (e) {
            console.error('Failed to resume:', e);
        } finally {
            setIsProcessing(false);
        }
    };

    const stopEpisode = async () => {
        // Stop the current episode (used from PAUSED state)
        setIsProcessing(true);
        try {
            await fetch(`${API_BASE}/hil/episode/stop`, { method: 'POST' });
        } catch (e) {
            console.error('Failed to stop episode:', e);
        } finally {
            setIsProcessing(false);
        }
    };

    const triggerRetrain = async () => {
        if (!confirm('Start retraining on intervention data?')) return;

        try {
            const res = await fetch(`${API_BASE}/hil/retrain`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            const data = await res.json();
            if (data.status === 'started') {
                alert(`Training job ${data.job_id} started! Check the training panel for progress.`);
            } else {
                alert(`Retrain failed: ${data.error}`);
            }
        } catch (e: any) {
            alert(`Retrain failed: ${e.message}`);
        }
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
                width: isMaximized ? 'calc(100vw - 40px)' : isMinimized ? 'auto' : `${size.width}px`,
                height: isMaximized ? 'calc(100vh - 100px)' : isMinimized ? 'auto' : `${size.height}px`,
                zIndex: isMaximized ? 100 : 50,
            }}
            className={`fixed flex flex-col overflow-hidden font-sans glass-panel shadow-2xl rounded-3xl bg-white/90 dark:bg-zinc-900/90 border border-white/50 dark:border-zinc-700/50 backdrop-blur-3xl transition-all duration-300 ${isMaximized ? 'top-20 left-5' : 'top-20 left-1/2 -translate-x-1/2'}`}
        >
            {/* Header */}
            <div
                onPointerDown={(e) => dragControls.start(e)}
                className="h-16 bg-white/30 dark:bg-zinc-800/30 border-b border-black/5 dark:border-white/5 flex items-center justify-between px-6 cursor-grab active:cursor-grabbing select-none flex-none"
            >
                <div className="flex items-center gap-4">
                    <div className="flex gap-2 mr-2">
                        <button onClick={onClose} className="w-3.5 h-3.5 rounded-full bg-[#FF5F57] hover:brightness-90 transition-all" />
                        <button onClick={() => setIsMinimized(!isMinimized)} className="w-3.5 h-3.5 rounded-full bg-[#FEBC2E] hover:brightness-90 transition-all" />
                        <button onClick={() => setMaximizedWindow(isMaximized ? null : 'hil')} className="w-3.5 h-3.5 rounded-full bg-[#28C840] hover:brightness-90 transition-all" />
                    </div>
                    <span className="font-semibold text-neutral-800 dark:text-zinc-200 text-lg tracking-tight flex items-center gap-2">
                        <Bot className="w-5 h-5 text-blue-500" /> Human-in-the-Loop
                    </span>
                </div>

                <div className="flex items-center gap-4">
                    {/* Voice Mute Toggle */}
                    {status?.active && (
                        <button
                            onClick={toggleVoice}
                            className="p-1.5 rounded-lg hover:bg-neutral-100 dark:hover:bg-zinc-800 transition-colors"
                            title={voiceMuted ? "Unmute voice feedback" : "Mute voice feedback"}
                        >
                            {voiceMuted ? <VolumeX className="w-4 h-4 text-neutral-400 dark:text-zinc-500" /> : <Volume2 className="w-4 h-4 text-blue-500" />}
                        </button>
                    )}
                    {status?.active && (
                        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold transition-all ${
                            status.mode === 'human'
                                ? 'bg-orange-100 dark:bg-orange-950/50 text-orange-700 dark:text-orange-300 border border-orange-200 dark:border-orange-800'
                                : status.mode === 'paused'
                                ? 'bg-yellow-100 dark:bg-yellow-950/50 text-yellow-700 dark:text-yellow-300 border border-yellow-200 dark:border-yellow-800'
                                : status.mode === 'autonomous'
                                ? 'bg-blue-100 dark:bg-blue-950/50 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800'
                                : 'bg-neutral-100 dark:bg-zinc-800 text-neutral-500 dark:text-zinc-400 border border-neutral-200 dark:border-zinc-700'
                        }`}>
                            {status.mode === 'human' ? <User className="w-3.5 h-3.5" /> :
                             status.mode === 'paused' ? <Pause className="w-3.5 h-3.5" /> :
                             <Bot className="w-3.5 h-3.5" />}
                            {status.mode === 'human' ? 'HUMAN CONTROL' :
                             status.mode === 'paused' ? 'PAUSED' :
                             status.mode === 'autonomous' ? 'AUTONOMOUS' : 'IDLE'}
                        </div>
                    )}
                    <EmergencyStop />
                </div>
            </div>

            {!isMinimized && (
                <div className="flex-1 flex flex-col p-6 overflow-hidden">
                    {step === 'setup' ? (
                        /* SETUP VIEW */
                        <div className="flex flex-col max-w-lg mx-auto w-full gap-5 animate-in fade-in slide-in-from-bottom-4">
                            <div className="text-center mb-2">
                                <h2 className="text-2xl font-bold text-neutral-900 dark:text-zinc-100 mb-1">Deploy with Human Oversight</h2>
                                <p className="text-neutral-500 dark:text-zinc-400 text-sm">Run a policy autonomously. Take over anytime to correct behavior.</p>
                            </div>

                            {/* Policy Selector */}
                            <div>
                                <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Select Policy</label>
                                <select
                                    value={selectedPolicy}
                                    onChange={e => setSelectedPolicy(e.target.value)}
                                    className="w-full px-4 py-3 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all text-neutral-900 dark:text-zinc-100"
                                >
                                    <option value="">Choose a trained policy...</option>
                                    {policies.map(p => (
                                        <option key={p.id} value={p.id}>
                                            {p.name} ({p.policy_type})
                                        </option>
                                    ))}
                                </select>
                                {policies.length === 0 && (
                                    <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-1">No trained policies available. Train a policy first.</p>
                                )}
                            </div>

                            {/* Intervention Dataset */}
                            <div>
                                <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Intervention Dataset</label>
                                <input
                                    type="text"
                                    value={interventionDataset}
                                    onChange={e => setInterventionDataset(e.target.value)}
                                    placeholder="hil_interventions/v1"
                                    className="w-full px-4 py-3 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all text-neutral-900 dark:text-zinc-100 placeholder-neutral-300 dark:placeholder-zinc-500"
                                />
                                <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-1">Human corrections will be saved here for retraining</p>
                            </div>

                            {/* Task Description */}
                            <div>
                                <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Task Description</label>
                                <input
                                    type="text"
                                    value={task}
                                    onChange={e => setTask(e.target.value)}
                                    placeholder="Describe the task..."
                                    className="w-full px-4 py-3 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all text-neutral-900 dark:text-zinc-100 placeholder-neutral-300 dark:placeholder-zinc-500"
                                />
                            </div>

                            {/* Movement Scale (Safety Limiter) */}
                            <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl p-4 border border-neutral-200 dark:border-zinc-700">
                                <div className="flex items-center justify-between mb-2">
                                    <label className="text-sm font-bold text-neutral-700 dark:text-zinc-300">
                                        Movement Scale
                                        <span className="ml-1 text-xs font-normal text-neutral-400 dark:text-zinc-500">(safety limiter)</span>
                                    </label>
                                    <span className={`text-sm font-bold ${
                                        movementScale >= 0.8 ? 'text-red-500' :
                                        movementScale >= 0.5 ? 'text-amber-500' : 'text-green-500'
                                    }`}>
                                        {Math.round(movementScale * 100)}%
                                    </span>
                                </div>
                                <input
                                    type="range"
                                    min={0.1}
                                    max={1.0}
                                    step={0.1}
                                    value={movementScale}
                                    onChange={(e) => setMovementScale(parseFloat(e.target.value))}
                                    className="w-full h-2 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                    style={{
                                        background: `linear-gradient(to right, #22c55e 0%, #22c55e 40%, #eab308 50%, #eab308 70%, #ef4444 80%, #ef4444 100%)`
                                    }}
                                />
                                <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-2">
                                    {movementScale < 0.5
                                        ? "Safe for testing — robot moves slowly toward target positions"
                                        : movementScale < 0.8
                                        ? "Moderate speed — verify policy behavior before increasing"
                                        : "Full speed — use only with tested, reliable policies"}
                                </p>
                            </div>

                            {error && (
                                <div className="bg-red-50 dark:bg-red-950/50 text-red-600 dark:text-red-400 px-4 py-3 rounded-xl text-sm flex items-center gap-2 border border-red-100 dark:border-red-900">
                                    <AlertCircle className="w-4 h-4 flex-shrink-0" /> {error}
                                </div>
                            )}

                            <button
                                onClick={startSession}
                                disabled={!selectedPolicy || isStarting}
                                className="w-full py-4 bg-blue-600 text-white rounded-2xl font-bold text-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-200 dark:shadow-blue-900/30 flex items-center justify-center gap-2"
                            >
                                {isStarting ? <Loader2 className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5" />}
                                {isStarting ? 'Starting...' : 'Start HIL Session'}
                            </button>
                        </div>
                    ) : (
                        /* RUNNING VIEW */
                        <div className="flex h-full gap-5">
                            {/* Left: Camera Feed - Only show policy-relevant cameras */}
                            <div className="flex-1 flex flex-col gap-4">
                                <div className="flex-1 bg-black rounded-2xl overflow-hidden relative min-h-0">
                                    <div className={`grid h-full ${activeCameras.length > 1 ? 'grid-cols-2' : 'grid-cols-1'} ${activeCameras.length > 2 ? 'grid-rows-2' : ''}`}>
                                        {activeCameras.map(cam => (
                                            <div key={cam.id} className="relative w-full h-full border-r border-white/10 last:border-0 overflow-hidden">
                                                <img
                                                    src={`${API_BASE}/video_feed/${cam.id}`}
                                                    className="w-full h-full object-cover"
                                                    alt={cam.id}
                                                />
                                                <div className="absolute bottom-2 left-2 bg-black/50 px-2 py-1 rounded text-xs text-white/80">
                                                    {cam.id}
                                                </div>
                                            </div>
                                        ))}
                                        {activeCameras.length === 0 && (
                                            <div className="flex items-center justify-center text-white/50">
                                                No cameras configured for this policy
                                            </div>
                                        )}
                                    </div>

                                    {/* Mode Indicator Overlay */}
                                    <div className={`absolute top-4 right-4 flex items-center gap-2 px-4 py-2 rounded-full text-sm font-bold shadow-lg backdrop-blur-md transition-all ${
                                        status?.mode === 'human'
                                            ? 'bg-orange-500/90 text-white'
                                            : status?.mode === 'paused'
                                            ? 'bg-yellow-500/90 text-white'
                                            : status?.mode === 'autonomous'
                                            ? 'bg-blue-500/90 text-white'
                                            : 'bg-neutral-500/90 text-white'
                                    }`}>
                                        {status?.mode === 'human' ? <User className="w-4 h-4" /> :
                                         status?.mode === 'paused' ? <Pause className="w-4 h-4" /> :
                                         <Bot className="w-4 h-4" />}
                                        {status?.mode === 'human' ? 'Human Control' :
                                         status?.mode === 'paused' ? 'Paused' :
                                         status?.mode === 'autonomous' ? 'Autonomous' : 'Idle'}
                                    </div>

                                    {/* Recording Indicator */}
                                    {status?.episode_active && (
                                        <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500/90 text-white px-3 py-1.5 rounded-full text-xs font-bold">
                                            <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                                            REC
                                        </div>
                                    )}

                                    {/* PAUSED Action Panel - shows decision buttons when system pauses after intervention */}
                                    {status?.mode === 'paused' && (
                                        <>
                                            {/* Pulsing yellow border overlay */}
                                            <div className="absolute inset-0 border-4 border-yellow-400 rounded-2xl animate-pulse pointer-events-none" />

                                            {/* Action buttons panel */}
                                            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-3 p-4 bg-yellow-50/95 border border-yellow-300 rounded-xl shadow-xl backdrop-blur-sm">
                                                <button
                                                    onClick={resumeAutonomous}
                                                    disabled={isProcessing}
                                                    className="px-5 py-2.5 bg-blue-500 text-white rounded-lg font-semibold hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-md"
                                                >
                                                    <Play className="w-4 h-4" /> Resume Autonomous
                                                </button>
                                                <button
                                                    onClick={stopEpisode}
                                                    disabled={isProcessing}
                                                    className="px-5 py-2.5 bg-red-500 text-white rounded-lg font-semibold hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-md"
                                                >
                                                    <StopCircle className="w-4 h-4" /> Stop Episode
                                                </button>
                                            </div>
                                        </>
                                    )}
                                </div>

                                {/* Control Bar */}
                                <div className="h-20 bg-white dark:bg-zinc-900 border border-neutral-100 dark:border-zinc-800 rounded-2xl shadow-sm flex items-center justify-between px-6 flex-shrink-0">
                                    <div>
                                        <span className="text-xs font-bold text-neutral-400 dark:text-zinc-500 uppercase">Episode</span>
                                        <span className="text-2xl font-bold text-neutral-900 dark:text-zinc-100 block">#{(status?.episode_count || 0) + (status?.episode_active ? 1 : 0)}</span>
                                    </div>

                                    <div className="flex items-center gap-3">
                                        {/* Main episode toggle button - disabled during processing, human mode, or paused */}
                                        <button
                                            onClick={toggleEpisode}
                                            disabled={isProcessing || status?.mode === 'human' || status?.mode === 'paused'}
                                            className={`w-16 h-16 rounded-full flex items-center justify-center transition-all shadow-lg ${
                                                isProcessing || status?.mode === 'human' || status?.mode === 'paused'
                                                    ? 'opacity-50 cursor-not-allowed'
                                                    : status?.episode_active
                                                    ? 'bg-white border-4 border-red-500 hover:scale-105'
                                                    : 'bg-red-500 border-4 border-red-100 hover:scale-105'
                                            }`}
                                            title={
                                                isProcessing ? 'Processing...' :
                                                status?.mode === 'human' ? 'Cannot stop during intervention' :
                                                status?.mode === 'paused' ? 'Use the action panel above' :
                                                status?.episode_active ? 'Stop Episode' : 'Start Episode'
                                            }
                                        >
                                            {isProcessing ? (
                                                <Loader2 className="w-6 h-6 text-neutral-400 animate-spin" />
                                            ) : status?.episode_active ? (
                                                <div className="w-6 h-6 bg-red-500 rounded-sm" />
                                            ) : (
                                                <Play className="w-6 h-6 text-white ml-1" />
                                            )}
                                        </button>

                                        {/* Next Episode button - stops current and starts new */}
                                        {status?.episode_active && status?.mode !== 'paused' && (
                                            <button
                                                onClick={nextEpisode}
                                                disabled={isProcessing || status?.mode === 'human'}
                                                className={`w-12 h-12 rounded-full flex items-center justify-center transition-all shadow-md ${
                                                    isProcessing || status?.mode === 'human'
                                                        ? 'bg-blue-300 cursor-not-allowed'
                                                        : 'bg-blue-500 hover:bg-blue-600 hover:scale-105'
                                                }`}
                                                title={
                                                    isProcessing ? 'Processing...' :
                                                    status?.mode === 'human' ? 'Cannot skip during intervention' :
                                                    'Stop current episode and start next one (robot tries again)'
                                                }
                                            >
                                                {isProcessing ? (
                                                    <Loader2 className="w-5 h-5 text-white animate-spin" />
                                                ) : (
                                                    <SkipForward className="w-5 h-5 text-white" />
                                                )}
                                            </button>
                                        )}
                                    </div>

                                    <button
                                        onClick={stopSession}
                                        disabled={isProcessing}
                                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                                            isProcessing
                                                ? 'text-neutral-300 dark:text-zinc-600 cursor-not-allowed'
                                                : 'text-neutral-500 dark:text-zinc-400 hover:bg-neutral-100 dark:hover:bg-zinc-800'
                                        }`}
                                    >
                                        Finish Session
                                    </button>
                                </div>
                            </div>

                            {/* Right: Stats Sidebar */}
                            <div className="w-56 flex flex-col gap-4 flex-shrink-0">
                                {/* Intervention Counter */}
                                <div className="bg-gradient-to-br from-orange-50 to-orange-100/50 dark:from-orange-950/30 dark:to-orange-900/20 rounded-2xl p-5 border border-orange-100 dark:border-orange-900">
                                    <h3 className="text-xs font-bold text-orange-400 uppercase tracking-widest mb-2">Interventions</h3>
                                    <div className="text-4xl font-bold text-orange-600 dark:text-orange-400">{status?.intervention_count || 0}</div>
                                    <p className="text-xs text-orange-500 dark:text-orange-400/70 mt-1">This episode: {status?.current_episode_interventions || 0}</p>
                                </div>

                                {/* Frame Stats */}
                                <div className="bg-white dark:bg-zinc-900 rounded-2xl p-5 border border-neutral-100 dark:border-zinc-800 shadow-sm">
                                    <h3 className="text-xs font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-widest mb-3">Frame Count</h3>
                                    <div className="space-y-3">
                                        <div className="flex justify-between items-center">
                                            <span className="text-sm text-neutral-600 dark:text-zinc-400 flex items-center gap-2">
                                                <Bot className="w-4 h-4 text-blue-500" /> Autonomous
                                            </span>
                                            <span className="font-mono font-bold text-neutral-900 dark:text-zinc-100">{status?.autonomous_frames || 0}</span>
                                        </div>
                                        <div className="flex justify-between items-center">
                                            <span className="text-sm text-neutral-600 dark:text-zinc-400 flex items-center gap-2">
                                                <User className="w-4 h-4 text-orange-500" /> Human
                                            </span>
                                            <span className="font-mono font-bold text-neutral-900 dark:text-zinc-100">{status?.human_frames || 0}</span>
                                        </div>
                                    </div>
                                </div>

                                {/* Movement Scale Indicator */}
                                <div className={`rounded-xl p-4 border ${
                                    (status?.movement_scale || 1) >= 0.8
                                        ? 'bg-red-50 dark:bg-red-950/30 border-red-100 dark:border-red-900'
                                        : (status?.movement_scale || 1) >= 0.5
                                        ? 'bg-amber-50 dark:bg-amber-950/30 border-amber-100 dark:border-amber-900'
                                        : 'bg-green-50 dark:bg-green-950/30 border-green-100 dark:border-green-900'
                                }`}>
                                    <div className="flex items-center justify-between">
                                        <span className={`text-xs font-bold uppercase tracking-widest ${
                                            (status?.movement_scale || 1) >= 0.8
                                                ? 'text-red-400'
                                                : (status?.movement_scale || 1) >= 0.5
                                                ? 'text-amber-400'
                                                : 'text-green-400'
                                        }`}>Move Scale</span>
                                        <span className={`text-lg font-bold ${
                                            (status?.movement_scale || 1) >= 0.8
                                                ? 'text-red-600 dark:text-red-400'
                                                : (status?.movement_scale || 1) >= 0.5
                                                ? 'text-amber-600 dark:text-amber-400'
                                                : 'text-green-600 dark:text-green-400'
                                        }`}>{Math.round((status?.movement_scale || 1) * 100)}%</span>
                                    </div>
                                </div>

                                {/* Retrain Button */}
                                <button
                                    onClick={triggerRetrain}
                                    disabled={!status?.intervention_count}
                                    className="w-full py-3 bg-purple-600 text-white rounded-xl font-semibold hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
                                >
                                    <RefreshCw className="w-4 h-4" /> Retrain on Data
                                </button>

                                {/* Policy Configuration Display */}
                                {status?.policy_config && (status?.policy_config?.cameras?.length > 0 || status?.policy_config?.arms?.length > 0) && (
                                    <div className="bg-purple-50 dark:bg-purple-950/30 rounded-xl p-4 border border-purple-100 dark:border-purple-900">
                                        <h3 className="text-xs font-bold text-purple-400 uppercase tracking-widest mb-2">Policy Config</h3>
                                        <div className="space-y-1 text-xs text-purple-700 dark:text-purple-300">
                                            {status.policy_config.type && (
                                                <p><strong>Type:</strong> {status.policy_config.type}</p>
                                            )}
                                            <p><strong>Cameras:</strong> {status.policy_config.cameras?.join(', ') || 'All'}</p>
                                            <p><strong>Arms:</strong> {status.policy_config.arms?.join(', ') || 'All'}</p>
                                        </div>
                                    </div>
                                )}

                                {/* Info Tip */}
                                <div className="bg-blue-50 dark:bg-blue-950/30 rounded-xl p-4 border border-blue-100 dark:border-blue-900 mt-auto">
                                    <p className="text-xs text-blue-700 dark:text-blue-300 leading-relaxed">
                                        <strong>Tip:</strong> Grab the leader arms to intervene.
                                        After you let go, the system will pause and wait for your decision:
                                        Resume autonomous or Stop episode.
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Resize Handle */}
            {!isMaximized && !isMinimized && (
                <div
                    onMouseDown={handleResizeMouseDown}
                    className="absolute bottom-2 right-2 w-4 h-4 cursor-se-resize opacity-30 hover:opacity-60 transition-opacity"
                >
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M22 22H20V20H22V22ZM22 18H20V16H22V18ZM18 22H16V20H18V22ZM22 14H20V12H22V14ZM18 18H16V16H18V18ZM14 22H12V20H14V22Z" />
                    </svg>
                </div>
            )}
        </motion.div>
    );
}
