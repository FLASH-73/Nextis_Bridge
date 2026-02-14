import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { User, Bot, Volume2, VolumeX, Pause } from 'lucide-react';
import { motion, useDragControls } from 'framer-motion';
import { useResizable } from '../../../hooks/useResizable';
import { usePolling } from '../../../hooks/usePolling';
import { EmergencyStop } from '../../EmergencyStop';
import HILSetupView from './HILSetupView';
import HILRunningView from './HILRunningView';
import { HILModalProps, Policy, HILStatus } from './types';
import { hilApi, policiesApi, camerasApi } from '../../../lib/api';

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
        }
    }, [isOpen]);

    const fetchPolicies = async () => {
        try {
            const data = await policiesApi.list();
            // Filter to only completed policies with checkpoints
            const deployable = data.filter((p: Policy) => p.status === 'completed' && p.checkpoint_path);
            setPolicies(deployable);
        } catch (e) {
            console.error('Failed to fetch policies:', e);
        }
    };

    const fetchCameras = async () => {
        try {
            const data = await camerasApi.config();
            if (Array.isArray(data)) setCameraConfigs(data);
        } catch (e) {
            console.error(e);
        }
    };

    const checkStatus = async () => {
        try {
            const data = await hilApi.status();
            setStatus(data);
            if (data.active) {
                setStep('running');
            }
        } catch (e) {
            console.error(e);
        }
    };

    // Poll for HIL status while running
    usePolling(async () => {
        try {
            const data = await hilApi.status();
            setStatus(data);
        } catch (e) {
            console.error(e);
        }
    }, 500, isOpen && step === 'running');

    const startSession = async () => {
        setError('');
        setIsStarting(true);

        if (!selectedPolicy) {
            setError('Please select a policy');
            setIsStarting(false);
            return;
        }

        try {
            const data = await hilApi.startSession({
                policy_id: selectedPolicy,
                intervention_dataset: interventionDataset,
                task,
                movement_scale: movementScale
            });
            if ((data as any).status === 'started') {
                setStep('running');
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
            await hilApi.stopSession();
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
                await hilApi.stopEpisode();
            } else {
                await hilApi.startEpisode();
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
            await hilApi.nextEpisode();
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
            await hilApi.resume({});
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
            await hilApi.stopEpisode();
        } catch (e) {
            console.error('Failed to stop episode:', e);
        } finally {
            setIsProcessing(false);
        }
    };

    const triggerRetrain = async () => {
        if (!confirm('Start retraining on intervention data?')) return;

        try {
            const data: any = await hilApi.retrain({});
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
                        <HILSetupView
                            policies={policies}
                            selectedPolicy={selectedPolicy}
                            setSelectedPolicy={setSelectedPolicy}
                            interventionDataset={interventionDataset}
                            setInterventionDataset={setInterventionDataset}
                            task={task}
                            setTask={setTask}
                            movementScale={movementScale}
                            setMovementScale={setMovementScale}
                            error={error}
                            isStarting={isStarting}
                            startSession={startSession}
                        />
                    ) : (
                        <HILRunningView
                            status={status}
                            activeCameras={activeCameras}
                            isProcessing={isProcessing}
                            toggleEpisode={toggleEpisode}
                            nextEpisode={nextEpisode}
                            resumeAutonomous={resumeAutonomous}
                            stopEpisode={stopEpisode}
                            stopSession={stopSession}
                            triggerRetrain={triggerRetrain}
                        />
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
