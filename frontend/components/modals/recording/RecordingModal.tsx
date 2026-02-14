import React, { useState, useEffect, useRef } from 'react';
import { X, Play, StopCircle, Video, Maximize2, Minimize2, Circle, AlertCircle, Save, Folder, Trash2, Loader2, Check } from 'lucide-react';
import { motion, useDragControls } from 'framer-motion';
import { useResizable } from '../../../hooks/useResizable';
import { EmergencyStop } from '../../EmergencyStop';
import RecordingView from './RecordingView';
import { recordingApi, camerasApi, teleopApi } from '../../../lib/api';

interface RecordingModalProps {
    isOpen: boolean;
    onClose: () => void;
    maximizedWindow: string | null;
    setMaximizedWindow: (window: string | null) => void;
}

export default function RecordingModal({ isOpen, onClose, maximizedWindow, setMaximizedWindow }: RecordingModalProps) {
    const dragControls = useDragControls();
    const { size, handleResizeMouseDown } = useResizable({ initialSize: { width: 1400, height: 900 } });
    const [isMinimized, setIsMinimized] = useState(false);
    const isMaximized = maximizedWindow === 'recording';

    // State
    const [step, setStep] = useState<'setup' | 'recording'>('setup');
    const [datasetConfig, setDatasetConfig] = useState({ repo_id: 'lerobot/dataset', task: 'Pick up the cube' });
    const [isSessionActive, setIsSessionActive] = useState(false);
    const [isRecordingEpisode, setIsRecordingEpisode] = useState(false);
    const [isSavingEpisode, setIsSavingEpisode] = useState(false);
    const [episodeCount, setEpisodeCount] = useState(0);
    const [cameraConfigs, setCameraConfigs] = useState<any[]>([]);
    const [error, setError] = useState('');

    // Camera/Arm Selection
    const [availableCameras, setAvailableCameras] = useState<{ id: string; name: string }[]>([]);
    const [availableArms, setAvailableArms] = useState<{ id: string; name: string; joints: number }[]>([]);
    const [selectedCameras, setSelectedCameras] = useState<string[]>([]);
    const [selectedArms, setSelectedArms] = useState<string[]>([]);
    const [optionsLoading, setOptionsLoading] = useState(true);

    // Timer
    const [timer, setTimer] = useState(0);
    const timerRef = useRef<NodeJS.Timeout | null>(null);

    // Initial Load
    useEffect(() => {
        if (isOpen) {
            checkStatus();
            // Fetch camera configs for preview
            camerasApi.config()
                .then(data => { if (Array.isArray(data)) setCameraConfigs(data); })
                .catch(console.error);

            // Fetch recording options (cameras + arms for selection)
            setOptionsLoading(true);
            recordingApi.options()
                .then((data: any) => {
                    setAvailableCameras(data.cameras || []);
                    setAvailableArms(data.arms || []);
                    // Default: select all
                    setSelectedCameras((data.cameras || []).map((c: any) => c.id));
                    setSelectedArms((data.arms || []).map((a: any) => a.id));
                    setOptionsLoading(false);
                })
                .catch(err => {
                    console.error('Failed to fetch recording options:', err);
                    setOptionsLoading(false);
                });
        } else {
            // Cleanup if closed abruptly? Better handling needed in future.
        }
    }, [isOpen]);

    // Timer Logic
    useEffect(() => {
        if (isRecordingEpisode) {
            timerRef.current = setInterval(() => setTimer(t => t + 0.1), 100);
        } else {
            if (timerRef.current) clearInterval(timerRef.current);
            setTimer(0);
        }
        return () => { if (timerRef.current) clearInterval(timerRef.current); };
    }, [isRecordingEpisode]);

    const checkStatus = async () => {
        try {
            // Check if session is already active
            const data: any = await recordingApi.status();
            if (data.session_active) {
                setStep('recording');
                setIsSessionActive(true);
                setIsRecordingEpisode(data.episode_active);
                setEpisodeCount(data.episode_count || 0);
                // We don't easily get repo_id back from status yet without caching in frontend or adding field to API
            }
        } catch (e) {
            console.error(e);
        }
    };

    const startSession = async () => {
        setError('');

        // Validate selections
        if (availableCameras.length > 0 && selectedCameras.length === 0) {
            setError('Please select at least one camera');
            return;
        }
        if (availableArms.length > 0 && selectedArms.length === 0) {
            setError('Please select at least one arm');
            return;
        }

        console.log('[RecordingModal] startSession called with config:', datasetConfig);
        console.log('[RecordingModal] Selected cameras:', selectedCameras);
        console.log('[RecordingModal] Selected arms:', selectedArms);

        try {
            console.log('[RecordingModal] Calling /recording/session/start...');
            const data: any = await recordingApi.startSession({
                ...datasetConfig,
                selected_cameras: selectedCameras.length === availableCameras.length ? null : selectedCameras,
                selected_arms: selectedArms.length === availableArms.length ? null : selectedArms,
            });
            console.log('[RecordingModal] /recording/session/start response:', data);

            if (data.status === 'success') {
                setIsSessionActive(true);
                setStep('recording');
                // Reset episode count from backend (handles both new and existing datasets)
                setEpisodeCount(data.episode_count ?? 0);

                // Also ensure Teleop is Started!
                console.log('[RecordingModal] Starting teleop...');
                const teleopData = await teleopApi.start({ force: false });
                console.log('[RecordingModal] /teleop/start response:', teleopData);

            } else {
                console.error('[RecordingModal] Session start failed:', data.message);
                setError(data.message);
            }
        } catch (e: any) {
            console.error('[RecordingModal] startSession error:', e);
            setError(e.message || "Failed to start session");
        }
    };

    const stopSession = async () => {
        if (!confirm("Finish Recording Session? This will finalize the dataset.")) return;
        console.log('[RecordingModal] stopSession called');
        try {
            console.log('[RecordingModal] Calling /recording/session/stop...');
            const data = await recordingApi.stopSession();
            console.log('[RecordingModal] /recording/session/stop response:', data);
            setIsSessionActive(false);
            setStep('setup');
            setEpisodeCount(0);
        } catch (e) {
            console.error('[RecordingModal] stopSession error:', e);
        }
    };

    const toggleEpisode = async () => {
        setError('');
        console.log('[RecordingModal] toggleEpisode called, isRecordingEpisode:', isRecordingEpisode, 'isSavingEpisode:', isSavingEpisode);

        // Prevent double-clicks while saving
        if (isSavingEpisode) {
            console.log('[RecordingModal] Still saving, ignoring click');
            return;
        }

        try {
            if (isRecordingEpisode) {
                // Stop - set saving state BEFORE API call
                console.log('[RecordingModal] Stopping episode...');
                setIsSavingEpisode(true);
                setIsRecordingEpisode(false);  // Update UI immediately

                const data: any = await recordingApi.stopEpisode();
                console.log('[RecordingModal] /recording/episode/stop response:', data);

                // Use actual count from backend instead of optimistic update
                if (data?.status === 'success' && data.episode_count !== undefined) {
                    setEpisodeCount(data.episode_count);
                } else if (data?.status === 'success') {
                    // Fallback to optimistic update if backend doesn't return count
                    setEpisodeCount(c => c + 1);
                } else {
                    // Save failed - show error
                    console.error('[RecordingModal] Episode save failed:', data?.message);
                    setError(data?.message || 'Failed to save episode');
                }

                setIsSavingEpisode(false);  // Allow new recordings
            } else {
                // Start
                console.log('[RecordingModal] Starting episode...');
                const data = await recordingApi.startEpisode();
                console.log('[RecordingModal] /recording/episode/start response:', data);

                if ((data as any).status === 'success') {
                    setIsRecordingEpisode(true);
                } else {
                    setError((data as any).message || 'Failed to start episode');
                }
            }
        } catch (e: any) {
            console.error('[RecordingModal] toggleEpisode error:', e);
            setIsSavingEpisode(false);
            setError(e.message);
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
                width: isMaximized ? 'calc(100vw - 40px)' : isMinimized ? 'auto' : `min(${size.width}px, 85vw)`,
                height: isMaximized ? 'calc(100vh - 100px)' : isMinimized ? 'auto' : `min(${size.height}px, 85vh)`,
                top: isMaximized ? '80px' : undefined,
                left: isMaximized ? '20px' : undefined,
                transform: isMaximized ? 'none' : undefined,
                zIndex: isMaximized ? 100 : 50,
            }}
            className={`fixed flex flex-col overflow-hidden font-sans glass-panel shadow-2xl rounded-3xl bg-white/90 dark:bg-zinc-900/90 border border-white/50 dark:border-zinc-700/50 backdrop-blur-3xl transition-all duration-300 ${isMaximized ? '' : 'top-20 left-1/2 -translate-x-1/2'}`}
        >
            {/* Header */}
            <div
                onPointerDown={(e) => dragControls.start(e)}
                className="h-16 bg-white/30 dark:bg-zinc-800/30 border-b border-black/5 dark:border-white/5 flex items-center justify-between px-6 cursor-grab active:cursor-grabbing select-none flex-none"
            >
                <div className="flex items-center gap-4">
                    <div className="flex gap-2 mr-2">
                        <button onClick={onClose} className="w-3.5 h-3.5 rounded-full bg-[#FF5F57] hover:brightness-90 btn-control transition-all" />
                        <button onClick={() => { setIsMinimized(!isMinimized); if (isMaximized) setMaximizedWindow(null); }} className="w-3.5 h-3.5 rounded-full bg-[#FEBC2E] hover:brightness-90 btn-control transition-all" />
                        <button onClick={() => { setMaximizedWindow(isMaximized ? null : 'recording'); if (isMinimized) setIsMinimized(false); }} className="w-3.5 h-3.5 rounded-full bg-[#28C840] hover:brightness-90 btn-control transition-all" />
                    </div>
                    <span className="font-semibold text-neutral-800 dark:text-zinc-200 text-lg tracking-tight flex items-center gap-2">
                        <Video className="w-5 h-5 text-red-500" /> Studio
                    </span>
                </div>

                <div className="flex items-center gap-4">
                    {step === 'recording' && (
                        <span className="text-xs font-medium text-neutral-500 dark:text-zinc-400 bg-neutral-100 dark:bg-zinc-800 px-3 py-1 rounded-full border border-neutral-200 dark:border-zinc-700">
                            {datasetConfig.repo_id}
                        </span>
                    )}
                    <EmergencyStop />
                </div>
            </div>

            {!isMinimized && (
                <div className="flex-1 flex flex-col p-8 overflow-hidden relative">
                    {step === 'setup' ? (
                        <div className="flex flex-col items-start max-w-lg mx-auto w-full mt-4 gap-6 animate-in fade-in slide-in-from-bottom-4 overflow-y-auto max-h-full pb-8">
                            <div>
                                <h2 className="text-3xl font-bold text-neutral-900 dark:text-zinc-100 mb-2">Create New Session</h2>
                                <p className="text-neutral-500 dark:text-zinc-400">Configure your dataset parameters before recording.</p>
                            </div>

                            <div className="w-full space-y-5">
                                <div>
                                    <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Repository ID</label>
                                    <input
                                        type="text"
                                        value={datasetConfig.repo_id}
                                        onChange={e => setDatasetConfig({ ...datasetConfig, repo_id: e.target.value })}
                                        className="w-full px-4 py-3 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-xl focus:ring-2 focus:ring-red-500 focus:border-red-500 outline-none transition-all placeholder-neutral-300 dark:placeholder-zinc-500 text-neutral-900 dark:text-zinc-100"
                                        placeholder="usage: user/dataset-name"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Task Description</label>
                                    <input
                                        type="text"
                                        value={datasetConfig.task}
                                        onChange={e => setDatasetConfig({ ...datasetConfig, task: e.target.value })}
                                        className="w-full px-4 py-3 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-xl focus:ring-2 focus:ring-red-500 focus:border-red-500 outline-none transition-all placeholder-neutral-300 dark:placeholder-zinc-500 text-neutral-900 dark:text-zinc-100"
                                        placeholder="e.g. Pick up the red cube"
                                    />
                                </div>

                                {/* Camera Selection */}
                                {!optionsLoading && availableCameras.length > 0 && (
                                    <div>
                                        <div className="flex justify-between items-center mb-2">
                                            <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300">Cameras to Record</label>
                                            <div className="flex gap-2">
                                                <button
                                                    onClick={() => setSelectedCameras(availableCameras.map(c => c.id))}
                                                    className="text-xs text-blue-600 hover:text-blue-400"
                                                >
                                                    Select All
                                                </button>
                                                <button
                                                    onClick={() => setSelectedCameras([])}
                                                    className="text-xs text-neutral-400 hover:text-neutral-600 dark:hover:text-zinc-300"
                                                >
                                                    Clear
                                                </button>
                                            </div>
                                        </div>
                                        <div className="grid grid-cols-2 gap-3">
                                            {availableCameras.map(cam => {
                                                const isSelected = selectedCameras.includes(cam.id);
                                                return (
                                                    <div
                                                        key={cam.id}
                                                        onClick={() => {
                                                            if (isSelected) {
                                                                setSelectedCameras(prev => prev.filter(id => id !== cam.id));
                                                            } else {
                                                                setSelectedCameras(prev => [...prev, cam.id]);
                                                            }
                                                        }}
                                                        className={`relative cursor-pointer rounded-xl overflow-hidden border-2 transition-all ${isSelected
                                                            ? 'border-red-500 ring-2 ring-red-200 dark:ring-red-900'
                                                            : 'border-neutral-200 dark:border-zinc-700 hover:border-neutral-300 dark:hover:border-zinc-600 opacity-50'
                                                            }`}
                                                    >
                                                        {/* Camera Preview */}
                                                        <div className="h-20 bg-neutral-900 relative">
                                                            <img
                                                                src={camerasApi.videoFeedUrl(cam.id)}
                                                                className="w-full h-full object-cover"
                                                                alt={cam.name}
                                                            />
                                                            {isSelected && (
                                                                <div className="absolute top-2 right-2 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center">
                                                                    <Check className="w-3 h-3 text-white" />
                                                                </div>
                                                            )}
                                                        </div>
                                                        <div className="p-2 bg-white dark:bg-zinc-800">
                                                            <span className={`text-xs font-semibold ${isSelected ? 'text-red-600 dark:text-red-400' : 'text-neutral-400 dark:text-zinc-500'}`}>
                                                                {cam.name}
                                                            </span>
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                )}

                                {/* Arm Selection */}
                                {!optionsLoading && availableArms.length > 0 && (
                                    <div>
                                        <div className="flex justify-between items-center mb-2">
                                            <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300">Arms to Record</label>
                                            <div className="flex gap-2">
                                                <button
                                                    onClick={() => setSelectedArms(availableArms.map(a => a.id))}
                                                    className="text-xs text-blue-600 hover:text-blue-400"
                                                >
                                                    Select All
                                                </button>
                                                <button
                                                    onClick={() => setSelectedArms([])}
                                                    className="text-xs text-neutral-400 hover:text-neutral-600 dark:hover:text-zinc-300"
                                                >
                                                    Clear
                                                </button>
                                            </div>
                                        </div>
                                        <div className="flex gap-3 justify-center">
                                            {availableArms.map(arm => {
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
                                                        className={`relative flex items-center gap-3 px-5 py-4 rounded-xl border-2 cursor-pointer transition-all min-w-[140px] ${isSelected
                                                            ? 'bg-neutral-900 dark:bg-zinc-100 border-neutral-900 dark:border-zinc-100 text-white dark:text-zinc-900 shadow-lg'
                                                            : 'bg-white dark:bg-zinc-800 border-neutral-200 dark:border-zinc-700 hover:border-neutral-300 dark:hover:border-zinc-600 text-neutral-400 dark:text-zinc-500'
                                                            }`}
                                                    >
                                                        <div className={`w-3 h-3 rounded-full ${isSelected ? 'bg-green-400' : 'bg-neutral-300'}`} />
                                                        <div className="flex flex-col">
                                                            <span className="font-semibold text-sm">{arm.name}</span>
                                                            <span className={`text-[10px] ${isSelected ? 'text-neutral-400 dark:text-zinc-500' : 'text-neutral-300 dark:text-zinc-600'}`}>
                                                                {arm.joints} DOF
                                                            </span>
                                                        </div>
                                                        {isSelected && (
                                                            <div className="absolute top-2 right-2 w-2 h-2 bg-blue-500 rounded-full" />
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                )}

                                {optionsLoading && (
                                    <div className="flex items-center justify-center py-4 text-neutral-400 dark:text-zinc-500">
                                        <Loader2 className="w-5 h-5 animate-spin mr-2" />
                                        Loading options...
                                    </div>
                                )}
                            </div>

                            {error && (
                                <div className="w-full bg-red-50 dark:bg-red-950/50 text-red-600 dark:text-red-400 px-4 py-3 rounded-xl text-sm flex items-center gap-2 border border-red-100 dark:border-red-900">
                                    <AlertCircle className="w-4 h-4" /> {error}
                                </div>
                            )}

                            <button
                                onClick={startSession}
                                className="w-full py-4 bg-black dark:bg-white text-white dark:text-black rounded-2xl font-bold text-lg hover:bg-neutral-800 dark:hover:bg-zinc-200 hover:scale-[1.02] active:scale-[0.98] transition-all shadow-xl shadow-neutral-200 dark:shadow-zinc-900"
                            >
                                Start Session
                            </button>
                        </div>
                    ) : (
                        <RecordingView
                            selectedCameras={selectedCameras}
                            cameraConfigs={cameraConfigs}
                            isRecordingEpisode={isRecordingEpisode}
                            isSavingEpisode={isSavingEpisode}
                            timer={timer}
                            episodeCount={episodeCount}
                            error={error}
                            toggleEpisode={toggleEpisode}
                            stopSession={stopSession}
                            setEpisodeCount={setEpisodeCount}
                        />
                    )}
                </div>
            )}


        </motion.div>
    );
}
