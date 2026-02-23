import React from 'react';
import { Play, StopCircle, User, Bot, RefreshCw, Loader2, SkipForward, Pause } from 'lucide-react';
import { HILStatus } from './types';
import CameraFeed from '../../ui/CameraFeed';
import { camerasApi } from '../../../lib/api';

interface HILRunningViewProps {
    status: HILStatus | null;
    activeCameras: any[];
    isProcessing: boolean;
    toggleEpisode: () => void;
    nextEpisode: () => void;
    resumeAutonomous: () => void;
    stopEpisode: () => void;
    stopSession: () => void;
    triggerRetrain: () => void;
}

export default function HILRunningView({
    status,
    activeCameras,
    isProcessing,
    toggleEpisode,
    nextEpisode,
    resumeAutonomous,
    stopEpisode,
    stopSession,
    triggerRetrain,
}: HILRunningViewProps) {
    return (
        /* RUNNING VIEW */
        <div className="flex h-full gap-5">
            {/* Left: Camera Feed - Only show policy-relevant cameras */}
            <div className="flex-1 flex flex-col gap-4">
                <div className="flex-1 bg-black rounded-2xl overflow-hidden relative min-h-0">
                    <div className="grid h-full gap-0.5" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${activeCameras.length <= 1 ? '100%' : '300px'}, 1fr))` }}>
                        {activeCameras.map(cam => (
                            <div key={cam.id} className="relative w-full h-full overflow-hidden">
                                <CameraFeed
                                    cameraId={cam.id}
                                    mode="contain"
                                    className="rounded-none border-0"
                                />
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
    );
}
