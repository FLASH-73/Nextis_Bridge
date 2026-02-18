import React from 'react';
import { StopCircle, Loader2, Save, Folder, Trash2 } from 'lucide-react';
import CameraFeed from '../../ui/CameraFeed';
import { recordingApi, camerasApi } from '../../../lib/api';

interface RecordingViewProps {
    selectedCameras: string[];
    cameraConfigs: any[];
    isRecordingEpisode: boolean;
    isSavingEpisode: boolean;
    timer: number;
    episodeCount: number;
    error: string;
    toggleEpisode: () => void;
    stopSession: () => void;
    setEpisodeCount: React.Dispatch<React.SetStateAction<number>>;
}

export default function RecordingView({
    selectedCameras,
    cameraConfigs,
    isRecordingEpisode,
    isSavingEpisode,
    timer,
    episodeCount,
    error,
    toggleEpisode,
    stopSession,
    setEpisodeCount,
}: RecordingViewProps) {
    return (
        <div className="flex h-full gap-6">
            {/* Left: Camera Feed & Controls */}
            <div className="flex-1 flex flex-col gap-6">
                {/* Camera Grid */}
                <div className="flex-1 bg-black rounded-3xl overflow-hidden relative shadow-inner group">
                    {/* Show selected cameras (or all if selection not loaded yet) */}
                    {(() => {
                        const camerasToDisplay = selectedCameras.length === 0
                            ? cameraConfigs
                            : cameraConfigs.filter(cam => selectedCameras.includes(cam.id));

                        // Dynamic grid layout based on camera count
                        const getGridClasses = (count: number) => {
                            switch (count) {
                                case 1: return 'grid-cols-1';
                                case 2: return 'grid-cols-2';
                                case 3: return 'grid-cols-3';
                                case 4: return 'grid-cols-2 grid-rows-2';
                                case 5:
                                case 6: return 'grid-cols-3 grid-rows-2';
                                default: return 'grid-cols-3';
                            }
                        };

                        return (
                            <div className={`grid h-full ${getGridClasses(camerasToDisplay.length)} gap-1`}>
                                {camerasToDisplay.map(cam => (
                                    <div
                                        key={cam.id}
                                        className="relative bg-black overflow-hidden"
                                    >
                                        <CameraFeed
                                            cameraId={cam.id}
                                            maxStreamWidth={800}
                                            mode="contain"
                                            className="rounded-none border-0"
                                        />
                                    </div>
                                ))}
                            </div>
                        );
                    })()}

                    {/* Recording Overlay Indicator */}
                    {isRecordingEpisode && (
                        <div className="absolute top-6 right-6 flex items-center gap-3 bg-red-500/90 backdrop-blur-md text-white px-4 py-2 rounded-full shadow-lg animate-in fade-in">
                            <div className="w-3 h-3 bg-white rounded-full animate-pulse" />
                            <span className="font-mono font-bold tracking-widest text-lg">{timer.toFixed(1)}s</span>
                        </div>
                    )}
                </div>

                {/* Control Bar */}
                <div className="h-24 bg-white dark:bg-zinc-900 border border-neutral-100 dark:border-zinc-800 rounded-3xl shadow-lg flex items-center justify-between px-8 flex-none">
                    <div className="flex flex-col">
                        <span className="text-xs font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-widest">Episode</span>
                        <span className="text-3xl font-bold text-neutral-900 dark:text-zinc-100">#{episodeCount + 1}</span>
                        {isSavingEpisode && (
                            <span className="text-xs text-amber-600 font-medium animate-pulse">Saving...</span>
                        )}
                    </div>

                    <div className="flex items-center gap-6">
                        <button
                            onClick={toggleEpisode}
                            disabled={isSavingEpisode}
                            className={`w-16 h-16 rounded-full flex items-center justify-center transition-all shadow-xl ${
                                isSavingEpisode
                                    ? 'bg-neutral-200 border-4 border-neutral-300 cursor-not-allowed opacity-60'
                                    : isRecordingEpisode
                                        ? 'bg-white border-4 border-red-500 hover:scale-105 active:scale-95'
                                        : 'bg-red-500 border-4 border-red-100 hover:scale-105 active:scale-95'
                            }`}
                        >
                            {isSavingEpisode ? (
                                <Loader2 className="w-6 h-6 text-neutral-400 animate-spin" />
                            ) : isRecordingEpisode ? (
                                <div className="w-6 h-6 bg-red-500 rounded-md" />
                            ) : (
                                <div className="w-6 h-6 bg-white rounded-full" />
                            )}
                        </button>
                    </div>

                    <div className="flex flex-col items-end">
                        <button
                            onClick={stopSession}
                            className="px-4 py-2 hover:bg-neutral-100 dark:hover:bg-zinc-800 rounded-lg text-sm text-neutral-500 dark:text-zinc-400 font-medium transition-colors flex items-center gap-2"
                        >
                            <Save className="w-4 h-4" /> Finish Session
                        </button>
                    </div>
                </div>
            </div>

            {/* Right: Sidebar */}
            <div className="w-64 flex flex-col gap-4">
                <div className="flex-1 bg-neutral-50 dark:bg-zinc-800/50 rounded-3xl border border-neutral-100 dark:border-zinc-700 p-4 overflow-y-auto">
                    <h3 className="font-bold text-neutral-400 dark:text-zinc-500 text-xs uppercase tracking-widest mb-4 flex items-center gap-2">
                        <Folder className="w-3.5 h-3.5" /> Library
                    </h3>
                    <div className="space-y-2">
                        {[...Array(episodeCount)].map((_, i) => (
                            <div key={i} className="flex items-center justify-between p-3 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-xl shadow-sm">
                                <div className="flex items-center gap-3">
                                    <div className="w-8 h-8 bg-neutral-100 dark:bg-zinc-800 rounded-lg flex items-center justify-center text-xs font-bold text-neutral-500 dark:text-zinc-400">
                                        #{i + 1}
                                    </div>
                                    <div className="flex flex-col">
                                        <span className="text-xs font-semibold text-neutral-800 dark:text-zinc-200">Episode {i + 1}</span>
                                        <span className="text-[10px] text-neutral-400 dark:text-zinc-500">Recorded</span>
                                    </div>
                                </div>
                                <button
                                    onClick={async () => {
                                        if (i !== episodeCount - 1) {
                                            alert("Can only delete the last episode!");
                                            return;
                                        }
                                        if (!confirm("Delete this episode? Files may persist on disk.")) return;
                                        try {
                                            await recordingApi.deleteLastEpisode();
                                            setEpisodeCount(c => Math.max(0, c - 1));
                                        } catch (e) { console.error(e); }
                                    }}
                                    className="text-neutral-300 dark:text-zinc-600 hover:text-red-500 transition-colors"
                                >
                                    <Trash2 className="w-3.5 h-3.5" />
                                </button>
                            </div>
                        ))}
                        {episodeCount === 0 && (
                            <div className="text-center py-8 text-neutral-400 dark:text-zinc-500 text-xs">
                                No episodes recorded yet.
                            </div>
                        )}
                    </div>
                </div>

                {/* Teleop Status Mini */}
                <div className="h-32 bg-white dark:bg-zinc-900 rounded-3xl border border-neutral-100 dark:border-zinc-800 p-4 shadow-sm relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-purple-500" />
                    <h3 className="font-bold text-neutral-900 dark:text-zinc-100 text-sm mb-1">Teleoperation</h3>
                    <p className="text-xs text-neutral-500 dark:text-zinc-400 mb-3">Service is active and syncing.</p>
                    <div className="flex items-center gap-2">
                        <div className="flex items-center gap-1.5 px-2 py-1 bg-green-50 dark:bg-green-950/50 text-green-700 dark:text-green-400 text-[10px] font-bold rounded-md border border-green-100 dark:border-green-900">
                            <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" /> ONLINE
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
