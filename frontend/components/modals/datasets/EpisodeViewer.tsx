import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Trash2, Play, Pause, SkipBack, SkipForward, Layers } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from 'recharts';
import { ColormapType } from '../../../utils/colormaps';
import DepthVideoCanvas from './DepthVideoCanvas';
import { API_BASE } from '../../../lib/api';

interface VideoMetadata {
    is_depth: boolean;
    from_timestamp?: number;
    to_timestamp?: number;
}

interface EpisodeDetail {
    index: number;
    length: number;
    actions: number[][];
    timestamps: number[];
    videos: Record<string, string>;
    video_metadata?: Record<string, VideoMetadata>;
    fps?: number;
}

interface CameraPair {
    rgb: { key: string; url: string; metadata?: VideoMetadata };
    depth?: { key: string; url: string; metadata?: VideoMetadata };
}

interface EpisodeViewerProps {
    episodeData: EpisodeDetail;
    onDeleteEpisode: () => void;
}

export default function EpisodeViewer({ episodeData, onDeleteEpisode }: EpisodeViewerProps) {
    // Video Playback
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentFrame, setCurrentFrame] = useState(0);
    const videoRefs = useRef<Record<string, HTMLVideoElement>>({});
    const primaryVideoRef = useRef<HTMLVideoElement | null>(null);
    const chartRef = useRef<HTMLDivElement>(null);

    // Depth visualization
    const [depthColormap, setDepthColormap] = useState<ColormapType>('turbo');

    // Video timestamp offset tracking for concatenated video files
    // LeRobot v3 stores multiple episodes in one file - need to offset seeks
    const [videoTimestampOffsets, setVideoTimestampOffsets] = useState<Record<string, number>>({});

    // FPS for time-to-frame conversion (from API or default 30)
    const fps = episodeData?.fps || 30;

    // Helper: Pair RGB cameras with their depth counterparts
    const pairCamerasWithDepth = useCallback((
        videos: Record<string, string>,
        metadata?: Record<string, VideoMetadata>
    ): CameraPair[] => {
        const pairs: CameraPair[] = [];
        const depthMap = new Map<string, { key: string; url: string; metadata?: VideoMetadata }>();

        // First pass: identify depth videos
        Object.entries(videos).forEach(([key, url]) => {
            const isDepth = metadata?.[key]?.is_depth || key.includes('_depth');
            if (isDepth) {
                // Extract base camera name (e.g., "camera_1" from "observation.images.camera_1_depth")
                const baseName = key.replace('observation.images.', '').replace('_depth', '');
                depthMap.set(baseName, { key, url, metadata: metadata?.[key] });
            }
        });

        // Second pass: create pairs (RGB with optional depth)
        Object.entries(videos).forEach(([key, url]) => {
            const isDepth = metadata?.[key]?.is_depth || key.includes('_depth');
            if (!isDepth) {
                const baseName = key.replace('observation.images.', '');
                pairs.push({
                    rgb: { key, url, metadata: metadata?.[key] },
                    depth: depthMap.get(baseName),
                });
            }
        });

        return pairs;
    }, []);

    // Check if any depth data exists
    const hasDepthData = episodeData?.video_metadata
        ? Object.values(episodeData.video_metadata).some(m => m.is_depth)
        : false;

    // Reset current frame and update video timestamp offsets when episode changes
    useEffect(() => {
        setCurrentFrame(0);
        setIsPlaying(false);

        // Extract from_timestamp for each video from episode metadata
        // LeRobot v3 stores episodes in concatenated files - need these offsets for seeking
        if (episodeData?.video_metadata) {
            const offsets: Record<string, number> = {};
            Object.entries(episodeData.video_metadata).forEach(([key, meta]) => {
                offsets[key] = meta.from_timestamp || 0;
            });
            setVideoTimestampOffsets(offsets);
        } else {
            setVideoTimestampOffsets({});
        }
    }, [episodeData]);

    // Space bar to play/pause
    useEffect(() => {
        if (!episodeData) return;

        const handleKeyDown = (e: KeyboardEvent) => {
            // Only handle space bar
            if (e.code === 'Space' || e.key === ' ') {
                // Don't trigger if user is typing in an input
                const target = e.target as HTMLElement;
                if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;

                e.preventDefault(); // Prevent page scroll
                // Toggle play/pause
                setIsPlaying(prev => {
                    if (!prev) {
                        // Starting playback - seek to current frame with offset
                        // LeRobot v3: videos are concatenated, need from_timestamp offset
                        const episodeTime = currentFrame / fps;
                        Object.entries(videoRefs.current).forEach(([key, v]) => {
                            if (v) {
                                const fromTs = videoTimestampOffsets[key] || 0;
                                v.currentTime = fromTs + episodeTime;
                            }
                        });
                    }
                    return !prev;
                });
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [episodeData, currentFrame, fps, videoTimestampOffsets]);

    // Handle video time updates - sync cursor with video playback
    // Must account for from_timestamp offset since LeRobot v3 concatenates episodes
    const handleTimeUpdate = useCallback((e: Event) => {
        const video = e.target as HTMLVideoElement;
        if (video && episodeData) {
            // Get the primary video's from_timestamp offset
            const primaryKey = Object.keys(episodeData.videos || {})[0];
            const fromTs = primaryKey ? (videoTimestampOffsets[primaryKey] || 0) : 0;

            // Convert absolute video time to episode-relative time
            const episodeTime = video.currentTime - fromTs;
            const frame = Math.floor(episodeTime * fps);
            const clampedFrame = Math.max(0, Math.min(frame, episodeData.length - 1));
            setCurrentFrame(clampedFrame);
        }
    }, [fps, episodeData, videoTimestampOffsets]);

    // Set up primary video ref and time update listener
    useEffect(() => {
        const videos = Object.values(videoRefs.current);
        if (videos.length > 0) {
            primaryVideoRef.current = videos[0];
            const primary = primaryVideoRef.current;
            if (primary) {
                primary.addEventListener('timeupdate', handleTimeUpdate);
                return () => {
                    primary.removeEventListener('timeupdate', handleTimeUpdate);
                };
            }
        }
    }, [episodeData, handleTimeUpdate]);

    // Sync all videos play/pause state
    useEffect(() => {
        const refs = Object.values(videoRefs.current);
        refs.forEach(v => {
            if (!v) return;
            if (isPlaying) v.play().catch(() => { });
            else v.pause();
        });
    }, [isPlaying]);

    // Seek all videos to a specific frame
    // Must account for from_timestamp offset since LeRobot v3 concatenates episodes
    const seekToFrame = useCallback((frame: number) => {
        if (!episodeData) return;
        const clampedFrame = Math.max(0, Math.min(frame, episodeData.length - 1));
        const episodeTime = clampedFrame / fps;

        setCurrentFrame(clampedFrame);

        // Seek each video with its specific from_timestamp offset
        Object.entries(videoRefs.current).forEach(([key, v]) => {
            if (v) {
                const fromTs = videoTimestampOffsets[key] || 0;
                v.currentTime = fromTs + episodeTime;
            }
        });
    }, [fps, episodeData, videoTimestampOffsets]);

    // Handle click on chart to seek
    const handleChartClick = useCallback((e: any) => {
        if (e && e.activeLabel !== undefined) {
            seekToFrame(e.activeLabel);
        }
    }, [seekToFrame]);

    // Handle slider change
    const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const frame = parseInt(e.target.value);
        seekToFrame(frame);
    };

    // Skip forward/backward
    const skipFrames = (delta: number) => {
        seekToFrame(currentFrame + delta);
    };

    // Toggle play/pause and start from current frame
    const togglePlay = () => {
        if (!isPlaying) {
            seekToFrame(currentFrame);
        }
        setIsPlaying(!isPlaying);
    };

    // Prepare chart data
    const chartData = episodeData?.actions.map((a, i) => ({
        frame: i,
        ...a.reduce((acc, v, j) => ({ ...acc, [`j${j}`]: v }), {})
    })) || [];

    return (
        <>
            {/* Toolbar */}
            <div className="h-12 shrink-0 border-b border-neutral-200/50 dark:border-zinc-700/50 flex items-center justify-between px-5 bg-white/70 dark:bg-zinc-800/70">
                <div className="flex items-center gap-4">
                    <span className="text-sm text-neutral-700 dark:text-zinc-200 font-semibold">Episode {episodeData.index}</span>
                    <span className="text-xs text-neutral-400 dark:text-zinc-500 bg-neutral-100 dark:bg-zinc-700 px-2 py-1 rounded-lg">{episodeData.length} Frames @ {fps}fps</span>
                    <span className="text-xs text-emerald-600 dark:text-emerald-400 font-mono bg-emerald-50 dark:bg-emerald-950 px-2 py-1 rounded-lg">Frame {currentFrame}</span>
                </div>
                <div className="flex items-center gap-1">
                    <button onClick={() => skipFrames(-10)} className="p-2 hover:bg-neutral-100 dark:hover:bg-zinc-700 rounded-lg text-neutral-500 dark:text-zinc-400 transition-colors" title="Back 10 frames">
                        <SkipBack className="w-4 h-4" />
                    </button>
                    <button onClick={togglePlay} className="p-2 hover:bg-neutral-100 dark:hover:bg-zinc-700 rounded-lg text-neutral-700 dark:text-zinc-300 transition-colors">
                        {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    </button>
                    <button onClick={() => skipFrames(10)} className="p-2 hover:bg-neutral-100 dark:hover:bg-zinc-700 rounded-lg text-neutral-500 dark:text-zinc-400 transition-colors" title="Forward 10 frames">
                        <SkipForward className="w-4 h-4" />
                    </button>

                    {/* Colormap selector - only show when depth data exists */}
                    {hasDepthData && (
                        <>
                            <div className="w-px h-5 bg-neutral-200 dark:bg-zinc-700 mx-2" />
                            <div className="flex items-center gap-2 bg-blue-50 dark:bg-blue-950/50 px-3 py-1.5 rounded-lg border border-blue-200 dark:border-blue-800">
                                <Layers className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
                                <span className="text-xs text-blue-600 dark:text-blue-400 font-medium">Depth:</span>
                                <select
                                    value={depthColormap}
                                    onChange={e => setDepthColormap(e.target.value as ColormapType)}
                                    className="text-xs bg-white dark:bg-zinc-800 border border-blue-200 dark:border-blue-700 rounded px-2 py-0.5 text-blue-700 dark:text-blue-300 font-medium focus:outline-none focus:ring-2 focus:ring-blue-300 dark:focus:ring-blue-700"
                                >
                                    <option value="turbo">Turbo</option>
                                    <option value="viridis">Viridis</option>
                                    <option value="grayscale">Grayscale</option>
                                </select>
                            </div>
                        </>
                    )}

                    <div className="w-px h-5 bg-neutral-200 dark:bg-zinc-700 mx-2" />
                    <button onClick={onDeleteEpisode} className="p-2 hover:bg-red-50 dark:hover:bg-red-950/50 hover:text-red-500 rounded-lg text-neutral-400 dark:text-zinc-500 transition-colors" title="Delete episode">
                        <Trash2 className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Video Grid - Paired RGB + Depth Layout */}
            <div className="flex-1 min-h-0 overflow-y-auto bg-neutral-100/50 dark:bg-zinc-800/50">
              <div className="p-4 grid grid-cols-2 gap-4">
                {pairCamerasWithDepth(episodeData.videos || {}, episodeData.video_metadata).map(({ rgb, depth }) => (
                    <div key={rgb.key} className="space-y-3">
                        {/* RGB Video */}
                        <div className="relative aspect-video bg-neutral-900 rounded-2xl border border-neutral-200 dark:border-zinc-700 overflow-hidden shadow-lg group">
                            <video
                                src={`${API_BASE}${rgb.url}`}
                                ref={el => { if (el) videoRefs.current[rgb.key] = el }}
                                className="w-full h-full object-contain"
                                loop
                                muted
                                playsInline
                                preload="auto"
                                onLoadedMetadata={(e) => {
                                    // Seek to correct position for concatenated videos
                                    const video = e.target as HTMLVideoElement;
                                    const fromTs = rgb.metadata?.from_timestamp;
                                    if (fromTs && fromTs > 0) {
                                        video.currentTime = fromTs;
                                    }
                                }}
                            />
                            <div className="absolute top-3 left-3 bg-black/60 px-3 py-1 rounded-lg text-xs text-white font-medium backdrop-blur-sm">
                                {rgb.key.replace('observation.images.', '')}
                            </div>
                        </div>

                        {/* Depth Video (if available) */}
                        {depth && (
                            <div className="relative aspect-video bg-neutral-900 rounded-2xl border-2 border-blue-500/30 overflow-hidden shadow-lg group">
                                <DepthVideoCanvas
                                    videoUrl={`${API_BASE}${depth.url}`}
                                    videoRef={el => { if (el) videoRefs.current[depth.key] = el }}
                                    colormap={depthColormap}
                                    isPlaying={isPlaying}
                                    fromTimestamp={depth.metadata?.from_timestamp}
                                />
                                <div className="absolute top-3 left-3 bg-blue-600/80 px-3 py-1 rounded-lg text-xs text-white font-medium backdrop-blur-sm flex items-center gap-1.5">
                                    <Layers className="w-3 h-3" />
                                    Depth
                                </div>
                            </div>
                        )}
                    </div>
                ))}
              </div>
            </div>

            {/* Telemetry Graph with Playhead */}
            <div className="h-44 shrink-0 border-t border-neutral-200/50 dark:border-zinc-700/50 bg-white/80 dark:bg-zinc-900/80 p-3" ref={chartRef}>
                <div className="h-full w-full cursor-crosshair bg-neutral-50 dark:bg-zinc-800 rounded-xl border border-neutral-200/50 dark:border-zinc-700/50">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                            data={chartData}
                            onClick={handleChartClick}
                            margin={{ top: 10, right: 10, left: 10, bottom: 5 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                            <XAxis
                                dataKey="frame"
                                tick={{ fill: '#999', fontSize: 10 }}
                                tickLine={{ stroke: '#ddd' }}
                                axisLine={{ stroke: '#ddd' }}
                            />
                            <YAxis hide domain={['auto', 'auto']} />
                            <Tooltip
                                contentStyle={{ backgroundColor: 'white', border: '1px solid #e5e5e5', borderRadius: '8px', fontSize: '12px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                itemStyle={{ color: '#666' }}
                                labelFormatter={(label) => `Frame ${label}`}
                            />
                            {/* Playhead - vertical line showing current frame */}
                            <ReferenceLine
                                x={currentFrame}
                                stroke="#10b981"
                                strokeWidth={2}
                                strokeDasharray="none"
                            />
                            {episodeData.actions[0]?.map((_, i) => (
                                <Line
                                    key={i}
                                    type="monotone"
                                    dataKey={`j${i}`}
                                    stroke={`hsl(${i * 25}, 65%, 55%)`}
                                    dot={false}
                                    strokeWidth={1.5}
                                    isAnimationActive={false}
                                />
                            ))}
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Timeline Scrubber */}
            <div className="h-14 shrink-0 border-t border-neutral-200/50 dark:border-zinc-700/50 bg-white/90 dark:bg-zinc-900/90 px-5 flex items-center gap-4">
                <span className="text-xs text-neutral-500 dark:text-zinc-400 w-14 text-right font-mono bg-neutral-100 dark:bg-zinc-800 px-2 py-1 rounded">
                    {currentFrame}
                </span>
                <div className="flex-1 relative">
                    <input
                        type="range"
                        min={0}
                        max={episodeData.length - 1}
                        value={currentFrame}
                        onChange={handleSliderChange}
                        className="w-full h-2 bg-neutral-200 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                        style={{
                            background: `linear-gradient(to right, #10b981 0%, #10b981 ${(currentFrame / (episodeData.length - 1)) * 100}%, #e5e5e5 ${(currentFrame / (episodeData.length - 1)) * 100}%, #e5e5e5 100%)`
                        }}
                    />
                </div>
                <span className="text-xs text-neutral-500 dark:text-zinc-400 w-14 font-mono bg-neutral-100 dark:bg-zinc-800 px-2 py-1 rounded">
                    {episodeData.length - 1}
                </span>
            </div>
        </>
    );
}
