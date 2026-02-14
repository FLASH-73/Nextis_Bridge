import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Trash2, Film, Play, Pause, Folder, ChevronRight, HardDrive, SkipBack, SkipForward, Database, CheckSquare, Square, Layers, Cloud, Loader2, CheckCircle, AlertCircle, Brain, Rocket, RotateCcw, Pencil, Check } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine, Legend } from 'recharts';
import { TURBO_COLORMAP, VIRIDIS_COLORMAP, applyColormap, ColormapType } from '../utils/colormaps';
import { useAuth } from '../lib/AuthContext';
import { supabase, isSupabaseConfigured } from '../lib/supabase';
import MergeModal from './MergeModal';

const API_BASE = "http://localhost:8000";

// Policy types
interface PolicyInfo {
    id: string;
    name: string;
    policy_type: string;
    status: "completed" | "training" | "failed";
    steps: number;
    total_steps: number;
    dataset_repo_id: string;
    created_at: string;
    final_loss: number | null;
    checkpoint_path: string;
    loss_history: [number, number][]; // [[step, loss], ...]
    output_dir: string;
}

type TabType = "datasets" | "policies";

interface DatasetInfo {
    repo_id: string;
    root: string;
    total_episodes: number;
    total_frames: number;
    fps?: number;
}

interface EpisodeSummary {
    index: number;
    episode_index?: number;
    length?: number;
    timestamp?: number;
}

interface VideoMetadata {
    is_depth: boolean;
    from_timestamp?: number;  // Start position in concatenated video
    to_timestamp?: number;    // End position in concatenated video
}

interface EpisodeDetail {
    index: number;
    length: number;
    actions: number[][]; // [frames][joints]
    timestamps: number[];
    videos: Record<string, string>; // key -> url
    video_metadata?: Record<string, VideoMetadata>; // key -> metadata
    fps?: number;
}

interface CameraPair {
    rgb: { key: string; url: string; metadata?: VideoMetadata };
    depth?: { key: string; url: string; metadata?: VideoMetadata };
}

/**
 * DepthVideoCanvas - Renders depth video with colormap applied via canvas
 */
function DepthVideoCanvas({
    videoUrl,
    videoRef,
    colormap,
    isPlaying,
    fromTimestamp,
}: {
    videoUrl: string;
    videoRef: (el: HTMLVideoElement | null) => void;
    colormap: ColormapType;
    isPlaying: boolean;
    fromTimestamp?: number;
}) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const hiddenVideoRef = useRef<HTMLVideoElement>(null);
    const animationRef = useRef<number | null>(null);

    // Get the appropriate colormap LUT
    const getLUT = useCallback(() => {
        if (colormap === 'turbo') return TURBO_COLORMAP;
        if (colormap === 'viridis') return VIRIDIS_COLORMAP;
        return null; // grayscale
    }, [colormap]);

    // Render a single frame with colormap
    const renderFrame = useCallback(() => {
        const video = hiddenVideoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || video.readyState < 2) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Match canvas size to video
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }

        // Draw video frame
        ctx.drawImage(video, 0, 0);

        // Apply colormap if not grayscale
        const lut = getLUT();
        if (lut) {
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;

            for (let i = 0; i < data.length; i += 4) {
                const gray = data[i]; // R channel (grayscale video)
                const [r, g, b] = applyColormap(gray, lut);
                data[i] = r;
                data[i + 1] = g;
                data[i + 2] = b;
                // Alpha stays unchanged
            }

            ctx.putImageData(imageData, 0, 0);
        }
    }, [getLUT]);

    // Animation loop for playing video
    useEffect(() => {
        const video = hiddenVideoRef.current;
        if (!video) return;

        const animate = () => {
            if (!video.paused && !video.ended) {
                renderFrame();
                animationRef.current = requestAnimationFrame(animate);
            }
        };

        if (isPlaying) {
            // Actually play the hidden video and start animation loop
            video.play().then(() => {
                animate();
            }).catch(e => {
                console.warn('Depth video play failed:', e);
            });
        } else {
            // Pause the video and render current frame
            video.pause();
            renderFrame();
        }

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [isPlaying, renderFrame]);

    // Re-render when colormap changes
    useEffect(() => {
        renderFrame();
    }, [colormap, renderFrame]);

    // Handle video events
    useEffect(() => {
        const video = hiddenVideoRef.current;
        if (!video) return;

        const handleSeeked = () => renderFrame();
        const handleLoadedData = () => {
            // Seek to correct position for concatenated videos
            if (fromTimestamp && fromTimestamp > 0) {
                video.currentTime = fromTimestamp;
            }
            renderFrame();
        };

        video.addEventListener('seeked', handleSeeked);
        video.addEventListener('loadeddata', handleLoadedData);

        return () => {
            video.removeEventListener('seeked', handleSeeked);
            video.removeEventListener('loadeddata', handleLoadedData);
        };
    }, [renderFrame, fromTimestamp]);

    return (
        <div className="relative w-full h-full">
            <video
                ref={el => {
                    hiddenVideoRef.current = el;
                    videoRef(el);
                }}
                src={videoUrl}
                className="hidden"
                muted
                loop
                playsInline
                preload="auto"
                crossOrigin="anonymous"
            />
            <canvas
                ref={canvasRef}
                className="w-full h-full object-contain"
            />
        </div>
    );
}

interface DatasetViewerModalProps {
    isOpen: boolean;
    onClose: () => void;
    maximizedWindow: string | null;
    setMaximizedWindow: (window: string | null) => void;
}

export default function DatasetViewerModal({ isOpen, onClose, maximizedWindow, setMaximizedWindow }: DatasetViewerModalProps) {
    const { user, session } = useAuth();

    // Tab state
    const [activeTab, setActiveTab] = useState<TabType>("datasets");

    const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
    const [selectedRepo, setSelectedRepo] = useState<string | null>(null);
    const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
    const [selectedEpisodeIndex, setSelectedEpisodeIndex] = useState<number | null>(null);
    const [episodeData, setEpisodeData] = useState<EpisodeDetail | null>(null);

    const [isLoadingRepo, setIsLoadingRepo] = useState(false);
    const [isLoadingEpisode, setIsLoadingEpisode] = useState(false);

    // Multi-select state
    const [selectedRepos, setSelectedRepos] = useState<Set<string>>(new Set());
    const [lastClickedRepoIndex, setLastClickedRepoIndex] = useState<number | null>(null);

    // Merge modal state
    const [isMergeModalOpen, setIsMergeModalOpen] = useState(false);

    // Policy state
    const [policies, setPolicies] = useState<PolicyInfo[]>([]);
    const [selectedPolicies, setSelectedPolicies] = useState<Set<string>>(new Set());
    const [isLoadingPolicies, setIsLoadingPolicies] = useState(false);
    const [editingPolicyId, setEditingPolicyId] = useState<string | null>(null);
    const [editingPolicyName, setEditingPolicyName] = useState("");
    const [isDeploying, setIsDeploying] = useState(false);
    const [showResumeModal, setShowResumeModal] = useState(false);
    const [resumeSteps, setResumeSteps] = useState(10000);
    const [resumingPolicyId, setResumingPolicyId] = useState<string | null>(null);

    // Upload state
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadingFileName, setUploadingFileName] = useState<string>('');
    const [uploadError, setUploadError] = useState<string | null>(null);
    const [uploadSuccess, setUploadSuccess] = useState(false);
    const [showUploadConfirm, setShowUploadConfirm] = useState(false);
    const [uploadingDataset, setUploadingDataset] = useState<DatasetInfo | null>(null);

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

    // Window state
    const isMaximized = maximizedWindow === 'datasetViewer';

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

    // Fetch Datasets on Open
    useEffect(() => {
        if (isOpen) fetchDatasets();
    }, [isOpen]);

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
        if (!isOpen || !episodeData) return;

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
    }, [isOpen, episodeData, currentFrame, fps, videoTimestampOffsets]);

    const fetchDatasets = async () => {
        try {
            const res = await fetch(`${API_BASE}/datasets`);
            const data = await res.json();
            setDatasets(data);
        } catch (e) {
            console.error("Failed to load datasets", e);
        }
    };

    // Policy functions
    const fetchPolicies = async () => {
        setIsLoadingPolicies(true);
        try {
            const res = await fetch(`${API_BASE}/policies`);
            const data = await res.json();
            setPolicies(data);
        } catch (e) {
            console.error("Failed to load policies", e);
        } finally {
            setIsLoadingPolicies(false);
        }
    };

    const handlePolicyClick = (policyId: string, e: React.MouseEvent) => {
        if (e.ctrlKey || e.metaKey) {
            // Toggle selection for comparison
            const newSelected = new Set(selectedPolicies);
            if (newSelected.has(policyId)) {
                newSelected.delete(policyId);
            } else {
                newSelected.add(policyId);
            }
            setSelectedPolicies(newSelected);
        } else {
            // Single select
            setSelectedPolicies(new Set([policyId]));
        }
    };

    const deletePolicy = async (policyId: string) => {
        if (!confirm(`Are you sure you want to delete this policy?\n\n${policyId}\n\nThis cannot be undone!`)) return;

        try {
            const res = await fetch(`${API_BASE}/policies/${policyId}`, { method: 'DELETE' });
            const result = await res.json();
            if (result.status === 'deleted') {
                setSelectedPolicies(prev => {
                    const next = new Set(prev);
                    next.delete(policyId);
                    return next;
                });
                fetchPolicies();
            } else {
                alert(`Failed to delete: ${result.error || 'Unknown error'}`);
            }
        } catch (e) {
            console.error("Delete failed:", e);
            alert("Failed to delete policy");
        }
    };

    const renamePolicy = async (policyId: string, newName: string) => {
        try {
            const res = await fetch(`${API_BASE}/policies/${policyId}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: newName })
            });
            const result = await res.json();
            if (result.status === 'updated') {
                setEditingPolicyId(null);
                fetchPolicies();
            } else {
                alert(`Failed to rename: ${result.error || 'Unknown error'}`);
            }
        } catch (e) {
            console.error("Rename failed:", e);
            alert("Failed to rename policy");
        }
    };

    const deployPolicy = async (policyId: string) => {
        setIsDeploying(true);
        try {
            const res = await fetch(`${API_BASE}/policies/${policyId}/deploy`, { method: 'POST' });
            const result = await res.json();
            if (result.status === 'deployed' || result.status === 'ready') {
                alert(`Policy deployed successfully!\n\nCheckpoint: ${result.checkpoint_path}`);
            } else {
                alert(`Failed to deploy: ${result.error || 'Unknown error'}`);
            }
        } catch (e) {
            console.error("Deploy failed:", e);
            alert("Failed to deploy policy");
        } finally {
            setIsDeploying(false);
        }
    };

    const resumeTraining = async (policyId: string, additionalSteps: number) => {
        try {
            const res = await fetch(`${API_BASE}/policies/${policyId}/resume`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ additional_steps: additionalSteps })
            });
            const result = await res.json();
            if (result.status === 'started') {
                alert(`Training resumed!\n\nJob ID: ${result.job_id}`);
                setShowResumeModal(false);
                fetchPolicies();
            } else {
                alert(`Failed to resume: ${result.error || 'Unknown error'}`);
            }
        } catch (e) {
            console.error("Resume failed:", e);
            alert("Failed to resume training");
        }
    };

    // Fetch policies when tab changes to policies
    useEffect(() => {
        if (isOpen && activeTab === "policies") {
            fetchPolicies();
        }
    }, [isOpen, activeTab]);

    // Get selected policies data for chart comparison
    const selectedPolicyData = policies.filter(p => selectedPolicies.has(p.id));
    const primarySelectedPolicy = selectedPolicyData.length > 0 ? selectedPolicyData[0] : null;

    const selectRepo = async (repo_id: string) => {
        setSelectedRepo(repo_id);
        setSelectedEpisodeIndex(null);
        setEpisodeData(null);
        setIsLoadingRepo(true);
        try {
            const res = await fetch(`${API_BASE}/datasets/${repo_id}/episodes`);
            const data = await res.json();
            if (Array.isArray(data)) {
                const sorted = data.sort((a, b) => (a.episode_index ?? a.index) - (b.episode_index ?? b.index));
                setEpisodes(sorted);
            }
        } catch (e) {
            console.error(e);
        } finally {
            setIsLoadingRepo(false);
        }
    };

    const selectEpisode = async (index: number) => {
        setSelectedEpisodeIndex(index);
        setIsLoadingEpisode(true);
        try {
            if (!selectedRepo) return;
            const res = await fetch(`${API_BASE}/datasets/${selectedRepo}/episode/${index}`);
            const data = await res.json();
            if (data.error) {
                console.error("Episode load error:", data.error);
                setEpisodeData(null);
            } else {
                setEpisodeData(data);
                setIsPlaying(false);
                setCurrentFrame(0);
            }
        } catch (e) {
            console.error(e);
            setEpisodeData(null);
        } finally {
            setIsLoadingEpisode(false);
        }
    };

    const deleteEpisode = async () => {
        if (selectedEpisodeIndex === null || !selectedRepo) return;
        if (!confirm(`Are you sure you want to delete Episode ${selectedEpisodeIndex}? This cannot be undone.`)) return;

        try {
            const res = await fetch(`${API_BASE}/datasets/${selectedRepo}/episode/${selectedEpisodeIndex}`, {
                method: 'DELETE'
            });
            const result = await res.json();
            if (result.status === 'success') {
                selectRepo(selectedRepo);
            } else {
                alert("Delete failed: " + result.message);
            }
        } catch (e) {
            console.error(e);
        }
    };

    const deleteDataset = async () => {
        if (!selectedRepo) return;
        if (!confirm(`Are you sure you want to delete the entire dataset "${selectedRepo}"? This will delete ALL episodes and cannot be undone!`)) return;

        try {
            const res = await fetch(`${API_BASE}/datasets/${selectedRepo}`, {
                method: 'DELETE'
            });
            const result = await res.json();
            if (result.status === 'success') {
                setSelectedRepo(null);
                setSelectedRepos(new Set());
                setEpisodes([]);
                setEpisodeData(null);
                fetchDatasets();
            } else {
                alert("Delete failed: " + (result.error || result.message));
            }
        } catch (e) {
            console.error(e);
        }
    };

    // Upload selected dataset to cloud - show confirmation
    const handleUploadToCloud = () => {
        if (!user || !session) {
            setUploadError('Please sign in to upload datasets');
            return;
        }
        if (!isSupabaseConfigured) {
            setUploadError('Supabase is not configured');
            return;
        }
        if (selectedRepos.size !== 1) {
            setUploadError('Please select exactly one dataset to upload');
            return;
        }

        const selectedRepoId = Array.from(selectedRepos)[0];
        const datasetInfo = datasets.find(d => d.repo_id === selectedRepoId);
        if (!datasetInfo) {
            setUploadError('Dataset not found');
            return;
        }

        setUploadingDataset(datasetInfo);
        setShowUploadConfirm(true);
        setUploadError(null);
        setUploadSuccess(false);
    };

    // Cancel upload confirmation
    const cancelUpload = () => {
        setShowUploadConfirm(false);
        setUploadingDataset(null);
    };

    // Confirm and start upload - fetch files from local API
    const confirmUpload = async () => {
        if (!uploadingDataset || !user) return;

        setShowUploadConfirm(false);
        setIsUploading(true);
        setUploadProgress(0);
        setUploadError(null);
        setUploadSuccess(false);

        try {
            // First, get the list of files from the local API
            setUploadingFileName('Fetching file list...');
            const filesRes = await fetch(`${API_BASE}/datasets/${uploadingDataset.repo_id}/files`);

            if (!filesRes.ok) {
                throw new Error('Failed to get file list. Make sure your local API supports the /files endpoint.');
            }

            const fileList: { path: string; size: number }[] = await filesRes.json();

            if (!fileList || fileList.length === 0) {
                throw new Error('No files found in dataset');
            }

            const userId = user.id;
            const datasetId = crypto.randomUUID();
            const basePath = `${userId}/${datasetId}`;
            const totalSize = fileList.reduce((sum, f) => sum + f.size, 0);
            let totalUploaded = 0;

            // Upload files sequentially
            for (const fileInfo of fileList) {
                setUploadingFileName(fileInfo.path);

                // Fetch file content from local API
                const fileRes = await fetch(`${API_BASE}/datasets/${uploadingDataset.repo_id}/file/${encodeURIComponent(fileInfo.path)}`);
                if (!fileRes.ok) {
                    throw new Error(`Failed to fetch ${fileInfo.path}`);
                }
                const fileBlob = await fileRes.blob();

                // Upload to Supabase
                const storagePath = `${basePath}/${fileInfo.path}`;
                const { error: uploadErr } = await supabase.storage
                    .from('datasets')
                    .upload(storagePath, fileBlob, {
                        cacheControl: '3600',
                        upsert: false,
                    });

                if (uploadErr) {
                    throw new Error(`Failed to upload ${fileInfo.path}: ${uploadErr.message}`);
                }

                totalUploaded += fileInfo.size;
                setUploadProgress(Math.round((totalUploaded / totalSize) * 100));
            }

            // Create database record
            const { error: dbError } = await supabase.from('datasets').insert({
                id: datasetId,
                owner_id: userId,
                name: uploadingDataset.repo_id,
                description: `Uploaded from local dataset: ${uploadingDataset.repo_id}`,
                status: 'ready',
                storage_path: basePath,
                file_size: totalSize,
                file_format: 'lerobot_v2',
                frame_count: uploadingDataset.total_frames,
            });

            if (dbError) {
                console.error('Database insert error:', dbError);
                throw new Error(`Database error: ${dbError.message}`);
            }

            setUploadSuccess(true);
            setUploadProgress(100);
        } catch (err) {
            console.error('Upload error:', err);
            setUploadError(err instanceof Error ? err.message : 'Upload failed');
        } finally {
            setIsUploading(false);
            setUploadingFileName('');
            setUploadingDataset(null);
        }
    };

    // Multi-select: Handle click with shift key for range selection
    const handleRepoClick = (repo_id: string, index: number, e: React.MouseEvent) => {
        if (e.shiftKey && lastClickedRepoIndex !== null) {
            // Range selection
            const start = Math.min(lastClickedRepoIndex, index);
            const end = Math.max(lastClickedRepoIndex, index);
            const newSelection = new Set(selectedRepos);
            for (let i = start; i <= end; i++) {
                newSelection.add(datasets[i].repo_id);
            }
            setSelectedRepos(newSelection);
        } else if (e.ctrlKey || e.metaKey) {
            // Toggle single item in selection
            const newSelection = new Set(selectedRepos);
            if (newSelection.has(repo_id)) {
                newSelection.delete(repo_id);
            } else {
                newSelection.add(repo_id);
            }
            setSelectedRepos(newSelection);
            setLastClickedRepoIndex(index);
        } else {
            // Normal click - select single repo and load it
            setSelectedRepos(new Set([repo_id]));
            setLastClickedRepoIndex(index);
            selectRepo(repo_id);
        }
    };

    // Delete all selected repos
    const deleteSelectedRepos = async () => {
        if (selectedRepos.size === 0) return;
        const repoList = Array.from(selectedRepos).join(', ');
        if (!confirm(`Are you sure you want to delete ${selectedRepos.size} dataset(s)?\n\n${repoList}\n\nThis cannot be undone!`)) return;

        let successCount = 0;
        let failCount = 0;

        for (const repo_id of selectedRepos) {
            try {
                const res = await fetch(`${API_BASE}/datasets/${repo_id}`, {
                    method: 'DELETE'
                });
                const result = await res.json();
                if (result.status === 'success') {
                    successCount++;
                } else {
                    failCount++;
                    console.error(`Failed to delete ${repo_id}:`, result);
                }
            } catch (e) {
                failCount++;
                console.error(`Error deleting ${repo_id}:`, e);
            }
        }

        // Reset state
        setSelectedRepo(null);
        setSelectedRepos(new Set());
        setEpisodes([]);
        setEpisodeData(null);
        fetchDatasets();

        if (failCount > 0) {
            alert(`Deleted ${successCount} dataset(s). ${failCount} failed.`);
        }
    };

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

    if (!isOpen) return null;

    // Prepare chart data
    const chartData = episodeData?.actions.map((a, i) => ({
        frame: i,
        ...a.reduce((acc, v, j) => ({ ...acc, [`j${j}`]: v }), {})
    })) || [];

    return (
        <>
        <AnimatePresence>
            <div className="fixed inset-0 flex items-center justify-center bg-black/20 backdrop-blur-sm" style={{ zIndex: isMaximized ? 100 : 50 }}>
                <motion.div
                    initial={{ opacity: 0, scale: 0.95, y: 20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: 20 }}
                    className={`bg-white/95 dark:bg-zinc-900/95 rounded-3xl border border-white/50 dark:border-zinc-700/50 shadow-2xl flex flex-col overflow-hidden backdrop-blur-3xl transition-all duration-300 ${isMaximized ? 'w-[calc(100vw-40px)] h-[calc(100vh-100px)]' : 'w-[90vw] h-[85vh]'}`}
                >
                    {/* Header */}
                    <div className="h-14 bg-white/50 dark:bg-zinc-800/50 border-b border-neutral-200/50 dark:border-zinc-700/50 flex items-center justify-between px-6">
                        <div className="flex items-center gap-3">
                            <div className="flex gap-2 mr-3">
                                <button onClick={onClose} className="w-3.5 h-3.5 rounded-full bg-[#FF5F57] hover:brightness-90 transition-all" />
                                <button className="w-3.5 h-3.5 rounded-full bg-[#FEBC2E] hover:brightness-90 transition-all" />
                                <button onClick={() => setMaximizedWindow(isMaximized ? null : 'datasetViewer')} className="w-3.5 h-3.5 rounded-full bg-[#28C840] hover:brightness-90 transition-all" />
                            </div>
                            {/* Tab Switcher */}
                            <div className="flex items-center gap-1 bg-neutral-100 dark:bg-zinc-800 rounded-xl p-1">
                                <button
                                    onClick={() => setActiveTab("datasets")}
                                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                                        activeTab === "datasets"
                                            ? "bg-white dark:bg-zinc-700 text-emerald-600 dark:text-emerald-400 shadow-sm"
                                            : "text-neutral-500 dark:text-zinc-400 hover:text-neutral-700 dark:hover:text-zinc-300"
                                    }`}
                                >
                                    <Database className="w-4 h-4" />
                                    Datasets
                                </button>
                                <button
                                    onClick={() => setActiveTab("policies")}
                                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                                        activeTab === "policies"
                                            ? "bg-white dark:bg-zinc-700 text-purple-600 dark:text-purple-400 shadow-sm"
                                            : "text-neutral-500 dark:text-zinc-400 hover:text-neutral-700 dark:hover:text-zinc-300"
                                    }`}
                                >
                                    <Brain className="w-4 h-4" />
                                    Policies
                                </button>
                            </div>
                        </div>
                        <button onClick={onClose} className="p-2 hover:bg-neutral-100 dark:hover:bg-zinc-800 rounded-xl transition-colors">
                            <X className="w-5 h-5 text-neutral-400 dark:text-zinc-500" />
                        </button>
                    </div>

                    {/* Content */}
                    <div className="flex flex-1 overflow-hidden">
                        {/* Sidebar */}
                        <div className="w-72 border-r border-neutral-200/50 dark:border-zinc-700/50 flex flex-col bg-neutral-50/50 dark:bg-zinc-800/50">
                            {activeTab === "datasets" ? (
                                <>
                            {/* Dataset List */}
                            <div className="p-4 border-b border-neutral-200/50 dark:border-zinc-700/50">
                                <div className="flex items-center justify-between mb-3">
                                    <h3 className="text-xs font-semibold text-neutral-500 dark:text-zinc-400 uppercase tracking-wider">
                                        Repositories {selectedRepos.size > 1 && <span className="text-emerald-600 dark:text-emerald-400">({selectedRepos.size})</span>}
                                    </h3>
                                    <div className="flex items-center gap-1">
                                        {selectedRepos.size === 1 && (
                                            <button
                                                onClick={handleUploadToCloud}
                                                disabled={isUploading}
                                                className="p-1.5 hover:bg-blue-100 rounded-lg text-blue-500 transition-colors disabled:opacity-50"
                                                title="Upload to Cloud"
                                            >
                                                {isUploading ? (
                                                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                ) : (
                                                    <Cloud className="w-3.5 h-3.5" />
                                                )}
                                            </button>
                                        )}
                                        {selectedRepos.size >= 2 && (
                                            <button
                                                onClick={() => setIsMergeModalOpen(true)}
                                                className="p-1.5 hover:bg-purple-100 dark:hover:bg-purple-900/30 rounded-lg text-purple-500 transition-colors"
                                                title={`Merge ${selectedRepos.size} datasets`}
                                            >
                                                <Layers className="w-3.5 h-3.5" />
                                            </button>
                                        )}
                                        {selectedRepos.size > 0 && (
                                            <button
                                                onClick={selectedRepos.size > 1 ? deleteSelectedRepos : deleteDataset}
                                                className="p-1.5 hover:bg-red-100 dark:hover:bg-red-900/30 rounded-lg text-red-500 transition-colors"
                                                title={selectedRepos.size > 1 ? `Delete ${selectedRepos.size} datasets` : "Delete dataset"}
                                            >
                                                <Trash2 className="w-3.5 h-3.5" />
                                            </button>
                                        )}
                                    </div>
                                </div>

                                {/* Upload Progress/Status */}
                                {(isUploading || uploadError || uploadSuccess) && (
                                    <div className={`mb-3 p-2.5 rounded-xl text-xs ${
                                        uploadError ? 'bg-red-50 dark:bg-red-950/50 border border-red-100 dark:border-red-900' :
                                        uploadSuccess ? 'bg-green-50 dark:bg-green-950/50 border border-green-100 dark:border-green-900' :
                                        'bg-blue-50 dark:bg-blue-950/50 border border-blue-100 dark:border-blue-900'
                                    }`}>
                                        {isUploading && (
                                            <div className="space-y-2">
                                                <div className="flex items-center gap-2 text-blue-600 dark:text-blue-400">
                                                    <Loader2 className="w-3 h-3 animate-spin" />
                                                    <span>Uploading to cloud...</span>
                                                </div>
                                                <div className="w-full bg-blue-200 dark:bg-blue-900 rounded-full h-1.5">
                                                    <div
                                                        className="bg-blue-500 h-1.5 rounded-full transition-all"
                                                        style={{ width: `${uploadProgress}%` }}
                                                    />
                                                </div>
                                                <p className="text-blue-500 dark:text-blue-400 truncate">{uploadingFileName}</p>
                                            </div>
                                        )}
                                        {uploadError && (
                                            <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
                                                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                                                <span>{uploadError}</span>
                                            </div>
                                        )}
                                        {uploadSuccess && !isUploading && (
                                            <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
                                                <CheckCircle className="w-3 h-3 flex-shrink-0" />
                                                <span>Upload complete!</span>
                                            </div>
                                        )}
                                    </div>
                                )}
                                <p className="text-[10px] text-neutral-400 dark:text-zinc-500 mb-2">Shift+click to select range, Ctrl+click to toggle</p>
                                <div className="space-y-1">
                                    {datasets.filter(ds => ds.repo_id).map((ds, index) => {
                                        const isSelected = selectedRepos.has(ds.repo_id);
                                        const isActive = selectedRepo === ds.repo_id;
                                        return (
                                            <button
                                                key={ds.repo_id || `dataset-${index}`}
                                                onClick={(e) => handleRepoClick(ds.repo_id, index, e)}
                                                className={`w-full text-left px-3 py-2.5 rounded-xl text-sm flex items-center justify-between group transition-all ${isActive
                                                    ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/20'
                                                    : isSelected
                                                        ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800'
                                                        : 'hover:bg-white dark:hover:bg-zinc-700 text-neutral-600 dark:text-zinc-300 hover:shadow-sm'
                                                    }`}
                                            >
                                                <div className="flex items-center gap-2">
                                                    {isSelected ? (
                                                        <CheckSquare className={`w-4 h-4 ${isActive ? 'text-white' : 'text-emerald-600 dark:text-emerald-400'}`} />
                                                    ) : (
                                                        <Folder className={`w-4 h-4 ${isActive ? 'text-white' : 'text-neutral-400 dark:text-zinc-500'}`} />
                                                    )}
                                                    <span className="truncate font-medium">{ds.repo_id}</span>
                                                </div>
                                                <span className={`text-[10px] px-2 py-0.5 rounded-full ${isActive ? 'bg-white/20 text-white' : isSelected ? 'bg-emerald-200 dark:bg-emerald-800 text-emerald-700 dark:text-emerald-300' : 'bg-neutral-100 dark:bg-zinc-700 text-neutral-500 dark:text-zinc-400'}`}>
                                                    {ds.total_episodes} ep
                                                </span>
                                            </button>
                                        );
                                    })}
                                    {datasets.length === 0 && (
                                        <div className="text-neutral-400 dark:text-zinc-500 text-sm italic px-3 py-4 text-center">No datasets found</div>
                                    )}
                                </div>
                            </div>

                            {/* Episode List */}
                            <div className="flex-1 overflow-y-auto p-4">
                                <h3 className="text-xs font-semibold text-neutral-500 dark:text-zinc-400 uppercase tracking-wider mb-3">
                                    Episodes {selectedRepo ? `(${episodes.length})` : ''}
                                </h3>
                                {isLoadingRepo ? (
                                    <div className="text-neutral-400 dark:text-zinc-500 text-sm animate-pulse px-3">Loading...</div>
                                ) : (
                                    <div className="space-y-1">
                                        {episodes.map((ep, i) => {
                                            const idx = ep.episode_index ?? ep.index ?? i;
                                            return (
                                                <button
                                                    key={i}
                                                    onClick={() => selectEpisode(idx)}
                                                    className={`w-full text-left px-3 py-2.5 rounded-xl text-sm flex items-center gap-2 transition-all ${selectedEpisodeIndex === idx
                                                        ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800'
                                                        : 'hover:bg-white dark:hover:bg-zinc-700 text-neutral-600 dark:text-zinc-300 hover:shadow-sm border border-transparent'
                                                        }`}
                                                >
                                                    <Film className={`w-4 h-4 ${selectedEpisodeIndex === idx ? 'text-emerald-600 dark:text-emerald-400' : 'text-neutral-400 dark:text-zinc-500'}`} />
                                                    <span className="font-medium">Episode {idx}</span>
                                                    {ep.length && (
                                                        <span className="ml-auto text-[10px] text-neutral-400 dark:text-zinc-500 bg-neutral-100 dark:bg-zinc-700 px-2 py-0.5 rounded-full">
                                                            {ep.length}f
                                                        </span>
                                                    )}
                                                </button>
                                            )
                                        })}
                                    </div>
                                )}
                            </div>
                                </>
                            ) : (
                                /* Policies Sidebar */
                                <>
                                    <div className="p-4 border-b border-neutral-200/50 dark:border-zinc-700/50">
                                        <div className="flex items-center justify-between mb-3">
                                            <h3 className="text-xs font-semibold text-neutral-500 dark:text-zinc-400 uppercase tracking-wider">
                                                Policies {selectedPolicies.size > 1 && <span className="text-purple-600 dark:text-purple-400">({selectedPolicies.size})</span>}
                                            </h3>
                                            {selectedPolicies.size === 1 && (
                                                <button
                                                    onClick={() => {
                                                        const policyId = Array.from(selectedPolicies)[0];
                                                        deletePolicy(policyId);
                                                    }}
                                                    className="p-1.5 hover:bg-red-100 rounded-lg text-red-500 transition-colors"
                                                    title="Delete policy"
                                                >
                                                    <Trash2 className="w-3.5 h-3.5" />
                                                </button>
                                            )}
                                        </div>
                                        <p className="text-[10px] text-neutral-400 dark:text-zinc-500 mb-2">Ctrl+click to compare multiple</p>
                                        <div className="space-y-1">
                                            {isLoadingPolicies ? (
                                                <div className="text-neutral-400 dark:text-zinc-500 text-sm animate-pulse px-3">Loading...</div>
                                            ) : policies.length === 0 ? (
                                                <div className="text-neutral-400 dark:text-zinc-500 text-sm italic px-3 py-4 text-center">No policies found</div>
                                            ) : (
                                                policies.map((policy) => {
                                                    const isSelected = selectedPolicies.has(policy.id);
                                                    const statusColor = policy.status === 'completed' ? 'bg-green-500' : policy.status === 'training' ? 'bg-purple-500' : 'bg-red-500';
                                                    return (
                                                        <button
                                                            key={policy.id}
                                                            onClick={(e) => handlePolicyClick(policy.id, e)}
                                                            className={`w-full text-left px-3 py-2.5 rounded-xl text-sm transition-all ${
                                                                isSelected
                                                                    ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-800'
                                                                    : 'hover:bg-white dark:hover:bg-zinc-700 text-neutral-600 dark:text-zinc-300 hover:shadow-sm border border-transparent'
                                                            }`}
                                                        >
                                                            <div className="flex items-center gap-2">
                                                                <div className={`w-2 h-2 rounded-full ${statusColor} ${policy.status === 'training' ? 'animate-pulse' : ''}`} />
                                                                <span className="truncate font-medium flex-1">{policy.name}</span>
                                                            </div>
                                                            <div className="flex items-center gap-2 mt-1 text-[10px] text-neutral-400 dark:text-zinc-500">
                                                                <span className="uppercase">{policy.policy_type}</span>
                                                                <span>•</span>
                                                                <span>{policy.status === 'training'
                                                                    ? `${Math.round((policy.steps / policy.total_steps) * 100)}%`
                                                                    : `${(policy.steps / 1000).toFixed(0)}K steps`
                                                                }</span>
                                                            </div>
                                                            {policy.status === 'training' && (
                                                                <div className="mt-1.5 w-full bg-purple-200 rounded-full h-1">
                                                                    <div
                                                                        className="bg-purple-500 h-1 rounded-full transition-all"
                                                                        style={{ width: `${(policy.steps / policy.total_steps) * 100}%` }}
                                                                    />
                                                                </div>
                                                            )}
                                                        </button>
                                                    );
                                                })
                                            )}
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>

                        {/* Main View */}
                        <div className="flex-1 flex flex-col bg-white/50 dark:bg-zinc-900/50 relative">
                            {activeTab === "datasets" ? (
                                /* Datasets Main View */
                                episodeData ? (
                                <>
                                    {/* Toolbar */}
                                    <div className="h-12 border-b border-neutral-200/50 dark:border-zinc-700/50 flex items-center justify-between px-5 bg-white/70 dark:bg-zinc-800/70">
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
                                            <button onClick={deleteEpisode} className="p-2 hover:bg-red-50 dark:hover:bg-red-950/50 hover:text-red-500 rounded-lg text-neutral-400 dark:text-zinc-500 transition-colors" title="Delete episode">
                                                <Trash2 className="w-4 h-4" />
                                            </button>
                                        </div>
                                    </div>

                                    {/* Video Grid - Paired RGB + Depth Layout */}
                                    <div className="flex-1 p-4 grid grid-cols-2 gap-4 overflow-y-auto bg-neutral-100/50 dark:bg-zinc-800/50">
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

                                    {/* Telemetry Graph with Playhead */}
                                    <div className="h-44 border-t border-neutral-200/50 dark:border-zinc-700/50 bg-white/80 dark:bg-zinc-900/80 p-3" ref={chartRef}>
                                        <div className="h-full w-full cursor-crosshair bg-neutral-50 dark:bg-zinc-800 rounded-xl border border-neutral-200/50 dark:border-zinc-700/50 overflow-hidden">
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
                                    <div className="h-14 border-t border-neutral-200/50 dark:border-zinc-700/50 bg-white/90 dark:bg-zinc-900/90 px-5 flex items-center gap-4">
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
                                ) : (
                                    <div className="flex-1 flex flex-col items-center justify-center text-neutral-400 dark:text-zinc-500 gap-4">
                                        <HardDrive className="w-16 h-16 text-neutral-200 dark:text-zinc-700" />
                                        <p className="text-neutral-500 dark:text-zinc-400">Select a dataset and episode to inspect</p>
                                    </div>
                                )
                            ) : (
                                /* Policies Main View */
                                selectedPolicyData.length > 0 ? (
                                    <div className="flex-1 flex flex-col overflow-hidden">
                                        {/* Policy Header with editable name */}
                                        <div className="h-12 border-b border-neutral-200/50 dark:border-zinc-700/50 flex items-center justify-between px-5 bg-white/70 dark:bg-zinc-800/70">
                                            <div className="flex items-center gap-3">
                                                {editingPolicyId === primarySelectedPolicy?.id ? (
                                                    <div className="flex items-center gap-2">
                                                        <input
                                                            type="text"
                                                            value={editingPolicyName}
                                                            onChange={(e) => setEditingPolicyName(e.target.value)}
                                                            onKeyDown={(e) => {
                                                                if (e.key === 'Enter' && primarySelectedPolicy) {
                                                                    renamePolicy(primarySelectedPolicy.id, editingPolicyName);
                                                                } else if (e.key === 'Escape') {
                                                                    setEditingPolicyId(null);
                                                                }
                                                            }}
                                                            className="text-sm font-semibold bg-purple-50 dark:bg-purple-950/50 border border-purple-200 dark:border-purple-800 rounded-lg px-2 py-1 focus:outline-none focus:ring-2 focus:ring-purple-300 dark:focus:ring-purple-700 text-neutral-900 dark:text-zinc-100"
                                                            autoFocus
                                                        />
                                                        <button
                                                            onClick={() => primarySelectedPolicy && renamePolicy(primarySelectedPolicy.id, editingPolicyName)}
                                                            className="p-1 hover:bg-purple-100 rounded text-purple-600"
                                                        >
                                                            <Check className="w-4 h-4" />
                                                        </button>
                                                    </div>
                                                ) : (
                                                    <button
                                                        onClick={() => {
                                                            if (primarySelectedPolicy) {
                                                                setEditingPolicyId(primarySelectedPolicy.id);
                                                                setEditingPolicyName(primarySelectedPolicy.name);
                                                            }
                                                        }}
                                                        className="flex items-center gap-2 group"
                                                    >
                                                        <span className="text-sm text-neutral-700 dark:text-zinc-200 font-semibold">{primarySelectedPolicy?.name}</span>
                                                        <Pencil className="w-3 h-3 text-neutral-300 dark:text-zinc-600 group-hover:text-purple-500 transition-colors" />
                                                    </button>
                                                )}
                                                <span className={`text-xs px-2 py-1 rounded-lg uppercase font-medium ${
                                                    primarySelectedPolicy?.status === 'completed' ? 'bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300' :
                                                    primarySelectedPolicy?.status === 'training' ? 'bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300' :
                                                    'bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300'
                                                }`}>
                                                    {primarySelectedPolicy?.status}
                                                </span>
                                                {selectedPolicyData.length > 1 && (
                                                    <span className="text-xs bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 px-2 py-1 rounded-lg">
                                                        Comparing {selectedPolicyData.length} policies
                                                    </span>
                                                )}
                                            </div>
                                            <div className="flex items-center gap-2">
                                                {primarySelectedPolicy?.status === 'completed' && primarySelectedPolicy?.checkpoint_path && (
                                                    <>
                                                        <button
                                                            onClick={() => primarySelectedPolicy && deployPolicy(primarySelectedPolicy.id)}
                                                            disabled={isDeploying}
                                                            className="flex items-center gap-1.5 px-3 py-1.5 bg-green-500 hover:bg-green-600 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
                                                        >
                                                            {isDeploying ? <Loader2 className="w-4 h-4 animate-spin" /> : <Rocket className="w-4 h-4" />}
                                                            Deploy
                                                        </button>
                                                        <button
                                                            onClick={() => {
                                                                if (primarySelectedPolicy) {
                                                                    setResumingPolicyId(primarySelectedPolicy.id);
                                                                    setShowResumeModal(true);
                                                                }
                                                            }}
                                                            className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-500 hover:bg-purple-600 text-white text-sm font-medium rounded-lg transition-colors"
                                                        >
                                                            <RotateCcw className="w-4 h-4" />
                                                            Resume
                                                        </button>
                                                    </>
                                                )}
                                                <button
                                                    onClick={() => primarySelectedPolicy && deletePolicy(primarySelectedPolicy.id)}
                                                    className="p-2 hover:bg-red-50 dark:hover:bg-red-950/50 hover:text-red-500 rounded-lg text-neutral-400 dark:text-zinc-500 transition-colors"
                                                    title="Delete policy"
                                                >
                                                    <Trash2 className="w-4 h-4" />
                                                </button>
                                            </div>
                                        </div>

                                        {/* Loss Curve Chart */}
                                        <div className="flex-1 p-4 overflow-auto">
                                            <div className="bg-white dark:bg-zinc-900 rounded-2xl border border-neutral-200/50 dark:border-zinc-700/50 p-4 shadow-sm">
                                                <h3 className="text-sm font-semibold text-neutral-700 dark:text-zinc-200 mb-3">Loss Curve</h3>
                                                <div className="h-64">
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <LineChart margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                                                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                                                            <XAxis
                                                                dataKey="step"
                                                                type="number"
                                                                domain={['dataMin', 'dataMax']}
                                                                tick={{ fill: '#999', fontSize: 10 }}
                                                                tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`}
                                                            />
                                                            <YAxis
                                                                tick={{ fill: '#999', fontSize: 10 }}
                                                                tickFormatter={(v) => v.toFixed(3)}
                                                            />
                                                            <Tooltip
                                                                contentStyle={{ backgroundColor: 'white', border: '1px solid #e5e5e5', borderRadius: '8px', fontSize: '12px' }}
                                                                formatter={(value) => [Number(value).toFixed(4), 'Loss']}
                                                                labelFormatter={(label) => `Step ${label}`}
                                                            />
                                                            {selectedPolicyData.length > 1 && <Legend />}
                                                            {selectedPolicyData.map((policy, idx) => {
                                                                const colors = ['#10b981', '#8b5cf6', '#f59e0b', '#ef4444'];
                                                                const chartData = policy.loss_history.map(([step, loss]) => ({ step, loss }));
                                                                return (
                                                                    <Line
                                                                        key={policy.id}
                                                                        data={chartData}
                                                                        type="monotone"
                                                                        dataKey="loss"
                                                                        name={selectedPolicyData.length > 1 ? policy.name : undefined}
                                                                        stroke={colors[idx % colors.length]}
                                                                        dot={false}
                                                                        strokeWidth={2}
                                                                        isAnimationActive={false}
                                                                    />
                                                                );
                                                            })}
                                                        </LineChart>
                                                    </ResponsiveContainer>
                                                </div>
                                                {selectedPolicyData[0]?.loss_history.length === 0 && (
                                                    <div className="text-center text-neutral-400 dark:text-zinc-500 py-8">
                                                        No loss history available for this policy
                                                    </div>
                                                )}
                                            </div>

                                            {/* Metadata Cards */}
                                            <div className="grid grid-cols-2 gap-4 mt-4">
                                                <div className="bg-white dark:bg-zinc-900 rounded-xl border border-neutral-200/50 dark:border-zinc-700/50 p-4">
                                                    <span className="text-xs text-neutral-500 dark:text-zinc-400 uppercase tracking-wide">Policy Type</span>
                                                    <p className="text-lg font-semibold text-neutral-800 dark:text-zinc-100 mt-1">{primarySelectedPolicy?.policy_type.toUpperCase()}</p>
                                                </div>
                                                <div className="bg-white dark:bg-zinc-900 rounded-xl border border-neutral-200/50 dark:border-zinc-700/50 p-4">
                                                    <span className="text-xs text-neutral-500 dark:text-zinc-400 uppercase tracking-wide">Training Steps</span>
                                                    <p className="text-lg font-semibold text-neutral-800 dark:text-zinc-100 mt-1">{primarySelectedPolicy?.steps.toLocaleString()}</p>
                                                </div>
                                                <div className="bg-white dark:bg-zinc-900 rounded-xl border border-neutral-200/50 dark:border-zinc-700/50 p-4">
                                                    <span className="text-xs text-neutral-500 dark:text-zinc-400 uppercase tracking-wide">Source Dataset</span>
                                                    <p className="text-lg font-semibold text-neutral-800 mt-1 truncate">{primarySelectedPolicy?.dataset_repo_id || 'Unknown'}</p>
                                                </div>
                                                <div className="bg-white dark:bg-zinc-900 rounded-xl border border-neutral-200/50 dark:border-zinc-700/50 p-4">
                                                    <span className="text-xs text-neutral-500 dark:text-zinc-400 uppercase tracking-wide">Final Loss</span>
                                                    <p className="text-lg font-semibold text-neutral-800 dark:text-zinc-100 mt-1">
                                                        {primarySelectedPolicy?.final_loss?.toFixed(4) || 'N/A'}
                                                    </p>
                                                </div>
                                            </div>

                                            {/* Created Date */}
                                            <div className="mt-4 text-xs text-neutral-400 dark:text-zinc-500 text-center">
                                                Created: {primarySelectedPolicy?.created_at ? new Date(primarySelectedPolicy.created_at).toLocaleString() : 'Unknown'}
                                            </div>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="flex-1 flex flex-col items-center justify-center text-neutral-400 dark:text-zinc-500 gap-4">
                                        <Brain className="w-16 h-16 text-neutral-200 dark:text-zinc-700" />
                                        <p className="text-neutral-500 dark:text-zinc-400">Select a policy to view details</p>
                                    </div>
                                )
                            )}
                        </div>
                    </div>
                </motion.div>

                {/* Resume Training Modal */}
                {showResumeModal && resumingPolicyId && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 flex items-center justify-center z-[200]"
                    >
                        <div className="absolute inset-0 bg-black/40" onClick={() => setShowResumeModal(false)} />
                        <motion.div
                            initial={{ scale: 0.95, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            className="relative bg-white dark:bg-zinc-900 rounded-2xl shadow-2xl p-6 max-w-md w-full mx-4 border border-transparent dark:border-zinc-700"
                        >
                            <div className="flex items-center gap-3 mb-4">
                                <div className="p-3 bg-purple-100 dark:bg-purple-900/50 rounded-xl">
                                    <RotateCcw className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                                </div>
                                <div>
                                    <h3 className="text-lg font-semibold text-neutral-900 dark:text-zinc-100">Resume Training</h3>
                                    <p className="text-sm text-neutral-500 dark:text-zinc-400">Continue training from checkpoint</p>
                                </div>
                            </div>

                            <div className="mb-4">
                                <label className="block text-sm font-medium text-neutral-700 dark:text-zinc-300 mb-2">
                                    Additional Training Steps
                                </label>
                                <input
                                    type="number"
                                    value={resumeSteps}
                                    onChange={(e) => setResumeSteps(parseInt(e.target.value) || 10000)}
                                    className="w-full px-3 py-2 border border-neutral-200 dark:border-zinc-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-300 dark:focus:ring-purple-700 bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                    min={1000}
                                    step={1000}
                                />
                                <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-1">
                                    Recommended: 10,000 - 50,000 steps
                                </p>
                            </div>

                            <div className="flex gap-3">
                                <button
                                    onClick={() => setShowResumeModal(false)}
                                    className="flex-1 py-2.5 px-4 border border-neutral-200 dark:border-zinc-700 rounded-xl text-neutral-600 dark:text-zinc-300 font-medium hover:bg-neutral-50 dark:hover:bg-zinc-800 transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={() => resumingPolicyId && resumeTraining(resumingPolicyId, resumeSteps)}
                                    className="flex-1 py-2.5 px-4 bg-purple-500 hover:bg-purple-600 rounded-xl text-white font-medium transition-colors flex items-center justify-center gap-2"
                                >
                                    <RotateCcw className="w-4 h-4" />
                                    Resume
                                </button>
                            </div>
                        </motion.div>
                    </motion.div>
                )}

                {/* Upload Confirmation Modal */}
                {showUploadConfirm && uploadingDataset && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 flex items-center justify-center z-[200]"
                    >
                        <div className="absolute inset-0 bg-black/40" onClick={cancelUpload} />
                        <motion.div
                            initial={{ scale: 0.95, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            className="relative bg-white dark:bg-zinc-900 rounded-2xl shadow-2xl p-6 max-w-md w-full mx-4 border border-transparent dark:border-zinc-700"
                        >
                            <div className="flex items-center gap-3 mb-4">
                                <div className="p-3 bg-blue-100 dark:bg-blue-900/50 rounded-xl">
                                    <Cloud className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                                </div>
                                <div>
                                    <h3 className="text-lg font-semibold text-neutral-900 dark:text-zinc-100">Upload to Cloud</h3>
                                    <p className="text-sm text-neutral-500 dark:text-zinc-400">Confirm dataset upload</p>
                                </div>
                            </div>

                            <div className="bg-neutral-50 dark:bg-zinc-800 rounded-xl p-4 mb-4 space-y-2">
                                <div className="flex justify-between text-sm">
                                    <span className="text-neutral-500 dark:text-zinc-400">Dataset</span>
                                    <span className="font-medium text-neutral-900 dark:text-zinc-100">{uploadingDataset.repo_id}</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                    <span className="text-neutral-500 dark:text-zinc-400">Episodes</span>
                                    <span className="font-medium text-neutral-900 dark:text-zinc-100">{uploadingDataset.total_episodes}</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                    <span className="text-neutral-500 dark:text-zinc-400">Total Frames</span>
                                    <span className="font-medium text-neutral-900 dark:text-zinc-100">{uploadingDataset.total_frames.toLocaleString()}</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                    <span className="text-neutral-500 dark:text-zinc-400">Location</span>
                                    <span className="font-mono text-xs text-neutral-600 dark:text-zinc-400 truncate max-w-[200px]" title={uploadingDataset.root}>
                                        {uploadingDataset.root}
                                    </span>
                                </div>
                            </div>

                            <p className="text-xs text-neutral-500 dark:text-zinc-400 mb-4">
                                This will upload all files from this dataset to your cloud storage.
                            </p>

                            <div className="flex gap-3">
                                <button
                                    onClick={cancelUpload}
                                    className="flex-1 py-2.5 px-4 border border-neutral-200 dark:border-zinc-700 rounded-xl text-neutral-600 dark:text-zinc-300 font-medium hover:bg-neutral-50 dark:hover:bg-zinc-800 transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={confirmUpload}
                                    className="flex-1 py-2.5 px-4 bg-blue-500 hover:bg-blue-600 rounded-xl text-white font-medium transition-colors flex items-center justify-center gap-2"
                                >
                                    <Cloud className="w-4 h-4" />
                                    Upload
                                </button>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </div>
        </AnimatePresence>

        {/* Merge Modal - Outside AnimatePresence to avoid duplicate key error */}
        <MergeModal
            isOpen={isMergeModalOpen}
            onClose={() => setIsMergeModalOpen(false)}
            selectedRepos={Array.from(selectedRepos)}
            onMergeComplete={() => {
                setIsMergeModalOpen(false);
                setSelectedRepos(new Set());
                fetchDatasets();
            }}
        />
        </>
    );
}
