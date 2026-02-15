import React, { useState } from 'react';
import { Trash2, Film, Folder, HardDrive, CheckSquare, Layers } from 'lucide-react';
import MergeModal from './MergeModal';
import EpisodeViewer from './EpisodeViewer';
import UploadSection from './UploadSection';
import { datasetsApi, api } from '../../../lib/api';
import type { DatasetInfo, EpisodeSummary, VideoMetadata } from '../../../lib/api/types';

interface EpisodeDetail {
    index: number;
    length: number;
    actions: number[][];
    timestamps: number[];
    videos: Record<string, string>;
    video_metadata?: Record<string, VideoMetadata>;
    fps?: number;
}

interface DatasetsTabProps {
    datasets: DatasetInfo[];
    fetchDatasets: () => void;
}

export default function DatasetsTab({ datasets, fetchDatasets }: DatasetsTabProps) {
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

    const selectRepo = async (repo_id: string) => {
        setSelectedRepo(repo_id);
        setSelectedEpisodeIndex(null);
        setEpisodeData(null);
        setIsLoadingRepo(true);
        try {
            const data = await datasetsApi.episodes(repo_id);
            const list = Array.isArray(data) ? data : data.episodes;
            if (Array.isArray(list)) {
                const sorted = list.sort((a: EpisodeSummary, b: EpisodeSummary) => (a.episode_index ?? a.index) - (b.episode_index ?? b.index));
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
            const data = await datasetsApi.episodeDetail(selectedRepo, index);
            if ((data as any).error) {
                console.error("Episode load error:", (data as any).error);
                setEpisodeData(null);
            } else {
                setEpisodeData(data as unknown as EpisodeDetail);
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
            const result = await datasetsApi.deleteEpisode(selectedRepo, selectedEpisodeIndex) as any;
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
            const result = await api.delete<any>(`/datasets/${encodeURIComponent(selectedRepo)}`);
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
                const result = await api.delete<any>(`/datasets/${encodeURIComponent(repo_id)}`);
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

    return (
        <>
            {/* Sidebar Content */}
            <div className="w-72 border-r border-neutral-200/50 dark:border-zinc-700/50 flex flex-col bg-neutral-50/50 dark:bg-zinc-800/50">
                {/* Dataset List */}
                <div className="p-4 border-b border-neutral-200/50 dark:border-zinc-700/50">
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="text-xs font-semibold text-neutral-500 dark:text-zinc-400 uppercase tracking-wider">
                            Repositories {selectedRepos.size > 1 && <span className="text-emerald-600 dark:text-emerald-400">({selectedRepos.size})</span>}
                        </h3>
                        <div className="flex items-center gap-1">
                            <UploadSection
                                datasets={datasets}
                                selectedRepos={selectedRepos}
                            />
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
            </div>

            {/* Main View */}
            <div className="flex-1 flex flex-col bg-white/50 dark:bg-zinc-900/50 relative">
                {episodeData ? (
                    <EpisodeViewer episodeData={episodeData} onDeleteEpisode={deleteEpisode} />
                ) : (
                    <div className="flex-1 flex flex-col items-center justify-center text-neutral-400 dark:text-zinc-500 gap-4">
                        <HardDrive className="w-16 h-16 text-neutral-200 dark:text-zinc-700" />
                        <p className="text-neutral-500 dark:text-zinc-400">Select a dataset and episode to inspect</p>
                    </div>
                )}
            </div>

            {/* Merge Modal - Outside main flow to avoid z-index issues */}
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
