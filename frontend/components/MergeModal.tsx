"use client";

import React, { useState, useEffect } from 'react';
import { X, Layers, AlertCircle, CheckCircle, Loader2, Database, GitMerge } from 'lucide-react';

const API_BASE = "http://127.0.0.1:8000";

interface MergeModalProps {
    isOpen: boolean;
    onClose: () => void;
    selectedRepos: string[];
    onMergeComplete: () => void;
}

interface DatasetInfo {
    repo_id: string;
    fps: number;
    robot_type: string;
    features: string[];
    total_episodes: number;
    total_frames: number;
}

interface ValidationResult {
    compatible: boolean;
    datasets: DatasetInfo[];
    merged_info: {
        total_episodes: number;
        total_frames: number;
        fps: number;
        robot_type: string;
        features: string[];
    } | null;
    errors: { type: string; message: string }[];
    warnings: string[];
}

interface MergeProgress {
    percent: number;
    message: string;
}

type MergeStatus = 'idle' | 'validating' | 'merging' | 'completed' | 'error';

export default function MergeModal({ isOpen, onClose, selectedRepos, onMergeComplete }: MergeModalProps) {
    const [outputName, setOutputName] = useState('');
    const [validation, setValidation] = useState<ValidationResult | null>(null);
    const [status, setStatus] = useState<MergeStatus>('idle');
    const [progress, setProgress] = useState<MergeProgress>({ percent: 0, message: '' });
    const [error, setError] = useState('');
    const [jobId, setJobId] = useState<string | null>(null);

    // Validate on open
    useEffect(() => {
        if (isOpen && selectedRepos.length >= 2) {
            validateMerge();
            // Suggest a default output name
            const timestamp = new Date().toISOString().slice(0, 10).replace(/-/g, '');
            setOutputName(`merged_${timestamp}`);
        }
        // Reset state when closed
        if (!isOpen) {
            setValidation(null);
            setStatus('idle');
            setProgress({ percent: 0, message: '' });
            setError('');
            setJobId(null);
        }
    }, [isOpen, selectedRepos]);

    // Poll for job status when merging
    useEffect(() => {
        if (status !== 'merging' || !jobId) return;

        const interval = setInterval(async () => {
            try {
                const res = await fetch(`${API_BASE}/datasets/merge/status/${jobId}`);
                const data = await res.json();

                setProgress(data.progress);

                if (data.status === 'completed') {
                    setStatus('completed');
                    clearInterval(interval);
                } else if (data.status === 'failed') {
                    setStatus('error');
                    setError(data.error || 'Merge failed');
                    clearInterval(interval);
                }
            } catch (e) {
                console.error('Failed to poll merge status:', e);
            }
        }, 1000);

        return () => clearInterval(interval);
    }, [status, jobId]);

    const validateMerge = async () => {
        setStatus('validating');
        setError('');

        try {
            const res = await fetch(`${API_BASE}/datasets/merge/validate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ repo_ids: selectedRepos })
            });
            const data: ValidationResult = await res.json();
            setValidation(data);
            setStatus('idle');
        } catch (e) {
            setError('Failed to validate datasets');
            setStatus('error');
        }
    };

    const startMerge = async () => {
        if (!outputName.trim()) {
            setError('Please enter an output name');
            return;
        }

        setStatus('merging');
        setError('');
        setProgress({ percent: 0, message: 'Starting merge...' });

        try {
            const res = await fetch(`${API_BASE}/datasets/merge/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    repo_ids: selectedRepos,
                    output_repo_id: outputName.trim()
                })
            });

            const data = await res.json();

            if (!res.ok) {
                setError(data.error || 'Failed to start merge');
                setStatus('error');
                return;
            }

            setJobId(data.job_id);
        } catch (e) {
            setError('Failed to start merge');
            setStatus('error');
        }
    };

    const handleClose = () => {
        if (status === 'merging') {
            if (!confirm('Merge is in progress. Are you sure you want to close?')) {
                return;
            }
        }
        if (status === 'completed') {
            onMergeComplete();
        } else {
            onClose();
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[70] flex items-center justify-center">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/50 dark:bg-black/70 backdrop-blur-sm"
                onClick={handleClose}
            />

            {/* Modal */}
            <div
                className="relative bg-white/95 dark:bg-zinc-900/95 backdrop-blur-2xl rounded-3xl shadow-2xl border border-white/50 dark:border-zinc-700/50 w-[650px] max-h-[85vh] overflow-hidden animate-in zoom-in-95 duration-200"
                onClick={e => e.stopPropagation()}
            >
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-5 border-b border-black/5 dark:border-white/5 bg-white/40 dark:bg-zinc-800/40">
                    <div className="flex items-center gap-3">
                        <div className="p-2.5 bg-purple-100 dark:bg-purple-900/40 rounded-xl">
                            <GitMerge className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-neutral-900 dark:text-zinc-100">
                                Merge Datasets
                            </h2>
                            <p className="text-sm text-neutral-500 dark:text-zinc-400">
                                {selectedRepos.length} datasets selected
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={handleClose}
                        className="p-2 hover:bg-black/5 dark:hover:bg-white/5 rounded-xl transition-colors"
                    >
                        <X className="w-5 h-5 text-neutral-400 dark:text-zinc-500" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-6 space-y-5 overflow-y-auto max-h-[60vh]">
                    {/* Validation Status */}
                    {status === 'validating' && (
                        <div className="flex items-center gap-3 p-4 bg-purple-50 dark:bg-purple-950/30 rounded-xl border border-purple-100 dark:border-purple-900/50">
                            <Loader2 className="w-5 h-5 text-purple-600 dark:text-purple-400 animate-spin" />
                            <span className="text-purple-700 dark:text-purple-300 font-medium">Validating compatibility...</span>
                        </div>
                    )}

                    {/* Validation Results */}
                    {validation && status !== 'validating' && (
                        <div className="space-y-4">
                            {/* Compatibility Status */}
                            <div className={`p-4 rounded-xl border ${
                                validation.compatible
                                    ? 'bg-green-50 dark:bg-green-950/30 border-green-200 dark:border-green-800'
                                    : 'bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-800'
                            }`}>
                                <div className="flex items-center gap-3">
                                    {validation.compatible ? (
                                        <>
                                            <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                                            <span className="font-medium text-green-700 dark:text-green-300">
                                                Datasets are compatible for merging
                                            </span>
                                        </>
                                    ) : (
                                        <>
                                            <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
                                            <span className="font-medium text-red-700 dark:text-red-300">
                                                Datasets are not compatible
                                            </span>
                                        </>
                                    )}
                                </div>
                                {validation.errors.length > 0 && (
                                    <ul className="mt-3 text-sm text-red-600 dark:text-red-400 space-y-1">
                                        {validation.errors.map((err, i) => (
                                            <li key={i} className="flex items-start gap-2">
                                                <span className="text-red-400">•</span>
                                                {err.message}
                                            </li>
                                        ))}
                                    </ul>
                                )}
                            </div>

                            {/* Dataset Details Table */}
                            <div className="border border-neutral-200 dark:border-zinc-700 rounded-xl overflow-hidden">
                                <table className="w-full text-sm">
                                    <thead className="bg-neutral-50 dark:bg-zinc-800">
                                        <tr>
                                            <th className="px-4 py-3 text-left font-medium text-neutral-600 dark:text-zinc-400">Dataset</th>
                                            <th className="px-4 py-3 text-center font-medium text-neutral-600 dark:text-zinc-400">FPS</th>
                                            <th className="px-4 py-3 text-center font-medium text-neutral-600 dark:text-zinc-400">Robot</th>
                                            <th className="px-4 py-3 text-center font-medium text-neutral-600 dark:text-zinc-400">Episodes</th>
                                            <th className="px-4 py-3 text-center font-medium text-neutral-600 dark:text-zinc-400">Frames</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {validation.datasets.map((ds, index) => (
                                            <tr key={ds.repo_id || `ds-${index}`} className="border-t border-neutral-100 dark:border-zinc-700">
                                                <td className="px-4 py-3 font-medium text-neutral-800 dark:text-zinc-200 truncate max-w-[180px]" title={ds.repo_id}>
                                                    {ds.repo_id}
                                                </td>
                                                <td className="px-4 py-3 text-center text-neutral-600 dark:text-zinc-400">{ds.fps}</td>
                                                <td className="px-4 py-3 text-center text-neutral-600 dark:text-zinc-400 text-xs">{ds.robot_type}</td>
                                                <td className="px-4 py-3 text-center text-neutral-600 dark:text-zinc-400">{ds.total_episodes}</td>
                                                <td className="px-4 py-3 text-center text-neutral-600 dark:text-zinc-400">{ds.total_frames}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>

                            {/* Merged Summary */}
                            {validation.compatible && validation.merged_info && (
                                <div className="p-4 bg-purple-50 dark:bg-purple-950/30 rounded-xl border border-purple-200 dark:border-purple-800">
                                    <div className="flex items-center gap-2 mb-3">
                                        <Database className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                                        <span className="font-semibold text-purple-700 dark:text-purple-300">Merged Dataset Preview</span>
                                    </div>
                                    <div className="grid grid-cols-4 gap-4 text-sm">
                                        <div className="bg-white/60 dark:bg-zinc-800/60 p-3 rounded-lg text-center">
                                            <div className="text-2xl font-bold text-neutral-800 dark:text-zinc-200">{validation.merged_info.total_episodes}</div>
                                            <div className="text-xs text-neutral-500 dark:text-zinc-400 mt-1">Episodes</div>
                                        </div>
                                        <div className="bg-white/60 dark:bg-zinc-800/60 p-3 rounded-lg text-center">
                                            <div className="text-2xl font-bold text-neutral-800 dark:text-zinc-200">{validation.merged_info.total_frames.toLocaleString()}</div>
                                            <div className="text-xs text-neutral-500 dark:text-zinc-400 mt-1">Frames</div>
                                        </div>
                                        <div className="bg-white/60 dark:bg-zinc-800/60 p-3 rounded-lg text-center">
                                            <div className="text-2xl font-bold text-neutral-800 dark:text-zinc-200">{validation.merged_info.fps}</div>
                                            <div className="text-xs text-neutral-500 dark:text-zinc-400 mt-1">FPS</div>
                                        </div>
                                        <div className="bg-white/60 dark:bg-zinc-800/60 p-3 rounded-lg text-center">
                                            <div className="text-sm font-bold text-neutral-800 dark:text-zinc-200 truncate" title={validation.merged_info.robot_type}>
                                                {validation.merged_info.robot_type.split('_').pop()}
                                            </div>
                                            <div className="text-xs text-neutral-500 dark:text-zinc-400 mt-1">Robot</div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Output Name Input */}
                    {validation?.compatible && status !== 'completed' && (
                        <div>
                            <label className="block text-sm font-medium text-neutral-700 dark:text-zinc-300 mb-2">
                                Output Dataset Name
                            </label>
                            <input
                                type="text"
                                value={outputName}
                                onChange={e => setOutputName(e.target.value)}
                                placeholder="merged_dataset"
                                disabled={status === 'merging'}
                                className="w-full px-4 py-3 border border-neutral-300 dark:border-zinc-600 rounded-xl
                                         bg-white dark:bg-zinc-800 text-neutral-800 dark:text-zinc-200
                                         focus:ring-2 focus:ring-purple-500 focus:border-transparent
                                         disabled:opacity-50 disabled:cursor-not-allowed
                                         placeholder:text-neutral-400 dark:placeholder:text-zinc-500"
                            />
                            <p className="mt-2 text-xs text-neutral-500 dark:text-zinc-400">
                                The merged dataset will be saved to: <code className="bg-neutral-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded">datasets/{outputName || '...'}</code>
                            </p>
                        </div>
                    )}

                    {/* Progress */}
                    {status === 'merging' && (
                        <div className="space-y-3 p-4 bg-purple-50 dark:bg-purple-950/30 rounded-xl border border-purple-200 dark:border-purple-800">
                            <div className="flex items-center justify-between text-sm">
                                <span className="text-purple-700 dark:text-purple-300 font-medium flex items-center gap-2">
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    {progress.message}
                                </span>
                                <span className="font-bold text-purple-600 dark:text-purple-400">{progress.percent}%</span>
                            </div>
                            <div className="w-full bg-purple-200 dark:bg-purple-900 rounded-full h-2.5 overflow-hidden">
                                <div
                                    className="bg-purple-500 h-2.5 rounded-full transition-all duration-500"
                                    style={{ width: `${progress.percent}%` }}
                                />
                            </div>
                        </div>
                    )}

                    {/* Completed State */}
                    {status === 'completed' && (
                        <div className="p-6 bg-green-50 dark:bg-green-950/30 rounded-xl border border-green-200 dark:border-green-800 text-center">
                            <div className="w-16 h-16 bg-green-100 dark:bg-green-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                                <CheckCircle className="w-8 h-8 text-green-500" />
                            </div>
                            <h3 className="text-xl font-semibold text-green-700 dark:text-green-300">Merge Complete!</h3>
                            <p className="text-sm text-green-600 dark:text-green-400 mt-2">
                                Dataset <code className="bg-green-100 dark:bg-green-900/50 px-2 py-0.5 rounded font-medium">{outputName}</code> has been created
                            </p>
                        </div>
                    )}

                    {/* Error State */}
                    {error && status === 'error' && (
                        <div className="p-4 bg-red-50 dark:bg-red-950/30 rounded-xl border border-red-200 dark:border-red-800">
                            <div className="flex items-start gap-3 text-red-600 dark:text-red-400">
                                <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                                <div>
                                    <span className="font-medium">Error</span>
                                    <p className="text-sm mt-1">{error}</p>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-black/5 dark:border-white/5 bg-neutral-50/50 dark:bg-zinc-800/50">
                    {status === 'completed' ? (
                        <button
                            onClick={() => onMergeComplete()}
                            className="px-6 py-2.5 bg-purple-600 text-white rounded-xl font-medium hover:bg-purple-700 transition-colors shadow-lg shadow-purple-200 dark:shadow-purple-900/30"
                        >
                            Done
                        </button>
                    ) : (
                        <>
                            <button
                                onClick={handleClose}
                                disabled={status === 'merging'}
                                className="px-5 py-2.5 text-neutral-600 dark:text-zinc-400 hover:bg-neutral-100 dark:hover:bg-zinc-700 rounded-xl transition-colors disabled:opacity-50"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={startMerge}
                                disabled={!validation?.compatible || status === 'merging' || status === 'validating'}
                                className="px-6 py-2.5 bg-purple-600 text-white rounded-xl font-medium hover:bg-purple-700 transition-colors
                                         disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2
                                         shadow-lg shadow-purple-200 dark:shadow-purple-900/30"
                            >
                                {status === 'merging' ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Merging...
                                    </>
                                ) : (
                                    <>
                                        <GitMerge className="w-4 h-4" />
                                        Merge Datasets
                                    </>
                                )}
                            </button>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
