import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Cloud, Loader2, AlertCircle, CheckCircle } from 'lucide-react';
import { useAuth } from '../../../lib/AuthContext';
import { supabase, isSupabaseConfigured } from '../../../lib/supabase';
import { datasetsApi, API_BASE } from '../../../lib/api';
import type { DatasetInfo } from '../../../lib/api/types';

interface UploadSectionProps {
    datasets: DatasetInfo[];
    selectedRepos: Set<string>;
}

export default function UploadSection({ datasets, selectedRepos }: UploadSectionProps) {
    const { user, session } = useAuth();

    // Upload state
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadingFileName, setUploadingFileName] = useState<string>('');
    const [uploadError, setUploadError] = useState<string | null>(null);
    const [uploadSuccess, setUploadSuccess] = useState(false);
    const [showUploadConfirm, setShowUploadConfirm] = useState(false);
    const [uploadingDataset, setUploadingDataset] = useState<DatasetInfo | null>(null);

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
            const fileList = await datasetsApi.files(uploadingDataset.repo_id) as unknown as { path: string; size: number }[];

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

    return (
        <>
            {/* Upload Button */}
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

            {/* Upload Progress/Status Inline */}
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
        </>
    );
}
