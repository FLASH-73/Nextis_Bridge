import React, { useState, useEffect, useRef } from 'react';
import { StopCircle, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { trainingApi } from '../../../lib/api';
import { usePolling } from '../../../hooks/usePolling';

export interface TrainingProgress {
    step: number;
    total_steps: number;
    loss: number | null;
    learning_rate: number | null;
    eta_seconds: number | null;
    epoch: number;
    loss_history: [number, number][];  // [[step, loss], ...]
}

export interface TrainingJob {
    id: string;
    status: 'pending' | 'validating' | 'training' | 'completed' | 'failed' | 'cancelled';
    policy_type: string;
    dataset_repo_id: string;
    config: Record<string, any>;
    progress: TrainingProgress;
    error: string | null;
    output_dir: string | null;
}

interface TrainingViewProps {
    jobId: string;
    totalSteps: number;
    onComplete: (job: TrainingJob) => void;
}

export default function TrainingView({ jobId, totalSteps, onComplete }: TrainingViewProps) {
    const [jobStatus, setJobStatus] = useState<TrainingJob | null>(null);
    const [logs, setLogs] = useState<string[]>([]);
    const [jobDone, setJobDone] = useState(false);
    const logsEndRef = useRef<HTMLDivElement>(null);

    // Poll for training progress
    usePolling(async () => {
        try {
            const data = await trainingApi.jobStatus(jobId) as TrainingJob;
            setJobStatus(data);

            // Fetch logs (get all available logs, up to 1000)
            const logsData = await trainingApi.jobLogs(jobId, 1000);
            setLogs(logsData.logs || []);

            if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
                setJobDone(true);
                onComplete(data);
            }
        } catch (e) {
            console.error('Failed to fetch job status:', e);
        }
    }, 2000, !!jobId && !jobDone);

    // Auto-scroll logs
    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    const formatETA = (seconds: number | null) => {
        if (!seconds) return '--';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        if (h > 0) return `${h}h ${m}m`;
        return `${m}m`;
    };

    const cancelTraining = async () => {
        if (!jobId) return;
        if (!confirm('Are you sure you want to cancel training?')) return;

        try {
            await trainingApi.cancelJob(jobId);
        } catch (e) {
            console.error('Failed to cancel training:', e);
        }
    };

    return (
        <div className="flex flex-col h-full gap-6 animate-in fade-in">
            {/* Progress Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold text-neutral-900 dark:text-zinc-100">Training in Progress</h2>
                    <p className="text-neutral-500 dark:text-zinc-400 text-sm">Job ID: {jobId}</p>
                </div>
                <button
                    onClick={cancelTraining}
                    className="px-4 py-2 bg-red-50 dark:bg-red-950/50 text-red-600 dark:text-red-400 rounded-xl font-medium hover:bg-red-100 dark:hover:bg-red-900/50 transition-colors flex items-center gap-2"
                >
                    <StopCircle className="w-4 h-4" /> Cancel
                </button>
            </div>

            {/* Progress Bar */}
            <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl p-6 border border-neutral-100 dark:border-zinc-700">
                <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                        <Loader2 className="w-5 h-5 text-purple-600 dark:text-purple-400 animate-spin" />
                        <span className="font-semibold text-neutral-700 dark:text-zinc-300">
                            Step {jobStatus?.progress.step?.toLocaleString() || 0} / {jobStatus?.progress.total_steps?.toLocaleString() || totalSteps.toLocaleString()}
                        </span>
                    </div>
                    <span className="text-sm text-neutral-500 dark:text-zinc-400">
                        ETA: {formatETA(jobStatus?.progress.eta_seconds || null)}
                    </span>
                </div>

                <div className="h-3 bg-neutral-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${((jobStatus?.progress.step || 0) / (jobStatus?.progress.total_steps || totalSteps)) * 100}%` }}
                        className="h-full bg-gradient-to-r from-purple-500 to-purple-600 rounded-full"
                    />
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-3 gap-4 mt-4">
                    <div className="text-center">
                        <div className="text-2xl font-bold text-neutral-900 dark:text-zinc-100">
                            {jobStatus?.progress.loss?.toFixed(4) || '--'}
                        </div>
                        <div className="text-xs text-neutral-500 dark:text-zinc-400">Loss</div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold text-neutral-900 dark:text-zinc-100">
                            {jobStatus?.progress.learning_rate?.toExponential(2) || '--'}
                        </div>
                        <div className="text-xs text-neutral-500 dark:text-zinc-400">Learning Rate</div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold text-neutral-900 dark:text-zinc-100">
                            {(((jobStatus?.progress.step || 0) / (jobStatus?.progress.total_steps || totalSteps)) * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-neutral-500 dark:text-zinc-400">Progress</div>
                    </div>
                </div>
            </div>

            {/* Loss Graph */}
            {jobStatus?.progress?.loss_history && jobStatus.progress.loss_history.length > 1 && (
                <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-neutral-200 dark:border-zinc-700">
                    <h3 className="text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-3">Training Loss</h3>
                    <div className="h-48">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={jobStatus.progress.loss_history.map(([step, loss]) => ({ step, loss }))}>
                                <XAxis
                                    dataKey="step"
                                    tick={{ fontSize: 10 }}
                                    tickFormatter={(v) => v >= 1000 ? `${(v/1000).toFixed(0)}K` : v}
                                />
                                <YAxis
                                    tick={{ fontSize: 10 }}
                                    domain={['auto', 'auto']}
                                    tickFormatter={(v) => v.toFixed(3)}
                                />
                                <Tooltip
                                    formatter={(value) => [Number(value).toFixed(4), 'Loss']}
                                    labelFormatter={(label) => `Step ${label}`}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="loss"
                                    stroke="#8b5cf6"
                                    strokeWidth={2}
                                    dot={false}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            {/* Logs */}
            <div className="flex-1 bg-neutral-900 dark:bg-zinc-950 rounded-xl p-4 overflow-hidden flex flex-col">
                <h3 className="text-xs font-bold text-neutral-500 dark:text-zinc-500 uppercase tracking-widest mb-2">Training Logs</h3>
                <div className="flex-1 overflow-auto font-mono text-xs text-green-400 space-y-1">
                    {logs.map((log, i) => (
                        <div key={i} className="whitespace-pre-wrap">{log}</div>
                    ))}
                    <div ref={logsEndRef} />
                </div>
            </div>
        </div>
    );
}
