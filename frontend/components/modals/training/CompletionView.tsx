import React from 'react';
import { StopCircle, AlertCircle, CheckCircle } from 'lucide-react';

export interface TrainingProgress {
    step: number;
    total_steps: number;
    loss: number | null;
    learning_rate: number | null;
    eta_seconds: number | null;
    epoch: number;
    loss_history: [number, number][];
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

interface CompletionViewProps {
    jobStatus: TrainingJob | null;
    logs: string[];
    onReset: () => void;
}

export default function CompletionView({ jobStatus, logs, onReset }: CompletionViewProps) {
    return (
        <div className="flex flex-col items-center justify-center h-full gap-6 animate-in fade-in">
            {jobStatus?.status === 'completed' ? (
                <>
                    <div className="w-20 h-20 bg-green-100 dark:bg-green-900/50 rounded-full flex items-center justify-center">
                        <CheckCircle className="w-10 h-10 text-green-600 dark:text-green-400" />
                    </div>
                    <div className="text-center">
                        <h2 className="text-2xl font-bold text-neutral-900 dark:text-zinc-100 mb-2">Training Complete!</h2>
                        <p className="text-neutral-500 dark:text-zinc-400">Your model has been trained successfully.</p>
                    </div>
                    <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl p-4 border border-neutral-100 dark:border-zinc-700 w-full max-w-md">
                        <div className="text-sm text-neutral-600 dark:text-zinc-400 space-y-2">
                            <div>Final Loss: <span className="font-medium text-neutral-900 dark:text-zinc-100">{jobStatus?.progress.loss?.toFixed(4) || '--'}</span></div>
                            <div>Total Steps: <span className="font-medium text-neutral-900 dark:text-zinc-100">{jobStatus?.progress.step?.toLocaleString()}</span></div>
                            <div>Output: <span className="font-mono text-xs text-neutral-700 dark:text-zinc-300">{jobStatus?.output_dir || '--'}</span></div>
                        </div>
                    </div>
                </>
            ) : jobStatus?.status === 'cancelled' ? (
                <>
                    <div className="w-20 h-20 bg-amber-100 dark:bg-amber-900/50 rounded-full flex items-center justify-center">
                        <StopCircle className="w-10 h-10 text-amber-600 dark:text-amber-400" />
                    </div>
                    <div className="text-center">
                        <h2 className="text-2xl font-bold text-neutral-900 dark:text-zinc-100 mb-2">Training Cancelled</h2>
                        <p className="text-neutral-500 dark:text-zinc-400">Training was stopped before completion.</p>
                    </div>
                </>
            ) : (
                <>
                    <div className="w-20 h-20 bg-red-100 dark:bg-red-900/50 rounded-full flex items-center justify-center">
                        <AlertCircle className="w-10 h-10 text-red-600 dark:text-red-400" />
                    </div>
                    <div className="text-center">
                        <h2 className="text-2xl font-bold text-neutral-900 dark:text-zinc-100 mb-2">Training Failed</h2>
                        <p className="text-neutral-500 dark:text-zinc-400">{jobStatus?.error || 'An error occurred during training.'}</p>
                    </div>

                    {/* Show logs on failure so user can see what went wrong */}
                    {logs.length > 0 && (
                        <div className="w-full max-w-4xl bg-neutral-900 dark:bg-zinc-950 rounded-xl p-4 max-h-[32rem] overflow-auto">
                            <h3 className="text-xs font-bold text-neutral-500 uppercase tracking-widest mb-2">Error Logs ({logs.length} lines)</h3>
                            <div className="font-mono text-xs text-red-400 space-y-1">
                                {logs.map((log, i) => (
                                    <div key={i} className="whitespace-pre-wrap">{log}</div>
                                ))}
                            </div>
                        </div>
                    )}
                </>
            )}

            <button
                onClick={onReset}
                className="px-8 py-3 bg-purple-600 text-white rounded-xl font-semibold hover:bg-purple-700 transition-all"
            >
                Start New Training
            </button>
        </div>
    );
}
