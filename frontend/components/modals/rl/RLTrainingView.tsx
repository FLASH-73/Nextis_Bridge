import React, { useState } from 'react';
import { Play, StopCircle, Pause, AlertCircle, CheckCircle, Hand, Brain, Eye } from 'lucide-react';
import { motion } from 'framer-motion';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts';
import { rlApi } from '../../../lib/api';
import { usePolling } from '../../../hooks/usePolling';
import type { RLStatus } from '../../../lib/api/types';

interface RLTrainingViewProps {
    rewardSource: 'sarm' | 'gvl' | 'classifier';
    onComplete: () => void;
    onReset: () => void;
}

export default function RLTrainingView({ rewardSource, onComplete, onReset }: RLTrainingViewProps) {
    const [rlStatus, setRlStatus] = useState<RLStatus | null>(null);
    const [movementScale, setMovementScale] = useState(0.5);
    const [viewState, setViewState] = useState<'training' | 'complete'>('training');

    const fetchRLStatus = async () => {
        try {
            const data = await rlApi.trainingStatus();
            setRlStatus(data);
            if (data.status === 'completed' || data.status === 'failed') {
                setViewState('complete');
                onComplete();
            }
        } catch (e) { console.error('Failed to fetch RL status', e); }
    };

    // Poll training status
    usePolling(fetchRLStatus, 1500, viewState === 'training');

    const stopTraining = async () => {
        if (!confirm('Stop RL training? Current policy will be saved.')) return;
        try {
            await rlApi.stopTraining();
        } catch (e) { console.error('Failed to stop training', e); }
    };

    const pauseTraining = async () => {
        try {
            await rlApi.pauseTraining();
        } catch (e) { console.error('Failed to pause training', e); }
    };

    const resumeTraining = async () => {
        try {
            await rlApi.resumeTraining();
        } catch (e) { console.error('Failed to resume training', e); }
    };

    const updateMovementScale = async (scale: number) => {
        setMovementScale(scale);
        try {
            await rlApi.updateSettings({ movement_scale: scale });
        } catch (e) { console.error('Failed to update movement scale', e); }
    };

    // COMPLETE VIEW
    if (viewState === 'complete') {
        return (
            <div className="flex flex-col items-center justify-center h-full gap-6 animate-in fade-in">
                {rlStatus?.status === 'completed' ? (
                    <>
                        <div className="w-20 h-20 bg-green-100 dark:bg-green-900/50 rounded-full flex items-center justify-center">
                            <CheckCircle className="w-10 h-10 text-green-600 dark:text-green-400" />
                        </div>
                        <div className="text-center">
                            <h2 className="text-2xl font-bold text-neutral-900 dark:text-zinc-100 mb-2">RL Training Complete!</h2>
                            <p className="text-neutral-500 dark:text-zinc-400">Policy has been saved.</p>
                        </div>
                        <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl p-4 border border-neutral-100 dark:border-zinc-700 w-full max-w-md space-y-2">
                            <div className="text-sm text-neutral-600 dark:text-zinc-400">
                                Episodes: <span className="font-medium text-neutral-900 dark:text-zinc-100">{rlStatus.episode}</span>
                            </div>
                            <div className="text-sm text-neutral-600 dark:text-zinc-400">
                                Avg Reward: <span className="font-medium text-neutral-900 dark:text-zinc-100">{rlStatus.avg_reward.toFixed(3)}</span>
                            </div>
                            <div className="text-sm text-neutral-600 dark:text-zinc-400">
                                Final Intervention Rate: <span className="font-medium text-neutral-900 dark:text-zinc-100">{(rlStatus.intervention_rate * 100).toFixed(0)}%</span>
                            </div>
                            <div className="text-sm text-neutral-600 dark:text-zinc-400">
                                Training Steps: <span className="font-medium text-neutral-900 dark:text-zinc-100">{rlStatus.training_step.toLocaleString()}</span>
                            </div>
                        </div>
                    </>
                ) : (
                    <>
                        <div className="w-20 h-20 bg-red-100 dark:bg-red-900/50 rounded-full flex items-center justify-center">
                            <AlertCircle className="w-10 h-10 text-red-600 dark:text-red-400" />
                        </div>
                        <div className="text-center">
                            <h2 className="text-2xl font-bold text-neutral-900 dark:text-zinc-100 mb-2">Training Failed</h2>
                            <p className="text-neutral-500 dark:text-zinc-400">{rlStatus?.error || 'An error occurred.'}</p>
                        </div>
                    </>
                )}

                <button
                    onClick={onReset}
                    className="px-8 py-3 bg-orange-600 text-white rounded-xl font-semibold hover:bg-orange-700 transition-all"
                >
                    Start New Training
                </button>
            </div>
        );
    }

    // TRAINING VIEW (waiting for first status)
    if (!rlStatus) {
        return (
            <div className="flex flex-col items-center justify-center h-full gap-4 animate-in fade-in">
                <div className="w-10 h-10 border-4 border-orange-200 border-t-orange-600 rounded-full animate-spin" />
                <p className="text-neutral-500 dark:text-zinc-400 text-sm">Waiting for training status...</p>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-full gap-4 animate-in fade-in">
            {/* Status Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-bold text-neutral-900 dark:text-zinc-100 flex items-center gap-2">
                        {rlStatus.is_human_intervening && <Hand className="w-5 h-5 text-amber-500 animate-pulse" />}
                        {rlStatus.status === 'paused' ? 'Training Paused' : 'RL Training Live'}
                    </h2>
                    <p className="text-neutral-500 dark:text-zinc-400 text-sm">
                        Episode {rlStatus.episode}/{rlStatus.total_episodes} | Step {rlStatus.episode_step}
                    </p>
                </div>
                <div className="flex gap-2">
                    {rlStatus.status === 'running' ? (
                        <button onClick={pauseTraining} className="px-3 py-2 bg-amber-50 dark:bg-amber-950/50 text-amber-600 dark:text-amber-400 rounded-xl font-medium hover:bg-amber-100 transition-colors flex items-center gap-1">
                            <Pause className="w-4 h-4" /> Pause
                        </button>
                    ) : (
                        <button onClick={resumeTraining} className="px-3 py-2 bg-green-50 dark:bg-green-950/50 text-green-600 dark:text-green-400 rounded-xl font-medium hover:bg-green-100 transition-colors flex items-center gap-1">
                            <Play className="w-4 h-4" /> Resume
                        </button>
                    )}
                    <button onClick={stopTraining} className="px-3 py-2 bg-red-50 dark:bg-red-950/50 text-red-600 dark:text-red-400 rounded-xl font-medium hover:bg-red-100 transition-colors flex items-center gap-1">
                        <StopCircle className="w-4 h-4" /> Stop
                    </button>
                </div>
            </div>

            {/* Progress Bar */}
            <div className="h-2 bg-neutral-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(rlStatus.episode / rlStatus.total_episodes) * 100}%` }}
                    className="h-full bg-gradient-to-r from-orange-500 to-orange-600 rounded-full"
                />
            </div>

            {/* Live Metrics Grid */}
            <div className="grid grid-cols-4 gap-3">
                <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl p-3 text-center border border-neutral-100 dark:border-zinc-700">
                    <div className="text-xl font-bold text-neutral-900 dark:text-zinc-100">
                        {rlStatus.avg_reward.toFixed(2)}
                    </div>
                    <div className="text-xs text-neutral-500 dark:text-zinc-400">Avg Reward</div>
                </div>
                <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl p-3 text-center border border-neutral-100 dark:border-zinc-700">
                    <div className="text-xl font-bold text-neutral-900 dark:text-zinc-100">
                        {(rlStatus.intervention_rate * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-neutral-500 dark:text-zinc-400">Intervention Rate</div>
                </div>
                <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl p-3 text-center border border-neutral-100 dark:border-zinc-700">
                    <div className="text-xl font-bold text-neutral-900 dark:text-zinc-100">
                        {rlStatus.loss_critic.toFixed(4)}
                    </div>
                    <div className="text-xs text-neutral-500 dark:text-zinc-400">Critic Loss</div>
                </div>
                <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl p-3 text-center border border-neutral-100 dark:border-zinc-700">
                    <div className="text-xl font-bold text-neutral-900 dark:text-zinc-100">
                        {rlStatus.online_buffer_size.toLocaleString()}
                    </div>
                    <div className="text-xs text-neutral-500 dark:text-zinc-400">Buffer Size</div>
                </div>
            </div>

            {/* Reward Chart */}
            {rlStatus.episode_rewards && rlStatus.episode_rewards.length > 1 && (
                <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-neutral-200 dark:border-zinc-700 flex-1 min-h-[180px]">
                    <h3 className="text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Episode Rewards</h3>
                    <div className="h-[160px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={rlStatus.episode_rewards.map((r, i) => ({ episode: i + 1, reward: r }))}>
                                <XAxis dataKey="episode" tick={{ fontSize: 10 }} />
                                <YAxis tick={{ fontSize: 10 }} domain={[0, 'auto']} />
                                <Tooltip formatter={(v) => [Number(v).toFixed(3), 'Reward']} />
                                <Area type="monotone" dataKey="reward" stroke="#f97316" fill="#fed7aa" fillOpacity={0.3} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            {/* Live Movement Scale */}
            <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl p-3 border border-neutral-100 dark:border-zinc-700">
                <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-neutral-600 dark:text-zinc-400">
                        Movement Scale: {movementScale.toFixed(1)}x
                    </span>
                    {rlStatus.is_human_intervening && (
                        <span className="text-xs text-amber-600 dark:text-amber-400 flex items-center gap-1">
                            <Hand className="w-3 h-3" /> Human intervening
                        </span>
                    )}
                </div>
                <input
                    type="range"
                    min={0.1} max={1.0} step={0.1}
                    value={movementScale}
                    onChange={(e) => updateMovementScale(parseFloat(e.target.value))}
                    className="w-full mt-1"
                />
            </div>

            {/* Reward Source Status */}
            <div className="text-xs text-neutral-500 dark:text-zinc-400 flex items-center gap-4">
                {rewardSource === 'sarm' && (
                    <>
                        <span className="flex items-center gap-1">
                            <Brain className="w-3 h-3" /> SARM
                        </span>
                        <span>Current Progress: {(rlStatus.current_reward * 100).toFixed(0)}%</span>
                    </>
                )}
                {rewardSource === 'gvl' && rlStatus.gvl_queries > 0 && (
                    <>
                        <span>GVL Queries: {rlStatus.gvl_queries}</span>
                        <span>Avg Latency: {rlStatus.gvl_avg_latency_ms.toFixed(0)}ms</span>
                        <span>Current Reward: {rlStatus.current_reward.toFixed(2)}</span>
                    </>
                )}
                {rewardSource === 'classifier' && (
                    <>
                        <span className="flex items-center gap-1">
                            <Eye className="w-3 h-3" /> Classifier
                        </span>
                        <span>Current Reward: {rlStatus.current_reward.toFixed(2)}</span>
                    </>
                )}
            </div>
        </div>
    );
}
