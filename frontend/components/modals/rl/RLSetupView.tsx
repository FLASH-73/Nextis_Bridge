import React, { useState, useEffect } from 'react';
import { Play, Loader2, AlertCircle, ChevronDown, Sparkles, Brain, Eye } from 'lucide-react';
import { rlApi, datasetsApi } from '../../../lib/api';
import { usePolling } from '../../../hooks/usePolling';
import type { RewardClassifier, SARMModel, SARMTrainingStatus, DatasetInfo } from '../../../lib/api/types';

interface RLSetupViewProps {
    onStartTraining: (config: any) => void;
}

export default function RLSetupView({ onStartTraining }: RLSetupViewProps) {
    // Setup state
    const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
    const [classifiers, setClassifiers] = useState<RewardClassifier[]>([]);
    const [sarmModels, setSarmModels] = useState<SARMModel[]>([]);
    const [rewardSource, setRewardSource] = useState<'sarm' | 'gvl' | 'classifier'>('sarm');
    const [selectedDataset, setSelectedDataset] = useState('');
    const [selectedClassifier, setSelectedClassifier] = useState('');
    const [selectedSarmModel, setSelectedSarmModel] = useState('');

    // GVL config
    const [taskDescription, setTaskDescription] = useState('');
    const [gvlQueryInterval, setGvlQueryInterval] = useState(5);
    const [gvlSuccessThreshold, setGvlSuccessThreshold] = useState(0.85);

    // Training config
    const [config, setConfig] = useState({
        max_episodes: 100,
        max_steps_per_episode: 300,
        fps: 30,
        discount: 0.99,
        actor_lr: 0.0003,
        critic_lr: 0.0003,
        batch_size: 32,
        movement_scale: 0.5,
        reset_time_s: 3.0,
        warmup_steps: 50,
    });

    const [error, setError] = useState('');
    const [showAdvanced, setShowAdvanced] = useState(false);

    // Classifier training
    const [isTrainingClassifier, setIsTrainingClassifier] = useState(false);
    const [classifierName, setClassifierName] = useState('');
    const [classifierDataset, setClassifierDataset] = useState('');
    const [classifierStatus, setClassifierStatus] = useState<any>(null);

    // SARM training
    const [isTrainingSarm, setIsTrainingSarm] = useState(false);
    const [sarmName, setSarmName] = useState('');
    const [sarmStatus, setSarmStatus] = useState<SARMTrainingStatus | null>(null);

    // Load datasets, classifiers, and SARM models on mount
    useEffect(() => {
        fetchDatasets();
        fetchClassifiers();
        fetchSarmModels();
    }, []);

    const fetchDatasets = async () => {
        try {
            const data = await datasetsApi.list();
            setDatasets(data || []);
        } catch (e) { console.error('Failed to fetch datasets', e); }
    };

    const fetchClassifiers = async () => {
        try {
            const data = await rlApi.listClassifiers();
            setClassifiers(data.classifiers || []);
        } catch (e) { console.error('Failed to fetch classifiers', e); }
    };

    const fetchSarmModels = async () => {
        try {
            const data = await rlApi.listSarmModels();
            setSarmModels(data.models || []);
        } catch (e) { console.error('Failed to fetch SARM models', e); }
    };

    const fetchSarmStatus = async () => {
        try {
            const data = await rlApi.sarmStatus();
            setSarmStatus(data);
            if (data.status === 'completed' || data.status === 'failed') {
                setIsTrainingSarm(false);
                fetchSarmModels();
                if (data.status === 'completed' && sarmName) {
                    setSelectedSarmModel(sarmName);
                }
            }
        } catch (e) { console.error('Failed to fetch SARM status', e); }
    };

    const fetchClassifierStatus = async () => {
        try {
            const data = await rlApi.classifierStatus();
            setClassifierStatus(data);
            if (data.status === 'completed' || data.status === 'failed') {
                setIsTrainingClassifier(false);
                fetchClassifiers();
            }
        } catch (e) { console.error('Failed to fetch classifier status', e); }
    };

    // Poll classifier training status
    usePolling(fetchClassifierStatus, 2000, isTrainingClassifier);

    // Poll SARM training status
    usePolling(fetchSarmStatus, 2000, isTrainingSarm);

    const trainClassifier = async () => {
        if (!classifierName || !classifierDataset) return;
        setIsTrainingClassifier(true);
        try {
            await rlApi.trainClassifier({
                dataset_repo_id: classifierDataset,
                name: classifierName,
                epochs: 50,
            });
        } catch (e: any) {
            setError(e.message);
            setIsTrainingClassifier(false);
        }
    };

    const trainSarm = async () => {
        if (!sarmName || !selectedDataset) return;
        setIsTrainingSarm(true);
        setSarmStatus(null);
        try {
            await rlApi.trainSarm({
                dataset_repo_id: selectedDataset,
                name: sarmName,
                config: {
                    annotation_mode: 'single_stage',
                    num_epochs: 10,
                },
            });
        } catch (e: any) {
            setError(e.message);
            setIsTrainingSarm(false);
        }
    };

    const startTraining = async () => {
        setError('');
        try {
            const trainingConfig: any = {
                ...config,
                reward_source: rewardSource,
                dataset_repo_id: selectedDataset,
                task_description: taskDescription,
            };

            if (rewardSource === 'sarm') {
                trainingConfig.sarm_model_name = selectedSarmModel;
            } else if (rewardSource === 'gvl') {
                trainingConfig.gvl_query_interval = gvlQueryInterval;
                trainingConfig.gvl_success_threshold = gvlSuccessThreshold;
            } else if (rewardSource === 'classifier') {
                trainingConfig.reward_classifier_name = selectedClassifier;
            }

            const data = await rlApi.startTraining(trainingConfig) as any;

            if (data.status === 'started') {
                onStartTraining(trainingConfig);
            } else {
                setError(data.error || data.message || 'Failed to start training');
            }
        } catch (e: any) {
            setError(e.message || 'Failed to start training');
        }
    };

    return (
        <div className="flex flex-col max-w-lg mx-auto w-full gap-5 animate-in fade-in slide-in-from-bottom-4">
            <div>
                <h2 className="text-2xl font-bold text-neutral-900 dark:text-zinc-100 mb-1">RL Training Setup</h2>
                <p className="text-neutral-500 dark:text-zinc-400 text-sm">Configure HIL-SERL training with online reinforcement learning.</p>
            </div>

            {/* Reward Source Selection - 3 options */}
            <div>
                <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Reward Source</label>
                <div className="grid grid-cols-3 gap-2">
                    {/* SARM - Recommended */}
                    <button
                        onClick={() => setRewardSource('sarm')}
                        className={`p-3 rounded-xl border-2 transition-all text-left ${rewardSource === 'sarm' ? 'border-orange-500 bg-orange-50 dark:bg-orange-900/30' : 'border-neutral-200 dark:border-zinc-700 hover:border-orange-300'}`}
                    >
                        <Brain className={`w-5 h-5 mb-1 ${rewardSource === 'sarm' ? 'text-orange-600 dark:text-orange-400' : 'text-neutral-400 dark:text-zinc-500'}`} />
                        <div className="font-semibold text-sm text-neutral-900 dark:text-zinc-100 flex items-center gap-1">
                            SARM
                            <span className="text-[10px] bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-400 px-1.5 py-0.5 rounded font-medium">Best</span>
                        </div>
                        <div className="text-xs text-neutral-400 dark:text-zinc-500">Learned from demos</div>
                    </button>
                    {/* GVL */}
                    <button
                        onClick={() => setRewardSource('gvl')}
                        className={`p-3 rounded-xl border-2 transition-all text-left ${rewardSource === 'gvl' ? 'border-orange-500 bg-orange-50 dark:bg-orange-900/30' : 'border-neutral-200 dark:border-zinc-700 hover:border-orange-300'}`}
                    >
                        <Sparkles className={`w-5 h-5 mb-1 ${rewardSource === 'gvl' ? 'text-orange-600 dark:text-orange-400' : 'text-neutral-400 dark:text-zinc-500'}`} />
                        <div className="font-semibold text-sm text-neutral-900 dark:text-zinc-100">GVL</div>
                        <div className="text-xs text-neutral-400 dark:text-zinc-500">Zero-shot (Gemini)</div>
                    </button>
                    {/* Classifier */}
                    <button
                        onClick={() => setRewardSource('classifier')}
                        className={`p-3 rounded-xl border-2 transition-all text-left ${rewardSource === 'classifier' ? 'border-orange-500 bg-orange-50 dark:bg-orange-900/30' : 'border-neutral-200 dark:border-zinc-700 hover:border-orange-300'}`}
                    >
                        <Eye className={`w-5 h-5 mb-1 ${rewardSource === 'classifier' ? 'text-orange-600 dark:text-orange-400' : 'text-neutral-400 dark:text-zinc-500'}`} />
                        <div className="font-semibold text-sm text-neutral-900 dark:text-zinc-100">Classifier</div>
                        <div className="text-xs text-neutral-400 dark:text-zinc-500">Binary (0/1)</div>
                    </button>
                </div>
            </div>

            {/* SARM Configuration */}
            {rewardSource === 'sarm' && (
                <div className="bg-orange-50 dark:bg-orange-950/30 rounded-xl p-4 border border-orange-200 dark:border-orange-900 space-y-3">
                    <h3 className="text-sm font-bold text-orange-700 dark:text-orange-300">SARM Reward Model</h3>
                    <p className="text-xs text-orange-600 dark:text-orange-400">
                        Stage-Aware Reward Model learns task progress (0→1) from your demos. Fast local inference, no API calls.
                    </p>

                    {sarmModels.length > 0 ? (
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Select Trained Model</label>
                            <select
                                value={selectedSarmModel}
                                onChange={(e) => setSelectedSarmModel(e.target.value)}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            >
                                <option value="">Select a SARM model...</option>
                                {sarmModels.map(m => (
                                    <option key={m.name} value={m.name}>
                                        {m.name} (from {m.dataset_repo_id})
                                    </option>
                                ))}
                            </select>
                        </div>
                    ) : (
                        <div className="text-sm text-orange-600 dark:text-orange-400 bg-orange-100 dark:bg-orange-900/30 rounded-lg p-2">
                            No SARM models trained yet. Train one below using your demo dataset.
                        </div>
                    )}

                    {/* Task Description for SARM */}
                    <div>
                        <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Task Description</label>
                        <textarea
                            value={taskDescription}
                            onChange={(e) => setTaskDescription(e.target.value)}
                            placeholder="e.g., Pick up the motor and insert it into the housing..."
                            className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100 min-h-[50px]"
                            rows={2}
                        />
                    </div>

                    {/* Train New SARM */}
                    <div className="border-t border-orange-200 dark:border-orange-800 pt-3 mt-2">
                        <h4 className="text-xs font-bold text-orange-600 dark:text-orange-400 mb-2">Train New SARM Model</h4>
                        <div className="flex gap-2">
                            <input
                                value={sarmName}
                                onChange={(e) => setSarmName(e.target.value)}
                                placeholder="Model name (e.g., assembly_v1)"
                                className="flex-1 px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            />
                            <button
                                onClick={trainSarm}
                                disabled={!selectedDataset || !sarmName || isTrainingSarm}
                                className="px-4 py-2 bg-orange-600 text-white rounded-lg text-sm font-medium hover:bg-orange-700 disabled:opacity-50 flex items-center gap-1"
                            >
                                {isTrainingSarm ? (
                                    <><Loader2 className="w-3 h-3 animate-spin" /> Training</>
                                ) : 'Train'}
                            </button>
                        </div>
                        <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-1">
                            Trains on selected dataset below. Takes ~5-10 min on GPU.
                        </p>
                    </div>

                    {/* Training Progress */}
                    {isTrainingSarm && sarmStatus && (
                        <div className="bg-orange-100 dark:bg-orange-900/30 rounded-lg p-3">
                            <div className="flex justify-between text-sm mb-1">
                                <span className="font-medium text-orange-700 dark:text-orange-300">Training SARM...</span>
                                <span className="text-orange-600 dark:text-orange-400">
                                    Epoch {sarmStatus.epoch}/{sarmStatus.total_epochs}
                                </span>
                            </div>
                            <div className="w-full bg-orange-200 dark:bg-orange-800 rounded-full h-2">
                                <div
                                    className="bg-orange-600 h-2 rounded-full transition-all"
                                    style={{ width: `${(sarmStatus.epoch / sarmStatus.total_epochs) * 100}%` }}
                                />
                            </div>
                            {sarmStatus.loss > 0 && (
                                <div className="text-xs text-orange-600 dark:text-orange-400 mt-1">
                                    Loss: {sarmStatus.loss.toFixed(4)}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}

            {/* GVL Configuration */}
            {rewardSource === 'gvl' && (
                <div className="bg-orange-50 dark:bg-orange-950/30 rounded-xl p-4 border border-orange-200 dark:border-orange-900 space-y-3">
                    <h3 className="text-sm font-bold text-orange-700 dark:text-orange-300">GVL Configuration</h3>
                    <div>
                        <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Task Description</label>
                        <textarea
                            value={taskDescription}
                            onChange={(e) => setTaskDescription(e.target.value)}
                            placeholder="e.g., Pick up the motor and insert it into the housing..."
                            className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100 min-h-[60px]"
                            rows={2}
                        />
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Query Interval (steps)</label>
                            <select
                                value={gvlQueryInterval}
                                onChange={(e) => setGvlQueryInterval(parseInt(e.target.value))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            >
                                <option value={3}>Every 3 steps (more API calls)</option>
                                <option value={5}>Every 5 steps (balanced)</option>
                                <option value={10}>Every 10 steps (fewer calls)</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Success Threshold</label>
                            <input
                                type="number"
                                value={gvlSuccessThreshold}
                                onChange={(e) => setGvlSuccessThreshold(parseFloat(e.target.value))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                min={0.5} max={1.0} step={0.05}
                            />
                        </div>
                    </div>
                </div>
            )}

            {/* Classifier Configuration */}
            {rewardSource === 'classifier' && (
                <div className="bg-blue-50 dark:bg-blue-950/30 rounded-xl p-4 border border-blue-200 dark:border-blue-900 space-y-3">
                    <h3 className="text-sm font-bold text-blue-700 dark:text-blue-300">Reward Classifier</h3>
                    {classifiers.length > 0 ? (
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Select Classifier</label>
                            <select
                                value={selectedClassifier}
                                onChange={(e) => setSelectedClassifier(e.target.value)}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            >
                                <option value="">Select...</option>
                                {classifiers.map(c => (
                                    <option key={c.name} value={c.name}>
                                        {c.name} ({(c.accuracy * 100).toFixed(0)}% accuracy)
                                    </option>
                                ))}
                            </select>
                        </div>
                    ) : (
                        <div className="text-sm text-neutral-500 dark:text-zinc-400">No classifiers trained yet.</div>
                    )}

                    {/* Train New Classifier */}
                    <div className="border-t border-blue-200 dark:border-blue-800 pt-3 mt-2">
                        <h4 className="text-xs font-bold text-blue-600 dark:text-blue-400 mb-2">Train New Classifier</h4>
                        <div className="grid grid-cols-2 gap-2">
                            <input
                                value={classifierName}
                                onChange={(e) => setClassifierName(e.target.value)}
                                placeholder="Classifier name"
                                className="px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            />
                            <select
                                value={classifierDataset}
                                onChange={(e) => setClassifierDataset(e.target.value)}
                                className="px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            >
                                <option value="">Dataset...</option>
                                {datasets.map(d => (
                                    <option key={d.repo_id} value={d.repo_id}>{d.repo_id} ({d.total_episodes} eps)</option>
                                ))}
                            </select>
                        </div>
                        <button
                            onClick={trainClassifier}
                            disabled={!classifierName || !classifierDataset || isTrainingClassifier}
                            className="mt-2 w-full py-2 bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded-lg text-sm font-medium hover:bg-blue-200 dark:hover:bg-blue-800/50 disabled:opacity-50 flex items-center justify-center gap-2"
                        >
                            {isTrainingClassifier ? (
                                <><Loader2 className="w-3 h-3 animate-spin" /> Training... {classifierStatus?.epoch || 0}/{classifierStatus?.total_epochs || 50}</>
                            ) : (
                                'Train Classifier'
                            )}
                        </button>
                    </div>
                </div>
            )}

            {/* Demo Dataset (Offline Buffer) */}
            <div>
                <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Demonstration Dataset</label>
                <select
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    className="w-full px-4 py-3 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-xl focus:ring-2 focus:ring-orange-500 outline-none transition-all text-neutral-900 dark:text-zinc-100"
                >
                    <option value="" disabled>Select a dataset...</option>
                    {datasets.map(d => (
                        <option key={d.repo_id} value={d.repo_id}>
                            {d.repo_id} ({d.total_episodes} episodes)
                        </option>
                    ))}
                </select>
                <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-1">Your collected demos are loaded into the offline replay buffer</p>
            </div>

            {/* Core Settings */}
            <div className="grid grid-cols-2 gap-3">
                <div>
                    <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Max Episodes</label>
                    <input
                        type="number"
                        value={config.max_episodes}
                        onChange={(e) => setConfig(c => ({ ...c, max_episodes: parseInt(e.target.value) || 100 }))}
                        className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                        min={1}
                    />
                </div>
                <div>
                    <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Steps per Episode</label>
                    <input
                        type="number"
                        value={config.max_steps_per_episode}
                        onChange={(e) => setConfig(c => ({ ...c, max_steps_per_episode: parseInt(e.target.value) || 300 }))}
                        className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                        min={10}
                    />
                </div>
            </div>

            {/* Movement Scale */}
            <div>
                <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">
                    Movement Scale (Safety): {config.movement_scale.toFixed(1)}x
                </label>
                <input
                    type="range"
                    min={0.1} max={1.0} step={0.1}
                    value={config.movement_scale}
                    onChange={(e) => setConfig(c => ({ ...c, movement_scale: parseFloat(e.target.value) }))}
                    className="w-full"
                />
                <div className="flex justify-between text-xs text-neutral-400 dark:text-zinc-500">
                    <span>0.1x (safest)</span>
                    <span>1.0x (full speed)</span>
                </div>
            </div>

            {/* Advanced Settings */}
            <div>
                <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex items-center gap-2 text-sm font-medium text-neutral-600 dark:text-zinc-400 hover:text-neutral-800 dark:hover:text-zinc-200"
                >
                    <ChevronDown className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
                    Advanced RL Settings
                </button>
                {showAdvanced && (
                    <div className="mt-3 p-4 bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700 space-y-3">
                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Discount (gamma)</label>
                                <input
                                    type="number"
                                    value={config.discount}
                                    onChange={(e) => setConfig(c => ({ ...c, discount: parseFloat(e.target.value) }))}
                                    className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                    min={0.9} max={0.999} step={0.01}
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Batch Size</label>
                                <select
                                    value={config.batch_size}
                                    onChange={(e) => setConfig(c => ({ ...c, batch_size: parseInt(e.target.value) }))}
                                    className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                >
                                    <option value={16}>16</option>
                                    <option value={32}>32 (default)</option>
                                    <option value={64}>64</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Actor LR</label>
                                <select
                                    value={config.actor_lr}
                                    onChange={(e) => setConfig(c => ({ ...c, actor_lr: parseFloat(e.target.value) }))}
                                    className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                >
                                    <option value={0.0001}>1e-4</option>
                                    <option value={0.0003}>3e-4 (default)</option>
                                    <option value={0.001}>1e-3</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Warmup Steps</label>
                                <input
                                    type="number"
                                    value={config.warmup_steps}
                                    onChange={(e) => setConfig(c => ({ ...c, warmup_steps: parseInt(e.target.value) }))}
                                    className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                    min={0}
                                />
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {error && (
                <div className="bg-red-50 dark:bg-red-950/50 text-red-600 dark:text-red-400 px-4 py-3 rounded-xl text-sm flex items-center gap-2 border border-red-100 dark:border-red-900">
                    <AlertCircle className="w-4 h-4" /> {error}
                </div>
            )}

            {/* Start Button */}
            <button
                onClick={startTraining}
                disabled={
                    !selectedDataset ||
                    (rewardSource === 'sarm' && !selectedSarmModel) ||
                    (rewardSource === 'gvl' && !taskDescription) ||
                    (rewardSource === 'classifier' && !selectedClassifier)
                }
                className="w-full py-4 bg-orange-600 text-white rounded-2xl font-bold text-lg hover:bg-orange-700 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-xl shadow-orange-200 dark:shadow-orange-900/30 flex items-center justify-center gap-2"
            >
                <Play className="w-5 h-5" /> Start RL Training
            </button>
        </div>
    );
}
