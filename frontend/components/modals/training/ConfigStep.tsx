import React, { useState } from 'react';
import { Play, AlertCircle, CheckCircle, Loader2, ChevronDown } from 'lucide-react';
import { trainingApi } from '../../../lib/api';

export type PolicyType = 'smolvla' | 'diffusion' | 'act' | 'pi05';

export interface HardwareDevice {
    id: string;
    type: 'cuda' | 'mps' | 'cpu';
    name: string;
    memory_gb: number | null;
    recommended: boolean;
}

export interface HardwareInfo {
    devices: HardwareDevice[];
    default: string;
}

export interface TrainingConfig {
    steps: number;
    batch_size: number;
    learning_rate: number;
    warmup_steps: number;
    freeze_vision_encoder: boolean;
    train_expert_only: boolean;
    device: string;
    // Diffusion-specific
    n_obs_steps: number;
    horizon: number;
    n_action_steps: number;
    noise_scheduler_type: string;
    num_train_timesteps: number;
    vision_backbone: string;
    // Pi0.5-specific
    pretrained_path: string;
    compile_model: boolean;
    gradient_checkpointing: boolean;
    dtype: string;
    use_quantile_normalization: boolean;
    chunk_size: number;
    n_action_steps_pi05: number;
    // LoRA settings
    lora_rank: number;
    lora_alpha: number;
    lora_dropout: number;
    // ACT-specific
    chunk_size_act: number;
    n_action_steps_act: number;
    use_vae: boolean;
    latent_dim: number;
    kl_weight: number;
    dim_model: number;
}

export interface QuantileStats {
    has_quantiles: boolean;
    missing_features: string[];
    message: string;
}

const PRESETS = {
    quick: { steps: 10000, batch_size: 8, description: 'Quick test (~10 min)' },
    standard: { steps: 100000, batch_size: 8, description: 'Standard (~2 hrs)' },
    full: { steps: 200000, batch_size: 16, description: 'Best quality (~4 hrs)' },
    custom: { steps: 100000, batch_size: 8, description: 'Your parameters' },
};

interface ConfigStepProps {
    policyType: PolicyType;
    selectedDataset: string;
    policyName: string;
    config: TrainingConfig;
    setConfig: React.Dispatch<React.SetStateAction<TrainingConfig>>;
    preset: 'quick' | 'standard' | 'full' | 'custom';
    setPreset: (preset: 'quick' | 'standard' | 'full' | 'custom') => void;
    hardware: HardwareInfo | null;
    quantileStats: QuantileStats | null;
    isComputingQuantiles: boolean;
    computeQuantileStats: () => Promise<void>;
    error: string;
    setError: (e: string) => void;
    onBack: () => void;
    onStartTraining: (config: Record<string, any>) => void;
}

export default function ConfigStep({
    policyType,
    selectedDataset,
    policyName,
    config,
    setConfig,
    preset,
    setPreset,
    hardware,
    quantileStats,
    isComputingQuantiles,
    computeQuantileStats,
    error,
    setError,
    onBack,
    onStartTraining,
}: ConfigStepProps) {
    const [showAdvanced, setShowAdvanced] = useState(false);

    const startTraining = async () => {
        setError('');

        try {
            // Build config based on policy type
            let trainingConfig: Record<string, any> = {
                ...config,
                policy_name: policyName || undefined,
            };

            // Filter and map policy-specific options
            if (policyType === 'pi05') {
                // Remove SmolVLA-specific options that are not valid for Pi0.5
                const {
                    freeze_vision_encoder,
                    train_expert_only,
                    train_state_proj,
                    // Also remove diffusion-specific options
                    horizon,
                    noise_scheduler_type,
                    num_train_timesteps,
                    vision_backbone,
                    // Remove ACT-specific options
                    chunk_size_act,
                    n_action_steps_act,
                    use_vae,
                    latent_dim,
                    kl_weight,
                    dim_model,
                    ...pi05Config
                } = trainingConfig;
                trainingConfig = {
                    ...pi05Config,
                    n_action_steps: config.n_action_steps_pi05,
                    // LoRA settings are already in config with correct names
                };
            } else if (policyType === 'act') {
                // Remove SmolVLA/Diffusion/Pi0.5-specific options
                const {
                    freeze_vision_encoder,
                    train_expert_only,
                    train_state_proj,
                    horizon,
                    noise_scheduler_type,
                    num_train_timesteps,
                    n_obs_steps,
                    // Remove Pi0.5-specific
                    pretrained_path,
                    compile_model,
                    gradient_checkpointing,
                    dtype,
                    use_quantile_normalization,
                    chunk_size,
                    n_action_steps_pi05,
                    lora_rank,
                    lora_alpha,
                    lora_dropout,
                    // Remove ACT internal keys (remap below)
                    chunk_size_act,
                    n_action_steps_act,
                    ...actConfig
                } = trainingConfig;
                trainingConfig = {
                    ...actConfig,
                    chunk_size: config.chunk_size_act,
                    n_action_steps: config.n_action_steps_act,
                    n_obs_steps: 1,
                };
            }

            const data: any = await trainingApi.start({
                dataset_repo_id: selectedDataset,
                policy_type: policyType,
                config: trainingConfig,
            });

            if (data.status === 'started') {
                onStartTraining({ jobId: data.job_id });
            } else {
                setError(data.error || 'Failed to start training');
            }
        } catch (e: any) {
            setError(e.message || 'Failed to start training');
        }
    };

    return (
        <div className="flex flex-col max-w-lg mx-auto w-full gap-6 animate-in fade-in slide-in-from-bottom-4">
            <div>
                <h2 className="text-2xl font-bold text-neutral-900 dark:text-zinc-100 mb-1">Training Configuration</h2>
                <p className="text-neutral-500 dark:text-zinc-400 text-sm">Select hardware, a preset, or customize settings.</p>
            </div>

            {/* Hardware Selection */}
            <div>
                <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Training Device</label>
                {hardware ? (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        {hardware.devices.map((device) => (
                            <button
                                key={device.id}
                                onClick={() => setConfig(c => ({ ...c, device: device.id }))}
                                className={`p-4 rounded-xl border-2 transition-all text-left ${
                                    config.device === device.id
                                        ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/30'
                                        : 'border-neutral-200 dark:border-zinc-700 hover:border-purple-300'
                                }`}
                            >
                                <div className="flex items-center gap-2">
                                    <div className={`w-2 h-2 rounded-full ${
                                        device.type === 'cuda' ? 'bg-green-500' :
                                        device.type === 'mps' ? 'bg-blue-500' : 'bg-neutral-400'
                                    }`} />
                                    <span className="font-semibold text-sm text-neutral-900 dark:text-zinc-100">{device.name}</span>
                                    {device.recommended && (
                                        <span className="text-xs bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300 px-1.5 py-0.5 rounded">Recommended</span>
                                    )}
                                </div>
                                <div className="text-xs text-neutral-400 dark:text-zinc-500 mt-1">
                                    {device.type === 'cuda' && device.memory_gb && `${device.memory_gb} GB VRAM`}
                                    {device.type === 'mps' && 'Apple Silicon GPU'}
                                    {device.type === 'cpu' && 'Slower, but works everywhere'}
                                </div>
                            </button>
                        ))}
                    </div>
                ) : (
                    <div className="text-sm text-neutral-500 dark:text-zinc-400">Detecting hardware...</div>
                )}
            </div>

            {/* Presets */}
            <div>
                <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Preset</label>
                <div className="grid grid-cols-4 gap-2">
                    {Object.entries(PRESETS).map(([key, val]) => (
                        <button
                            key={key}
                            onClick={() => setPreset(key as any)}
                            className={`p-3 rounded-xl border-2 transition-all text-left ${
                                preset === key ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/30' : 'border-neutral-200 dark:border-zinc-700 hover:border-purple-300'
                            }`}
                        >
                            <div className="font-semibold text-sm capitalize text-neutral-900 dark:text-zinc-100">{key}</div>
                            <div className="text-xs text-neutral-400 dark:text-zinc-500">{val.description}</div>
                        </button>
                    ))}
                </div>
            </div>

            {/* Custom Parameters - shown when custom preset selected */}
            {preset === 'custom' && (
                <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl p-4 border border-neutral-200 dark:border-zinc-700 space-y-4">
                    <h3 className="text-sm font-bold text-neutral-700 dark:text-zinc-300">Custom Parameters</h3>

                    <div className="grid grid-cols-2 gap-4">
                        {/* Steps */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Training Steps</label>
                            <input
                                type="number"
                                value={config.steps}
                                onChange={(e) => setConfig(c => ({ ...c, steps: parseInt(e.target.value) || 10000 }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                min={1000}
                                step={1000}
                            />
                        </div>

                        {/* Batch Size */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Batch Size</label>
                            <select
                                value={config.batch_size}
                                onChange={(e) => setConfig(c => ({ ...c, batch_size: parseInt(e.target.value) }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            >
                                <option value={4}>4 (low VRAM)</option>
                                <option value={8}>8 (default)</option>
                                <option value={16}>16 (high VRAM)</option>
                                <option value={32}>32 (very high VRAM)</option>
                            </select>
                        </div>

                        {/* Learning Rate */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Learning Rate</label>
                            <select
                                value={config.learning_rate}
                                onChange={(e) => setConfig(c => ({ ...c, learning_rate: parseFloat(e.target.value) }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            >
                                <option value={0.00001}>1e-5 (conservative)</option>
                                <option value={0.0001}>1e-4 (default)</option>
                                <option value={0.001}>1e-3 (aggressive)</option>
                            </select>
                        </div>

                        {/* Warmup Steps */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Warmup Steps</label>
                            <input
                                type="number"
                                value={config.warmup_steps}
                                onChange={(e) => setConfig(c => ({ ...c, warmup_steps: parseInt(e.target.value) || 1000 }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                min={0}
                                step={100}
                            />
                        </div>
                    </div>
                </div>
            )}

            {/* Diffusion Policy Settings */}
            {policyType === 'diffusion' && (
                <div className="bg-blue-50 dark:bg-blue-950/30 rounded-xl p-4 border border-blue-200 dark:border-blue-900 space-y-4">
                    <h3 className="text-sm font-bold text-blue-700 dark:text-blue-300">Diffusion Policy Settings</h3>
                    <div className="grid grid-cols-2 gap-4">
                        {/* Observation Steps */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Observation Steps</label>
                            <input
                                type="number"
                                value={config.n_obs_steps}
                                onChange={(e) => setConfig(c => ({ ...c, n_obs_steps: parseInt(e.target.value) || 2 }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                min={1}
                                max={10}
                            />
                        </div>

                        {/* Horizon */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Horizon (Action Chunk)</label>
                            <select
                                value={config.horizon}
                                onChange={(e) => setConfig(c => ({ ...c, horizon: parseInt(e.target.value) }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            >
                                <option value={8}>8 (short)</option>
                                <option value={16}>16 (default)</option>
                                <option value={32}>32 (long)</option>
                            </select>
                        </div>

                        {/* Noise Scheduler */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Noise Scheduler</label>
                            <select
                                value={config.noise_scheduler_type}
                                onChange={(e) => setConfig(c => ({ ...c, noise_scheduler_type: e.target.value }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            >
                                <option value="DDPM">DDPM (default)</option>
                                <option value="DDIM">DDIM (faster inference)</option>
                            </select>
                        </div>

                        {/* Diffusion Steps */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Diffusion Steps</label>
                            <select
                                value={config.num_train_timesteps}
                                onChange={(e) => setConfig(c => ({ ...c, num_train_timesteps: parseInt(e.target.value) }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            >
                                <option value={50}>50 (fast)</option>
                                <option value={100}>100 (default)</option>
                                <option value={200}>200 (high quality)</option>
                            </select>
                        </div>
                    </div>
                </div>
            )}

            {/* Pi0.5 Policy Settings */}
            {policyType === 'pi05' && (
                <div className="bg-purple-50 dark:bg-purple-950/30 rounded-xl p-4 border border-purple-200 dark:border-purple-900 space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="text-sm font-bold text-purple-700 dark:text-purple-300">Pi0.5 Policy Settings</h3>
                        <div className={`flex items-center gap-2 text-xs ${config.lora_rank > 0 ? 'text-green-600 dark:text-green-400' : 'text-amber-600 dark:text-amber-400'}`}>
                            {config.lora_rank > 0 ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
                            {config.lora_rank > 0 ? `LoRA enabled (~22GB)` : 'Full fine-tune (~40GB+)'}
                        </div>
                    </div>

                    {/* Quantile Stats Warning */}
                    {quantileStats && !quantileStats.has_quantiles && config.use_quantile_normalization && (
                        <div className="bg-amber-50 dark:bg-amber-950/50 border border-amber-200 dark:border-amber-800 rounded-lg p-3">
                            <div className="flex items-center gap-2 text-amber-700 dark:text-amber-300 text-sm font-medium mb-2">
                                <AlertCircle className="w-4 h-4" />
                                Missing Quantile Statistics
                            </div>
                            <p className="text-xs text-amber-600 dark:text-amber-400 mb-2">
                                {quantileStats.message}
                            </p>
                            <div className="flex gap-2">
                                <button
                                    onClick={computeQuantileStats}
                                    disabled={isComputingQuantiles}
                                    className="px-3 py-1.5 bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-300 rounded-lg text-xs font-medium hover:bg-amber-200 dark:hover:bg-amber-800/50 disabled:opacity-50 flex items-center gap-1"
                                >
                                    {isComputingQuantiles ? <><Loader2 className="w-3 h-3 animate-spin" /> Computing...</> : 'Compute Quantiles'}
                                </button>
                                <button
                                    onClick={() => setConfig(c => ({ ...c, use_quantile_normalization: false }))}
                                    className="px-3 py-1.5 bg-neutral-100 dark:bg-zinc-800 text-neutral-600 dark:text-zinc-400 rounded-lg text-xs font-medium hover:bg-neutral-200 dark:hover:bg-zinc-700"
                                >
                                    Use MEAN_STD Fallback
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Quantile Stats Success */}
                    {quantileStats?.has_quantiles && config.use_quantile_normalization && (
                        <div className="bg-green-50 dark:bg-green-950/50 border border-green-200 dark:border-green-800 rounded-lg p-3">
                            <div className="flex items-center gap-2 text-green-700 dark:text-green-300 text-sm font-medium">
                                <CheckCircle className="w-4 h-4" />
                                Quantile statistics available
                            </div>
                        </div>
                    )}

                    <div className="grid grid-cols-2 gap-4">
                        {/* Pretrained Model */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Pretrained Model</label>
                            <select
                                value={config.pretrained_path}
                                onChange={(e) => setConfig(c => ({ ...c, pretrained_path: e.target.value }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            >
                                <option value="lerobot/pi05_base">Pi0.5 Base (General)</option>
                                <option value="lerobot/pi05_libero">Pi0.5 Libero (Sim-trained)</option>
                            </select>
                        </div>

                        {/* Data Type */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Precision (dtype)</label>
                            <select
                                value={config.dtype}
                                onChange={(e) => setConfig(c => ({ ...c, dtype: e.target.value }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            >
                                <option value="bfloat16">bfloat16 (recommended)</option>
                                <option value="float32">float32 (higher memory)</option>
                            </select>
                        </div>

                        {/* Chunk Size */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Chunk Size (Action Horizon)</label>
                            <select
                                value={config.chunk_size}
                                onChange={(e) => setConfig(c => ({ ...c, chunk_size: parseInt(e.target.value) }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            >
                                <option value={25}>25 (short horizon)</option>
                                <option value={50}>50 (default)</option>
                                <option value={100}>100 (long horizon)</option>
                            </select>
                        </div>

                        {/* Action Steps */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Action Steps to Execute</label>
                            <input
                                type="number"
                                value={config.n_action_steps_pi05}
                                onChange={(e) => setConfig(c => ({ ...c, n_action_steps_pi05: parseInt(e.target.value) || 50 }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                min={1}
                                max={config.chunk_size}
                            />
                        </div>
                    </div>

                    {/* LoRA Settings */}
                    <div className="bg-green-50 dark:bg-green-950/30 border border-green-200 dark:border-green-800 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                            <label className="flex items-center gap-2 text-sm font-medium text-green-700 dark:text-green-300">
                                <input
                                    type="checkbox"
                                    checked={config.lora_rank > 0}
                                    onChange={e => setConfig({ ...config, lora_rank: e.target.checked ? 8 : 0 })}
                                    className="rounded"
                                />
                                Enable LoRA (Low-Rank Adaptation)
                            </label>
                            <span className="text-xs text-green-600 dark:text-green-400">
                                Recommended for small datasets
                            </span>
                        </div>
                        {config.lora_rank > 0 && (
                            <div className="flex items-center gap-4 mt-2">
                                <div>
                                    <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">LoRA Rank</label>
                                    <select
                                        value={config.lora_rank}
                                        onChange={(e) => setConfig(c => ({ ...c, lora_rank: parseInt(e.target.value) }))}
                                        className="px-2 py-1 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                    >
                                        <option value={4}>4 (minimal)</option>
                                        <option value={8}>8 (recommended)</option>
                                        <option value={16}>16 (more capacity)</option>
                                        <option value={32}>32 (high capacity)</option>
                                    </select>
                                </div>
                                <div className="text-xs text-neutral-500 dark:text-zinc-500">
                                    Higher rank = more trainable params, better quality, more memory
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Checkboxes */}
                    <div className="flex flex-wrap items-center gap-4 pt-2">
                        <label className="flex items-center gap-2 text-sm text-neutral-700 dark:text-zinc-300">
                            <input
                                type="checkbox"
                                checked={config.gradient_checkpointing}
                                onChange={e => setConfig({ ...config, gradient_checkpointing: e.target.checked })}
                                className="rounded"
                            />
                            Gradient Checkpointing (saves VRAM)
                        </label>
                        <label className="flex items-center gap-2 text-sm text-neutral-700 dark:text-zinc-300">
                            <input
                                type="checkbox"
                                checked={config.compile_model}
                                onChange={e => setConfig({ ...config, compile_model: e.target.checked })}
                                className="rounded"
                            />
                            Compile Model (faster training)
                        </label>
                        <label className="flex items-center gap-2 text-sm text-neutral-700 dark:text-zinc-300">
                            <input
                                type="checkbox"
                                checked={config.use_quantile_normalization}
                                onChange={e => setConfig({ ...config, use_quantile_normalization: e.target.checked })}
                                className="rounded"
                            />
                            Use Quantile Normalization
                        </label>
                    </div>

                    {/* Memory Info */}
                    <div className={`text-xs px-3 py-2 rounded-lg ${config.lora_rank > 0 ? 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30' : 'text-amber-600 dark:text-amber-400 bg-amber-100 dark:bg-amber-900/30'}`}>
                        <strong>Memory Requirements:</strong> {config.lora_rank > 0
                            ? `With LoRA rank ${config.lora_rank}, Pi0.5 requires ~22-24GB GPU VRAM. Suitable for RTX 4090, RTX A6000, or 32GB+ GPUs.`
                            : 'Without LoRA, Pi0.5 requires ~40GB+ GPU VRAM. Enable LoRA for consumer GPUs (24-32GB).'
                        }
                    </div>
                </div>
            )}

            {/* ACT Policy Settings */}
            {policyType === 'act' && (
                <div className="bg-emerald-50 dark:bg-emerald-950/30 rounded-xl p-4 border border-emerald-200 dark:border-emerald-900 space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="text-sm font-bold text-emerald-700 dark:text-emerald-300">ACT Policy Settings</h3>
                        <span className="text-xs text-emerald-600 dark:text-emerald-400">~8GB VRAM</span>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        {/* Chunk Size */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Chunk Size (Action Horizon)</label>
                            <select
                                value={config.chunk_size_act}
                                onChange={(e) => setConfig(c => ({ ...c, chunk_size_act: parseInt(e.target.value) }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            >
                                <option value={25}>25 (short horizon)</option>
                                <option value={50}>50 (medium)</option>
                                <option value={100}>100 (default, 2s at 50Hz)</option>
                            </select>
                        </div>

                        {/* Action Steps */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Action Steps to Execute</label>
                            <input
                                type="number"
                                value={config.n_action_steps_act}
                                onChange={(e) => setConfig(c => ({ ...c, n_action_steps_act: parseInt(e.target.value) || 100 }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                min={1}
                                max={config.chunk_size_act}
                            />
                        </div>

                        {/* Vision Backbone */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Vision Backbone</label>
                            <select
                                value={config.vision_backbone}
                                onChange={(e) => setConfig(c => ({ ...c, vision_backbone: e.target.value }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            >
                                <option value="resnet18">ResNet18 (default, lighter)</option>
                                <option value="resnet34">ResNet34 (more capacity)</option>
                            </select>
                        </div>

                        {/* Model Dimension */}
                        <div>
                            <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Transformer Dim</label>
                            <select
                                value={config.dim_model}
                                onChange={(e) => setConfig(c => ({ ...c, dim_model: parseInt(e.target.value) }))}
                                className="w-full px-3 py-2 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                            >
                                <option value={256}>256 (smaller)</option>
                                <option value={512}>512 (default)</option>
                            </select>
                        </div>
                    </div>

                    {/* VAE Settings */}
                    <div className="bg-emerald-100 dark:bg-emerald-900/30 border border-emerald-200 dark:border-emerald-800 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                            <label className="flex items-center gap-2 text-sm font-medium text-emerald-700 dark:text-emerald-300">
                                <input
                                    type="checkbox"
                                    checked={config.use_vae}
                                    onChange={e => setConfig({ ...config, use_vae: e.target.checked })}
                                    className="rounded"
                                />
                                Enable VAE (handles multimodal demos)
                            </label>
                            <span className="text-xs text-emerald-600 dark:text-emerald-400">Recommended</span>
                        </div>
                        {config.use_vae && (
                            <div className="flex items-center gap-4 mt-2">
                                <div>
                                    <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Latent Dim</label>
                                    <select
                                        value={config.latent_dim}
                                        onChange={(e) => setConfig(c => ({ ...c, latent_dim: parseInt(e.target.value) }))}
                                        className="px-2 py-1 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                    >
                                        <option value={16}>16</option>
                                        <option value={32}>32 (default)</option>
                                        <option value={64}>64</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">KL Weight</label>
                                    <select
                                        value={config.kl_weight}
                                        onChange={(e) => setConfig(c => ({ ...c, kl_weight: parseFloat(e.target.value) }))}
                                        className="px-2 py-1 rounded-lg border border-neutral-200 dark:border-zinc-700 text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                    >
                                        <option value={1.0}>1.0 (low)</option>
                                        <option value={10.0}>10.0 (default)</option>
                                        <option value={50.0}>50.0 (high)</option>
                                    </select>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Info */}
                    <div className="text-xs px-3 py-2 rounded-lg text-emerald-600 dark:text-emerald-400 bg-emerald-100 dark:bg-emerald-900/30">
                        <strong>ACT</strong> is designed for bimanual fine manipulation. Best with 10-50 clean demos. Predicts action chunks to reduce compounding errors.
                    </div>
                </div>
            )}

            {/* Advanced Settings */}
            <div>
                <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex items-center gap-2 text-sm font-medium text-neutral-600 dark:text-zinc-400 hover:text-neutral-800 dark:hover:text-zinc-200 transition-colors"
                >
                    <ChevronDown className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
                    Advanced Settings
                </button>

                {showAdvanced && (
                    <div className="mt-4 space-y-4 p-4 bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700">
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Training Steps</label>
                                <input
                                    type="number"
                                    value={config.steps}
                                    onChange={e => setConfig({ ...config, steps: parseInt(e.target.value) || 0 })}
                                    className="w-full px-3 py-2 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Batch Size</label>
                                <input
                                    type="number"
                                    value={config.batch_size}
                                    onChange={e => setConfig({ ...config, batch_size: parseInt(e.target.value) || 1 })}
                                    className="w-full px-3 py-2 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Learning Rate</label>
                                <input
                                    type="number"
                                    step="0.00001"
                                    value={config.learning_rate}
                                    onChange={e => setConfig({ ...config, learning_rate: parseFloat(e.target.value) || 0 })}
                                    className="w-full px-3 py-2 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-neutral-600 dark:text-zinc-400 mb-1">Warmup Steps</label>
                                <input
                                    type="number"
                                    value={config.warmup_steps}
                                    onChange={e => setConfig({ ...config, warmup_steps: parseInt(e.target.value) || 0 })}
                                    className="w-full px-3 py-2 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                />
                            </div>
                        </div>

                        <div className="flex items-center gap-4 pt-2">
                            <label className="flex items-center gap-2 text-sm text-neutral-700 dark:text-zinc-300">
                                <input
                                    type="checkbox"
                                    checked={config.freeze_vision_encoder}
                                    onChange={e => setConfig({ ...config, freeze_vision_encoder: e.target.checked })}
                                    className="rounded"
                                />
                                Freeze Vision Encoder
                            </label>
                            <label className="flex items-center gap-2 text-sm text-neutral-700 dark:text-zinc-300">
                                <input
                                    type="checkbox"
                                    checked={config.train_expert_only}
                                    onChange={e => setConfig({ ...config, train_expert_only: e.target.checked })}
                                    className="rounded"
                                />
                                Train Expert Only
                            </label>
                        </div>
                    </div>
                )}
            </div>

            {/* Summary */}
            <div className="p-4 bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700">
                <h3 className="font-semibold text-neutral-700 dark:text-zinc-300 mb-2">Training Summary</h3>
                <div className="text-sm text-neutral-600 dark:text-zinc-400 space-y-1">
                    <div>Dataset: <span className="font-medium text-neutral-900 dark:text-zinc-100">{selectedDataset}</span></div>
                    <div>Policy: <span className="font-medium text-neutral-900 dark:text-zinc-100 capitalize">{policyType === 'pi05' ? 'Pi0.5' : policyType === 'act' ? 'ACT' : policyType}</span></div>
                    <div>Steps: <span className="font-medium text-neutral-900 dark:text-zinc-100">{config.steps.toLocaleString()}</span></div>
                    <div>Batch Size: <span className="font-medium text-neutral-900 dark:text-zinc-100">{config.batch_size}</span></div>
                    {policyType === 'diffusion' && (
                        <>
                            <div>Horizon: <span className="font-medium text-neutral-900 dark:text-zinc-100">{config.horizon}</span></div>
                            <div>Diffusion Steps: <span className="font-medium text-neutral-900 dark:text-zinc-100">{config.num_train_timesteps}</span></div>
                        </>
                    )}
                    {policyType === 'pi05' && (
                        <>
                            <div>Pretrained: <span className="font-medium text-neutral-900 dark:text-zinc-100">{config.pretrained_path.split('/').pop()}</span></div>
                            <div>LoRA: <span className={`font-medium ${config.lora_rank > 0 ? 'text-green-600 dark:text-green-400' : 'text-neutral-900 dark:text-zinc-100'}`}>{config.lora_rank > 0 ? `Enabled (rank ${config.lora_rank})` : 'Disabled'}</span></div>
                            <div>Dtype: <span className="font-medium text-neutral-900 dark:text-zinc-100">{config.dtype}</span></div>
                            <div>Normalization: <span className="font-medium text-neutral-900 dark:text-zinc-100">{config.use_quantile_normalization ? 'Quantile' : 'MEAN_STD'}</span></div>
                        </>
                    )}
                    {policyType === 'act' && (
                        <>
                            <div>Chunk Size: <span className="font-medium text-neutral-900 dark:text-zinc-100">{config.chunk_size_act}</span></div>
                            <div>Action Steps: <span className="font-medium text-neutral-900 dark:text-zinc-100">{config.n_action_steps_act}</span></div>
                            <div>VAE: <span className={`font-medium ${config.use_vae ? 'text-green-600 dark:text-green-400' : 'text-neutral-900 dark:text-zinc-100'}`}>{config.use_vae ? 'Enabled' : 'Disabled'}</span></div>
                            <div>Backbone: <span className="font-medium text-neutral-900 dark:text-zinc-100">{config.vision_backbone}</span></div>
                        </>
                    )}
                </div>
            </div>

            {error && (
                <div className="bg-red-50 dark:bg-red-950/50 text-red-600 dark:text-red-400 px-4 py-3 rounded-xl text-sm flex items-center gap-2 border border-red-100 dark:border-red-900">
                    <AlertCircle className="w-4 h-4" /> {error}
                </div>
            )}

            {/* Buttons */}
            <div className="flex gap-3">
                <button
                    onClick={onBack}
                    className="flex-1 py-3 bg-neutral-100 dark:bg-zinc-800 text-neutral-700 dark:text-zinc-300 rounded-xl font-semibold hover:bg-neutral-200 dark:hover:bg-zinc-700 transition-all"
                >
                    Back
                </button>
                <button
                    onClick={startTraining}
                    className="flex-[2] py-4 bg-purple-600 text-white rounded-2xl font-bold text-lg hover:bg-purple-700 hover:scale-[1.02] active:scale-[0.98] transition-all shadow-xl shadow-purple-200 dark:shadow-purple-900/30 flex items-center justify-center gap-2"
                >
                    <Play className="w-5 h-5" /> Start Training
                </button>
            </div>
        </div>
    );
}
