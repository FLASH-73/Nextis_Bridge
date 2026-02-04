import React, { useState, useEffect, useRef } from 'react';
import { X, Play, StopCircle, Sparkles, Maximize2, Minimize2, AlertCircle, CheckCircle, Loader2, ChevronRight, ChevronDown, Settings, Zap, Database, Brain } from 'lucide-react';
import { motion, useDragControls } from 'framer-motion';
import { useResizable } from '../hooks/useResizable';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const API_BASE = typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : 'http://127.0.0.1:8000';

interface TrainModalProps {
    isOpen: boolean;
    onClose: () => void;
    maximizedWindow: string | null;
    setMaximizedWindow: (window: string | null) => void;
}

interface Dataset {
    repo_id: string;
    root: string;
    fps: number;
    robot_type: string;
    total_episodes: number;
    total_frames: number;
}

interface ValidationResult {
    valid: boolean;
    errors: string[];
    warnings: string[];
    features: {
        detected: string[];
        total_episodes: number;
        total_frames: number;
        fps: number;
        robot_type: string;
    };
}

interface TrainingProgress {
    step: number;
    total_steps: number;
    loss: number | null;
    learning_rate: number | null;
    eta_seconds: number | null;
    epoch: number;
    loss_history: [number, number][];  // [[step, loss], ...]
}

interface TrainingJob {
    id: string;
    status: 'pending' | 'validating' | 'training' | 'completed' | 'failed' | 'cancelled';
    policy_type: string;
    dataset_repo_id: string;
    config: Record<string, any>;
    progress: TrainingProgress;
    error: string | null;
    output_dir: string | null;
}

interface HardwareDevice {
    id: string;
    type: 'cuda' | 'mps' | 'cpu';
    name: string;
    memory_gb: number | null;
    recommended: boolean;
}

interface HardwareInfo {
    devices: HardwareDevice[];
    default: string;
}

const PRESETS = {
    quick: { steps: 10000, batch_size: 8, description: 'Quick test (~10 min)' },
    standard: { steps: 100000, batch_size: 8, description: 'Standard (~2 hrs)' },
    full: { steps: 200000, batch_size: 16, description: 'Best quality (~4 hrs)' },
    custom: { steps: 100000, batch_size: 8, description: 'Your parameters' },
};

export default function TrainModal({ isOpen, onClose, maximizedWindow, setMaximizedWindow }: TrainModalProps) {
    const dragControls = useDragControls();
    const { size, handleResizeMouseDown } = useResizable({ initialSize: { width: 800, height: 650 } });
    const [isMinimized, setIsMinimized] = useState(false);
    const isMaximized = maximizedWindow === 'train';

    // Wizard State
    const [step, setStep] = useState<'setup' | 'config' | 'training' | 'complete'>('setup');

    // Setup State
    const [datasets, setDatasets] = useState<Dataset[]>([]);
    const [selectedDataset, setSelectedDataset] = useState('');
    const [policyName, setPolicyName] = useState('');
    const [policyType, setPolicyType] = useState<'smolvla' | 'diffusion' | 'act' | 'pi05'>('smolvla');
    const [validation, setValidation] = useState<ValidationResult | null>(null);
    const [isValidating, setIsValidating] = useState(false);

    // Config State
    const [preset, setPreset] = useState<'quick' | 'standard' | 'full' | 'custom'>('standard');
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [hardware, setHardware] = useState<HardwareInfo | null>(null);
    const [config, setConfig] = useState({
        steps: 100000,
        batch_size: 8,
        learning_rate: 0.0001,
        warmup_steps: 1000,
        freeze_vision_encoder: true,
        train_expert_only: true,
        device: 'auto',
        // Diffusion-specific (only used when policyType === 'diffusion')
        n_obs_steps: 2,
        horizon: 16,
        n_action_steps: 8,
        noise_scheduler_type: 'DDPM',
        num_train_timesteps: 100,
        vision_backbone: 'resnet18',
        // Pi0.5-specific (only used when policyType === 'pi05')
        pretrained_path: 'lerobot/pi05_base',
        compile_model: false,  // Disabled by default to save memory
        gradient_checkpointing: true,
        dtype: 'bfloat16',
        use_quantile_normalization: true,
        chunk_size: 50,
        n_action_steps_pi05: 50,
        // LoRA settings (memory-efficient fine-tuning)
        lora_rank: 8,  // 0 = disabled, 8 = recommended
        lora_alpha: 16,
        lora_dropout: 0.1,
        // ACT-specific
        chunk_size_act: 100,
        n_action_steps_act: 100,
        use_vae: true,
        latent_dim: 32,
        kl_weight: 10.0,
        dim_model: 512,
    });

    // Quantile stats state (for Pi0.5)
    const [quantileStats, setQuantileStats] = useState<{
        has_quantiles: boolean;
        missing_features: string[];
        message: string;
    } | null>(null);
    const [isCheckingQuantiles, setIsCheckingQuantiles] = useState(false);
    const [isComputingQuantiles, setIsComputingQuantiles] = useState(false);

    // Training State
    const [jobId, setJobId] = useState<string | null>(null);
    const [jobStatus, setJobStatus] = useState<TrainingJob | null>(null);
    const [logs, setLogs] = useState<string[]>([]);
    const [error, setError] = useState('');

    // Refs
    const logsEndRef = useRef<HTMLDivElement>(null);

    // Load datasets and hardware on open
    useEffect(() => {
        if (isOpen) {
            fetchDatasets();
            fetchHardware();
        }
    }, [isOpen]);

    // Apply preset changes (except for custom)
    useEffect(() => {
        if (preset !== 'custom') {
            const p = PRESETS[preset];
            setConfig(c => ({ ...c, steps: p.steps, batch_size: p.batch_size }));
        }
    }, [preset]);

    // Reset config defaults when switching policy types
    useEffect(() => {
        if (policyType === 'diffusion') {
            setConfig(c => ({
                ...c,
                steps: 50000,
                batch_size: 32,
                warmup_steps: 500,
                n_obs_steps: 2,
                horizon: 16,
                n_action_steps: 8,
                noise_scheduler_type: 'DDPM',
                num_train_timesteps: 100,
            }));
            setPreset('standard');
            setQuantileStats(null);
        } else if (policyType === 'smolvla') {
            setConfig(c => ({
                ...c,
                steps: 100000,
                batch_size: 8,
                warmup_steps: 1000,
            }));
            setPreset('standard');
            setQuantileStats(null);
        } else if (policyType === 'pi05') {
            setConfig(c => ({
                ...c,
                steps: 5000,  // Optimized for small datasets (20-50 episodes)
                batch_size: 8,  // Reduced for LoRA memory efficiency
                learning_rate: 0.000025,
                warmup_steps: 1000,
                pretrained_path: 'lerobot/pi05_base',
                compile_model: false,  // Disabled to save memory
                gradient_checkpointing: true,
                dtype: 'bfloat16',
                use_quantile_normalization: true,
                chunk_size: 50,
                n_action_steps_pi05: 50,
                // LoRA settings (enabled by default for memory efficiency)
                lora_rank: 8,
                lora_alpha: 16,
                lora_dropout: 0.1,
            }));
            setPreset('standard');
            // Check quantile stats when switching to Pi0.5
            if (selectedDataset) {
                checkQuantileStats(selectedDataset);
            }
        } else if (policyType === 'act') {
            setConfig(c => ({
                ...c,
                steps: 100000,
                batch_size: 8,
                learning_rate: 0.00001,
                warmup_steps: 500,
                chunk_size_act: 100,
                n_action_steps_act: 100,
                use_vae: true,
                latent_dim: 32,
                kl_weight: 10.0,
                vision_backbone: 'resnet18',
                dim_model: 512,
            }));
            setPreset('standard');
            setQuantileStats(null);
        }
    }, [policyType]);

    // Poll for training progress
    useEffect(() => {
        if (step === 'training' && jobId) {
            const interval = setInterval(async () => {
                try {
                    const res = await fetch(`${API_BASE}/training/jobs/${jobId}`);
                    const data: TrainingJob = await res.json();
                    setJobStatus(data);

                    // Fetch logs (get all available logs, up to 1000)
                    const logsRes = await fetch(`${API_BASE}/training/jobs/${jobId}/logs?limit=1000`);
                    const logsData = await logsRes.json();
                    setLogs(logsData.logs || []);

                    if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
                        setStep('complete');
                        clearInterval(interval);
                    }
                } catch (e) {
                    console.error('Failed to fetch job status:', e);
                }
            }, 2000);
            return () => clearInterval(interval);
        }
    }, [step, jobId]);

    // Auto-scroll logs
    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    const fetchDatasets = async () => {
        try {
            const res = await fetch(`${API_BASE}/datasets`);
            const data = await res.json();
            setDatasets(data || []);
        } catch (e) {
            console.error('Failed to fetch datasets:', e);
        }
    };

    const fetchHardware = async () => {
        try {
            const res = await fetch(`${API_BASE}/training/hardware`);
            const data: HardwareInfo = await res.json();
            setHardware(data);
            // Set default device
            if (data.default) {
                setConfig(c => ({ ...c, device: data.default }));
            }
        } catch (e) {
            console.error('Failed to fetch hardware:', e);
            // Fallback to CPU
            setHardware({
                devices: [{ id: 'cpu', type: 'cpu', name: 'CPU', memory_gb: null, recommended: true }],
                default: 'cpu'
            });
        }
    };

    const checkQuantileStats = async (datasetId: string) => {
        setIsCheckingQuantiles(true);
        try {
            const res = await fetch(`${API_BASE}/training/dataset/${encodeURIComponent(datasetId)}/quantiles`);
            const data = await res.json();
            setQuantileStats(data);
        } catch (e) {
            console.error('Failed to check quantile stats:', e);
            setQuantileStats({ has_quantiles: false, missing_features: [], message: 'Failed to check quantile stats' });
        } finally {
            setIsCheckingQuantiles(false);
        }
    };

    const computeQuantileStats = async () => {
        if (!selectedDataset) return;
        setIsComputingQuantiles(true);
        setError('');
        try {
            const res = await fetch(`${API_BASE}/training/dataset/${encodeURIComponent(selectedDataset)}/compute-quantiles`, {
                method: 'POST',
            });
            const data = await res.json();
            if (data.status === 'success') {
                // Re-check quantile stats after computing
                await checkQuantileStats(selectedDataset);
            } else {
                setError(data.message || 'Failed to compute quantiles');
            }
        } catch (e: any) {
            setError(e.message || 'Failed to compute quantiles');
        } finally {
            setIsComputingQuantiles(false);
        }
    };

    const validateDataset = async () => {
        if (!selectedDataset) return;

        setIsValidating(true);
        setValidation(null);
        setError('');

        try {
            const res = await fetch(`${API_BASE}/training/validate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_repo_id: selectedDataset, policy_type: policyType }),
            });
            const data: ValidationResult = await res.json();
            setValidation(data);

            // Also check quantile stats for Pi0.5
            if (policyType === 'pi05') {
                await checkQuantileStats(selectedDataset);
            }
        } catch (e: any) {
            setError(e.message || 'Validation failed');
        } finally {
            setIsValidating(false);
        }
    };

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

            const res = await fetch(`${API_BASE}/training/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_repo_id: selectedDataset,
                    policy_type: policyType,
                    config: trainingConfig,
                }),
            });
            const data = await res.json();

            if (data.status === 'started') {
                setJobId(data.job_id);
                setStep('training');
            } else {
                setError(data.error || 'Failed to start training');
            }
        } catch (e: any) {
            setError(e.message || 'Failed to start training');
        }
    };

    const cancelTraining = async () => {
        if (!jobId) return;
        if (!confirm('Are you sure you want to cancel training?')) return;

        try {
            await fetch(`${API_BASE}/training/jobs/${jobId}/cancel`, { method: 'POST' });
        } catch (e) {
            console.error('Failed to cancel training:', e);
        }
    };

    const resetWizard = () => {
        setStep('setup');
        setSelectedDataset('');
        setPolicyName('');
        setValidation(null);
        setJobId(null);
        setJobStatus(null);
        setLogs([]);
        setError('');
    };

    const formatETA = (seconds: number | null) => {
        if (!seconds) return '--';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        if (h > 0) return `${h}h ${m}m`;
        return `${m}m`;
    };

    if (!isOpen) return null;

    return (
        <motion.div
            drag
            dragControls={dragControls}
            dragListener={false}
            dragMomentum={false}
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            style={{
                width: isMaximized ? 'calc(100vw - 40px)' : isMinimized ? 'auto' : `${size.width}px`,
                height: isMaximized ? 'calc(100vh - 100px)' : isMinimized ? 'auto' : `${size.height}px`,
                top: isMaximized ? '80px' : undefined,
                left: isMaximized ? '20px' : undefined,
                transform: isMaximized ? 'none' : undefined,
                zIndex: isMaximized ? 100 : 50,
            }}
            className={`fixed flex flex-col overflow-hidden font-sans glass-panel shadow-2xl rounded-3xl bg-white/90 dark:bg-zinc-900/90 border border-white/50 dark:border-zinc-700/50 backdrop-blur-3xl transition-all duration-300 ${isMaximized ? '' : 'top-20 left-1/2 -translate-x-1/2'}`}
        >
            {/* Header */}
            <div
                onPointerDown={(e) => dragControls.start(e)}
                className="h-16 bg-white/30 dark:bg-zinc-800/30 border-b border-black/5 dark:border-white/5 flex items-center justify-between px-6 cursor-grab active:cursor-grabbing select-none flex-none"
            >
                <div className="flex items-center gap-4">
                    <div className="flex gap-2 mr-2">
                        <button onClick={onClose} className="w-3.5 h-3.5 rounded-full bg-[#FF5F57] hover:brightness-90 btn-control transition-all" />
                        <button onClick={() => { setIsMinimized(!isMinimized); if (isMaximized) setMaximizedWindow(null); }} className="w-3.5 h-3.5 rounded-full bg-[#FEBC2E] hover:brightness-90 btn-control transition-all" />
                        <button onClick={() => { setMaximizedWindow(isMaximized ? null : 'train'); if (isMinimized) setIsMinimized(false); }} className="w-3.5 h-3.5 rounded-full bg-[#28C840] hover:brightness-90 btn-control transition-all" />
                    </div>
                    <span className="font-semibold text-neutral-800 dark:text-zinc-200 text-lg tracking-tight flex items-center gap-2">
                        <Sparkles className="w-5 h-5 text-purple-500" /> Train Model
                    </span>
                </div>

                {/* Step Indicator */}
                <div className="flex items-center gap-2 text-xs">
                    {['setup', 'config', 'training', 'complete'].map((s, i) => (
                        <React.Fragment key={s}>
                            <div className={`px-3 py-1 rounded-full font-medium ${step === s ? 'bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300' : 'text-neutral-400 dark:text-zinc-500'}`}>
                                {s.charAt(0).toUpperCase() + s.slice(1)}
                            </div>
                            {i < 3 && <ChevronRight className="w-3 h-3 text-neutral-300 dark:text-zinc-600" />}
                        </React.Fragment>
                    ))}
                </div>
            </div>

            {!isMinimized && (
                <div className="flex-1 flex flex-col p-8 overflow-auto">
                    {/* SETUP STEP */}
                    {step === 'setup' && (
                        <div className="flex flex-col max-w-lg mx-auto w-full gap-6 animate-in fade-in slide-in-from-bottom-4">
                            <div>
                                <h2 className="text-2xl font-bold text-neutral-900 dark:text-zinc-100 mb-1">Select Dataset</h2>
                                <p className="text-neutral-500 dark:text-zinc-400 text-sm">Choose a dataset and validate compatibility.</p>
                            </div>

                            {/* Dataset Selector */}
                            <div>
                                <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Dataset</label>
                                <select
                                    value={selectedDataset}
                                    onChange={e => { setSelectedDataset(e.target.value); setValidation(null); }}
                                    className="w-full px-4 py-3 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all text-neutral-900 dark:text-zinc-100"
                                >
                                    <option value="">Select a dataset...</option>
                                    {datasets.map(d => (
                                        <option key={d.repo_id} value={d.repo_id}>
                                            {d.repo_id} ({d.total_episodes} episodes, {d.total_frames} frames)
                                        </option>
                                    ))}
                                </select>
                            </div>

                            {/* Policy Name */}
                            <div>
                                <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Policy Name</label>
                                <input
                                    type="text"
                                    value={policyName}
                                    onChange={(e) => setPolicyName(e.target.value)}
                                    placeholder="e.g., pick-and-place-v1"
                                    className="w-full px-4 py-3 rounded-xl border border-neutral-200 dark:border-zinc-700 focus:border-purple-500 focus:ring-2 focus:ring-purple-100 dark:focus:ring-purple-900/50 outline-none transition-all bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100 placeholder-neutral-300 dark:placeholder-zinc-500"
                                />
                                <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-1">Give your trained policy a memorable name (optional)</p>
                            </div>

                            {/* Policy Selector */}
                            <div>
                                <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Policy Type</label>
                                <div className="grid grid-cols-2 gap-3">
                                    {[
                                        { id: 'smolvla', name: 'SmolVLA', icon: Brain, desc: 'Vision-Language-Action' },
                                        { id: 'diffusion', name: 'Diffusion', icon: Zap, desc: 'Diffusion Policy' },
                                        { id: 'pi05', name: 'Pi0.5', icon: Sparkles, desc: 'Open-World VLA', badge: '22GB+ LoRA' },
                                        { id: 'act', name: 'ACT', icon: Settings, desc: 'Action Chunking', badge: '~8GB' },
                                    ].map(p => (
                                        <button
                                            key={p.id}
                                            onClick={() => !p.disabled && setPolicyType(p.id as any)}
                                            disabled={p.disabled}
                                            className={`p-4 rounded-xl border-2 transition-all text-left ${
                                                policyType === p.id
                                                    ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/30'
                                                    : p.disabled
                                                        ? 'border-neutral-100 dark:border-zinc-800 bg-neutral-50 dark:bg-zinc-800/50 opacity-50 cursor-not-allowed'
                                                        : 'border-neutral-200 dark:border-zinc-700 hover:border-purple-300'
                                            }`}
                                        >
                                            <div className="flex items-start justify-between">
                                                <p.icon className={`w-5 h-5 mb-2 ${policyType === p.id ? 'text-purple-600 dark:text-purple-400' : 'text-neutral-400 dark:text-zinc-500'}`} />
                                                {p.badge && (
                                                    <span className="text-[10px] bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-300 px-1.5 py-0.5 rounded font-medium">{p.badge}</span>
                                                )}
                                            </div>
                                            <div className="font-semibold text-sm text-neutral-900 dark:text-zinc-100">{p.name}</div>
                                            <div className="text-xs text-neutral-400 dark:text-zinc-500">{p.desc}</div>
                                            {p.disabled && <div className="text-xs text-neutral-400 dark:text-zinc-500 mt-1">(Coming soon)</div>}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Validate Button */}
                            <button
                                onClick={validateDataset}
                                disabled={!selectedDataset || isValidating}
                                className="w-full py-3 bg-neutral-100 dark:bg-zinc-800 text-neutral-700 dark:text-zinc-300 rounded-xl font-semibold hover:bg-neutral-200 dark:hover:bg-zinc-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
                            >
                                {isValidating ? (
                                    <><Loader2 className="w-4 h-4 animate-spin" /> Validating...</>
                                ) : (
                                    <><Database className="w-4 h-4" /> Validate Dataset</>
                                )}
                            </button>

                            {/* Validation Results */}
                            {validation && (
                                <div className={`p-4 rounded-xl border ${validation.valid ? 'bg-green-50 dark:bg-green-950/50 border-green-200 dark:border-green-900' : 'bg-red-50 dark:bg-red-950/50 border-red-200 dark:border-red-900'}`}>
                                    <div className="flex items-center gap-2 mb-3">
                                        {validation.valid ? (
                                            <><CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" /><span className="font-semibold text-green-700 dark:text-green-300">Dataset Compatible</span></>
                                        ) : (
                                            <><AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" /><span className="font-semibold text-red-700 dark:text-red-300">Validation Failed</span></>
                                        )}
                                    </div>

                                    {validation.errors.length > 0 && (
                                        <ul className="text-sm text-red-600 dark:text-red-400 space-y-1 mb-2">
                                            {validation.errors.map((e, i) => <li key={i}>• {e}</li>)}
                                        </ul>
                                    )}

                                    {validation.warnings.length > 0 && (
                                        <ul className="text-sm text-amber-600 dark:text-amber-400 space-y-1 mb-2">
                                            {validation.warnings.map((w, i) => <li key={i}>⚠ {w}</li>)}
                                        </ul>
                                    )}

                                    <div className="text-xs text-neutral-500 dark:text-zinc-400 mt-2">
                                        {validation.features.total_episodes} episodes • {validation.features.total_frames} frames • {validation.features.fps} FPS
                                    </div>
                                </div>
                            )}

                            {error && (
                                <div className="bg-red-50 dark:bg-red-950/50 text-red-600 dark:text-red-400 px-4 py-3 rounded-xl text-sm flex items-center gap-2 border border-red-100 dark:border-red-900">
                                    <AlertCircle className="w-4 h-4" /> {error}
                                </div>
                            )}

                            {/* Next Button */}
                            <button
                                onClick={() => setStep('config')}
                                disabled={!validation?.valid}
                                className="w-full py-4 bg-purple-600 text-white rounded-2xl font-bold text-lg hover:bg-purple-700 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-xl shadow-purple-200 dark:shadow-purple-900/30"
                            >
                                Continue to Configuration
                            </button>
                        </div>
                    )}

                    {/* CONFIG STEP */}
                    {step === 'config' && (
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
                                    onClick={() => setStep('setup')}
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
                    )}

                    {/* TRAINING STEP */}
                    {step === 'training' && (
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
                                            Step {jobStatus?.progress.step?.toLocaleString() || 0} / {jobStatus?.progress.total_steps?.toLocaleString() || config.steps.toLocaleString()}
                                        </span>
                                    </div>
                                    <span className="text-sm text-neutral-500 dark:text-zinc-400">
                                        ETA: {formatETA(jobStatus?.progress.eta_seconds || null)}
                                    </span>
                                </div>

                                <div className="h-3 bg-neutral-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: `${((jobStatus?.progress.step || 0) / (jobStatus?.progress.total_steps || config.steps)) * 100}%` }}
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
                                            {(((jobStatus?.progress.step || 0) / (jobStatus?.progress.total_steps || config.steps)) * 100).toFixed(1)}%
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
                                                    formatter={(value: number) => [value.toFixed(4), 'Loss']}
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
                    )}

                    {/* COMPLETE STEP */}
                    {step === 'complete' && (
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
                                onClick={resetWizard}
                                className="px-8 py-3 bg-purple-600 text-white rounded-xl font-semibold hover:bg-purple-700 transition-all"
                            >
                                Start New Training
                            </button>
                        </div>
                    )}
                </div>
            )}
        </motion.div>
    );
}
