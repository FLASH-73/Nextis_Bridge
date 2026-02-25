import React, { useState, useEffect } from 'react';
import { X, Sparkles, ChevronRight } from 'lucide-react';
import { motion, useDragControls } from 'framer-motion';
import { useResizable } from '../../../hooks/useResizable';
import SetupStep, { Dataset, ValidationResult, PolicyType, SetupData } from './SetupStep';
import ConfigStep, { TrainingConfig, HardwareInfo, QuantileStats } from './ConfigStep';
import TrainingView, { TrainingJob } from './TrainingView';
import CompletionView from './CompletionView';
import { trainingApi, datasetsApi } from '../../../lib/api';

interface TrainModalProps {
    isOpen: boolean;
    onClose: () => void;
    maximizedWindow: string | null;
    setMaximizedWindow: (window: string | null) => void;
}

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
    const [policyType, setPolicyType] = useState<PolicyType>('smolvla');
    const [validation, setValidation] = useState<ValidationResult | null>(null);

    // Config State
    const [preset, setPreset] = useState<'quick' | 'standard' | 'full' | 'custom'>('standard');
    const [hardware, setHardware] = useState<HardwareInfo | null>(null);
    const [config, setConfig] = useState<TrainingConfig>({
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
        // System / dataloader (Custom mode)
        num_workers: 4,
        save_freq: 20000,
        eval_freq: 20000,
        // Dataset episodes filter (Custom mode) — e.g. "0:76" or "0,1,2,5"
        dataset_episodes: '',
        // Model (Custom mode)
        image_size: 0,          // 0 = not set (use policy default)
        resize_size: '',        // empty = not set
        // Evaluation (Custom mode)
        eval_n_episodes: 0,     // 0 = not set
        // Output (Custom mode)
        output_dir_custom: '',  // empty = auto-generated
    });

    // Quantile stats state (for Pi0.5)
    const [quantileStats, setQuantileStats] = useState<QuantileStats | null>(null);
    const [isCheckingQuantiles, setIsCheckingQuantiles] = useState(false);
    const [isComputingQuantiles, setIsComputingQuantiles] = useState(false);

    // Training State
    const [jobId, setJobId] = useState<string | null>(null);
    const [jobStatus, setJobStatus] = useState<TrainingJob | null>(null);
    const [logs, setLogs] = useState<string[]>([]);
    const [error, setError] = useState('');

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
            const PRESETS: Record<string, { steps: number; batch_size: number }> = {
                quick: { steps: 10000, batch_size: 8 },
                standard: { steps: 100000, batch_size: 8 },
                full: { steps: 200000, batch_size: 16 },
            };
            const p = PRESETS[preset];
            if (p) {
                setConfig(c => ({ ...c, steps: p.steps, batch_size: p.batch_size }));
            }
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

    const fetchDatasets = async () => {
        try {
            const data = await datasetsApi.list();
            setDatasets((data || []) as Dataset[]);
        } catch (e) {
            console.error('Failed to fetch datasets:', e);
        }
    };

    const fetchHardware = async () => {
        try {
            const data = await trainingApi.hardware() as unknown as HardwareInfo;
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
            const data = await trainingApi.datasetQuantiles(encodeURIComponent(datasetId));
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
            const data: any = await trainingApi.computeQuantiles(encodeURIComponent(selectedDataset));
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

    const handleSetupNext = (data: SetupData) => {
        setStep('config');
    };

    const handleStartTraining = (result: Record<string, any>) => {
        setJobId(result.jobId);
        setStep('training');
    };

    const handleTrainingComplete = (job: TrainingJob) => {
        setJobStatus(job);
        // Fetch final logs
        trainingApi.jobLogs(job.id, 1000)
            .then(data => setLogs(data.logs || []))
            .catch(() => {});
        setStep('complete');
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
                        <SetupStep
                            datasets={datasets}
                            selectedDataset={selectedDataset}
                            setSelectedDataset={setSelectedDataset}
                            policyType={policyType}
                            setPolicyType={setPolicyType}
                            policyName={policyName}
                            setPolicyName={setPolicyName}
                            validation={validation}
                            setValidation={setValidation}
                            error={error}
                            setError={setError}
                            onNext={handleSetupNext}
                            checkQuantileStats={checkQuantileStats}
                        />
                    )}

                    {/* CONFIG STEP */}
                    {step === 'config' && (
                        <ConfigStep
                            policyType={policyType}
                            selectedDataset={selectedDataset}
                            policyName={policyName}
                            config={config}
                            setConfig={setConfig}
                            preset={preset}
                            setPreset={setPreset}
                            hardware={hardware}
                            quantileStats={quantileStats}
                            isComputingQuantiles={isComputingQuantiles}
                            computeQuantileStats={computeQuantileStats}
                            error={error}
                            setError={setError}
                            onBack={() => setStep('setup')}
                            onStartTraining={handleStartTraining}
                        />
                    )}

                    {/* TRAINING STEP */}
                    {step === 'training' && jobId && (
                        <TrainingView
                            jobId={jobId}
                            totalSteps={config.steps}
                            onComplete={handleTrainingComplete}
                        />
                    )}

                    {/* COMPLETE STEP */}
                    {step === 'complete' && (
                        <CompletionView
                            jobStatus={jobStatus}
                            logs={logs}
                            onReset={resetWizard}
                        />
                    )}
                </div>
            )}
        </motion.div>
    );
}
