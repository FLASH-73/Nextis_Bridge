import React, { useState, useEffect } from 'react';
import { AlertCircle, CheckCircle, Loader2, Database, Brain, Zap, Sparkles, Settings } from 'lucide-react';
import { trainingApi } from '../../../lib/api';

export interface Dataset {
    repo_id: string;
    root: string;
    fps: number;
    robot_type: string;
    total_episodes: number;
    total_frames: number;
}

export interface ValidationResult {
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

export type PolicyType = 'smolvla' | 'diffusion' | 'act' | 'pi05';

export interface SetupData {
    selectedDataset: string;
    policyType: PolicyType;
    policyName: string;
    validation: ValidationResult;
}

interface SetupStepProps {
    datasets: Dataset[];
    selectedDataset: string;
    setSelectedDataset: (dataset: string) => void;
    policyType: PolicyType;
    setPolicyType: (type: PolicyType) => void;
    policyName: string;
    setPolicyName: (name: string) => void;
    validation: ValidationResult | null;
    setValidation: (v: ValidationResult | null) => void;
    error: string;
    setError: (e: string) => void;
    onNext: (data: SetupData) => void;
    checkQuantileStats: (datasetId: string) => Promise<void>;
}

export default function SetupStep({
    datasets,
    selectedDataset,
    setSelectedDataset,
    policyType,
    setPolicyType,
    policyName,
    setPolicyName,
    validation,
    setValidation,
    error,
    setError,
    onNext,
    checkQuantileStats,
}: SetupStepProps) {
    const [isValidating, setIsValidating] = useState(false);

    const validateDataset = async () => {
        if (!selectedDataset) return;

        setIsValidating(true);
        setValidation(null);
        setError('');

        try {
            const data = await trainingApi.validate({ dataset_repo_id: selectedDataset, policy_type: policyType }) as ValidationResult;
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

    return (
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
                    {([
                        { id: 'smolvla', name: 'SmolVLA', icon: Brain, desc: 'Vision-Language-Action', badge: undefined as string | undefined, disabled: false },
                        { id: 'diffusion', name: 'Diffusion', icon: Zap, desc: 'Diffusion Policy', badge: undefined as string | undefined, disabled: false },
                        { id: 'pi05', name: 'Pi0.5', icon: Sparkles, desc: 'Open-World VLA', badge: '22GB+ LoRA', disabled: false },
                        { id: 'act', name: 'ACT', icon: Settings, desc: 'Action Chunking', badge: '~8GB', disabled: false },
                    ]).map(p => (
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
                onClick={() => {
                    if (validation?.valid) {
                        onNext({
                            selectedDataset,
                            policyType,
                            policyName,
                            validation,
                        });
                    }
                }}
                disabled={!validation?.valid}
                className="w-full py-4 bg-purple-600 text-white rounded-2xl font-bold text-lg hover:bg-purple-700 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-xl shadow-purple-200 dark:shadow-purple-900/30"
            >
                Continue to Configuration
            </button>
        </div>
    );
}
