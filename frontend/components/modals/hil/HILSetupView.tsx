import React from 'react';
import { Play, AlertCircle, Loader2 } from 'lucide-react';
import { Policy } from './types';

interface HILSetupViewProps {
    policies: Policy[];
    selectedPolicy: string;
    setSelectedPolicy: (value: string) => void;
    interventionDataset: string;
    setInterventionDataset: (value: string) => void;
    task: string;
    setTask: (value: string) => void;
    movementScale: number;
    setMovementScale: (value: number) => void;
    error: string;
    isStarting: boolean;
    startSession: () => void;
}

export default function HILSetupView({
    policies,
    selectedPolicy,
    setSelectedPolicy,
    interventionDataset,
    setInterventionDataset,
    task,
    setTask,
    movementScale,
    setMovementScale,
    error,
    isStarting,
    startSession,
}: HILSetupViewProps) {
    return (
        /* SETUP VIEW */
        <div className="flex flex-col max-w-lg mx-auto w-full gap-5 animate-in fade-in slide-in-from-bottom-4">
            <div className="text-center mb-2">
                <h2 className="text-2xl font-bold text-neutral-900 dark:text-zinc-100 mb-1">Deploy with Human Oversight</h2>
                <p className="text-neutral-500 dark:text-zinc-400 text-sm">Run a policy autonomously. Take over anytime to correct behavior.</p>
            </div>

            {/* Policy Selector */}
            <div>
                <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Select Policy</label>
                <select
                    value={selectedPolicy}
                    onChange={e => setSelectedPolicy(e.target.value)}
                    className="w-full px-4 py-3 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all text-neutral-900 dark:text-zinc-100"
                >
                    <option value="">Choose a trained policy...</option>
                    {policies.map(p => (
                        <option key={p.id} value={p.id}>
                            {p.name} ({p.policy_type})
                        </option>
                    ))}
                </select>
                {policies.length === 0 && (
                    <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-1">No trained policies available. Train a policy first.</p>
                )}
            </div>

            {/* Intervention Dataset */}
            <div>
                <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Intervention Dataset</label>
                <input
                    type="text"
                    value={interventionDataset}
                    onChange={e => setInterventionDataset(e.target.value)}
                    placeholder="hil_interventions/v1"
                    className="w-full px-4 py-3 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all text-neutral-900 dark:text-zinc-100 placeholder-neutral-300 dark:placeholder-zinc-500"
                />
                <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-1">Human corrections will be saved here for retraining</p>
            </div>

            {/* Task Description */}
            <div>
                <label className="block text-sm font-bold text-neutral-700 dark:text-zinc-300 mb-2">Task Description</label>
                <input
                    type="text"
                    value={task}
                    onChange={e => setTask(e.target.value)}
                    placeholder="Describe the task..."
                    className="w-full px-4 py-3 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all text-neutral-900 dark:text-zinc-100 placeholder-neutral-300 dark:placeholder-zinc-500"
                />
            </div>

            {/* Movement Scale (Safety Limiter) */}
            <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl p-4 border border-neutral-200 dark:border-zinc-700">
                <div className="flex items-center justify-between mb-2">
                    <label className="text-sm font-bold text-neutral-700 dark:text-zinc-300">
                        Movement Scale
                        <span className="ml-1 text-xs font-normal text-neutral-400 dark:text-zinc-500">(safety limiter)</span>
                    </label>
                    <span className={`text-sm font-bold ${
                        movementScale >= 0.8 ? 'text-red-500' :
                        movementScale >= 0.5 ? 'text-amber-500' : 'text-green-500'
                    }`}>
                        {Math.round(movementScale * 100)}%
                    </span>
                </div>
                <input
                    type="range"
                    min={0.1}
                    max={1.0}
                    step={0.1}
                    value={movementScale}
                    onChange={(e) => setMovementScale(parseFloat(e.target.value))}
                    className="w-full h-2 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    style={{
                        background: `linear-gradient(to right, #22c55e 0%, #22c55e 40%, #eab308 50%, #eab308 70%, #ef4444 80%, #ef4444 100%)`
                    }}
                />
                <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-2">
                    {movementScale < 0.5
                        ? "Safe for testing — robot moves slowly toward target positions"
                        : movementScale < 0.8
                        ? "Moderate speed — verify policy behavior before increasing"
                        : "Full speed — use only with tested, reliable policies"}
                </p>
            </div>

            {error && (
                <div className="bg-red-50 dark:bg-red-950/50 text-red-600 dark:text-red-400 px-4 py-3 rounded-xl text-sm flex items-center gap-2 border border-red-100 dark:border-red-900">
                    <AlertCircle className="w-4 h-4 flex-shrink-0" /> {error}
                </div>
            )}

            <button
                onClick={startSession}
                disabled={!selectedPolicy || isStarting}
                className="w-full py-4 bg-blue-600 text-white rounded-2xl font-bold text-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-200 dark:shadow-blue-900/30 flex items-center justify-center gap-2"
            >
                {isStarting ? <Loader2 className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5" />}
                {isStarting ? 'Starting...' : 'Start HIL Session'}
            </button>
        </div>
    );
}
