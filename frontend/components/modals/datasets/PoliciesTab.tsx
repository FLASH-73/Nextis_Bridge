import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Trash2, Brain, Rocket, RotateCcw, Pencil, Check, Loader2 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';
import { policiesApi, api } from '../../../lib/api';
import type { PolicyInfo } from '../../../lib/api/types';

interface PoliciesTabProps {
    isOpen: boolean;
}

export default function PoliciesTab({ isOpen }: PoliciesTabProps) {
    const [policies, setPolicies] = useState<PolicyInfo[]>([]);
    const [selectedPolicies, setSelectedPolicies] = useState<Set<string>>(new Set());
    const [isLoadingPolicies, setIsLoadingPolicies] = useState(false);
    const [editingPolicyId, setEditingPolicyId] = useState<string | null>(null);
    const [editingPolicyName, setEditingPolicyName] = useState("");
    const [isDeploying, setIsDeploying] = useState(false);
    const [showResumeModal, setShowResumeModal] = useState(false);
    const [resumeSteps, setResumeSteps] = useState(10000);
    const [resumingPolicyId, setResumingPolicyId] = useState<string | null>(null);

    const fetchPolicies = async () => {
        setIsLoadingPolicies(true);
        try {
            const data = await policiesApi.list();
            setPolicies(data);
        } catch (e) {
            console.error("Failed to load policies", e);
        } finally {
            setIsLoadingPolicies(false);
        }
    };

    useEffect(() => {
        if (isOpen) fetchPolicies();
    }, [isOpen]);

    const handlePolicyClick = (policyId: string, e: React.MouseEvent) => {
        if (e.ctrlKey || e.metaKey) {
            const newSelected = new Set(selectedPolicies);
            if (newSelected.has(policyId)) {
                newSelected.delete(policyId);
            } else {
                newSelected.add(policyId);
            }
            setSelectedPolicies(newSelected);
        } else {
            setSelectedPolicies(new Set([policyId]));
        }
    };

    const deletePolicy = async (policyId: string) => {
        if (!confirm(`Are you sure you want to delete this policy?\n\n${policyId}\n\nThis cannot be undone!`)) return;

        try {
            const result = await policiesApi.delete(policyId) as any;
            if (result.status === 'deleted') {
                setSelectedPolicies(prev => {
                    const next = new Set(prev);
                    next.delete(policyId);
                    return next;
                });
                fetchPolicies();
            } else {
                alert(`Failed to delete: ${result.error || 'Unknown error'}`);
            }
        } catch (e) {
            console.error("Delete failed:", e);
            alert("Failed to delete policy");
        }
    };

    const renamePolicy = async (policyId: string, newName: string) => {
        try {
            const result = await policiesApi.update(policyId, { name: newName }) as any;
            if (result.status === 'updated') {
                setEditingPolicyId(null);
                fetchPolicies();
            } else {
                alert(`Failed to rename: ${result.error || 'Unknown error'}`);
            }
        } catch (e) {
            console.error("Rename failed:", e);
            alert("Failed to rename policy");
        }
    };

    const deployPolicy = async (policyId: string) => {
        setIsDeploying(true);
        try {
            const result = await policiesApi.deploy(policyId) as any;
            if (result.status === 'deployed' || result.status === 'ready') {
                alert(`Policy deployed successfully!\n\nCheckpoint: ${result.checkpoint_path}`);
            } else {
                alert(`Failed to deploy: ${result.error || 'Unknown error'}`);
            }
        } catch (e) {
            console.error("Deploy failed:", e);
            alert("Failed to deploy policy");
        } finally {
            setIsDeploying(false);
        }
    };

    const resumeTraining = async (policyId: string, additionalSteps: number) => {
        try {
            const result = await api.post<any>(`/policies/${policyId}/resume`, { additional_steps: additionalSteps });
            if (result.status === 'started') {
                alert(`Training resumed!\n\nJob ID: ${result.job_id}`);
                setShowResumeModal(false);
                fetchPolicies();
            } else {
                alert(`Failed to resume: ${result.error || 'Unknown error'}`);
            }
        } catch (e) {
            console.error("Resume failed:", e);
            alert("Failed to resume training");
        }
    };

    const selectedPolicyData = policies.filter(p => selectedPolicies.has(p.id));
    const primarySelectedPolicy = selectedPolicyData.length > 0 ? selectedPolicyData[0] : null;

    return (
        <>
            {/* Sidebar */}
            <div className="w-72 border-r border-neutral-200/50 dark:border-zinc-700/50 flex flex-col bg-neutral-50/50 dark:bg-zinc-800/50">
                <div className="p-4 border-b border-neutral-200/50 dark:border-zinc-700/50">
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="text-xs font-semibold text-neutral-500 dark:text-zinc-400 uppercase tracking-wider">
                            Policies {selectedPolicies.size > 1 && <span className="text-purple-600 dark:text-purple-400">({selectedPolicies.size})</span>}
                        </h3>
                        {selectedPolicies.size === 1 && (
                            <button
                                onClick={() => {
                                    const policyId = Array.from(selectedPolicies)[0];
                                    deletePolicy(policyId);
                                }}
                                className="p-1.5 hover:bg-red-100 rounded-lg text-red-500 transition-colors"
                                title="Delete policy"
                            >
                                <Trash2 className="w-3.5 h-3.5" />
                            </button>
                        )}
                    </div>
                    <p className="text-[10px] text-neutral-400 dark:text-zinc-500 mb-2">Ctrl+click to compare multiple</p>
                    <div className="space-y-1">
                        {isLoadingPolicies ? (
                            <div className="text-neutral-400 dark:text-zinc-500 text-sm animate-pulse px-3">Loading...</div>
                        ) : policies.length === 0 ? (
                            <div className="text-neutral-400 dark:text-zinc-500 text-sm italic px-3 py-4 text-center">No policies found</div>
                        ) : (
                            policies.map((policy) => {
                                const isSelected = selectedPolicies.has(policy.id);
                                const statusColor = policy.status === 'completed' ? 'bg-green-500' : policy.status === 'training' ? 'bg-purple-500' : 'bg-red-500';
                                return (
                                    <button
                                        key={policy.id}
                                        onClick={(e) => handlePolicyClick(policy.id, e)}
                                        className={`w-full text-left px-3 py-2.5 rounded-xl text-sm transition-all ${
                                            isSelected
                                                ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-800'
                                                : 'hover:bg-white dark:hover:bg-zinc-700 text-neutral-600 dark:text-zinc-300 hover:shadow-sm border border-transparent'
                                        }`}
                                    >
                                        <div className="flex items-center gap-2">
                                            <div className={`w-2 h-2 rounded-full ${statusColor} ${policy.status === 'training' ? 'animate-pulse' : ''}`} />
                                            <span className="truncate font-medium flex-1">{policy.name}</span>
                                        </div>
                                        <div className="flex items-center gap-2 mt-1 text-[10px] text-neutral-400 dark:text-zinc-500">
                                            <span className="uppercase">{policy.policy_type}</span>
                                            <span>•</span>
                                            <span>{policy.status === 'training'
                                                ? `${Math.round((policy.steps / policy.total_steps) * 100)}%`
                                                : `${(policy.steps / 1000).toFixed(0)}K steps`
                                            }</span>
                                        </div>
                                        {policy.status === 'training' && (
                                            <div className="mt-1.5 w-full bg-purple-200 rounded-full h-1">
                                                <div
                                                    className="bg-purple-500 h-1 rounded-full transition-all"
                                                    style={{ width: `${(policy.steps / policy.total_steps) * 100}%` }}
                                                />
                                            </div>
                                        )}
                                    </button>
                                );
                            })
                        )}
                    </div>
                </div>
            </div>

            {/* Main View */}
            <div className="flex-1 flex flex-col bg-white/50 dark:bg-zinc-900/50 relative">
                {selectedPolicyData.length > 0 ? (
                    <div className="flex-1 flex flex-col overflow-hidden">
                        {/* Policy Header with editable name */}
                        <div className="h-12 border-b border-neutral-200/50 dark:border-zinc-700/50 flex items-center justify-between px-5 bg-white/70 dark:bg-zinc-800/70">
                            <div className="flex items-center gap-3">
                                {editingPolicyId === primarySelectedPolicy?.id ? (
                                    <div className="flex items-center gap-2">
                                        <input
                                            type="text"
                                            value={editingPolicyName}
                                            onChange={(e) => setEditingPolicyName(e.target.value)}
                                            onKeyDown={(e) => {
                                                if (e.key === 'Enter' && primarySelectedPolicy) {
                                                    renamePolicy(primarySelectedPolicy.id, editingPolicyName);
                                                } else if (e.key === 'Escape') {
                                                    setEditingPolicyId(null);
                                                }
                                            }}
                                            className="text-sm font-semibold bg-purple-50 dark:bg-purple-950/50 border border-purple-200 dark:border-purple-800 rounded-lg px-2 py-1 focus:outline-none focus:ring-2 focus:ring-purple-300 dark:focus:ring-purple-700 text-neutral-900 dark:text-zinc-100"
                                            autoFocus
                                        />
                                        <button
                                            onClick={() => primarySelectedPolicy && renamePolicy(primarySelectedPolicy.id, editingPolicyName)}
                                            className="p-1 hover:bg-purple-100 rounded text-purple-600"
                                        >
                                            <Check className="w-4 h-4" />
                                        </button>
                                    </div>
                                ) : (
                                    <button
                                        onClick={() => {
                                            if (primarySelectedPolicy) {
                                                setEditingPolicyId(primarySelectedPolicy.id);
                                                setEditingPolicyName(primarySelectedPolicy.name);
                                            }
                                        }}
                                        className="flex items-center gap-2 group"
                                    >
                                        <span className="text-sm text-neutral-700 dark:text-zinc-200 font-semibold">{primarySelectedPolicy?.name}</span>
                                        <Pencil className="w-3 h-3 text-neutral-300 dark:text-zinc-600 group-hover:text-purple-500 transition-colors" />
                                    </button>
                                )}
                                <span className={`text-xs px-2 py-1 rounded-lg uppercase font-medium ${
                                    primarySelectedPolicy?.status === 'completed' ? 'bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300' :
                                    primarySelectedPolicy?.status === 'training' ? 'bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300' :
                                    'bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300'
                                }`}>
                                    {primarySelectedPolicy?.status}
                                </span>
                                {selectedPolicyData.length > 1 && (
                                    <span className="text-xs bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 px-2 py-1 rounded-lg">
                                        Comparing {selectedPolicyData.length} policies
                                    </span>
                                )}
                            </div>
                            <div className="flex items-center gap-2">
                                {primarySelectedPolicy?.status === 'completed' && primarySelectedPolicy?.checkpoint_path && (
                                    <>
                                        <button
                                            onClick={() => primarySelectedPolicy && deployPolicy(primarySelectedPolicy.id)}
                                            disabled={isDeploying}
                                            className="flex items-center gap-1.5 px-3 py-1.5 bg-green-500 hover:bg-green-600 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
                                        >
                                            {isDeploying ? <Loader2 className="w-4 h-4 animate-spin" /> : <Rocket className="w-4 h-4" />}
                                            Deploy
                                        </button>
                                        <button
                                            onClick={() => {
                                                if (primarySelectedPolicy) {
                                                    setResumingPolicyId(primarySelectedPolicy.id);
                                                    setShowResumeModal(true);
                                                }
                                            }}
                                            className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-500 hover:bg-purple-600 text-white text-sm font-medium rounded-lg transition-colors"
                                        >
                                            <RotateCcw className="w-4 h-4" />
                                            Resume
                                        </button>
                                    </>
                                )}
                                <button
                                    onClick={() => primarySelectedPolicy && deletePolicy(primarySelectedPolicy.id)}
                                    className="p-2 hover:bg-red-50 dark:hover:bg-red-950/50 hover:text-red-500 rounded-lg text-neutral-400 dark:text-zinc-500 transition-colors"
                                    title="Delete policy"
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>
                            </div>
                        </div>

                        {/* Loss Curve Chart */}
                        <div className="flex-1 p-4 overflow-auto">
                            <div className="bg-white dark:bg-zinc-900 rounded-2xl border border-neutral-200/50 dark:border-zinc-700/50 p-4 shadow-sm">
                                <h3 className="text-sm font-semibold text-neutral-700 dark:text-zinc-200 mb-3">Loss Curve</h3>
                                <div className="h-64">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                                            <XAxis
                                                dataKey="step"
                                                type="number"
                                                domain={['dataMin', 'dataMax']}
                                                tick={{ fill: '#999', fontSize: 10 }}
                                                tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`}
                                            />
                                            <YAxis
                                                tick={{ fill: '#999', fontSize: 10 }}
                                                tickFormatter={(v) => v.toFixed(3)}
                                            />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: 'white', border: '1px solid #e5e5e5', borderRadius: '8px', fontSize: '12px' }}
                                                formatter={(value) => [Number(value).toFixed(4), 'Loss']}
                                                labelFormatter={(label) => `Step ${label}`}
                                            />
                                            {selectedPolicyData.length > 1 && <Legend />}
                                            {selectedPolicyData.map((policy, idx) => {
                                                const colors = ['#10b981', '#8b5cf6', '#f59e0b', '#ef4444'];
                                                const chartData = policy.loss_history.map(([step, loss]) => ({ step, loss }));
                                                return (
                                                    <Line
                                                        key={policy.id}
                                                        data={chartData}
                                                        type="monotone"
                                                        dataKey="loss"
                                                        name={selectedPolicyData.length > 1 ? policy.name : undefined}
                                                        stroke={colors[idx % colors.length]}
                                                        dot={false}
                                                        strokeWidth={2}
                                                        isAnimationActive={false}
                                                    />
                                                );
                                            })}
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                                {selectedPolicyData[0]?.loss_history.length === 0 && (
                                    <div className="text-center text-neutral-400 dark:text-zinc-500 py-8">
                                        No loss history available for this policy
                                    </div>
                                )}
                            </div>

                            {/* Metadata Cards */}
                            <div className="grid grid-cols-2 gap-4 mt-4">
                                <div className="bg-white dark:bg-zinc-900 rounded-xl border border-neutral-200/50 dark:border-zinc-700/50 p-4">
                                    <span className="text-xs text-neutral-500 dark:text-zinc-400 uppercase tracking-wide">Policy Type</span>
                                    <p className="text-lg font-semibold text-neutral-800 dark:text-zinc-100 mt-1">{primarySelectedPolicy?.policy_type.toUpperCase()}</p>
                                </div>
                                <div className="bg-white dark:bg-zinc-900 rounded-xl border border-neutral-200/50 dark:border-zinc-700/50 p-4">
                                    <span className="text-xs text-neutral-500 dark:text-zinc-400 uppercase tracking-wide">Training Steps</span>
                                    <p className="text-lg font-semibold text-neutral-800 dark:text-zinc-100 mt-1">{primarySelectedPolicy?.steps.toLocaleString()}</p>
                                </div>
                                <div className="bg-white dark:bg-zinc-900 rounded-xl border border-neutral-200/50 dark:border-zinc-700/50 p-4">
                                    <span className="text-xs text-neutral-500 dark:text-zinc-400 uppercase tracking-wide">Source Dataset</span>
                                    <p className="text-lg font-semibold text-neutral-800 mt-1 truncate">{primarySelectedPolicy?.dataset_repo_id || 'Unknown'}</p>
                                </div>
                                <div className="bg-white dark:bg-zinc-900 rounded-xl border border-neutral-200/50 dark:border-zinc-700/50 p-4">
                                    <span className="text-xs text-neutral-500 dark:text-zinc-400 uppercase tracking-wide">Final Loss</span>
                                    <p className="text-lg font-semibold text-neutral-800 dark:text-zinc-100 mt-1">
                                        {primarySelectedPolicy?.final_loss?.toFixed(4) || 'N/A'}
                                    </p>
                                </div>
                            </div>

                            {/* Created Date */}
                            <div className="mt-4 text-xs text-neutral-400 dark:text-zinc-500 text-center">
                                Created: {primarySelectedPolicy?.created_at ? new Date(primarySelectedPolicy.created_at).toLocaleString() : 'Unknown'}
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="flex-1 flex flex-col items-center justify-center text-neutral-400 dark:text-zinc-500 gap-4">
                        <Brain className="w-16 h-16 text-neutral-200 dark:text-zinc-700" />
                        <p className="text-neutral-500 dark:text-zinc-400">Select a policy to view details</p>
                    </div>
                )}
            </div>

            {/* Resume Training Modal */}
            {showResumeModal && resumingPolicyId && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 flex items-center justify-center z-[200]"
                >
                    <div className="absolute inset-0 bg-black/40" onClick={() => setShowResumeModal(false)} />
                    <motion.div
                        initial={{ scale: 0.95, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        className="relative bg-white dark:bg-zinc-900 rounded-2xl shadow-2xl p-6 max-w-md w-full mx-4 border border-transparent dark:border-zinc-700"
                    >
                        <div className="flex items-center gap-3 mb-4">
                            <div className="p-3 bg-purple-100 dark:bg-purple-900/50 rounded-xl">
                                <RotateCcw className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                            </div>
                            <div>
                                <h3 className="text-lg font-semibold text-neutral-900 dark:text-zinc-100">Resume Training</h3>
                                <p className="text-sm text-neutral-500 dark:text-zinc-400">Continue training from checkpoint</p>
                            </div>
                        </div>

                        <div className="mb-4">
                            <label className="block text-sm font-medium text-neutral-700 dark:text-zinc-300 mb-2">
                                Additional Training Steps
                            </label>
                            <input
                                type="number"
                                value={resumeSteps}
                                onChange={(e) => setResumeSteps(parseInt(e.target.value) || 10000)}
                                className="w-full px-3 py-2 border border-neutral-200 dark:border-zinc-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-300 dark:focus:ring-purple-700 bg-white dark:bg-zinc-800 text-neutral-900 dark:text-zinc-100"
                                min={1000}
                                step={1000}
                            />
                            <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-1">
                                Recommended: 10,000 - 50,000 steps
                            </p>
                        </div>

                        <div className="flex gap-3">
                            <button
                                onClick={() => setShowResumeModal(false)}
                                className="flex-1 py-2.5 px-4 border border-neutral-200 dark:border-zinc-700 rounded-xl text-neutral-600 dark:text-zinc-300 font-medium hover:bg-neutral-50 dark:hover:bg-zinc-800 transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={() => resumingPolicyId && resumeTraining(resumingPolicyId, resumeSteps)}
                                className="flex-1 py-2.5 px-4 bg-purple-500 hover:bg-purple-600 rounded-xl text-white font-medium transition-colors flex items-center justify-center gap-2"
                            >
                                <RotateCcw className="w-4 h-4" />
                                Resume
                            </button>
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </>
    );
}
