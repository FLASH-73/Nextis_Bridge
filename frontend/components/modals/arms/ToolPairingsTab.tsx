"use client";

import React, { useState } from 'react';
import { Link2, Unlink, Play, Square, AlertTriangle, Plus } from 'lucide-react';
import type { Tool, Trigger, ToolPairing, Port } from '@/lib/api/types';

interface ToolPairingsTabProps {
    tools: Tool[];
    triggers: Trigger[];
    toolPairings: ToolPairing[];
    newToolPairing: { trigger_id: string; tool_id: string; name: string; action: string };
    setNewToolPairing: (pairing: { trigger_id: string; tool_id: string; name: string; action: string }) => void;
    loading: boolean;
    listenerRunning: boolean;
    ports: Port[];
    onCreateToolPairing: () => void;
    onRemoveToolPairing: (triggerId: string, toolId: string) => void;
    onStartListener: () => void;
    onStopListener: () => void;
    onAddTrigger: (trigger: {
        id: string; name: string; trigger_type: string;
        port: string; pin: number; active_low: boolean;
    }) => void;
    onScanPorts: () => void;
}

export default function ToolPairingsTab({
    tools,
    triggers,
    toolPairings,
    newToolPairing,
    setNewToolPairing,
    loading,
    listenerRunning,
    ports,
    onCreateToolPairing,
    onRemoveToolPairing,
    onStartListener,
    onStopListener,
    onAddTrigger,
    onScanPorts,
}: ToolPairingsTabProps) {
    const [newTrigger, setNewTrigger] = useState({
        id: '', name: '', trigger_type: 'gpio_switch', port: '', pin: 0, active_low: true,
    });

    const handleAddTrigger = () => {
        onAddTrigger(newTrigger);
        setNewTrigger({ id: '', name: '', trigger_type: 'gpio_switch', port: '', pin: 0, active_low: true });
    };

    return (
        <div className="space-y-6">
            {/* Listener Control */}
            <div className="flex items-center justify-between p-4 bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700">
                <div className="flex items-center gap-3">
                    <div className={`w-2.5 h-2.5 rounded-full ${listenerRunning ? 'bg-green-500' : 'bg-neutral-400 dark:bg-zinc-600'}`} />
                    <span className="text-sm font-medium text-black dark:text-white">
                        Trigger Listener {listenerRunning ? 'Running' : 'Stopped'}
                    </span>
                </div>
                {listenerRunning ? (
                    <button
                        onClick={onStopListener}
                        disabled={loading}
                        className="px-4 py-2 text-sm font-medium text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/50 hover:bg-red-100 dark:hover:bg-red-950 rounded-lg transition-colors flex items-center gap-2"
                    >
                        <Square className="w-3.5 h-3.5" /> Stop
                    </button>
                ) : (
                    <button
                        onClick={onStartListener}
                        disabled={loading || triggers.length === 0}
                        className="px-4 py-2 text-sm font-medium text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-950/50 hover:bg-green-100 dark:hover:bg-green-950 rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50"
                    >
                        <Play className="w-3.5 h-3.5" /> Start
                    </button>
                )}
            </div>

            {/* Prerequisites Warning */}
            {!listenerRunning && (triggers.length === 0 || toolPairings.length === 0) && (
                <div className="p-4 bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded-xl">
                    <div className="flex items-start gap-2">
                        <AlertTriangle className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0" />
                        <div className="text-sm text-amber-700 dark:text-amber-300 space-y-1">
                            {triggers.length === 0 && (
                                <p>No trigger devices registered. Add a trigger below to use the listener.</p>
                            )}
                            {triggers.length > 0 && toolPairings.length === 0 && (
                                <p>No tool pairings configured. Create a pairing below to link a trigger to a tool.</p>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Active Tool Pairings */}
            <div>
                <h3 className="text-xs font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-wider mb-3">
                    Active Tool Pairings ({toolPairings.length})
                </h3>
                <div className="space-y-2">
                    {toolPairings.length === 0 ? (
                        <div className="text-center py-8 text-neutral-400 dark:text-zinc-500">
                            No tool pairings configured. Create one below.
                        </div>
                    ) : (
                        toolPairings.map((p, i) => (
                            <div
                                key={i}
                                className="flex items-center justify-between p-4 bg-white/60 dark:bg-zinc-800/60 rounded-xl border border-neutral-100 dark:border-zinc-700"
                            >
                                <div className="flex items-center gap-3">
                                    <div className="px-3 py-1.5 bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-300 rounded-lg text-sm font-medium">
                                        {triggers.find(t => t.id === p.trigger_id)?.name || p.trigger_id}
                                    </div>
                                    <Link2 className="w-4 h-4 text-neutral-400" />
                                    <div className="px-3 py-1.5 bg-teal-100 dark:bg-teal-900/50 text-teal-700 dark:text-teal-300 rounded-lg text-sm font-medium">
                                        {tools.find(t => t.id === p.tool_id)?.name || p.tool_id}
                                    </div>
                                    <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-neutral-100 dark:bg-zinc-800 text-neutral-600 dark:text-zinc-400 rounded">
                                        {p.action}
                                    </span>
                                </div>
                                <button
                                    onClick={() => onRemoveToolPairing(p.trigger_id, p.tool_id)}
                                    className="p-2 text-red-500 hover:bg-red-50 dark:hover:bg-red-950 rounded-lg transition-colors"
                                    title="Remove Pairing"
                                >
                                    <Unlink className="w-4 h-4" />
                                </button>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Create New Tool Pairing */}
            <div className="p-4 bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700">
                <h4 className="font-medium text-black dark:text-white mb-4">Create Tool Pairing</h4>
                <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Trigger</label>
                        <select
                            value={newToolPairing.trigger_id}
                            onChange={(e) => setNewToolPairing({ ...newToolPairing, trigger_id: e.target.value })}
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        >
                            <option value="">Select trigger...</option>
                            {triggers.map(t => (
                                <option key={t.id} value={t.id}>{t.name}</option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Tool</label>
                        <select
                            value={newToolPairing.tool_id}
                            onChange={(e) => setNewToolPairing({ ...newToolPairing, tool_id: e.target.value })}
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        >
                            <option value="">Select tool...</option>
                            {tools.map(t => (
                                <option key={t.id} value={t.id}>{t.name}</option>
                            ))}
                        </select>
                    </div>
                </div>
                <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Action</label>
                        <select
                            value={newToolPairing.action}
                            onChange={(e) => setNewToolPairing({ ...newToolPairing, action: e.target.value })}
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        >
                            <option value="toggle">Toggle (press to toggle on/off)</option>
                            <option value="hold">Hold (active while pressed)</option>
                            <option value="pulse">Pulse (brief activation)</option>
                        </select>
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Name (optional)</label>
                        <input
                            type="text"
                            value={newToolPairing.name}
                            onChange={(e) => setNewToolPairing({ ...newToolPairing, name: e.target.value })}
                            placeholder="e.g., Button → Screwdriver"
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        />
                    </div>
                </div>
                <button
                    onClick={onCreateToolPairing}
                    disabled={!newToolPairing.trigger_id || !newToolPairing.tool_id || loading}
                    className="w-full py-2 bg-black dark:bg-white text-white dark:text-black rounded-lg font-medium text-sm disabled:opacity-50 hover:bg-neutral-800 dark:hover:bg-zinc-200 transition-colors flex items-center justify-center gap-2"
                >
                    <Link2 className="w-4 h-4" /> Create Tool Pairing
                </button>
            </div>

            {/* Add Trigger Device */}
            <div className="p-4 bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700">
                <h4 className="font-medium text-black dark:text-white mb-4">Add Trigger Device</h4>
                <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Trigger ID *</label>
                        <input
                            type="text"
                            value={newTrigger.id}
                            onChange={(e) => setNewTrigger({ ...newTrigger, id: e.target.value.replace(/\s+/g, '_').toLowerCase() })}
                            placeholder="e.g., button_1"
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        />
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Display Name *</label>
                        <input
                            type="text"
                            value={newTrigger.name}
                            onChange={(e) => setNewTrigger({ ...newTrigger, name: e.target.value })}
                            placeholder="e.g., Foot Pedal"
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        />
                    </div>
                </div>
                <div className="grid grid-cols-3 gap-4 mb-4">
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">
                            Port *
                            <button
                                onClick={onScanPorts}
                                disabled={loading}
                                className="ml-2 text-blue-500 hover:text-blue-600 text-xs"
                            >
                                {loading ? 'Scanning...' : 'Scan'}
                            </button>
                        </label>
                        <select
                            value={newTrigger.port}
                            onChange={(e) => setNewTrigger({ ...newTrigger, port: e.target.value })}
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        >
                            <option value="">Select port...</option>
                            {ports.map(p => (
                                <option key={p.device} value={p.device} disabled={p.in_use}>
                                    {p.device} - {p.description}{p.in_use ? ' (in use)' : ''}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">GPIO Pin *</label>
                        <input
                            type="number"
                            min={0}
                            max={28}
                            value={newTrigger.pin}
                            onChange={(e) => setNewTrigger({ ...newTrigger, pin: parseInt(e.target.value) || 0 })}
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        />
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Active Low</label>
                        <label className="flex items-center gap-2 mt-2">
                            <input
                                type="checkbox"
                                checked={newTrigger.active_low}
                                onChange={(e) => setNewTrigger({ ...newTrigger, active_low: e.target.checked })}
                                className="rounded"
                            />
                            <span className="text-xs text-neutral-500 dark:text-zinc-400">Pin LOW = pressed</span>
                        </label>
                    </div>
                </div>
                <button
                    onClick={handleAddTrigger}
                    disabled={!newTrigger.id || !newTrigger.name || !newTrigger.port || loading}
                    className="w-full py-2 bg-black dark:bg-white text-white dark:text-black rounded-lg font-medium text-sm disabled:opacity-50 hover:bg-neutral-800 dark:hover:bg-zinc-200 transition-colors flex items-center justify-center gap-2"
                >
                    <Plus className="w-4 h-4" /> Add Trigger
                </button>
            </div>
        </div>
    );
}

export type { ToolPairingsTabProps };
