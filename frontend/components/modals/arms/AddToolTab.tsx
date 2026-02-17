"use client";

import React from 'react';
import { Plus } from 'lucide-react';
import type { Port } from '@/lib/api/types';

interface NewToolForm {
    id: string;
    name: string;
    tool_type: string;
    motor_type: string;
    port: string;
    motor_id: number;
}

interface AddToolTabProps {
    newTool: NewToolForm;
    setNewTool: (tool: NewToolForm) => void;
    ports: Port[];
    loading: boolean;
    onScanPorts: () => void;
    onAddTool: () => void;
}

export default function AddToolTab({
    newTool,
    setNewTool,
    ports,
    loading,
    onScanPorts,
    onAddTool,
}: AddToolTabProps) {
    return (
        <div className="space-y-6">
            <div className="p-4 bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700">
                <h4 className="font-medium text-black dark:text-white mb-4">Add New Tool</h4>

                <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Tool ID *</label>
                        <input
                            type="text"
                            value={newTool.id}
                            onChange={(e) => setNewTool({ ...newTool, id: e.target.value.replace(/\s+/g, '_').toLowerCase() })}
                            placeholder="e.g., screwdriver_1"
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        />
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Display Name *</label>
                        <input
                            type="text"
                            value={newTool.name}
                            onChange={(e) => setNewTool({ ...newTool, name: e.target.value })}
                            placeholder="e.g., Main Screwdriver"
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        />
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Tool Type *</label>
                        <select
                            value={newTool.tool_type}
                            onChange={(e) => setNewTool({ ...newTool, tool_type: e.target.value })}
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        >
                            <option value="screwdriver">Screwdriver</option>
                            <option value="gripper">Gripper</option>
                            <option value="pump">Pump</option>
                            <option value="custom">Custom</option>
                        </select>
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Motor Type *</label>
                        <select
                            value={newTool.motor_type}
                            onChange={(e) => setNewTool({ ...newTool, motor_type: e.target.value })}
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        >
                            <option value="sts3215">STS3215 (Feetech)</option>
                            <option value="dynamixel_xl330">Dynamixel XL330</option>
                            <option value="dynamixel_xl430">Dynamixel XL430</option>
                        </select>
                    </div>
                </div>

                <div className="mb-4">
                    <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">
                        Serial Port *
                        <button
                            onClick={onScanPorts}
                            disabled={loading}
                            className="ml-2 text-blue-500 hover:text-blue-600"
                        >
                            {loading ? 'Scanning...' : 'Scan Ports'}
                        </button>
                    </label>
                    <select
                        value={newTool.port}
                        onChange={(e) => setNewTool({ ...newTool, port: e.target.value })}
                        className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                    >
                        <option value="">Select port...</option>
                        {ports.map(p => (
                            <option key={p.device} value={p.device} disabled={p.in_use}>
                                {p.device} - {p.description} {p.in_use ? '(in use)' : ''}
                            </option>
                        ))}
                        <option value="custom">Enter manually...</option>
                    </select>
                    {newTool.port === 'custom' && (
                        <input
                            type="text"
                            placeholder="/dev/ttyUSB0"
                            onChange={(e) => setNewTool({ ...newTool, port: e.target.value })}
                            className="w-full mt-2 px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        />
                    )}
                </div>

                <div className="mb-4">
                    <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Motor ID *</label>
                    <input
                        type="number"
                        min={1}
                        max={253}
                        value={newTool.motor_id}
                        onChange={(e) => setNewTool({ ...newTool, motor_id: parseInt(e.target.value) || 1 })}
                        className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                    />
                    <p className="text-[10px] text-neutral-400 dark:text-zinc-500 mt-1">
                        Motor ID on the serial bus (1-253). Use Motor Setup tab to scan for IDs.
                    </p>
                </div>

                <button
                    onClick={onAddTool}
                    disabled={!newTool.id || !newTool.name || !newTool.port || loading}
                    className="w-full py-3 bg-black dark:bg-white text-white dark:text-black rounded-xl font-medium disabled:opacity-50 hover:bg-neutral-800 dark:hover:bg-zinc-200 transition-colors flex items-center justify-center gap-2"
                >
                    <Plus className="w-4 h-4" /> Add Tool
                </button>
            </div>
        </div>
    );
}

export type { AddToolTabProps, NewToolForm };
