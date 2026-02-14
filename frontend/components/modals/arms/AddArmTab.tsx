"use client";

import React from 'react';
import { Plus } from 'lucide-react';
import type { Port } from '@/lib/api/types';

interface NewArmForm {
    id: string;
    name: string;
    role: 'leader' | 'follower';
    motor_type: string;
    port: string;
    structural_design: string;
}

interface AddArmTabProps {
    newArm: NewArmForm;
    setNewArm: (arm: NewArmForm) => void;
    ports: Port[];
    loading: boolean;
    onScanPorts: () => void;
    onAddArm: () => void;
}

export default function AddArmTab({
    newArm,
    setNewArm,
    ports,
    loading,
    onScanPorts,
    onAddArm,
}: AddArmTabProps) {
    return (
        <div className="space-y-6">
            <div className="p-4 bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700">
                <h4 className="font-medium text-black dark:text-white mb-4">Add New Arm</h4>

                <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Arm ID *</label>
                        <input
                            type="text"
                            value={newArm.id}
                            onChange={(e) => setNewArm({ ...newArm, id: e.target.value.replace(/\s+/g, '_').toLowerCase() })}
                            placeholder="e.g., damiao_main"
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        />
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Display Name *</label>
                        <input
                            type="text"
                            value={newArm.name}
                            onChange={(e) => setNewArm({ ...newArm, name: e.target.value })}
                            placeholder="e.g., Damiao Assembly Arm"
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        />
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Role *</label>
                        <select
                            value={newArm.role}
                            onChange={(e) => setNewArm({ ...newArm, role: e.target.value as 'leader' | 'follower' })}
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        >
                            <option value="follower">Follower</option>
                            <option value="leader">Leader</option>
                        </select>
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Motor Type *</label>
                        <select
                            value={newArm.motor_type}
                            onChange={(e) => setNewArm({ ...newArm, motor_type: e.target.value })}
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        >
                            <option value="sts3215">STS3215 (Feetech)</option>
                            <option value="damiao">Damiao (CAN)</option>
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
                        value={newArm.port}
                        onChange={(e) => setNewArm({ ...newArm, port: e.target.value })}
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
                    {newArm.port === 'custom' && (
                        <input
                            type="text"
                            placeholder="/dev/ttyUSB0"
                            onChange={(e) => setNewArm({ ...newArm, port: e.target.value })}
                            className="w-full mt-2 px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        />
                    )}
                </div>

                <div className="mb-4">
                    <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Structural Design (optional)</label>
                    <select
                        value={newArm.structural_design}
                        onChange={(e) => setNewArm({ ...newArm, structural_design: e.target.value })}
                        className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                    >
                        <option value="">Select design...</option>
                        <option value="umbra_7dof">Umbra 7-DOF</option>
                        <option value="damiao_7dof">Damiao 7-DOF</option>
                        <option value="custom">Custom</option>
                    </select>
                    <p className="text-[10px] text-neutral-400 dark:text-zinc-500 mt-1">
                        Helps identify compatible leader-follower pairs
                    </p>
                </div>

                <button
                    onClick={onAddArm}
                    disabled={!newArm.id || !newArm.name || !newArm.port || loading}
                    className="w-full py-3 bg-black dark:bg-white text-white dark:text-black rounded-xl font-medium disabled:opacity-50 hover:bg-neutral-800 dark:hover:bg-zinc-200 transition-colors flex items-center justify-center gap-2"
                >
                    <Plus className="w-4 h-4" /> Add Arm
                </button>
            </div>
        </div>
    );
}

export type { AddArmTabProps, NewArmForm, Port };
