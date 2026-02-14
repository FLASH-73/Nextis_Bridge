"use client";

import React from 'react';
import { X, Power, PowerOff, Trash2, Edit3, Save, Home } from 'lucide-react';
import type { Arm } from '@/lib/api/types';

const MOTOR_TYPE_COLORS: Record<string, string> = {
    sts3215: 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300',
    damiao: 'bg-orange-100 text-orange-700 dark:bg-orange-900/50 dark:text-orange-300',
    dynamixel_xl330: 'bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300',
    dynamixel_xl430: 'bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300',
};

const MOTOR_TYPE_LABELS: Record<string, string> = {
    sts3215: 'STS3215',
    damiao: 'Damiao',
    dynamixel_xl330: 'XL330',
    dynamixel_xl430: 'XL430',
};

interface ArmCardProps {
    arm: Arm;
    onConnect: (id: string) => void;
    onDisconnect: (id: string) => void;
    onDelete: (id: string) => void;
    onEdit: (id: string) => void;
    onSetHome: (id: string) => void;
    isEditing: boolean;
    editName: string;
    setEditName: (name: string) => void;
    editPort: string;
    setEditPort: (port: string) => void;
    onSaveEdit: (id: string) => void;
    onCancelEdit: () => void;
}

export default function ArmCard({
    arm,
    onConnect,
    onDisconnect,
    onDelete,
    onEdit,
    onSetHome,
    isEditing,
    editName,
    setEditName,
    editPort,
    setEditPort,
    onSaveEdit,
    onCancelEdit,
}: ArmCardProps) {
    const statusColor = {
        connected: 'bg-green-500',
        connecting: 'bg-yellow-500 animate-pulse',
        error: 'bg-red-500',
        disconnected: 'bg-neutral-400 dark:bg-zinc-600',
    }[arm.status];

    const motorTypeColor = MOTOR_TYPE_COLORS[arm.motor_type] || 'bg-neutral-100 text-neutral-700 dark:bg-zinc-800 dark:text-zinc-300';
    const motorTypeLabel = MOTOR_TYPE_LABELS[arm.motor_type] || arm.motor_type;

    return (
        <div className="flex items-center justify-between p-4 bg-white/60 dark:bg-zinc-800/60 rounded-xl border border-neutral-100 dark:border-zinc-700 hover:border-neutral-200 dark:hover:border-zinc-600 transition-colors group">
            <div className="flex items-center gap-3">
                <div className={`w-2.5 h-2.5 rounded-full ${statusColor}`} />
                <div>
                    {isEditing ? (
                        <div className="flex items-center gap-2">
                            <input
                                type="text"
                                value={editName}
                                onChange={(e) => setEditName(e.target.value)}
                                className="px-2 py-1 text-sm border border-neutral-200 dark:border-zinc-600 rounded bg-white dark:bg-zinc-900"
                                placeholder="Name"
                                autoFocus
                            />
                            <input
                                type="text"
                                value={editPort}
                                onChange={(e) => setEditPort(e.target.value)}
                                className="px-2 py-1 text-sm border border-neutral-200 dark:border-zinc-600 rounded bg-white dark:bg-zinc-900 font-mono"
                                placeholder="Port"
                            />
                            <button onClick={() => onSaveEdit(arm.id)} className="p-1 text-green-500 hover:bg-green-50 dark:hover:bg-green-950 rounded">
                                <Save className="w-4 h-4" />
                            </button>
                            <button onClick={onCancelEdit} className="p-1 text-neutral-400 hover:bg-neutral-100 dark:hover:bg-zinc-800 rounded">
                                <X className="w-4 h-4" />
                            </button>
                        </div>
                    ) : (
                        <div className="flex items-center gap-2">
                            <span className="font-medium text-black dark:text-white">{arm.name}</span>
                            <span className={`px-1.5 py-0.5 text-[10px] font-semibold rounded ${motorTypeColor}`}>
                                {motorTypeLabel}
                            </span>
                            {arm.calibrated && (
                                <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300 rounded">
                                    CALIBRATED
                                </span>
                            )}
                        </div>
                    )}
                    <div className="text-xs text-neutral-400 dark:text-zinc-500">
                        {arm.port}
                    </div>
                </div>
            </div>

            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                {arm.status === 'connected' && (
                    <button
                        onClick={() => onSetHome(arm.id)}
                        className="p-2 text-blue-500 hover:bg-blue-50 dark:hover:bg-blue-950 rounded-lg transition-colors"
                        title="Set current position as home"
                    >
                        <Home className="w-4 h-4" />
                    </button>
                )}
                {arm.status === 'connected' ? (
                    <button
                        onClick={() => onDisconnect(arm.id)}
                        className="p-2 text-orange-500 hover:bg-orange-50 dark:hover:bg-orange-950 rounded-lg transition-colors"
                        title="Disconnect"
                    >
                        <PowerOff className="w-4 h-4" />
                    </button>
                ) : (
                    <button
                        onClick={() => onConnect(arm.id)}
                        className="p-2 text-green-500 hover:bg-green-50 dark:hover:bg-green-950 rounded-lg transition-colors"
                        title="Connect"
                    >
                        <Power className="w-4 h-4" />
                    </button>
                )}
                <button
                    onClick={() => onEdit(arm.id)}
                    className="p-2 text-neutral-400 hover:text-black dark:hover:text-white hover:bg-neutral-100 dark:hover:bg-zinc-800 rounded-lg transition-colors"
                    title="Edit"
                >
                    <Edit3 className="w-4 h-4" />
                </button>
                <button
                    onClick={() => onDelete(arm.id)}
                    className="p-2 text-red-500 hover:bg-red-50 dark:hover:bg-red-950 rounded-lg transition-colors"
                    title="Delete"
                >
                    <Trash2 className="w-4 h-4" />
                </button>
            </div>
        </div>
    );
}

export type { Arm, ArmCardProps };
