"use client";

import React, { useRef } from 'react';
import { Power, PowerOff, Play, Trash2, RefreshCw } from 'lucide-react';
import type { Tool } from '@/lib/api/types';

const TOOL_TYPE_COLORS: Record<string, string> = {
    screwdriver: 'bg-teal-100 text-teal-700 dark:bg-teal-900/50 dark:text-teal-300',
    gripper: 'bg-cyan-100 text-cyan-700 dark:bg-cyan-900/50 dark:text-cyan-300',
    pump: 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/50 dark:text-indigo-300',
    custom: 'bg-neutral-100 text-neutral-700 dark:bg-zinc-800 dark:text-zinc-300',
};

const MOTOR_TYPE_LABELS: Record<string, string> = {
    sts3215: 'STS3215',
    dynamixel_xl330: 'XL330',
    dynamixel_xl430: 'XL430',
};

interface ToolsTabProps {
    tools: Tool[];
    loading: boolean;
    onConnect: (id: string) => void;
    onDisconnect: (id: string) => void;
    onActivate: (id: string) => void;
    onDeactivate: (id: string) => void;
    onRemove: (id: string) => void;
    onRefresh: () => void;
}

export default function ToolsTab({
    tools,
    loading,
    onConnect,
    onDisconnect,
    onActivate,
    onDeactivate,
    onRemove,
    onRefresh,
}: ToolsTabProps) {
    const testTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

    const handleTest = (id: string) => {
        // Clear any existing timer for this tool
        if (testTimers.current[id]) {
            clearTimeout(testTimers.current[id]);
        }
        onActivate(id);
        testTimers.current[id] = setTimeout(() => {
            onDeactivate(id);
            delete testTimers.current[id];
        }, 2000);
    };

    return (
        <div className="space-y-6">
            <div>
                <h3 className="text-xs font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-wider mb-3">
                    Registered Tools ({tools.length})
                </h3>
                <div className="space-y-2">
                    {tools.length === 0 ? (
                        <div className="text-center py-8 text-neutral-400 dark:text-zinc-500">
                            No tools registered. Add one from the &quot;Add Tool&quot; tab.
                        </div>
                    ) : (
                        tools.map(tool => {
                            const statusColor = {
                                connected: 'bg-green-500',
                                error: 'bg-red-500',
                                disconnected: 'bg-neutral-400 dark:bg-zinc-600',
                            }[tool.status];

                            const typeColor = TOOL_TYPE_COLORS[tool.tool_type] || TOOL_TYPE_COLORS.custom;
                            const motorLabel = MOTOR_TYPE_LABELS[tool.motor_type] || tool.motor_type;

                            return (
                                <div
                                    key={tool.id}
                                    className="flex items-center justify-between p-4 bg-white/60 dark:bg-zinc-800/60 rounded-xl border border-neutral-100 dark:border-zinc-700 hover:border-neutral-200 dark:hover:border-zinc-600 transition-colors group"
                                >
                                    <div className="flex items-center gap-3">
                                        <div className={`w-2.5 h-2.5 rounded-full ${statusColor}`} />
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <span className="font-medium text-black dark:text-white">{tool.name}</span>
                                                <span className={`px-1.5 py-0.5 text-[10px] font-semibold rounded ${typeColor}`}>
                                                    {tool.tool_type}
                                                </span>
                                            </div>
                                            <div className="text-xs text-neutral-400 dark:text-zinc-500">
                                                {motorLabel} · ID {tool.motor_id} · {tool.port}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                        {tool.status === 'connected' && (
                                            <button
                                                onClick={() => handleTest(tool.id)}
                                                className="p-2 text-blue-500 hover:bg-blue-50 dark:hover:bg-blue-950 rounded-lg transition-colors"
                                                title="Test (activate 2s)"
                                            >
                                                <Play className="w-4 h-4" />
                                            </button>
                                        )}
                                        {tool.status === 'connected' ? (
                                            <button
                                                onClick={() => onDisconnect(tool.id)}
                                                className="p-2 text-orange-500 hover:bg-orange-50 dark:hover:bg-orange-950 rounded-lg transition-colors"
                                                title="Disconnect"
                                            >
                                                <PowerOff className="w-4 h-4" />
                                            </button>
                                        ) : (
                                            <button
                                                onClick={() => onConnect(tool.id)}
                                                className="p-2 text-green-500 hover:bg-green-50 dark:hover:bg-green-950 rounded-lg transition-colors"
                                                title="Connect"
                                            >
                                                <Power className="w-4 h-4" />
                                            </button>
                                        )}
                                        <button
                                            onClick={() => onRemove(tool.id)}
                                            className="p-2 text-red-500 hover:bg-red-50 dark:hover:bg-red-950 rounded-lg transition-colors"
                                            title="Remove"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>
                            );
                        })
                    )}
                </div>
            </div>

            <div className="flex justify-center">
                <button
                    onClick={onRefresh}
                    disabled={loading}
                    className="px-4 py-2 text-sm text-neutral-500 dark:text-zinc-400 hover:text-black dark:hover:text-white hover:bg-neutral-100 dark:hover:bg-zinc-800 rounded-lg transition-colors flex items-center gap-2"
                >
                    <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} /> Refresh
                </button>
            </div>
        </div>
    );
}

export type { ToolsTabProps };
