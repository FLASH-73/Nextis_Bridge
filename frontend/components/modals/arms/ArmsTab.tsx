"use client";

import React from 'react';
import { RefreshCw } from 'lucide-react';
import type { Arm } from '@/lib/api/types';
import ArmCard from './ArmCard';

interface ArmsTabProps {
    arms: Arm[];
    editingArm: string | null;
    editName: string;
    editPort: string;
    setEditingArm: (id: string | null) => void;
    setEditName: (name: string) => void;
    setEditPort: (port: string) => void;
    onConnect: (id: string) => void;
    onDisconnect: (id: string) => void;
    onDelete: (id: string) => void;
    onSetHome: (id: string) => void;
    onSaveEdit: (id: string) => void;
    onRefresh: () => void;
}

export default function ArmsTab({
    arms,
    editingArm,
    editName,
    editPort,
    setEditingArm,
    setEditName,
    setEditPort,
    onConnect,
    onDisconnect,
    onDelete,
    onSetHome,
    onSaveEdit,
    onRefresh,
}: ArmsTabProps) {
    const leaders = arms.filter(a => a.role === 'leader');
    const followers = arms.filter(a => a.role === 'follower');

    return (
        <div className="space-y-6">
            {/* Followers Section */}
            <div>
                <h3 className="text-xs font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-wider mb-3">
                    Followers ({followers.length})
                </h3>
                <div className="space-y-2">
                    {followers.length === 0 ? (
                        <div className="text-center py-8 text-neutral-400 dark:text-zinc-500">
                            No follower arms configured
                        </div>
                    ) : (
                        followers.map(arm => (
                            <ArmCard
                                key={arm.id}
                                arm={arm}
                                onConnect={onConnect}
                                onDisconnect={onDisconnect}
                                onDelete={onDelete}
                                onEdit={(id) => { setEditingArm(id); setEditName(arm.name); setEditPort(arm.port); }}
                                onSetHome={onSetHome}
                                isEditing={editingArm === arm.id}
                                editName={editName}
                                setEditName={setEditName}
                                editPort={editPort}
                                setEditPort={setEditPort}
                                onSaveEdit={onSaveEdit}
                                onCancelEdit={() => setEditingArm(null)}
                            />
                        ))
                    )}
                </div>
            </div>

            {/* Leaders Section */}
            <div>
                <h3 className="text-xs font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-wider mb-3">
                    Leaders ({leaders.length})
                </h3>
                <div className="space-y-2">
                    {leaders.length === 0 ? (
                        <div className="text-center py-8 text-neutral-400 dark:text-zinc-500">
                            No leader arms configured
                        </div>
                    ) : (
                        leaders.map(arm => (
                            <ArmCard
                                key={arm.id}
                                arm={arm}
                                onConnect={onConnect}
                                onDisconnect={onDisconnect}
                                onDelete={onDelete}
                                onEdit={(id) => { setEditingArm(id); setEditName(arm.name); setEditPort(arm.port); }}
                                onSetHome={onSetHome}
                                isEditing={editingArm === arm.id}
                                editName={editName}
                                setEditName={setEditName}
                                editPort={editPort}
                                setEditPort={setEditPort}
                                onSaveEdit={onSaveEdit}
                                onCancelEdit={() => setEditingArm(null)}
                            />
                        ))
                    )}
                </div>
            </div>

            {/* Refresh Button */}
            <div className="flex justify-center pt-4">
                <button
                    onClick={onRefresh}
                    className="flex items-center gap-2 px-4 py-2 text-sm text-neutral-600 dark:text-zinc-400 hover:text-black dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/5 rounded-lg transition-colors"
                >
                    <RefreshCw className="w-4 h-4" /> Refresh
                </button>
            </div>
        </div>
    );
}

export type { ArmsTabProps };
