"use client";

import React from 'react';
import { Link2, Unlink } from 'lucide-react';
import type { Arm, Pairing } from '@/lib/api/types';

interface PairingsTabProps {
    arms: Arm[];
    pairings: Pairing[];
    newPairing: { leader_id: string; follower_id: string; name: string };
    setNewPairing: (pairing: { leader_id: string; follower_id: string; name: string }) => void;
    loading: boolean;
    onCreatePairing: () => void;
    onRemovePairing: (leaderId: string, followerId: string) => void;
}

export default function PairingsTab({
    arms,
    pairings,
    newPairing,
    setNewPairing,
    loading,
    onCreatePairing,
    onRemovePairing,
}: PairingsTabProps) {
    const leaders = arms.filter(a => a.role === 'leader');
    const followers = arms.filter(a => a.role === 'follower');

    return (
        <div className="space-y-6">
            {/* Existing Pairings */}
            <div>
                <h3 className="text-xs font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-wider mb-3">
                    Active Pairings ({pairings.length})
                </h3>
                <div className="space-y-2">
                    {pairings.length === 0 ? (
                        <div className="text-center py-8 text-neutral-400 dark:text-zinc-500">
                            No pairings configured. Create one below.
                        </div>
                    ) : (
                        pairings.map((p, i) => (
                            <div
                                key={i}
                                className="flex items-center justify-between p-4 bg-white/60 dark:bg-zinc-800/60 rounded-xl border border-neutral-100 dark:border-zinc-700"
                            >
                                <div className="flex items-center gap-3">
                                    <div className="px-3 py-1.5 bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300 rounded-lg text-sm font-medium">
                                        {arms.find(a => a.id === p.leader_id)?.name || p.leader_id}
                                    </div>
                                    <Link2 className="w-4 h-4 text-neutral-400" />
                                    <div className="px-3 py-1.5 bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded-lg text-sm font-medium">
                                        {arms.find(a => a.id === p.follower_id)?.name || p.follower_id}
                                    </div>
                                </div>
                                <button
                                    onClick={() => onRemovePairing(p.leader_id, p.follower_id)}
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

            {/* Create New Pairing */}
            <div className="p-4 bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700">
                <h4 className="font-medium text-black dark:text-white mb-4">Create New Pairing</h4>
                <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Leader</label>
                        <select
                            value={newPairing.leader_id}
                            onChange={(e) => setNewPairing({ ...newPairing, leader_id: e.target.value })}
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        >
                            <option value="">Select leader...</option>
                            {leaders.map(l => (
                                <option key={l.id} value={l.id}>{l.name}</option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Follower</label>
                        <select
                            value={newPairing.follower_id}
                            onChange={(e) => setNewPairing({ ...newPairing, follower_id: e.target.value })}
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        >
                            <option value="">Select follower...</option>
                            {followers.map(f => (
                                <option key={f.id} value={f.id}>{f.name}</option>
                            ))}
                        </select>
                    </div>
                </div>
                <div className="mb-4">
                    <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Pairing Name (optional)</label>
                    <input
                        type="text"
                        value={newPairing.name}
                        onChange={(e) => setNewPairing({ ...newPairing, name: e.target.value })}
                        placeholder="e.g., Damiao Assembly Setup"
                        className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                    />
                </div>
                <button
                    onClick={onCreatePairing}
                    disabled={!newPairing.leader_id || !newPairing.follower_id || loading}
                    className="w-full py-2 bg-black dark:bg-white text-white dark:text-black rounded-lg font-medium text-sm disabled:opacity-50 hover:bg-neutral-800 dark:hover:bg-zinc-200 transition-colors flex items-center justify-center gap-2"
                >
                    <Link2 className="w-4 h-4" /> Create Pairing
                </button>
            </div>
        </div>
    );
}

export type { PairingsTabProps };
