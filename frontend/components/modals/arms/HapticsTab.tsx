"use client";

import React from 'react';

interface HapticsTabProps {
    ffGripper: boolean;
    ffJoint: boolean;
    ffLoading: boolean;
    onToggleForceFeedback: (key: 'gripper' | 'joint', value: boolean) => void;
}

export default function HapticsTab({
    ffGripper,
    ffJoint,
    ffLoading,
    onToggleForceFeedback,
}: HapticsTabProps) {
    return (
        <div className="space-y-4">
            <p className="text-sm text-neutral-500 dark:text-zinc-400">
                Toggle force feedback channels on the leader arm during teleoperation.
            </p>

            {/* Gripper Force Feedback */}
            <div className="flex items-center justify-between p-4 bg-white/60 dark:bg-zinc-800/60 rounded-xl border border-neutral-100 dark:border-zinc-700">
                <div>
                    <h4 className="text-sm font-medium text-black dark:text-white">Gripper Force Feedback</h4>
                    <p className="text-xs text-neutral-500 dark:text-zinc-400 mt-1">
                        Resistance on leader gripper when follower contacts objects
                    </p>
                </div>
                <button
                    onClick={() => onToggleForceFeedback('gripper', !ffGripper)}
                    disabled={ffLoading}
                    className={`relative w-11 h-6 rounded-full transition-colors ${
                        ffGripper ? 'bg-green-500' : 'bg-neutral-300 dark:bg-zinc-600'
                    } ${ffLoading ? 'opacity-50' : ''}`}
                >
                    <span className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                        ffGripper ? 'translate-x-5' : 'translate-x-0'
                    }`} />
                </button>
            </div>

            {/* Joint Force Feedback */}
            <div className="flex items-center justify-between p-4 bg-white/60 dark:bg-zinc-800/60 rounded-xl border border-neutral-100 dark:border-zinc-700">
                <div>
                    <h4 className="text-sm font-medium text-black dark:text-white">Joint Force Feedback (link3)</h4>
                    <p className="text-xs text-neutral-500 dark:text-zinc-400 mt-1">
                        Virtual spring resistance when follower arm is blocked
                    </p>
                </div>
                <button
                    onClick={() => onToggleForceFeedback('joint', !ffJoint)}
                    disabled={ffLoading}
                    className={`relative w-11 h-6 rounded-full transition-colors ${
                        ffJoint ? 'bg-green-500' : 'bg-neutral-300 dark:bg-zinc-600'
                    } ${ffLoading ? 'opacity-50' : ''}`}
                >
                    <span className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                        ffJoint ? 'translate-x-5' : 'translate-x-0'
                    }`} />
                </button>
            </div>
        </div>
    );
}

export type { HapticsTabProps };
