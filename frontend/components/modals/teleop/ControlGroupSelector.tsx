import React from 'react';
import { AlertCircle, Settings, Link2 } from 'lucide-react';

interface Pairing {
    leader_id: string;
    follower_id: string;
    name: string;
}

interface ControlGroupSelectorProps {
    arms: any[];
    selectedArms: string[];
    setSelectedArms: React.Dispatch<React.SetStateAction<string[]>>;
    pairings: Pairing[];
    handleSettingsOpen: (armId: string) => void;
    handleReset: () => void;
}

export default function ControlGroupSelector({
    arms,
    selectedArms,
    setSelectedArms,
    pairings,
    handleSettingsOpen,
    handleReset,
}: ControlGroupSelectorProps) {
    // Get active pairings based on current selection
    const activePairings = pairings.filter(p =>
        selectedArms.includes(p.leader_id) && selectedArms.includes(p.follower_id)
    );

    return (
        <div className="flex flex-col gap-3">
            <div className="flex justify-between items-end px-1">
                <span className="text-xs font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-widest">Active Control Group</span>
                <div className="flex gap-3">
                    <button
                        onClick={() => setSelectedArms(arms.map(a => a.id))}
                        className="text-[10px] uppercase font-bold text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
                    >
                        Select All
                    </button>
                    <button
                        onClick={() => setSelectedArms([])}
                        className="text-[10px] uppercase font-bold text-neutral-400 dark:text-zinc-500 hover:text-neutral-600 dark:hover:text-zinc-400 transition-colors"
                    >
                        Clear
                    </button>
                </div>
            </div>

            <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-hide justify-center">
                {arms.length > 0 ? (
                    arms.map(arm => {
                        const isSelected = selectedArms.includes(arm.id);
                        return (
                            <div
                                key={arm.id}
                                onClick={() => {
                                    if (isSelected) {
                                        setSelectedArms(prev => prev.filter(id => id !== arm.id));
                                    } else {
                                        setSelectedArms(prev => [...prev, arm.id]);
                                    }
                                }}
                                className={`group relative flex items-center gap-3 px-4 py-3 rounded-xl border transition-all min-w-[180px] cursor-pointer select-none
                                    ${isSelected
                                        ? 'bg-neutral-900 dark:bg-zinc-100 border-neutral-900 dark:border-zinc-100 shadow-lg shadow-neutral-200 dark:shadow-zinc-800'
                                        : 'bg-white dark:bg-zinc-800 border-neutral-200 dark:border-zinc-700 hover:border-neutral-300 dark:hover:border-zinc-600 hover:shadow-sm'}`}
                            >
                                <div className={`w-2.5 h-2.5 rounded-full ring-2 transition-all
                                    ${arm.calibrated
                                        ? 'bg-green-500 ring-green-500/20'
                                        : 'bg-amber-500 ring-amber-500/20'}`}
                                />

                                <div className="flex flex-col">
                                    <span className={`font-semibold text-sm transition-colors ${isSelected ? 'text-white dark:text-zinc-900' : 'text-neutral-700 dark:text-zinc-300'}`}>
                                        {arm.name}
                                    </span>
                                    <span className={`text-[10px] uppercase tracking-wider font-medium transition-colors ${isSelected ? 'text-neutral-400 dark:text-zinc-500' : 'text-neutral-400 dark:text-zinc-500'}`}>
                                        {arm.calibrated ? 'Ready' : 'Uncalibrated'}
                                    </span>
                                </div>

                                {/* Config Button (stopPropagation) */}
                                <button
                                    onClick={(e) => { e.stopPropagation(); handleSettingsOpen(arm.id); }}
                                    className={`ml-auto p-1.5 rounded-lg transition-all
                                        ${isSelected ? 'text-neutral-500 dark:text-zinc-600 hover:text-white dark:hover:text-zinc-900 hover:bg-white/10 dark:hover:bg-zinc-900/10' : 'text-neutral-300 dark:text-zinc-600 hover:text-blue-500 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-950'}`}
                                    title="Configure Motors"
                                >
                                    <Settings className="w-4 h-4" />
                                </button>

                                {/* Selection Checkmark Indicator */}
                                {isSelected && (
                                    <div className="absolute top-2 right-2 w-2 h-2 bg-blue-500 rounded-full animate-in zoom-in" />
                                )}
                            </div>
                        );
                    })
                ) : (
                    <div className="flex-1 flex items-center justify-between bg-amber-50 dark:bg-amber-950/50 border border-amber-100 dark:border-amber-900 rounded-xl px-4 py-2">
                        <span className="text-amber-700 dark:text-amber-400 text-sm font-medium flex items-center gap-2">
                            <AlertCircle className="w-4 h-4" /> No Robots Found
                        </span>
                        <button
                            onClick={handleReset}
                            className="px-3 py-1 bg-white dark:bg-zinc-800 border border-amber-200 dark:border-amber-700 text-amber-800 dark:text-amber-400 rounded-lg text-xs font-bold hover:bg-amber-100 dark:hover:bg-amber-950 transition-colors shadow-sm"
                        >
                            Reconnect Hardware
                        </button>
                    </div>
                )}
            </div>

            {/* Active Pairings Display */}
            {activePairings.length > 0 && (
                <div className="mt-3 flex items-center justify-center gap-3 flex-wrap">
                    <span className="text-[10px] font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-widest">Active Pairings:</span>
                    {activePairings.map((p, i) => (
                        <div key={i} className="flex items-center gap-1.5 px-2.5 py-1 bg-blue-50 dark:bg-blue-950/50 border border-blue-100 dark:border-blue-900 rounded-full">
                            <span className="text-xs font-medium text-purple-600 dark:text-purple-400">
                                {arms.find(a => a.id === p.leader_id)?.name || p.leader_id}
                            </span>
                            <Link2 className="w-3 h-3 text-blue-400" />
                            <span className="text-xs font-medium text-blue-600 dark:text-blue-400">
                                {arms.find(a => a.id === p.follower_id)?.name || p.follower_id}
                            </span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
