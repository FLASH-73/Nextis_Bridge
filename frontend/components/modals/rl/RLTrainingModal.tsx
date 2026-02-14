import React, { useState } from 'react';
import { ChevronRight, Activity } from 'lucide-react';
import { motion, useDragControls } from 'framer-motion';
import { useResizable } from '../../../hooks/useResizable';
import RLSetupView from './RLSetupView';
import RLTrainingView from './RLTrainingView';

interface RLTrainingModalProps {
    isOpen: boolean;
    onClose: () => void;
    maximizedWindow: string | null;
    setMaximizedWindow: (window: string | null) => void;
}

export default function RLTrainingModal({ isOpen, onClose, maximizedWindow, setMaximizedWindow }: RLTrainingModalProps) {
    const dragControls = useDragControls();
    const { size, handleResizeMouseDown } = useResizable({ initialSize: { width: 850, height: 700 } });
    const [isMinimized, setIsMinimized] = useState(false);
    const isMaximized = maximizedWindow === 'rl-training';

    // View state
    const [view, setView] = useState<'setup' | 'training' | 'complete'>('setup');

    // Shared state between setup and training views
    const [rewardSource, setRewardSource] = useState<'sarm' | 'gvl' | 'classifier'>('sarm');

    const handleStartTraining = (config: any) => {
        // Capture the reward source from the config for the training view
        if (config.reward_source) {
            setRewardSource(config.reward_source);
        }
        setView('training');
    };

    const handleComplete = () => {
        setView('complete');
    };

    const handleReset = () => {
        setView('setup');
    };

    if (!isOpen) return null;

    return (
        <motion.div
            drag
            dragControls={dragControls}
            dragListener={false}
            dragMomentum={false}
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            style={{
                width: isMaximized ? 'calc(100vw - 40px)' : isMinimized ? 'auto' : `${size.width}px`,
                height: isMaximized ? 'calc(100vh - 100px)' : isMinimized ? 'auto' : `${size.height}px`,
                top: isMaximized ? '80px' : undefined,
                left: isMaximized ? '20px' : undefined,
                transform: isMaximized ? 'none' : undefined,
                zIndex: isMaximized ? 100 : 50,
            }}
            className={`fixed flex flex-col overflow-hidden font-sans glass-panel shadow-2xl rounded-3xl bg-white/90 dark:bg-zinc-900/90 border border-white/50 dark:border-zinc-700/50 backdrop-blur-3xl transition-all duration-300 ${isMaximized ? '' : 'top-20 left-1/2 -translate-x-1/2'}`}
        >
            {/* Header */}
            <div
                onPointerDown={(e) => dragControls.start(e)}
                className="h-16 bg-white/30 dark:bg-zinc-800/30 border-b border-black/5 dark:border-white/5 flex items-center justify-between px-6 cursor-grab active:cursor-grabbing select-none flex-none"
            >
                <div className="flex items-center gap-4">
                    <div className="flex gap-2 mr-2">
                        <button onClick={onClose} className="w-3.5 h-3.5 rounded-full bg-[#FF5F57] hover:brightness-90 btn-control transition-all" />
                        <button onClick={() => { setIsMinimized(!isMinimized); if (isMaximized) setMaximizedWindow(null); }} className="w-3.5 h-3.5 rounded-full bg-[#FEBC2E] hover:brightness-90 btn-control transition-all" />
                        <button onClick={() => { setMaximizedWindow(isMaximized ? null : 'rl-training'); if (isMinimized) setIsMinimized(false); }} className="w-3.5 h-3.5 rounded-full bg-[#28C840] hover:brightness-90 btn-control transition-all" />
                    </div>
                    <span className="font-semibold text-neutral-800 dark:text-zinc-200 text-lg tracking-tight flex items-center gap-2">
                        <Activity className="w-5 h-5 text-orange-500" /> RL Training (HIL-SERL)
                    </span>
                </div>

                <div className="flex items-center gap-2 text-xs">
                    {['setup', 'training', 'complete'].map((s, i) => (
                        <React.Fragment key={s}>
                            <div className={`px-3 py-1 rounded-full font-medium ${view === s ? 'bg-orange-100 dark:bg-orange-900/50 text-orange-700 dark:text-orange-300' : 'text-neutral-400 dark:text-zinc-500'}`}>
                                {s.charAt(0).toUpperCase() + s.slice(1)}
                            </div>
                            {i < 2 && <ChevronRight className="w-3 h-3 text-neutral-300 dark:text-zinc-600" />}
                        </React.Fragment>
                    ))}
                </div>
            </div>

            {!isMinimized && (
                <div className="flex-1 flex flex-col p-6 overflow-auto">
                    {view === 'setup' && (
                        <RLSetupView onStartTraining={handleStartTraining} />
                    )}

                    {(view === 'training' || view === 'complete') && (
                        <RLTrainingView
                            rewardSource={rewardSource}
                            onComplete={handleComplete}
                            onReset={handleReset}
                        />
                    )}
                </div>
            )}
        </motion.div>
    );
}
