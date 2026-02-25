import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Database, Brain } from 'lucide-react';
import DatasetsTab from './DatasetsTab';
import PoliciesTab from './PoliciesTab';
import { datasetsApi } from '../../../lib/api';
import type { DatasetInfo } from '../../../lib/api/types';

type TabType = "datasets" | "policies";

interface DatasetViewerModalProps {
    isOpen: boolean;
    onClose: () => void;
    maximizedWindow: string | null;
    setMaximizedWindow: (window: string | null) => void;
}

export default function DatasetViewerModal({ isOpen, onClose, maximizedWindow, setMaximizedWindow }: DatasetViewerModalProps) {
    const [activeTab, setActiveTab] = useState<TabType>("datasets");
    const [datasets, setDatasets] = useState<DatasetInfo[]>([]);

    const isMaximized = maximizedWindow === 'datasetViewer';

    const fetchDatasets = async () => {
        try {
            const data = await datasetsApi.list();
            setDatasets(data);
        } catch (e) {
            console.error("Failed to load datasets", e);
        }
    };

    useEffect(() => {
        if (isOpen) fetchDatasets();
    }, [isOpen]);

    if (!isOpen) return null;

    return (
        <AnimatePresence>
            <div className="fixed inset-0 flex items-center justify-center bg-black/20 backdrop-blur-sm" style={{ zIndex: isMaximized ? 100 : 50 }}>
                <motion.div
                    initial={{ opacity: 0, scale: 0.95, y: 20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: 20 }}
                    className={`bg-white/95 dark:bg-zinc-900/95 rounded-3xl border border-white/50 dark:border-zinc-700/50 shadow-2xl flex flex-col overflow-hidden backdrop-blur-3xl transition-all duration-300 ${isMaximized ? 'w-[calc(100vw-40px)] h-[calc(100vh-100px)]' : 'w-[90vw] h-[85vh]'}`}
                >
                    {/* Header */}
                    <div className="h-14 bg-white/50 dark:bg-zinc-800/50 border-b border-neutral-200/50 dark:border-zinc-700/50 flex items-center justify-between px-6">
                        <div className="flex items-center gap-3">
                            <div className="flex gap-2 mr-3">
                                <button onClick={onClose} className="w-3.5 h-3.5 rounded-full bg-[#FF5F57] hover:brightness-90 transition-all" />
                                <button className="w-3.5 h-3.5 rounded-full bg-[#FEBC2E] hover:brightness-90 transition-all" />
                                <button onClick={() => setMaximizedWindow(isMaximized ? null : 'datasetViewer')} className="w-3.5 h-3.5 rounded-full bg-[#28C840] hover:brightness-90 transition-all" />
                            </div>
                            {/* Tab Switcher */}
                            <div className="flex items-center gap-1 bg-neutral-100 dark:bg-zinc-800 rounded-xl p-1">
                                <button
                                    onClick={() => setActiveTab("datasets")}
                                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                                        activeTab === "datasets"
                                            ? "bg-white dark:bg-zinc-700 text-emerald-600 dark:text-emerald-400 shadow-sm"
                                            : "text-neutral-500 dark:text-zinc-400 hover:text-neutral-700 dark:hover:text-zinc-300"
                                    }`}
                                >
                                    <Database className="w-4 h-4" />
                                    Datasets
                                </button>
                                <button
                                    onClick={() => setActiveTab("policies")}
                                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                                        activeTab === "policies"
                                            ? "bg-white dark:bg-zinc-700 text-purple-600 dark:text-purple-400 shadow-sm"
                                            : "text-neutral-500 dark:text-zinc-400 hover:text-neutral-700 dark:hover:text-zinc-300"
                                    }`}
                                >
                                    <Brain className="w-4 h-4" />
                                    Policies
                                </button>
                            </div>
                        </div>
                        <button onClick={onClose} className="p-2 hover:bg-neutral-100 dark:hover:bg-zinc-800 rounded-xl transition-colors">
                            <X className="w-5 h-5 text-neutral-400 dark:text-zinc-500" />
                        </button>
                    </div>

                    {/* Content */}
                    <div className="flex flex-1 overflow-hidden">
                        {activeTab === "datasets" ? (
                            <DatasetsTab datasets={datasets} fetchDatasets={fetchDatasets} />
                        ) : (
                            <PoliciesTab isOpen={isOpen && activeTab === "policies"} />
                        )}
                    </div>
                </motion.div>
            </div>
        </AnimatePresence>
    );
}
