"use client";

import React, { useState, useEffect, useRef } from 'react';
import { X, Minimize2, Maximize2, Cpu, RefreshCw, ChevronDown } from 'lucide-react';
import { motion, useDragControls } from 'framer-motion';
import { useResizable } from '../hooks/useResizable';

const API_BASE = typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : 'http://127.0.0.1:8000';

interface MotorData {
    name: string;
    id: number;
    model: string;
    position: number | null;
    velocity: number | null;
    current: number | null;
    temperature: number | null;
    voltage: number | null;
    load: number | null;
    error: number;
    error_names: string[];
}

interface Arm {
    id: string;
    name: string;
    role: string;
    motor_type: string;
    status: string;
}

interface MotorMonitorModalProps {
    isOpen: boolean;
    onClose: () => void;
    maximizedWindow: string | null;
    setMaximizedWindow: (window: string | null) => void;
}

export default function MotorMonitorModal({ isOpen, onClose, maximizedWindow, setMaximizedWindow }: MotorMonitorModalProps) {
    const dragControls = useDragControls();
    const { size, handleResizeMouseDown } = useResizable({ initialSize: { width: 620, height: 420 }, minSize: { width: 480, height: 280 } });

    const [isMinimized, setIsMinimized] = useState(false);
    const isMaximized = maximizedWindow === 'motor-monitor';

    const [arms, setArms] = useState<Arm[]>([]);
    const [selectedArm, setSelectedArm] = useState<string>('');
    const [motors, setMotors] = useState<MotorData[]>([]);
    const [motorType, setMotorType] = useState<string>('');
    const [polling, setPolling] = useState(true);
    const [lastUpdate, setLastUpdate] = useState<number>(0);
    const pollRef = useRef<NodeJS.Timeout | null>(null);
    const [dropdownOpen, setDropdownOpen] = useState(false);

    // Fetch arms list
    useEffect(() => {
        if (!isOpen) return;
        const fetchArms = async () => {
            try {
                const res = await fetch(`${API_BASE}/arms`);
                const data = await res.json();
                const armsList: Arm[] = (data.arms || []).filter((a: Arm) => a.status === 'connected');
                setArms(armsList);
                if (armsList.length > 0 && !selectedArm) {
                    setSelectedArm(armsList[0].id);
                }
            } catch { /* ignore */ }
        };
        fetchArms();
        const interval = setInterval(fetchArms, 5000);
        return () => clearInterval(interval);
    }, [isOpen]);

    // Poll motor diagnostics
    useEffect(() => {
        if (!isOpen || !selectedArm || !polling) {
            if (pollRef.current) clearInterval(pollRef.current);
            return;
        }

        const fetchDiagnostics = async () => {
            try {
                const res = await fetch(`${API_BASE}/arms/${selectedArm}/motors/diagnostics`);
                if (!res.ok) return;
                const data = await res.json();
                setMotors(data.motors || []);
                setMotorType(data.motor_type || '');
                setLastUpdate(Date.now());
            } catch { /* ignore */ }
        };

        fetchDiagnostics();
        pollRef.current = setInterval(fetchDiagnostics, 500);
        return () => { if (pollRef.current) clearInterval(pollRef.current); };
    }, [isOpen, selectedArm, polling]);

    // Clear motors when arm changes
    useEffect(() => { setMotors([]); }, [selectedArm]);

    if (!isOpen) return null;

    const getStatusColor = (m: MotorData) => {
        if (m.error > 0) return 'bg-red-500';
        if (m.temperature !== null && m.temperature > 55) return 'bg-amber-500';
        if (m.temperature !== null && m.temperature > 45) return 'bg-yellow-400';
        return 'bg-emerald-500';
    };

    const getTempClass = (temp: number | null) => {
        if (temp === null) return 'text-neutral-400 dark:text-zinc-500';
        if (temp > 55) return 'text-red-500 font-semibold';
        if (temp > 45) return 'text-amber-500';
        return 'text-neutral-700 dark:text-zinc-300';
    };

    const selectedArmData = arms.find(a => a.id === selectedArm);

    return (
        <motion.div
            drag
            dragControls={dragControls}
            dragListener={false}
            dragMomentum={false}
            initial={{ x: 60, y: 120, opacity: 0, scale: 0.95 }}
            animate={{
                opacity: 1,
                scale: 1,
                x: isMaximized ? 20 : undefined,
                y: isMaximized ? 80 : undefined,
            }}
            style={{
                width: isMaximized ? 'calc(100vw - 40px)' : isMinimized ? 'auto' : `${size.width}px`,
                height: isMaximized ? 'calc(100vh - 100px)' : isMinimized ? 'auto' : `${size.height}px`,
                zIndex: isMaximized ? 100 : 45,
            }}
            className="fixed flex flex-col overflow-hidden font-sans shadow-2xl rounded-2xl bg-white/95 dark:bg-zinc-900/95 border border-white/50 dark:border-zinc-700/50 backdrop-blur-3xl transition-all duration-300"
        >
            {/* Title Bar */}
            <div
                onPointerDown={(e) => dragControls.start(e)}
                className="h-11 bg-white/40 dark:bg-zinc-800/40 border-b border-black/5 dark:border-white/5 flex items-center justify-between px-4 cursor-grab active:cursor-grabbing select-none shrink-0"
            >
                <div className="flex items-center gap-2">
                    <div className="flex gap-1.5 mr-3">
                        <button onClick={onClose} className="w-3 h-3 rounded-full bg-[#FF5F57] hover:brightness-90 transition-all" />
                        <button onClick={() => { setIsMinimized(!isMinimized); if (isMaximized) setMaximizedWindow(null); }} className="w-3 h-3 rounded-full bg-[#FEBC2E] hover:brightness-90 transition-all" />
                        <button onClick={() => { setMaximizedWindow(isMaximized ? null : 'motor-monitor'); if (isMinimized) setIsMinimized(false); }} className="w-3 h-3 rounded-full bg-[#28C840] hover:brightness-90 transition-all" />
                    </div>
                </div>

                <span className="font-semibold text-neutral-800 dark:text-zinc-200 text-xs tracking-wide flex items-center gap-1.5">
                    <Cpu className="w-3.5 h-3.5 text-neutral-400 dark:text-zinc-500" />
                    Motor Monitor
                </span>

                <div className="flex items-center gap-2">
                    {lastUpdate > 0 && (
                        <div className={`w-1.5 h-1.5 rounded-full ${polling ? 'bg-emerald-500 animate-pulse' : 'bg-neutral-300 dark:bg-zinc-600'}`} />
                    )}
                    <button
                        onClick={() => setPolling(!polling)}
                        className={`p-1 rounded-md transition-colors ${polling ? 'text-emerald-600 dark:text-emerald-400' : 'text-neutral-400 dark:text-zinc-500 hover:text-neutral-600 dark:hover:text-zinc-300'}`}
                        title={polling ? 'Pause polling' : 'Resume polling'}
                    >
                        <RefreshCw className={`w-3 h-3 ${polling ? 'animate-spin' : ''}`} style={polling ? { animationDuration: '2s' } : {}} />
                    </button>
                </div>
            </div>

            {/* Content */}
            {!isMinimized && (
                <div className="flex-1 flex flex-col overflow-hidden">
                    {/* Arm Selector */}
                    <div className="px-4 py-2.5 border-b border-black/5 dark:border-white/5 bg-white/20 dark:bg-zinc-800/20">
                        <div className="relative">
                            <button
                                onClick={() => setDropdownOpen(!dropdownOpen)}
                                className="w-full flex items-center justify-between px-3 py-1.5 bg-white/60 dark:bg-zinc-800/60 border border-neutral-200/60 dark:border-zinc-700/60 rounded-lg text-xs transition-colors hover:border-neutral-300 dark:hover:border-zinc-600"
                            >
                                <span className="text-neutral-800 dark:text-zinc-200">
                                    {selectedArmData ? (
                                        <span className="flex items-center gap-2">
                                            <span className={`w-1.5 h-1.5 rounded-full ${selectedArmData.status === 'connected' ? 'bg-emerald-500' : 'bg-neutral-300'}`} />
                                            {selectedArmData.name}
                                            <span className="text-neutral-400 dark:text-zinc-500">{selectedArmData.motor_type}</span>
                                        </span>
                                    ) : (
                                        <span className="text-neutral-400 dark:text-zinc-500">No connected arms</span>
                                    )}
                                </span>
                                <ChevronDown className={`w-3 h-3 text-neutral-400 transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} />
                            </button>
                            {dropdownOpen && (
                                <>
                                    <div className="fixed inset-0 z-10" onClick={() => setDropdownOpen(false)} />
                                    <div className="absolute top-full left-0 right-0 mt-1 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-lg shadow-xl z-20 overflow-hidden">
                                        {arms.length === 0 && (
                                            <div className="px-3 py-2 text-xs text-neutral-400 dark:text-zinc-500">No connected arms</div>
                                        )}
                                        {arms.map(arm => (
                                            <button
                                                key={arm.id}
                                                onClick={() => { setSelectedArm(arm.id); setDropdownOpen(false); }}
                                                className={`w-full px-3 py-2 text-xs text-left flex items-center gap-2 transition-colors ${
                                                    selectedArm === arm.id
                                                        ? 'bg-black dark:bg-white text-white dark:text-black'
                                                        : 'text-neutral-700 dark:text-zinc-300 hover:bg-neutral-50 dark:hover:bg-zinc-700'
                                                }`}
                                            >
                                                <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${arm.status === 'connected' ? 'bg-emerald-500' : 'bg-neutral-300'}`} />
                                                <span className="truncate">{arm.name}</span>
                                                <span className="ml-auto text-[10px] opacity-60">{arm.motor_type}</span>
                                            </button>
                                        ))}
                                    </div>
                                </>
                            )}
                        </div>
                    </div>

                    {/* Motor Table */}
                    <div className="flex-1 overflow-auto">
                        {motors.length === 0 ? (
                            <div className="flex items-center justify-center h-full text-neutral-400 dark:text-zinc-500 text-xs">
                                {selectedArm ? 'Reading motors...' : 'Select a connected arm'}
                            </div>
                        ) : (
                            <table className="w-full">
                                <thead>
                                    <tr className="border-b border-black/5 dark:border-white/5 sticky top-0 bg-white/90 dark:bg-zinc-900/90 backdrop-blur-sm">
                                        <th className="text-[10px] uppercase tracking-wider text-neutral-400 dark:text-zinc-500 font-medium text-left pl-4 pr-2 py-2"></th>
                                        <th className="text-[10px] uppercase tracking-wider text-neutral-400 dark:text-zinc-500 font-medium text-left px-2 py-2">Motor</th>
                                        <th className="text-[10px] uppercase tracking-wider text-neutral-400 dark:text-zinc-500 font-medium text-right px-2 py-2">ID</th>
                                        <th className="text-[10px] uppercase tracking-wider text-neutral-400 dark:text-zinc-500 font-medium text-right px-2 py-2">Position</th>
                                        <th className="text-[10px] uppercase tracking-wider text-neutral-400 dark:text-zinc-500 font-medium text-right px-2 py-2">Vel</th>
                                        <th className="text-[10px] uppercase tracking-wider text-neutral-400 dark:text-zinc-500 font-medium text-right px-2 py-2">Temp</th>
                                        <th className="text-[10px] uppercase tracking-wider text-neutral-400 dark:text-zinc-500 font-medium text-right px-2 py-2">Volt</th>
                                        <th className="text-[10px] uppercase tracking-wider text-neutral-400 dark:text-zinc-500 font-medium text-right px-2 py-2">mA</th>
                                        <th className="text-[10px] uppercase tracking-wider text-neutral-400 dark:text-zinc-500 font-medium text-right px-2 py-2 pr-4">Err</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {motors.map((m) => (
                                        <tr
                                            key={m.name}
                                            className={`border-b border-black/[0.03] dark:border-white/[0.03] transition-colors hover:bg-black/[0.02] dark:hover:bg-white/[0.02] ${
                                                m.error > 0 ? 'bg-red-50/50 dark:bg-red-950/20' : ''
                                            }`}
                                        >
                                            <td className="pl-4 pr-2 py-1.5">
                                                <div className={`w-1.5 h-1.5 rounded-full ${getStatusColor(m)}`} />
                                            </td>
                                            <td className="px-2 py-1.5 text-xs text-neutral-800 dark:text-zinc-200 font-medium">{m.name}</td>
                                            <td className="px-2 py-1.5 text-xs font-mono tabular-nums text-neutral-500 dark:text-zinc-400 text-right">{m.id}</td>
                                            <td className="px-2 py-1.5 text-xs font-mono tabular-nums text-neutral-700 dark:text-zinc-300 text-right">{m.position ?? '—'}</td>
                                            <td className="px-2 py-1.5 text-xs font-mono tabular-nums text-neutral-500 dark:text-zinc-400 text-right">{m.velocity ?? '—'}</td>
                                            <td className={`px-2 py-1.5 text-xs font-mono tabular-nums text-right ${getTempClass(m.temperature)}`}>
                                                {m.temperature !== null ? `${m.temperature}°` : '—'}
                                            </td>
                                            <td className="px-2 py-1.5 text-xs font-mono tabular-nums text-neutral-500 dark:text-zinc-400 text-right">
                                                {m.voltage !== null ? `${m.voltage}` : '—'}
                                            </td>
                                            <td className="px-2 py-1.5 text-xs font-mono tabular-nums text-neutral-500 dark:text-zinc-400 text-right">
                                                {m.current ?? m.load ?? '—'}
                                            </td>
                                            <td className="px-2 py-1.5 pr-4 text-right">
                                                {m.error > 0 ? (
                                                    <span
                                                        className="inline-block px-1.5 py-0.5 bg-red-100 dark:bg-red-900/50 text-red-600 dark:text-red-400 rounded text-[10px] font-semibold cursor-default"
                                                        title={m.error_names.join(', ')}
                                                    >
                                                        {m.error_names[0] || `0x${m.error.toString(16)}`}
                                                    </span>
                                                ) : (
                                                    <span className="text-xs text-neutral-300 dark:text-zinc-600">—</span>
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        )}
                    </div>

                    {/* Footer */}
                    <div className="px-4 py-1.5 border-t border-black/5 dark:border-white/5 bg-white/20 dark:bg-zinc-800/20 flex items-center justify-between shrink-0">
                        <span className="text-[10px] text-neutral-400 dark:text-zinc-500">
                            {motors.length} motors · {motorType || '—'}
                        </span>
                        <span className="text-[10px] text-neutral-400 dark:text-zinc-500">
                            {polling ? '500ms' : 'paused'}
                        </span>
                    </div>
                </div>
            )}

            {/* Resize Handle */}
            {!isMinimized && !isMaximized && (
                <div
                    onMouseDown={handleResizeMouseDown}
                    className="absolute bottom-0 right-0 w-6 h-6 cursor-nwse-resize flex items-end justify-end p-1 z-50 group pointer-events-auto"
                >
                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className="text-neutral-300 dark:text-zinc-600 group-hover:text-blue-500 transition-colors">
                        <path d="M11 1L11 11L1 11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                        <path d="M8 4L8 8L4 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity="0.5" />
                    </svg>
                </div>
            )}
        </motion.div>
    );
}
