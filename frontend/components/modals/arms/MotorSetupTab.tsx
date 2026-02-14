"use client";

import React from 'react';
import { RefreshCw, AlertTriangle, Check, Zap } from 'lucide-react';
import type { Port } from '@/lib/api/types';

interface MotorSetupState {
    port: string;
    motor_type: string;
    current_id: number;
    new_id: number;
}

interface MotorInfoEntry {
    model_number: number;
    model_name: string;
    baudrate: number;
    has_error?: boolean;
    error_status?: number;
    error_names?: string[];
}

interface RecoveryLogEntry {
    step: number;
    action: string;
    status: string;
    detail?: string;
}

interface RecoveryResultData {
    recovered: boolean;
    message: string;
    motor?: { id: number; model_name: string; baudrate: number };
}

interface MotorSetupTabProps {
    motorSetup: MotorSetupState;
    setMotorSetup: (setup: MotorSetupState) => void;
    ports: Port[];
    loading: boolean;
    foundMotorIds: number[];
    motorInfo: Record<number, MotorInfoEntry>;
    motorSetupStatus: string | null;
    scanningMotors: boolean;
    recovering: boolean;
    recoveryLog: RecoveryLogEntry[];
    recoveryResult: RecoveryResultData | null;
    onScanPorts: () => void;
    onScanMotors: () => void;
    onSetMotorId: () => void;
    onRecoverMotor: () => void;
}

export default function MotorSetupTab({
    motorSetup,
    setMotorSetup,
    ports,
    loading,
    foundMotorIds,
    motorInfo,
    motorSetupStatus,
    scanningMotors,
    recovering,
    recoveryLog,
    recoveryResult,
    onScanPorts,
    onScanMotors,
    onSetMotorId,
    onRecoverMotor,
}: MotorSetupTabProps) {
    return (
        <div className="p-6">
            <div className="max-w-lg mx-auto">
                {/* Warning Banner */}
                <div className="mb-6 p-4 bg-orange-50 dark:bg-orange-950/50 border border-orange-200 dark:border-orange-800 rounded-xl">
                    <div className="flex items-center gap-2 text-orange-600 dark:text-orange-400 font-medium text-sm mb-2">
                        <AlertTriangle className="w-4 h-4" />
                        Important: Connect ONE Motor at a Time
                    </div>
                    <p className="text-orange-700/70 dark:text-orange-300/70 text-xs leading-relaxed">
                        Motor ID configuration requires connecting motors one at a time.
                        Disconnect all other motors before scanning or setting IDs.
                        New motors typically have ID=1 by default.
                    </p>
                </div>

                {/* Motor Recovery Section */}
                <details className="mb-6">
                    <summary className="cursor-pointer text-sm font-medium text-red-600 dark:text-red-400 flex items-center gap-2 p-3 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded-xl hover:bg-red-100 dark:hover:bg-red-950/50 transition-colors">
                        <Zap className="w-4 h-4" />
                        Recover Unresponsive Motor
                    </summary>
                    <div className="mt-2 p-4 bg-red-50/50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-xl">
                        <p className="text-xs text-red-600/80 dark:text-red-400/80 mb-3">
                            Attempts to recover a motor stuck in hardware error state or not responding to normal commands.
                            Tries reboot, then factory reset across all baud rates. Select port and motor type above first.
                        </p>
                        <button
                            onClick={onRecoverMotor}
                            disabled={!motorSetup.port || recovering}
                            className="w-full py-2.5 bg-red-500 text-white rounded-lg font-medium disabled:opacity-50 hover:bg-red-600 transition-colors flex items-center justify-center gap-2"
                        >
                            <Zap className={`w-4 h-4 ${recovering ? 'animate-pulse' : ''}`} />
                            {recovering ? 'Recovery in progress...' : 'Attempt Recovery'}
                        </button>

                        {/* Recovery Log */}
                        {recoveryLog.length > 0 && (
                            <div className="mt-3 space-y-1.5 max-h-48 overflow-y-auto">
                                {recoveryLog.map((entry, i) => (
                                    <div key={i} className="flex items-start gap-2 text-xs">
                                        <span className={`mt-0.5 w-2 h-2 rounded-full shrink-0 ${
                                            entry.status === 'success' ? 'bg-green-500' :
                                            entry.status === 'failed' ? 'bg-red-500' :
                                            entry.status === 'warning' ? 'bg-yellow-500' :
                                            'bg-blue-500 animate-pulse'
                                        }`} />
                                        <div>
                                            <span className="text-neutral-700 dark:text-zinc-300">{entry.action}</span>
                                            {entry.detail && (
                                                <p className={`mt-0.5 ${
                                                    entry.status === 'success' ? 'text-green-600 dark:text-green-400' :
                                                    entry.status === 'failed' ? 'text-red-600 dark:text-red-400' :
                                                    entry.status === 'warning' ? 'text-yellow-600 dark:text-yellow-400' :
                                                    'text-neutral-500 dark:text-zinc-500'
                                                }`}>{entry.detail}</p>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Recovery Result */}
                        {recoveryResult && (
                            <div className={`mt-3 p-3 rounded-lg ${
                                recoveryResult.recovered
                                    ? 'bg-green-50 dark:bg-green-950/50 border border-green-200 dark:border-green-800'
                                    : 'bg-red-100 dark:bg-red-950/50 border border-red-300 dark:border-red-700'
                            }`}>
                                <div className="flex items-center gap-2">
                                    {recoveryResult.recovered ? (
                                        <Check className="w-4 h-4 text-green-600 dark:text-green-400" />
                                    ) : (
                                        <AlertTriangle className="w-4 h-4 text-red-600 dark:text-red-400" />
                                    )}
                                    <span className={`text-sm font-medium ${
                                        recoveryResult.recovered
                                            ? 'text-green-700 dark:text-green-300'
                                            : 'text-red-700 dark:text-red-300'
                                    }`}>
                                        {recoveryResult.message}
                                    </span>
                                </div>
                                {!recoveryResult.recovered && (
                                    <p className="mt-2 text-xs text-red-600/70 dark:text-red-400/70">
                                        Try Dynamixel Wizard 2.0 for firmware-level recovery, or the motor may be physically damaged.
                                    </p>
                                )}
                            </div>
                        )}
                    </div>
                </details>

                {/* Port and Motor Type Selection */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">
                            Motor Type
                        </label>
                        <select
                            value={motorSetup.motor_type}
                            onChange={(e) => setMotorSetup({ ...motorSetup, motor_type: e.target.value })}
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        >
                            <option value="dynamixel_xl330">Dynamixel XL330</option>
                            <option value="dynamixel_xl430">Dynamixel XL430</option>
                            <option value="sts3215">STS3215 (Feetech)</option>
                            <option value="damiao">Damiao (CAN)</option>
                        </select>
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">
                            Serial Port
                            <button
                                onClick={onScanPorts}
                                disabled={loading}
                                className="ml-2 text-blue-500 hover:text-blue-600"
                            >
                                {loading ? 'Scanning...' : 'Refresh'}
                            </button>
                        </label>
                        <select
                            value={motorSetup.port}
                            onChange={(e) => setMotorSetup({ ...motorSetup, port: e.target.value })}
                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                        >
                            <option value="">Select port...</option>
                            {ports.map(p => (
                                <option key={p.device} value={p.device}>
                                    {p.device} - {p.description}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>

                {/* Scan for Motors */}
                <div className="mb-6 p-4 bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700">
                    <h4 className="text-sm font-medium text-black dark:text-white mb-3">Step 1: Scan for Connected Motor</h4>
                    <button
                        onClick={onScanMotors}
                        disabled={!motorSetup.port || scanningMotors}
                        className="w-full py-2.5 bg-blue-500 text-white rounded-lg font-medium disabled:opacity-50 hover:bg-blue-600 transition-colors flex items-center justify-center gap-2"
                    >
                        <RefreshCw className={`w-4 h-4 ${scanningMotors ? 'animate-spin' : ''}`} />
                        {scanningMotors ? 'Scanning all baud rates...' : 'Scan for Motors'}
                    </button>
                    {scanningMotors && (
                        <p className="mt-2 text-xs text-neutral-500 dark:text-zinc-500">
                            Scanning multiple baud rates - this may take 10-30 seconds
                        </p>
                    )}
                    {foundMotorIds.length > 0 && (
                        <div className="mt-3 p-3 bg-green-50 dark:bg-green-950/50 rounded-lg space-y-2">
                            {foundMotorIds.map(id => {
                                const info = motorInfo[id];
                                return (
                                    <div key={id} className={`p-2 rounded ${info?.has_error ? 'bg-red-100 dark:bg-red-950/50' : ''}`}>
                                        <div className="flex items-center justify-between text-sm">
                                            <span className={`font-medium ${info?.has_error ? 'text-red-700 dark:text-red-300' : 'text-green-700 dark:text-green-300'}`}>
                                                Motor ID: {id}
                                            </span>
                                            {info && (
                                                <span className={`text-xs ${info.has_error ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}`}>
                                                    {info.model_name} @ {info.baudrate} baud
                                                </span>
                                            )}
                                        </div>
                                        {info?.has_error && (
                                            <div className="mt-1 flex items-center gap-1 text-xs text-red-600 dark:text-red-400">
                                                <AlertTriangle className="w-3 h-3" />
                                                <span>Hardware error: {info.error_names?.join(', ') || 'Unknown'} - will attempt auto-fix</span>
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    )}
                    {motorSetupStatus && foundMotorIds.length === 0 && (
                        <p className="mt-2 text-sm text-neutral-600 dark:text-zinc-400">{motorSetupStatus}</p>
                    )}
                </div>

                {/* Set Motor ID */}
                <div className="p-4 bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700">
                    <h4 className="text-sm font-medium text-black dark:text-white mb-3">Step 2: Change Motor ID</h4>
                    <div className="grid grid-cols-2 gap-4 mb-4">
                        <div>
                            <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Current ID</label>
                            <input
                                type="number"
                                min="1"
                                max="253"
                                value={motorSetup.current_id}
                                onChange={(e) => setMotorSetup({ ...motorSetup, current_id: parseInt(e.target.value) || 1 })}
                                className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                            />
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">New ID</label>
                            <input
                                type="number"
                                min="1"
                                max="253"
                                value={motorSetup.new_id}
                                onChange={(e) => setMotorSetup({ ...motorSetup, new_id: parseInt(e.target.value) || 1 })}
                                className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                            />
                        </div>
                    </div>
                    <button
                        onClick={onSetMotorId}
                        disabled={!motorSetup.port || loading}
                        className="w-full py-2.5 bg-black dark:bg-white text-white dark:text-black rounded-lg font-medium disabled:opacity-50 hover:bg-neutral-800 dark:hover:bg-zinc-200 transition-colors"
                    >
                        Set Motor ID
                    </button>
                    {motorSetupStatus && motorSetupStatus.includes('changed') && (
                        <div className="mt-3 p-2 bg-green-50 dark:bg-green-950/50 rounded-lg">
                            <p className="text-green-600 dark:text-green-400 text-sm flex items-center gap-2">
                                <Check className="w-4 h-4" /> {motorSetupStatus}
                            </p>
                        </div>
                    )}
                </div>

                {/* 7-DOF Setup Guide */}
                <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-950/50 rounded-xl border border-blue-100 dark:border-blue-900">
                    <h4 className="text-sm font-medium text-blue-700 dark:text-blue-300 mb-2">7-DOF Arm Setup Guide</h4>
                    <ol className="text-xs text-blue-600/80 dark:text-blue-400/80 space-y-1 list-decimal list-inside">
                        <li>Connect only the <strong>base</strong> motor → Set ID to <strong>1</strong></li>
                        <li>Disconnect, connect <strong>shoulder</strong> motor → Set ID to <strong>2</strong></li>
                        <li>Continue for elbow (3), wrist1 (4), wrist2 (5), wrist3 (6)</li>
                        <li>Finally, connect <strong>gripper</strong> motor → Set ID to <strong>7</strong></li>
                    </ol>
                </div>
            </div>
        </div>
    );
}

export type { MotorSetupTabProps, MotorSetupState, MotorInfoEntry, RecoveryLogEntry, RecoveryResultData };
