"use client";

import React, { useState, useEffect } from 'react';
import { X, Settings, AlertTriangle } from 'lucide-react';
import { armsApi, motorsApi, teleopApi, toolsApi, triggersApi, toolPairingsApi, apiFetch } from '@/lib/api';
import type { Arm, Pairing, Port, Tool, Trigger, ToolPairing } from '@/lib/api/types';
import ArmsTab from './ArmsTab';
import PairingsTab from './PairingsTab';
import AddArmTab from './AddArmTab';
import MotorSetupTab from './MotorSetupTab';
import HapticsTab from './HapticsTab';
import ToolsTab from './ToolsTab';
import AddToolTab from './AddToolTab';
import ToolPairingsTab from './ToolPairingsTab';

interface ArmManagerModalProps {
    isOpen: boolean;
    onClose: () => void;
}

type Tab = 'arms' | 'pairings' | 'add' | 'motor_setup' | 'haptics' | 'tools' | 'add_tool' | 'tool_pairings';

export default function ArmManagerModal({ isOpen, onClose }: ArmManagerModalProps) {
    const [activeTab, setActiveTab] = useState<Tab>('arms');
    const [arms, setArms] = useState<Arm[]>([]);
    const [pairings, setPairings] = useState<Pairing[]>([]);
    const [ports, setPorts] = useState<Port[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Add arm form state
    const [newArm, setNewArm] = useState({
        id: '',
        name: '',
        role: 'follower' as 'leader' | 'follower',
        motor_type: 'sts3215',
        port: '',
        structural_design: '',
    });

    // Create pairing state
    const [newPairing, setNewPairing] = useState({
        leader_id: '',
        follower_id: '',
        name: '',
    });

    // Edit arm state
    const [editingArm, setEditingArm] = useState<string | null>(null);
    const [editName, setEditName] = useState('');
    const [editPort, setEditPort] = useState('');

    // Motor setup state
    const [motorSetup, setMotorSetup] = useState({
        port: '',
        motor_type: 'dynamixel_xl330',
        current_id: 1,
        new_id: 1,
    });
    const [foundMotorIds, setFoundMotorIds] = useState<number[]>([]);
    const [motorInfo, setMotorInfo] = useState<Record<number, {
        model_number: number;
        model_name: string;
        baudrate: number;
        has_error?: boolean;
        error_status?: number;
        error_names?: string[];
    }>>({});
    const [motorSetupStatus, setMotorSetupStatus] = useState<string | null>(null);
    const [scanningMotors, setScanningMotors] = useState(false);

    // Recovery state
    const [recovering, setRecovering] = useState(false);
    const [recoveryLog, setRecoveryLog] = useState<Array<{
        step: number;
        action: string;
        status: string;
        detail?: string;
    }>>([]);
    const [recoveryResult, setRecoveryResult] = useState<{
        recovered: boolean;
        message: string;
        motor?: { id: number; model_name: string; baudrate: number };
    } | null>(null);

    // Force feedback (haptics) state
    const [ffGripper, setFfGripper] = useState(true);
    const [ffJoint, setFfJoint] = useState(true);
    const [ffLoading, setFfLoading] = useState(false);

    // Tool state
    const [tools, setTools] = useState<Tool[]>([]);
    const [triggers, setTriggers] = useState<Trigger[]>([]);
    const [toolPairings, setToolPairings] = useState<ToolPairing[]>([]);
    const [listenerRunning, setListenerRunning] = useState(false);
    const [newTool, setNewTool] = useState({
        id: '', name: '', tool_type: 'screwdriver', motor_type: 'sts3215', port: '', motor_id: 1,
    });
    const [newToolPairing, setNewToolPairing] = useState({
        trigger_id: '', tool_id: '', name: '', action: 'toggle',
    });

    // Fetch data on open
    useEffect(() => {
        if (isOpen) {
            fetchArms();
            fetchPairings();
            fetchTools();
            fetchTriggers();
            fetchToolPairings();
            fetchListenerStatus();
        }
    }, [isOpen]);

    // Fetch force feedback state when haptics tab is selected
    useEffect(() => {
        if (activeTab === 'haptics') {
            fetchForceFeedback();
        }
    }, [activeTab]);

    const fetchArms = async () => {
        try {
            const data = await armsApi.list();
            setArms(data.arms || []);
        } catch (e) {
            console.error('Failed to fetch arms:', e);
        }
    };

    const fetchForceFeedback = async () => {
        try {
            const data = await teleopApi.getForceFeedback();
            setFfGripper(data.gripper);
            setFfJoint(data.joint);
        } catch (e) {
            console.error('Failed to fetch force feedback state:', e);
        }
    };

    const toggleForceFeedback = async (key: 'gripper' | 'joint', value: boolean) => {
        setFfLoading(true);
        try {
            await teleopApi.setForceFeedback({ [key]: value });
            // Re-fetch to get the updated state
            const data = await teleopApi.getForceFeedback();
            setFfGripper(data.gripper);
            setFfJoint(data.joint);
        } catch (e: any) {
            setError(e.message || 'Failed to toggle force feedback');
        } finally {
            setFfLoading(false);
        }
    };

    const fetchPairings = async () => {
        try {
            const data = await armsApi.listPairings();
            setPairings(data.pairings || []);
        } catch (e) {
            console.error('Failed to fetch pairings:', e);
        }
    };

    const scanPorts = async () => {
        setLoading(true);
        try {
            const data = await armsApi.scanPorts();
            setPorts(data.ports || []);
        } catch (e) {
            console.error('Failed to scan ports:', e);
        } finally {
            setLoading(false);
        }
    };

    const connectArm = async (armId: string) => {
        try {
            const data = await armsApi.connect(armId);
            if (!data.success) {
                setError(data.error || null);
            }
            fetchArms();
        } catch (e) {
            console.error('Failed to connect arm:', e);
        }
    };

    const disconnectArm = async (armId: string) => {
        try {
            await armsApi.disconnect(armId);
            fetchArms();
        } catch (e) {
            console.error('Failed to disconnect arm:', e);
        }
    };

    const deleteArm = async (armId: string) => {
        if (!confirm(`Are you sure you want to remove "${armId}"?`)) return;
        try {
            await armsApi.delete(armId);
            fetchArms();
            fetchPairings();
        } catch (e) {
            console.error('Failed to delete arm:', e);
        }
    };

    const setHomePosition = async (armId: string) => {
        try {
            const data = await armsApi.setHome(armId);
            if (data.success) {
                const joints = Object.entries(data.home_position || {})
                    .map(([k, v]) => `${k}: ${(v as number).toFixed(2)}`)
                    .join(', ');
                alert(`Home position saved for ${armId}\n\n${joints}`);
            } else {
                alert(`Failed: ${data.error}`);
            }
        } catch (e) {
            alert(`Error setting home position: ${e}`);
        }
    };

    const addArm = async () => {
        if (!newArm.id || !newArm.name || !newArm.port) {
            setError('Please fill in all required fields');
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const data = await armsApi.create(newArm);
            if (!data.success) {
                setError(data.error || null);
            } else {
                setNewArm({ id: '', name: '', role: 'follower', motor_type: 'sts3215', port: '', structural_design: '' });
                setActiveTab('arms');
                fetchArms();
            }
        } catch (e) {
            setError('Failed to add arm');
        } finally {
            setLoading(false);
        }
    };

    const updateArm = async (armId: string) => {
        try {
            await armsApi.update(armId, { name: editName, port: editPort });
            setEditingArm(null);
            fetchArms();
        } catch (e) {
            console.error('Failed to update arm:', e);
        }
    };

    const createPairing = async () => {
        if (!newPairing.leader_id || !newPairing.follower_id) {
            setError('Please select a leader and follower');
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const data = await armsApi.createPairing(newPairing);
            if (!data.success) {
                setError(data.error || null);
            } else {
                if (data.warning) {
                    alert(`Warning: ${data.warning}`);
                }
                setNewPairing({ leader_id: '', follower_id: '', name: '' });
                fetchPairings();
            }
        } catch (e) {
            setError('Failed to create pairing');
        } finally {
            setLoading(false);
        }
    };

    const removePairing = async (leaderId: string, followerId: string) => {
        try {
            await armsApi.removePairing(leaderId, followerId);
            fetchPairings();
        } catch (e) {
            console.error('Failed to remove pairing:', e);
        }
    };

    // Tool functions
    const fetchTools = async () => {
        try {
            const data = await toolsApi.list();
            setTools(data || []);
        } catch (e) {
            console.error('Failed to fetch tools:', e);
        }
    };

    const fetchTriggers = async () => {
        try {
            const data = await triggersApi.list();
            setTriggers(data || []);
        } catch (e) {
            console.error('Failed to fetch triggers:', e);
        }
    };

    const fetchToolPairings = async () => {
        try {
            const data = await toolPairingsApi.list();
            setToolPairings(data || []);
        } catch (e) {
            console.error('Failed to fetch tool pairings:', e);
        }
    };

    const fetchListenerStatus = async () => {
        try {
            const data = await toolPairingsApi.listenerStatus();
            setListenerRunning(data.running);
        } catch (e) {
            console.error('Failed to fetch listener status:', e);
        }
    };

    const connectTool = async (toolId: string) => {
        try {
            const data = await toolsApi.connect(toolId);
            if (!data.success) {
                setError(data.error || null);
            }
            fetchTools();
        } catch (e) {
            console.error('Failed to connect tool:', e);
        }
    };

    const disconnectTool = async (toolId: string) => {
        try {
            await toolsApi.disconnect(toolId);
            fetchTools();
        } catch (e) {
            console.error('Failed to disconnect tool:', e);
        }
    };

    const activateTool = async (toolId: string) => {
        try {
            await toolsApi.activate(toolId);
        } catch (e) {
            console.error('Failed to activate tool:', e);
        }
    };

    const deactivateTool = async (toolId: string) => {
        try {
            await toolsApi.deactivate(toolId);
        } catch (e) {
            console.error('Failed to deactivate tool:', e);
        }
    };

    const updateToolConfig = async (toolId: string, config: Record<string, unknown>) => {
        try {
            const tool = tools.find(t => t.id === toolId);
            const merged = { ...tool?.config, ...config };
            await toolsApi.update(toolId, { config: merged });
            fetchTools();
        } catch (e: any) {
            setError(e.message || 'Failed to update tool config');
        }
    };

    const removeTool = async (toolId: string) => {
        if (!confirm(`Are you sure you want to remove tool "${toolId}"?`)) return;
        try {
            await toolsApi.remove(toolId);
            fetchTools();
            fetchToolPairings();
        } catch (e) {
            console.error('Failed to remove tool:', e);
        }
    };

    const addTool = async () => {
        if (!newTool.id || !newTool.name || !newTool.port) {
            setError('Please fill in all required fields');
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const data = await toolsApi.create(newTool);
            if (!data.success) {
                setError(data.error || null);
            } else {
                setNewTool({ id: '', name: '', tool_type: 'screwdriver', motor_type: 'sts3215', port: '', motor_id: 1 });
                setActiveTab('tools');
                fetchTools();
            }
        } catch (e) {
            setError('Failed to add tool');
        } finally {
            setLoading(false);
        }
    };

    const createToolPairing = async () => {
        if (!newToolPairing.trigger_id || !newToolPairing.tool_id) {
            setError('Please select a trigger and tool');
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const data = await toolPairingsApi.create(newToolPairing);
            if (!data.success) {
                setError(data.error || null);
            } else {
                setNewToolPairing({ trigger_id: '', tool_id: '', name: '', action: 'toggle' });
                fetchToolPairings();
            }
        } catch (e) {
            setError('Failed to create tool pairing');
        } finally {
            setLoading(false);
        }
    };

    const removeToolPairing = async (triggerId: string, toolId: string) => {
        try {
            await toolPairingsApi.remove(triggerId, toolId);
            fetchToolPairings();
        } catch (e) {
            console.error('Failed to remove tool pairing:', e);
        }
    };

    const startListener = async () => {
        try {
            const data = await toolPairingsApi.startListener();
            if (data.warning) {
                setError(data.warning);
            }
            fetchListenerStatus();
        } catch (e: any) {
            setError(e.message || 'Failed to start trigger listener');
        }
    };

    const stopListener = async () => {
        try {
            await toolPairingsApi.stopListener();
            fetchListenerStatus();
        } catch (e) {
            console.error('Failed to stop listener:', e);
        }
    };

    const addTrigger = async (trigger: {
        id: string; name: string; trigger_type: string;
        port: string; pin: number; active_low: boolean;
    }) => {
        setLoading(true);
        setError(null);
        try {
            const data = await triggersApi.create(trigger);
            if (!data.success) {
                setError(data.error || 'Failed to add trigger');
            } else {
                fetchTriggers();
            }
        } catch (e: any) {
            setError(e.message || 'Failed to add trigger');
        } finally {
            setLoading(false);
        }
    };

    // Motor setup functions
    const scanMotors = async () => {
        if (!motorSetup.port) {
            setError('Please select a port first');
            return;
        }
        setScanningMotors(true);
        setMotorSetupStatus(null);
        setFoundMotorIds([]);
        setMotorInfo({});
        try {
            const data = await motorsApi.scan(motorSetup.port, motorSetup.motor_type);
            const foundIds = data.found_ids || [];
            const info = (data.motor_info || {}) as Record<number, {
                model_number: number;
                model_name: string;
                baudrate: number;
                has_error?: boolean;
                error_status?: number;
                error_names?: string[];
            }>;
            setFoundMotorIds(foundIds);
            setMotorInfo(info);
            if (foundIds.length === 0) {
                setMotorSetupStatus('No motors found. Check connection.');
            } else if (foundIds.length === 1) {
                const id = foundIds[0];
                const motorEntry = info[id];
                setMotorSetup(prev => ({ ...prev, current_id: id }));
                if (motorEntry) {
                    setMotorSetupStatus(`Found: ID=${id}, Model=${motorEntry.model_name} at ${motorEntry.baudrate} baud`);
                } else {
                    setMotorSetupStatus(`Found motor with ID ${id}`);
                }
            } else {
                setMotorSetupStatus(`Found ${foundIds.length} motors: ${foundIds.join(', ')}`);
            }
        } catch (e: any) {
            setError(e.message || 'Failed to scan motors');
        } finally {
            setScanningMotors(false);
        }
    };

    const setMotorId = async () => {
        setLoading(true);
        setMotorSetupStatus(null);
        try {
            const data = await motorsApi.setId({
                port: motorSetup.port,
                motor_type: motorSetup.motor_type,
                current_id: motorSetup.current_id,
                new_id: motorSetup.new_id,
            });
            if (data.success) {
                setMotorSetupStatus(`Motor ID changed from ${motorSetup.current_id} to ${motorSetup.new_id}`);
                setFoundMotorIds([]);
            } else {
                setError(data.error || 'Failed to set motor ID');
            }
        } catch (e: any) {
            setError(e.message || 'Failed to set motor ID');
        } finally {
            setLoading(false);
        }
    };

    const recoverMotor = async () => {
        if (!motorSetup.port) {
            setError('Please select a port first');
            return;
        }
        setRecovering(true);
        setRecoveryLog([]);
        setRecoveryResult(null);
        try {
            const data = await apiFetch<{
                recovered: boolean;
                message: string;
                motor?: { id: number; model_name: string; baudrate: number };
                log: Array<{ step: number; action: string; status: string; detail?: string }>;
            }>('/motors/recover', {
                method: 'POST',
                body: JSON.stringify({
                    port: motorSetup.port,
                    motor_type: motorSetup.motor_type,
                }),
            });
            setRecoveryLog(data.log || []);
            setRecoveryResult({
                recovered: data.recovered || false,
                message: data.message || 'Unknown result',
                motor: data.motor || undefined,
            });
        } catch (e) {
            setRecoveryResult({
                recovered: false,
                message: 'Failed to connect to recovery endpoint',
            });
        } finally {
            setRecovering(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 flex items-center justify-center z-[60]">
            <div className="absolute inset-0 bg-black/30 backdrop-blur-sm" onClick={onClose} />
            <div className="bg-white/95 dark:bg-zinc-900/95 backdrop-blur-2xl border border-white/60 dark:border-zinc-700/60 rounded-3xl w-[800px] max-h-[85vh] flex flex-col shadow-2xl overflow-hidden relative animate-in zoom-in-95 duration-200">

                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-black/5 dark:border-white/5 bg-white/40 dark:bg-zinc-800/40">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-black/5 dark:bg-white/5 rounded-lg">
                            <Settings className="w-5 h-5 text-black/70 dark:text-white/70" />
                        </div>
                        <h2 className="text-xl font-light tracking-tight text-black dark:text-white">Arm Manager</h2>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-black/5 dark:hover:bg-white/5 rounded-full transition-colors">
                        <X className="w-5 h-5 text-neutral-400" />
                    </button>
                </div>

                {/* Tabs */}
                <div className="flex gap-1 px-6 py-3 border-b border-black/5 dark:border-white/5 bg-white/20 dark:bg-zinc-800/20">
                    {[
                        { id: 'arms', label: 'Arms' },
                        { id: 'pairings', label: 'Pairings' },
                        { id: 'add', label: 'Add Arm' },
                        { id: 'motor_setup', label: 'Motor Setup' },
                        { id: 'haptics', label: 'Haptics' },
                        { id: 'tools', label: 'Tools' },
                        { id: 'add_tool', label: 'Add Tool' },
                        { id: 'tool_pairings', label: 'Tool Pairings' },
                    ].map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id as Tab)}
                            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                                activeTab === tab.id
                                    ? 'bg-black dark:bg-white text-white dark:text-black'
                                    : 'text-neutral-600 dark:text-zinc-400 hover:bg-black/5 dark:hover:bg-white/5'
                            }`}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* Error Banner */}
                {error && (
                    <div className="mx-6 mt-4 p-3 bg-red-50 dark:bg-red-950/50 border border-red-200 dark:border-red-800 rounded-xl flex items-center gap-2 text-red-600 dark:text-red-400 text-sm">
                        <AlertTriangle className="w-4 h-4" />
                        {error}
                        <button onClick={() => setError(null)} className="ml-auto hover:text-red-800 dark:hover:text-red-300">
                            <X className="w-4 h-4" />
                        </button>
                    </div>
                )}

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6">
                    {activeTab === 'arms' && (
                        <ArmsTab
                            arms={arms}
                            editingArm={editingArm}
                            editName={editName}
                            editPort={editPort}
                            setEditingArm={setEditingArm}
                            setEditName={setEditName}
                            setEditPort={setEditPort}
                            onConnect={connectArm}
                            onDisconnect={disconnectArm}
                            onDelete={deleteArm}
                            onSetHome={setHomePosition}
                            onSaveEdit={updateArm}
                            onRefresh={fetchArms}
                        />
                    )}

                    {activeTab === 'pairings' && (
                        <PairingsTab
                            arms={arms}
                            pairings={pairings}
                            newPairing={newPairing}
                            setNewPairing={setNewPairing}
                            loading={loading}
                            onCreatePairing={createPairing}
                            onRemovePairing={removePairing}
                        />
                    )}

                    {activeTab === 'add' && (
                        <AddArmTab
                            newArm={newArm}
                            setNewArm={setNewArm}
                            ports={ports}
                            loading={loading}
                            onScanPorts={scanPorts}
                            onAddArm={addArm}
                        />
                    )}

                    {activeTab === 'motor_setup' && (
                        <MotorSetupTab
                            motorSetup={motorSetup}
                            setMotorSetup={setMotorSetup}
                            ports={ports}
                            loading={loading}
                            foundMotorIds={foundMotorIds}
                            motorInfo={motorInfo}
                            motorSetupStatus={motorSetupStatus}
                            scanningMotors={scanningMotors}
                            recovering={recovering}
                            recoveryLog={recoveryLog}
                            recoveryResult={recoveryResult}
                            onScanPorts={scanPorts}
                            onScanMotors={scanMotors}
                            onSetMotorId={setMotorId}
                            onRecoverMotor={recoverMotor}
                        />
                    )}

                    {activeTab === 'haptics' && (
                        <HapticsTab
                            ffGripper={ffGripper}
                            ffJoint={ffJoint}
                            ffLoading={ffLoading}
                            onToggleForceFeedback={toggleForceFeedback}
                        />
                    )}

                    {activeTab === 'tools' && (
                        <ToolsTab
                            tools={tools}
                            loading={loading}
                            onConnect={connectTool}
                            onDisconnect={disconnectTool}
                            onActivate={activateTool}
                            onDeactivate={deactivateTool}
                            onRemove={removeTool}
                            onRefresh={fetchTools}
                            onUpdateConfig={updateToolConfig}
                        />
                    )}

                    {activeTab === 'add_tool' && (
                        <AddToolTab
                            newTool={newTool}
                            setNewTool={setNewTool}
                            ports={ports}
                            loading={loading}
                            onScanPorts={scanPorts}
                            onAddTool={addTool}
                        />
                    )}

                    {activeTab === 'tool_pairings' && (
                        <ToolPairingsTab
                            tools={tools}
                            triggers={triggers}
                            toolPairings={toolPairings}
                            newToolPairing={newToolPairing}
                            setNewToolPairing={setNewToolPairing}
                            loading={loading}
                            listenerRunning={listenerRunning}
                            ports={ports}
                            onCreateToolPairing={createToolPairing}
                            onRemoveToolPairing={removeToolPairing}
                            onStartListener={startListener}
                            onStopListener={stopListener}
                            onAddTrigger={addTrigger}
                            onScanPorts={scanPorts}
                        />
                    )}
                </div>
            </div>
        </div>
    );
}
