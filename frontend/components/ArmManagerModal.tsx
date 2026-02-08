"use client";

import React, { useState, useEffect } from 'react';
import { X, Plus, Link2, Unlink, RefreshCw, Power, PowerOff, Settings, AlertTriangle, Check, Trash2, Edit3, Save, Zap, Home } from 'lucide-react';

const API_BASE = "http://127.0.0.1:8000";

interface Arm {
    id: string;
    name: string;
    role: 'leader' | 'follower';
    motor_type: string;
    port: string;
    enabled: boolean;
    status: 'connected' | 'disconnected' | 'connecting' | 'error';
    calibrated: boolean;
    structural_design?: string;
}

interface Pairing {
    leader_id: string;
    follower_id: string;
    name: string;
}

interface Port {
    device: string;
    name: string;
    description: string;
    manufacturer?: string;
    in_use: boolean;
}

interface ArmManagerModalProps {
    isOpen: boolean;
    onClose: () => void;
}

type Tab = 'arms' | 'pairings' | 'add' | 'motor_setup';

const MOTOR_TYPE_COLORS: Record<string, string> = {
    sts3215: 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300',
    damiao: 'bg-orange-100 text-orange-700 dark:bg-orange-900/50 dark:text-orange-300',
    dynamixel_xl330: 'bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300',
    dynamixel_xl430: 'bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300',
};

const MOTOR_TYPE_LABELS: Record<string, string> = {
    sts3215: 'STS3215',
    damiao: 'Damiao',
    dynamixel_xl330: 'XL330',
    dynamixel_xl430: 'XL430',
};

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

    // Fetch data on open
    useEffect(() => {
        if (isOpen) {
            fetchArms();
            fetchPairings();
        }
    }, [isOpen]);

    const fetchArms = async () => {
        try {
            const res = await fetch(`${API_BASE}/arms`);
            const data = await res.json();
            setArms(data.arms || []);
        } catch (e) {
            console.error('Failed to fetch arms:', e);
        }
    };

    const fetchPairings = async () => {
        try {
            const res = await fetch(`${API_BASE}/arms/pairings`);
            const data = await res.json();
            setPairings(data.pairings || []);
        } catch (e) {
            console.error('Failed to fetch pairings:', e);
        }
    };

    const scanPorts = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/arms/scan-ports`);
            const data = await res.json();
            setPorts(data.ports || []);
        } catch (e) {
            console.error('Failed to scan ports:', e);
        } finally {
            setLoading(false);
        }
    };

    const connectArm = async (armId: string) => {
        try {
            const res = await fetch(`${API_BASE}/arms/${armId}/connect`, { method: 'POST' });
            const data = await res.json();
            if (!data.success) {
                setError(data.error);
            }
            fetchArms();
        } catch (e) {
            console.error('Failed to connect arm:', e);
        }
    };

    const disconnectArm = async (armId: string) => {
        try {
            await fetch(`${API_BASE}/arms/${armId}/disconnect`, { method: 'POST' });
            fetchArms();
        } catch (e) {
            console.error('Failed to disconnect arm:', e);
        }
    };

    const deleteArm = async (armId: string) => {
        if (!confirm(`Are you sure you want to remove "${armId}"?`)) return;
        try {
            await fetch(`${API_BASE}/arms/${armId}`, { method: 'DELETE' });
            fetchArms();
            fetchPairings();
        } catch (e) {
            console.error('Failed to delete arm:', e);
        }
    };

    const setHomePosition = async (armId: string) => {
        try {
            const res = await fetch(`${API_BASE}/arms/${armId}/set-home`, { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                const joints = Object.entries(data.home_position)
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
            const res = await fetch(`${API_BASE}/arms`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newArm),
            });
            const data = await res.json();
            if (!data.success) {
                setError(data.error);
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

    const updateArmName = async (armId: string) => {
        try {
            await fetch(`${API_BASE}/arms/${armId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: editName }),
            });
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
            const res = await fetch(`${API_BASE}/arms/pairings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newPairing),
            });
            const data = await res.json();
            if (!data.success) {
                setError(data.error);
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
            await fetch(`${API_BASE}/arms/pairings`, {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ leader_id: leaderId, follower_id: followerId }),
            });
            fetchPairings();
        } catch (e) {
            console.error('Failed to remove pairing:', e);
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'connected': return 'bg-green-500';
            case 'connecting': return 'bg-yellow-500 animate-pulse';
            case 'error': return 'bg-red-500';
            default: return 'bg-neutral-400 dark:bg-zinc-600';
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
            const res = await fetch(`${API_BASE}/motors/scan`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    port: motorSetup.port,
                    motor_type: motorSetup.motor_type,
                }),
            });
            const data = await res.json();
            if (data.success) {
                setFoundMotorIds(data.found_ids || []);
                setMotorInfo(data.motor_info || {});
                if (data.found_ids?.length === 0) {
                    setMotorSetupStatus('No motors found. Check connection.');
                } else if (data.found_ids?.length === 1) {
                    const id = data.found_ids[0];
                    const info = data.motor_info?.[id];
                    setMotorSetup(prev => ({ ...prev, current_id: id }));
                    if (info) {
                        setMotorSetupStatus(`Found: ID=${id}, Model=${info.model_name} at ${info.baudrate} baud`);
                    } else {
                        setMotorSetupStatus(`Found motor with ID ${id}`);
                    }
                } else {
                    setMotorSetupStatus(`Found ${data.found_ids.length} motors: ${data.found_ids.join(', ')}`);
                }
            } else {
                setError(data.error || 'Scan failed');
            }
        } catch (e) {
            setError('Failed to scan motors');
        } finally {
            setScanningMotors(false);
        }
    };

    const setMotorId = async () => {
        setLoading(true);
        setMotorSetupStatus(null);
        try {
            const res = await fetch(`${API_BASE}/motors/set-id`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    port: motorSetup.port,
                    motor_type: motorSetup.motor_type,
                    current_id: motorSetup.current_id,
                    new_id: motorSetup.new_id,
                }),
            });
            const data = await res.json();
            if (data.success) {
                setMotorSetupStatus(`Motor ID changed from ${motorSetup.current_id} to ${data.new_id}`);
                setFoundMotorIds([]);
            } else {
                setError(data.error || 'Failed to set motor ID');
            }
        } catch (e) {
            setError('Failed to set motor ID');
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
            const res = await fetch(`${API_BASE}/motors/recover`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    port: motorSetup.port,
                    motor_type: motorSetup.motor_type,
                }),
            });
            const data = await res.json();
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

    const leaders = arms.filter(a => a.role === 'leader');
    const followers = arms.filter(a => a.role === 'follower');

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
                    {/* Arms Tab */}
                    {activeTab === 'arms' && (
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
                                                onConnect={connectArm}
                                                onDisconnect={disconnectArm}
                                                onDelete={deleteArm}
                                                onEdit={(id) => { setEditingArm(id); setEditName(arm.name); }}
                                                onSetHome={setHomePosition}
                                                isEditing={editingArm === arm.id}
                                                editName={editName}
                                                setEditName={setEditName}
                                                onSaveEdit={updateArmName}
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
                                                onConnect={connectArm}
                                                onDisconnect={disconnectArm}
                                                onDelete={deleteArm}
                                                onEdit={(id) => { setEditingArm(id); setEditName(arm.name); }}
                                                onSetHome={setHomePosition}
                                                isEditing={editingArm === arm.id}
                                                editName={editName}
                                                setEditName={setEditName}
                                                onSaveEdit={updateArmName}
                                                onCancelEdit={() => setEditingArm(null)}
                                            />
                                        ))
                                    )}
                                </div>
                            </div>

                            {/* Refresh Button */}
                            <div className="flex justify-center pt-4">
                                <button
                                    onClick={fetchArms}
                                    className="flex items-center gap-2 px-4 py-2 text-sm text-neutral-600 dark:text-zinc-400 hover:text-black dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/5 rounded-lg transition-colors"
                                >
                                    <RefreshCw className="w-4 h-4" /> Refresh
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Pairings Tab */}
                    {activeTab === 'pairings' && (
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
                                                    onClick={() => removePairing(p.leader_id, p.follower_id)}
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
                                    onClick={createPairing}
                                    disabled={!newPairing.leader_id || !newPairing.follower_id || loading}
                                    className="w-full py-2 bg-black dark:bg-white text-white dark:text-black rounded-lg font-medium text-sm disabled:opacity-50 hover:bg-neutral-800 dark:hover:bg-zinc-200 transition-colors flex items-center justify-center gap-2"
                                >
                                    <Link2 className="w-4 h-4" /> Create Pairing
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Add Arm Tab */}
                    {activeTab === 'add' && (
                        <div className="space-y-6">
                            <div className="p-4 bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700">
                                <h4 className="font-medium text-black dark:text-white mb-4">Add New Arm</h4>

                                <div className="grid grid-cols-2 gap-4 mb-4">
                                    <div>
                                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Arm ID *</label>
                                        <input
                                            type="text"
                                            value={newArm.id}
                                            onChange={(e) => setNewArm({ ...newArm, id: e.target.value.replace(/\s+/g, '_').toLowerCase() })}
                                            placeholder="e.g., damiao_main"
                                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Display Name *</label>
                                        <input
                                            type="text"
                                            value={newArm.name}
                                            onChange={(e) => setNewArm({ ...newArm, name: e.target.value })}
                                            placeholder="e.g., Damiao Assembly Arm"
                                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                                        />
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-4 mb-4">
                                    <div>
                                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Role *</label>
                                        <select
                                            value={newArm.role}
                                            onChange={(e) => setNewArm({ ...newArm, role: e.target.value as 'leader' | 'follower' })}
                                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                                        >
                                            <option value="follower">Follower</option>
                                            <option value="leader">Leader</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Motor Type *</label>
                                        <select
                                            value={newArm.motor_type}
                                            onChange={(e) => setNewArm({ ...newArm, motor_type: e.target.value })}
                                            className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                                        >
                                            <option value="sts3215">STS3215 (Feetech)</option>
                                            <option value="damiao">Damiao (CAN)</option>
                                            <option value="dynamixel_xl330">Dynamixel XL330</option>
                                            <option value="dynamixel_xl430">Dynamixel XL430</option>
                                        </select>
                                    </div>
                                </div>

                                <div className="mb-4">
                                    <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">
                                        Serial Port *
                                        <button
                                            onClick={scanPorts}
                                            disabled={loading}
                                            className="ml-2 text-blue-500 hover:text-blue-600"
                                        >
                                            {loading ? 'Scanning...' : 'Scan Ports'}
                                        </button>
                                    </label>
                                    <select
                                        value={newArm.port}
                                        onChange={(e) => setNewArm({ ...newArm, port: e.target.value })}
                                        className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                                    >
                                        <option value="">Select port...</option>
                                        {ports.map(p => (
                                            <option key={p.device} value={p.device} disabled={p.in_use}>
                                                {p.device} - {p.description} {p.in_use ? '(in use)' : ''}
                                            </option>
                                        ))}
                                        <option value="custom">Enter manually...</option>
                                    </select>
                                    {newArm.port === 'custom' && (
                                        <input
                                            type="text"
                                            placeholder="/dev/ttyUSB0"
                                            onChange={(e) => setNewArm({ ...newArm, port: e.target.value })}
                                            className="w-full mt-2 px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                                        />
                                    )}
                                </div>

                                <div className="mb-4">
                                    <label className="block text-xs font-medium text-neutral-500 dark:text-zinc-400 mb-1">Structural Design (optional)</label>
                                    <select
                                        value={newArm.structural_design}
                                        onChange={(e) => setNewArm({ ...newArm, structural_design: e.target.value })}
                                        className="w-full px-3 py-2 bg-white dark:bg-zinc-900 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm"
                                    >
                                        <option value="">Select design...</option>
                                        <option value="umbra_7dof">Umbra 7-DOF</option>
                                        <option value="damiao_7dof">Damiao 7-DOF</option>
                                        <option value="custom">Custom</option>
                                    </select>
                                    <p className="text-[10px] text-neutral-400 dark:text-zinc-500 mt-1">
                                        Helps identify compatible leader-follower pairs
                                    </p>
                                </div>

                                <button
                                    onClick={addArm}
                                    disabled={!newArm.id || !newArm.name || !newArm.port || loading}
                                    className="w-full py-3 bg-black dark:bg-white text-white dark:text-black rounded-xl font-medium disabled:opacity-50 hover:bg-neutral-800 dark:hover:bg-zinc-200 transition-colors flex items-center justify-center gap-2"
                                >
                                    <Plus className="w-4 h-4" /> Add Arm
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Motor Setup Tab */}
                    {activeTab === 'motor_setup' && (
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
                                            onClick={recoverMotor}
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
                                                onClick={scanPorts}
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
                                        onClick={scanMotors}
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
                                        onClick={setMotorId}
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
                    )}
                </div>
            </div>
        </div>
    );
}

// Arm Card Component
interface ArmCardProps {
    arm: Arm;
    onConnect: (id: string) => void;
    onDisconnect: (id: string) => void;
    onDelete: (id: string) => void;
    onEdit: (id: string) => void;
    onSetHome: (id: string) => void;
    isEditing: boolean;
    editName: string;
    setEditName: (name: string) => void;
    onSaveEdit: (id: string) => void;
    onCancelEdit: () => void;
}

function ArmCard({
    arm,
    onConnect,
    onDisconnect,
    onDelete,
    onEdit,
    onSetHome,
    isEditing,
    editName,
    setEditName,
    onSaveEdit,
    onCancelEdit,
}: ArmCardProps) {
    const statusColor = {
        connected: 'bg-green-500',
        connecting: 'bg-yellow-500 animate-pulse',
        error: 'bg-red-500',
        disconnected: 'bg-neutral-400 dark:bg-zinc-600',
    }[arm.status];

    const motorTypeColor = MOTOR_TYPE_COLORS[arm.motor_type] || 'bg-neutral-100 text-neutral-700 dark:bg-zinc-800 dark:text-zinc-300';
    const motorTypeLabel = MOTOR_TYPE_LABELS[arm.motor_type] || arm.motor_type;

    return (
        <div className="flex items-center justify-between p-4 bg-white/60 dark:bg-zinc-800/60 rounded-xl border border-neutral-100 dark:border-zinc-700 hover:border-neutral-200 dark:hover:border-zinc-600 transition-colors group">
            <div className="flex items-center gap-3">
                <div className={`w-2.5 h-2.5 rounded-full ${statusColor}`} />
                <div>
                    {isEditing ? (
                        <div className="flex items-center gap-2">
                            <input
                                type="text"
                                value={editName}
                                onChange={(e) => setEditName(e.target.value)}
                                className="px-2 py-1 text-sm border border-neutral-200 dark:border-zinc-600 rounded bg-white dark:bg-zinc-900"
                                autoFocus
                            />
                            <button onClick={() => onSaveEdit(arm.id)} className="p-1 text-green-500 hover:bg-green-50 dark:hover:bg-green-950 rounded">
                                <Save className="w-4 h-4" />
                            </button>
                            <button onClick={onCancelEdit} className="p-1 text-neutral-400 hover:bg-neutral-100 dark:hover:bg-zinc-800 rounded">
                                <X className="w-4 h-4" />
                            </button>
                        </div>
                    ) : (
                        <div className="flex items-center gap-2">
                            <span className="font-medium text-black dark:text-white">{arm.name}</span>
                            <span className={`px-1.5 py-0.5 text-[10px] font-semibold rounded ${motorTypeColor}`}>
                                {motorTypeLabel}
                            </span>
                            {arm.calibrated && (
                                <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300 rounded">
                                    CALIBRATED
                                </span>
                            )}
                        </div>
                    )}
                    <div className="text-xs text-neutral-400 dark:text-zinc-500">
                        {arm.port}
                    </div>
                </div>
            </div>

            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                {arm.status === 'connected' && (
                    <button
                        onClick={() => onSetHome(arm.id)}
                        className="p-2 text-blue-500 hover:bg-blue-50 dark:hover:bg-blue-950 rounded-lg transition-colors"
                        title="Set current position as home"
                    >
                        <Home className="w-4 h-4" />
                    </button>
                )}
                {arm.status === 'connected' ? (
                    <button
                        onClick={() => onDisconnect(arm.id)}
                        className="p-2 text-orange-500 hover:bg-orange-50 dark:hover:bg-orange-950 rounded-lg transition-colors"
                        title="Disconnect"
                    >
                        <PowerOff className="w-4 h-4" />
                    </button>
                ) : (
                    <button
                        onClick={() => onConnect(arm.id)}
                        className="p-2 text-green-500 hover:bg-green-50 dark:hover:bg-green-950 rounded-lg transition-colors"
                        title="Connect"
                    >
                        <Power className="w-4 h-4" />
                    </button>
                )}
                <button
                    onClick={() => onEdit(arm.id)}
                    className="p-2 text-neutral-400 hover:text-black dark:hover:text-white hover:bg-neutral-100 dark:hover:bg-zinc-800 rounded-lg transition-colors"
                    title="Edit"
                >
                    <Edit3 className="w-4 h-4" />
                </button>
                <button
                    onClick={() => onDelete(arm.id)}
                    className="p-2 text-red-500 hover:bg-red-50 dark:hover:bg-red-950 rounded-lg transition-colors"
                    title="Delete"
                >
                    <Trash2 className="w-4 h-4" />
                </button>
            </div>
        </div>
    );
}
