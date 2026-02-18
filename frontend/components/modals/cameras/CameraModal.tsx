import React, { useState, useEffect } from 'react';
import { X, Camera } from 'lucide-react';
import CameraGrid from './CameraGrid';
import { usePolling } from '../../../hooks/usePolling';
import type { CameraConfig, CameraDevice } from './CameraGrid';
import type { CameraStatusEntry, CameraCapabilities } from '../../../lib/api/types';

interface CameraModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function CameraModal({ isOpen, onClose }: CameraModalProps) {
    const [activeTab, setActiveTab] = useState<'assign' | 'preview'>('assign');
    const [availableCameras, setAvailableCameras] = useState<CameraDevice[]>([]);
    const [configs, setConfigs] = useState<CameraConfig[]>([]);
    const [isScanning, setIsScanning] = useState(false);
    const [scanError, setScanError] = useState('');

    const [roles, setRoles] = useState<string[]>(['camera_1', 'camera_2', 'camera_3']);
    const [editingRole, setEditingRole] = useState<string | null>(null);
    const [tempRoleName, setTempRoleName] = useState('');

    // Camera connection status
    const [cameraStatus, setCameraStatus] = useState<Record<string, CameraStatusEntry>>({});

    // Resolution capabilities
    const [capabilitiesCache, setCapabilitiesCache] = useState<Record<string, CameraCapabilities>>({});
    const [loadingCapabilities, setLoadingCapabilities] = useState<Record<string, boolean>>({});
    const [selectedResolutions, setSelectedResolutions] = useState<Record<string, { width: number; height: number; fps: number }>>({});

    useEffect(() => {
        if (isOpen) {
            loadConfigs();
            scanCameras();
            fetchCameraStatus();
        }
    }, [isOpen]);

    // Sync roles with loaded configs, but preserve user-added roles that might not be assigned yet
    useEffect(() => {
        if (configs.length > 0) {
            const configIds = configs.map(c => c.id);
            setRoles(prev => {
                const unique = Array.from(new Set([...prev, ...configIds]));
                return unique;
            });
        }
    }, [configs]);

    const fetchCameraStatus = async () => {
        try {
            const res = await fetch('http://127.0.0.1:8000/cameras/status');
            if (res.ok) {
                const data = await res.json();
                setCameraStatus(prev => {
                    const merged = { ...prev };
                    for (const [key, val] of Object.entries(data)) {
                        const entry = val as CameraStatusEntry;
                        // Don't overwrite local 'connecting'/'disconnecting' states
                        if (merged[key]?.status === 'connecting' || merged[key]?.status === 'disconnecting') continue;
                        merged[key] = entry;
                    }
                    return merged;
                });
            }
        } catch {
            // Silently handle - backend may be reloading
        }
    };

    // Poll camera status when preview tab is active
    usePolling(fetchCameraStatus, 5000, isOpen && activeTab === 'preview');

    const connectCamera = async (cameraKey: string) => {
        setCameraStatus(prev => ({ ...prev, [cameraKey]: { status: 'connecting', error: '' } }));
        try {
            const res = await fetch(`http://127.0.0.1:8000/cameras/${cameraKey}/connect`, { method: 'POST' });
            const data = await res.json();
            if (data.status === 'connected') {
                setCameraStatus(prev => ({ ...prev, [cameraKey]: { status: 'connected', error: '' } }));
            } else {
                setCameraStatus(prev => ({ ...prev, [cameraKey]: { status: 'error', error: data.message || 'Unknown error' } }));
            }
        } catch (e: any) {
            setCameraStatus(prev => ({ ...prev, [cameraKey]: { status: 'error', error: e.message || 'Connection failed' } }));
        }
    };

    const disconnectCamera = async (cameraKey: string) => {
        setCameraStatus(prev => ({ ...prev, [cameraKey]: { status: 'disconnecting', error: '' } }));
        try {
            await fetch(`http://127.0.0.1:8000/cameras/${cameraKey}/disconnect`, { method: 'POST' });
            setCameraStatus(prev => ({ ...prev, [cameraKey]: { status: 'disconnected', error: '' } }));
        } catch {
            setCameraStatus(prev => ({ ...prev, [cameraKey]: { status: 'disconnected', error: '' } }));
        }
    };

    const fetchCapabilities = async (deviceType: string, deviceId: string | number) => {
        const key = `${deviceType}:${deviceId}`;
        if (capabilitiesCache[key]) return capabilitiesCache[key];

        setLoadingCapabilities(prev => ({ ...prev, [key]: true }));
        try {
            const res = await fetch(`http://127.0.0.1:8000/cameras/capabilities/${deviceType}/${deviceId}`);
            const data: CameraCapabilities = await res.json();
            setCapabilitiesCache(prev => ({ ...prev, [key]: data }));

            // Pre-select default: 1280x720@30 if available, else native, else first
            const resolutions = data.resolutions || [];
            const default720 = resolutions.find(r => r.width === 1280 && r.height === 720);
            const nativeRes = data.native
                ? resolutions.find(r => r.width === data.native!.width && r.height === data.native!.height)
                : null;
            const selected = default720 || nativeRes || resolutions[0];

            if (selected) {
                const fps = selected.fps.includes(30) ? 30 : selected.fps[0];
                setSelectedResolutions(prev => ({
                    ...prev,
                    [key]: { width: selected.width, height: selected.height, fps }
                }));
            }

            return data;
        } catch {
            return null;
        } finally {
            setLoadingCapabilities(prev => ({ ...prev, [key]: false }));
        }
    };

    const scanCameras = async () => {
        setIsScanning(true);
        setScanError('');
        try {
            const res = await fetch('http://127.0.0.1:8000/cameras/scan');
            if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
            }
            const data = await res.json();

            // Backend returns { opencv: [], realsense: [] }
            const opencv = Array.isArray(data.opencv) ? data.opencv : [];
            const realsense = Array.isArray(data.realsense) ? data.realsense : [];

            // Helper to get ID consistently
            const getId = (c: any) => c.id || c.index_or_path || c.serial_number_or_name;

            const mapped: CameraDevice[] = [
                ...opencv.map((c: any) => ({
                    id: getId(c),
                    name: c.name || `Camera ${getId(c)}`,
                    path: c.index_or_path,
                    type: 'opencv' as const
                })),
                ...realsense.map((c: any) => ({
                    id: getId(c),
                    name: c.name || `RealSense ${getId(c)}`,
                    path: c.serial_number_or_name,
                    type: 'intelrealsense' as const
                }))
            ];

            setAvailableCameras(mapped);
            setScanError('');
        } catch (e) {
            // Silently handle errors - don't clear available cameras on transient failures
            console.warn('Camera scan failed (backend may be reloading):', e);
            // Only show error if we have no cameras at all
            if (availableCameras.length === 0) {
                setScanError('Failed to scan for cameras. Click Rescan to retry.');
            }
        } finally {
            setIsScanning(false);
        }
    };

    const loadConfigs = async () => {
        try {
            const res = await fetch('http://127.0.0.1:8000/cameras/config');
            const data = await res.json();
            const loaded = Array.isArray(data) ? data : [];
            setConfigs(loaded);

            // Should we overwrite roles? Better to just ensure they exist.
            if (loaded.length > 0) {
                const loadedIds = loaded.map((c: any) => c.id);
                // Only add if not present (simple merge)
                setRoles(prev => Array.from(new Set([...prev, ...loadedIds])));
            }
        } catch (e) {
            console.error("Failed to load camera config");
        }
    };

    const saveConfig = async (newConfigs: CameraConfig[]) => {
        try {
            await fetch('http://127.0.0.1:8000/cameras/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newConfigs)
            });
            setConfigs(newConfigs);
        } catch (e) {
            console.error("Failed to save config");
        }
    };

    const addRole = () => {
        const base = "new_camera";
        let name = base;
        let i = 1;
        while (roles.includes(name)) {
            name = `${base}_${i}`;
            i++;
        }
        setRoles([...roles, name]);
    };

    const deleteRole = (roleToDelete: string) => {
        // 1. Remove from local roles
        setRoles(roles.filter(r => r !== roleToDelete));

        // 2. Remove assignment (and save to backend)
        const newConfigs = configs.filter(c => c.id !== roleToDelete);
        if (newConfigs.length !== configs.length) {
            saveConfig(newConfigs);
        }
    };

    const startEditing = (role: string) => {
        setEditingRole(role);
        setTempRoleName(role);
    };

    const saveRoleName = (oldName: string) => {
        if (!tempRoleName.trim() || tempRoleName === oldName) {
            setEditingRole(null);
            return;
        }

        if (roles.includes(tempRoleName)) {
            alert('Role name already exists!');
            return;
        }

        // 1. Update roles list
        setRoles(roles.map(r => r === oldName ? tempRoleName : r));

        // 2. Update config if assigned
        const configIndex = configs.findIndex(c => c.id === oldName);
        if (configIndex >= 0) {
            const newConfigs = [...configs];
            newConfigs[configIndex] = { ...newConfigs[configIndex], id: tempRoleName };
            saveConfig(newConfigs);
        }

        setEditingRole(null);
    };

    const assignCamera = (
        deviceId: number | string,
        cameraName: string,
        deviceType: 'opencv' | 'intelrealsense',
        resolution?: { width: number; height: number; fps: number }
    ) => {
        const key = `${deviceType}:${deviceId}`;
        const res = resolution || selectedResolutions[key] || { width: 1280, height: 720, fps: 30 };

        const newConfigs = [...configs];
        const existingIndex = newConfigs.findIndex(c => c.id === cameraName);

        const newEntry: CameraConfig = {
            id: cameraName,
            video_device_id: deviceId,
            width: res.width,
            height: res.height,
            fps: res.fps,
            type: deviceType,
            // Default use_depth to false for RealSense cameras (user can toggle it)
            ...(deviceType === 'intelrealsense' ? { use_depth: false } : {})
        };

        if (existingIndex >= 0) {
            newConfigs[existingIndex] = newEntry;
        } else {
            newConfigs.push(newEntry);
        }
        saveConfig(newConfigs);
    };

    // Toggle depth setting for a camera
    const toggleDepth = (cameraId: string) => {
        const newConfigs = configs.map(c => {
            if (c.id === cameraId && c.type === 'intelrealsense') {
                return { ...c, use_depth: !c.use_depth };
            }
            return c;
        });
        saveConfig(newConfigs);
    };

    const removeAssignment = (cameraName: string) => {
        const newConfigs = configs.filter(c => c.id !== cameraName);
        saveConfig(newConfigs);
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 flex items-center justify-center z-[60]">
            <div className="absolute inset-0 bg-white/30 dark:bg-black/30 backdrop-blur-sm" onClick={onClose} />
            <div className="bg-white/90 dark:bg-zinc-900/90 backdrop-blur-2xl border border-white/60 dark:border-zinc-700/60 rounded-3xl w-[90vw] max-w-[1100px] h-[85vh] max-h-[800px] flex flex-col shadow-2xl overflow-hidden relative animate-in zoom-in-95 duration-200">

                {/* Header */}
                <div className="flex items-center justify-between px-8 py-6 border-b border-black/5 dark:border-white/5 bg-white/40 dark:bg-zinc-800/40">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-black/5 dark:bg-white/5 rounded-lg">
                            <Camera className="w-5 h-5 text-black/70 dark:text-white/70" />
                        </div>
                        <h2 className="text-xl font-light tracking-tight text-black dark:text-white">Camera Configuration</h2>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-black/5 dark:hover:bg-white/5 rounded-full text-neutral-400 hover:text-black dark:hover:text-white transition-colors">
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Tabs */}
                <div className="flex px-8 border-b border-black/5 dark:border-white/5 bg-white/20 dark:bg-zinc-800/20">
                    <button
                        onClick={() => setActiveTab('assign')}
                        className={`py-4 px-2 mr-6 text-sm font-medium border-b-2 transition-all ${activeTab === 'assign' ? 'border-black dark:border-white text-black dark:text-white' : 'border-transparent text-neutral-400 dark:text-zinc-500 hover:text-black dark:hover:text-white'
                            }`}
                    >
                        Assignments
                    </button>
                    <button
                        onClick={() => setActiveTab('preview')}
                        className={`py-4 px-2 text-sm font-medium border-b-2 transition-all ${activeTab === 'preview' ? 'border-black dark:border-white text-black dark:text-white' : 'border-transparent text-neutral-400 dark:text-zinc-500 hover:text-black dark:hover:text-white'
                            }`}
                    >
                        Live Preview
                    </button>
                </div>

                {/* Content */}
                <CameraGrid
                    activeTab={activeTab}
                    setActiveTab={setActiveTab}
                    roles={roles}
                    configs={configs}
                    availableCameras={availableCameras}
                    isScanning={isScanning}
                    editingRole={editingRole}
                    tempRoleName={tempRoleName}
                    scanCameras={scanCameras}
                    addRole={addRole}
                    deleteRole={deleteRole}
                    startEditing={startEditing}
                    setTempRoleName={setTempRoleName}
                    saveRoleName={saveRoleName}
                    assignCamera={assignCamera}
                    removeAssignment={removeAssignment}
                    toggleDepth={toggleDepth}
                    cameraStatus={cameraStatus}
                    connectCamera={connectCamera}
                    disconnectCamera={disconnectCamera}
                    capabilitiesCache={capabilitiesCache}
                    loadingCapabilities={loadingCapabilities}
                    selectedResolutions={selectedResolutions}
                    fetchCapabilities={fetchCapabilities}
                    setSelectedResolutions={setSelectedResolutions}
                />
            </div>
        </div>
    );
}
