import React, { useState, useEffect } from 'react';
import { X, RefreshCw, Camera, Check, Settings, Image as ImageIcon, ExternalLink, Trash2, ArrowRight, Plus, Pencil, Layers } from 'lucide-react';

interface CameraModalProps {
    isOpen: boolean;
    onClose: () => void;
}

interface CameraConfig {
    id: string; // "phone_wrist" etc
    video_device_id: number | string; // /dev/videoX or index
    width: number;
    height: number;
    fps: number;
    type?: string; // "opencv" or "intelrealsense"
    use_depth?: boolean; // Only for RealSense cameras
}

interface CameraDevice {
    id: number | string;
    name: string;
    path: string;
    type: 'opencv' | 'intelrealsense'; // Camera type
}

export default function CameraModal({ isOpen, onClose }: CameraModalProps) {
    const [activeTab, setActiveTab] = useState<'assign' | 'preview'>('assign');
    const [availableCameras, setAvailableCameras] = useState<CameraDevice[]>([]);
    const [configs, setConfigs] = useState<CameraConfig[]>([]);
    const [isScanning, setIsScanning] = useState(false);
    const [scanError, setScanError] = useState('');
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    const [roles, setRoles] = useState<string[]>(['camera_1', 'camera_2', 'camera_3']);
    const [editingRole, setEditingRole] = useState<string | null>(null);
    const [tempRoleName, setTempRoleName] = useState('');

    useEffect(() => {
        if (isOpen) {
            loadConfigs();
            scanCameras();
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

    const assignCamera = (deviceId: number | string, cameraName: string, deviceType: 'opencv' | 'intelrealsense') => {
        const newConfigs = [...configs];
        const existingIndex = newConfigs.findIndex(c => c.id === cameraName);

        const newEntry: CameraConfig = {
            id: cameraName,
            video_device_id: deviceId,
            width: 1280,
            height: 720,
            fps: 30,
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
            <div className="bg-white/90 dark:bg-zinc-900/90 backdrop-blur-2xl border border-white/60 dark:border-zinc-700/60 rounded-3xl w-[900px] h-[650px] flex flex-col shadow-2xl overflow-hidden relative animate-in zoom-in-95 duration-200">

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
                <div className="flex-1 overflow-hidden relative">

                    {activeTab === 'assign' && (
                        <div className="h-full flex flex-col p-8 overflow-y-auto bg-gradient-to-b from-white/40 dark:from-zinc-800/40 to-white/10 dark:to-zinc-900/10">
                            <div className="flex justify-between items-center mb-6">
                                <div>
                                    <h3 className="text-lg font-medium text-black dark:text-white">Device Mapping</h3>
                                    <p className="text-sm text-neutral-500 dark:text-zinc-400">Map physical USB devices to logical roles.</p>
                                </div>
                                <button
                                    onClick={scanCameras}
                                    disabled={isScanning}
                                    className="px-4 py-2 bg-white dark:bg-zinc-800 border border-black/10 dark:border-white/10 rounded-full text-sm font-medium text-black dark:text-white shadow-sm hover:shadow-md transition-all flex items-center gap-2 hover:bg-neutral-50 dark:hover:bg-zinc-700"
                                >
                                    <RefreshCw className={`w-4 h-4 ${isScanning ? 'animate-spin' : ''}`} />
                                    {isScanning ? 'Scanning...' : 'Rescan Devices'}
                                </button>
                            </div>

                            <div className="grid grid-cols-2 gap-8">
                                {/* Left Column: Configured Roles */}
                                <div className="space-y-4">
                                    <div className="flex justify-between items-center mb-2">
                                        <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 dark:text-zinc-500">Logical Roles</h4>
                                        <button
                                            onClick={addRole}
                                            className="text-xs flex items-center gap-1 text-black dark:text-white bg-black/5 dark:bg-white/5 hover:bg-black/10 dark:hover:bg-white/10 px-2 py-1 rounded transition-colors"
                                        >
                                            <Plus className="w-3 h-3" /> Add Role
                                        </button>
                                    </div>

                                    {roles.map(roleId => {
                                        const assigned = configs.find(c => c.id === roleId);
                                        const isEditing = editingRole === roleId;

                                        return (
                                            <div key={roleId} className="bg-white/60 dark:bg-zinc-800/60 p-4 rounded-xl border border-white/50 dark:border-zinc-700/50 shadow-sm flex flex-col gap-3 group">
                                                <div className="flex justify-between items-center">
                                                    <div className="flex items-center gap-2 flex-1">
                                                        <div className={`w-2 h-2 rounded-full ${assigned ? 'bg-green-500' : 'bg-orange-300'}`} />

                                                        {isEditing ? (
                                                            <input
                                                                autoFocus
                                                                value={tempRoleName}
                                                                onChange={(e) => setTempRoleName(e.target.value)}
                                                                onBlur={() => saveRoleName(roleId)}
                                                                onKeyDown={(e) => e.key === 'Enter' && saveRoleName(roleId)}
                                                                className="flex-1 bg-white dark:bg-zinc-900 border border-blue-500 rounded px-1 py-0.5 text-sm outline-none text-black dark:text-white"
                                                            />
                                                        ) : (
                                                            <div className="flex items-center gap-2 group-hover:bg-white/50 dark:group-hover:bg-zinc-700/50 rounded px-1 -ml-1 transition-colors cursor-pointer" onClick={() => startEditing(roleId)}>
                                                                <span className="font-medium text-black dark:text-white">{roleId}</span>
                                                                <Pencil className="w-3 h-3 text-neutral-300 dark:text-zinc-600 opacity-0 group-hover:opacity-100" />
                                                            </div>
                                                        )}
                                                    </div>

                                                    <div className="flex items-center gap-1">
                                                        {assigned && (
                                                            <button
                                                                onClick={() => removeAssignment(roleId)}
                                                                className="p-1.5 rounded hover:bg-black/5 dark:hover:bg-white/5 text-neutral-300 dark:text-zinc-600 hover:text-orange-500 transition-colors"
                                                                title="Unassign Camera"
                                                            >
                                                                <RefreshCw className="w-3 h-3" />
                                                            </button>
                                                        )}
                                                        <button
                                                            onClick={() => deleteRole(roleId)}
                                                            className="p-1.5 rounded hover:bg-black/5 dark:hover:bg-white/5 text-neutral-300 dark:text-zinc-600 hover:text-red-500 transition-colors"
                                                            title="Delete Role"
                                                        >
                                                            <Trash2 className="w-3 h-3" />
                                                        </button>
                                                    </div>
                                                </div>

                                                {assigned ? (
                                                    <div className="space-y-2">
                                                        <div className="text-xs text-neutral-500 dark:text-zinc-400 bg-white/50 dark:bg-zinc-900/50 p-2 rounded-lg border border-black/5 dark:border-white/5 flex items-center justify-between">
                                                            <span className="truncate max-w-[150px]">Assigned to <strong>{typeof assigned.video_device_id === 'number' ? `/dev/video${assigned.video_device_id}` : assigned.video_device_id}</strong></span>
                                                            <Check className="w-3 h-3 text-green-500" />
                                                        </div>
                                                        {/* Depth toggle - only for RealSense cameras */}
                                                        {assigned.type === 'intelrealsense' && (
                                                            <button
                                                                onClick={() => toggleDepth(assigned.id)}
                                                                className={`w-full text-xs p-2 rounded-lg border flex items-center justify-between transition-all ${
                                                                    assigned.use_depth
                                                                        ? 'bg-blue-50 dark:bg-blue-950/50 border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300'
                                                                        : 'bg-white/50 dark:bg-zinc-900/50 border-black/5 dark:border-white/5 text-neutral-500 dark:text-zinc-400 hover:bg-neutral-50 dark:hover:bg-zinc-800'
                                                                }`}
                                                            >
                                                                <span className="flex items-center gap-2">
                                                                    <Layers className="w-3 h-3" />
                                                                    Record Depth Data
                                                                </span>
                                                                <div className={`w-8 h-4 rounded-full transition-colors ${assigned.use_depth ? 'bg-blue-500' : 'bg-neutral-300 dark:bg-zinc-600'}`}>
                                                                    <div className={`w-3 h-3 bg-white rounded-full m-0.5 transition-transform ${assigned.use_depth ? 'translate-x-4' : ''}`} />
                                                                </div>
                                                            </button>
                                                        )}
                                                    </div>
                                                ) : (
                                                    <div className="text-xs text-neutral-400 dark:text-zinc-500 italic">No device assigned</div>
                                                )}

                                                {!assigned && (
                                                    <div className="mt-1 pt-3 border-t border-black/5 dark:border-white/5">
                                                        <p className="text-[10px] text-neutral-400 dark:text-zinc-500 mb-2 uppercase">Available Devices:</p>
                                                        <div className="flex flex-wrap gap-2">
                                                            {(availableCameras || []).map((cam, idx) => (
                                                                <button
                                                                    key={`${cam.id}-${idx}`}
                                                                    onClick={() => assignCamera(cam.id, roleId, cam.type)}
                                                                    className="px-2 py-1 bg-black/5 dark:bg-white/5 hover:bg-black/10 dark:hover:bg-white/10 rounded text-xs text-black dark:text-white transition-colors"
                                                                >
                                                                    {typeof cam.id === 'string' && cam.id.includes('video') ? cam.id : (cam.name.length < 15 ? cam.name : cam.id)}
                                                                    {cam.type === 'intelrealsense' && <span className="ml-1 text-blue-500">(D)</span>}
                                                                </button>
                                                            ))}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        )
                                    })}
                                </div>

                                {/* Right Column: Discovered Devices */}
                                <div>
                                    <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 dark:text-zinc-500 mb-4">Physical Devices (/dev/video*)</h4>
                                    {isScanning ? (
                                        <div className="flex items-center justify-center h-32 text-neutral-400 dark:text-zinc-500">
                                            Scanning...
                                        </div>
                                    ) : (
                                        <div className="space-y-3">
                                            {(availableCameras || []).map((cam, idx) => (
                                                <div key={`${cam.id}-${idx}`} className="p-3 bg-neutral-50/50 dark:bg-zinc-800/50 rounded-lg border border-neutral-100 dark:border-zinc-700 flex items-center justify-between">
                                                    <div className="flex items-center gap-3">
                                                        <div className="w-8 h-8 rounded bg-neutral-200 dark:bg-zinc-700 flex items-center justify-center">
                                                            <Camera className="w-4 h-4 text-neutral-500 dark:text-zinc-400" />
                                                        </div>
                                                        <div>
                                                            <p className="text-sm font-medium text-black dark:text-white">/dev/video{cam.id}</p>
                                                            <p className="text-xs text-neutral-500 dark:text-zinc-400 truncate max-w-[150px]">{cam.name}</p>
                                                        </div>
                                                    </div>
                                                    {/* Check if assigned */}
                                                    {configs.some(c => c.video_device_id === cam.id) ? (
                                                        <span className="text-[10px] font-bold text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-950/50 px-2 py-1 rounded-full border border-green-100 dark:border-green-900">Mapped</span>
                                                    ) : (
                                                        <span className="text-[10px] text-neutral-400 dark:text-zinc-500 border border-neutral-200 dark:border-zinc-700 px-2 py-1 rounded-full">Unused</span>
                                                    )}
                                                </div>
                                            ))}
                                            {(availableCameras || []).length === 0 && (
                                                <p className="text-sm text-neutral-400 dark:text-zinc-500 italic">No cameras found.</p>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {activeTab === 'preview' && (
                        <div className="h-full bg-black/90 dark:bg-black/95 p-8 flex flex-col overflow-y-auto">
                            <div className="grid grid-cols-2 gap-6">
                                {configs.map(conf => (
                                    <div key={conf.id} className="bg-neutral-900 rounded-2xl overflow-hidden border border-white/10 shadow-2xl relative group">
                                        <img
                                            src={`http://127.0.0.1:8000/video_feed/${conf.id}`}
                                            alt={conf.id}
                                            className="w-full aspect-video object-cover bg-neutral-800"
                                            onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
                                        />
                                        <div className="absolute top-4 left-4">
                                            <span className="px-3 py-1 bg-black/50 backdrop-blur-md rounded-full text-xs text-white border border-white/20 font-medium flex items-center gap-2">
                                                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                                                {conf.id}
                                            </span>
                                        </div>

                                        {/* Overlay Stats */}
                                        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-sm">
                                            <div className="text-center">
                                                <p className="text-white font-medium mb-2">{conf.width}x{conf.height} @ {conf.fps}fps</p>
                                                <p className="text-neutral-400 text-xs">/dev/video{conf.video_device_id}</p>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                                {configs.length === 0 && (
                                    <div className="col-span-2 flex flex-col items-center justify-center p-12 text-neutral-500 border-2 border-dashed border-neutral-800 rounded-3xl">
                                        <ImageIcon className="w-12 h-12 mb-4 opacity-20" />
                                        <p>No cameras configured.</p>
                                        <button onClick={() => setActiveTab('assign')} className="mt-4 text-blue-400 hover:text-blue-300 text-sm flex items-center gap-1">
                                            Go to Assignments <ArrowRight className="w-3 h-3" />
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
