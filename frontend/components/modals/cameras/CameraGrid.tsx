import React, { useState } from 'react';
import { RefreshCw, Camera, Check, Plus, Pencil, Layers, Power, Loader2, AlertCircle, ArrowRight, Trash2, Image as ImageIcon, ChevronDown } from 'lucide-react';
import CameraFeed from '../../../components/ui/CameraFeed';
import type { CameraCapabilities, CameraStatusEntry } from '../../../lib/api/types';

export type { CameraConnectionStatus, CameraStatusEntry } from '../../../lib/api/types';

// ── Shared types ──────────────────────────────────────────────────────

export interface CameraConfig {
    id: string; // "phone_wrist" etc
    video_device_id: number | string; // /dev/videoX or index
    width: number;
    height: number;
    fps: number;
    type?: string; // "opencv" or "intelrealsense"
    use_depth?: boolean; // Only for RealSense cameras
}

export interface CameraDevice {
    id: number | string;
    name: string;
    path: string;
    type: 'opencv' | 'intelrealsense'; // Camera type
}

// ── Props ─────────────────────────────────────────────────────────────

interface CameraGridProps {
    activeTab: 'assign' | 'preview';
    setActiveTab: (tab: 'assign' | 'preview') => void;

    // Assign-tab state
    roles: string[];
    configs: CameraConfig[];
    availableCameras: CameraDevice[];
    isScanning: boolean;
    editingRole: string | null;
    tempRoleName: string;

    // Assign-tab callbacks
    scanCameras: () => void;
    addRole: () => void;
    deleteRole: (roleToDelete: string) => void;
    startEditing: (role: string) => void;
    setTempRoleName: (name: string) => void;
    saveRoleName: (oldName: string) => void;
    assignCamera: (deviceId: number | string, cameraName: string, deviceType: 'opencv' | 'intelrealsense', resolution?: { width: number; height: number; fps: number }) => void;
    removeAssignment: (cameraName: string) => void;
    toggleDepth: (cameraId: string) => void;

    // Preview-tab state & callbacks
    cameraStatus: Record<string, CameraStatusEntry>;
    connectCamera: (cameraKey: string) => void;
    disconnectCamera: (cameraKey: string) => void;

    // Resolution capabilities
    capabilitiesCache: Record<string, CameraCapabilities>;
    loadingCapabilities: Record<string, boolean>;
    selectedResolutions: Record<string, { width: number; height: number; fps: number }>;
    fetchCapabilities: (deviceType: string, deviceId: string | number) => Promise<any>;
    setSelectedResolutions: React.Dispatch<React.SetStateAction<Record<string, { width: number; height: number; fps: number }>>>;
}

// ── Component ─────────────────────────────────────────────────────────

export default function CameraGrid({
    activeTab,
    setActiveTab,
    roles,
    configs,
    availableCameras,
    isScanning,
    editingRole,
    tempRoleName,
    scanCameras,
    addRole,
    deleteRole,
    startEditing,
    setTempRoleName,
    saveRoleName,
    assignCamera,
    removeAssignment,
    toggleDepth,
    cameraStatus,
    connectCamera,
    disconnectCamera,
    capabilitiesCache,
    loadingCapabilities,
    selectedResolutions,
    fetchCapabilities,
    setSelectedResolutions,
}: CameraGridProps) {

    // Track which device is being configured per role (for resolution selector)
    const [selectedDeviceForRole, setSelectedDeviceForRole] = useState<Record<string, { id: number | string; type: 'opencv' | 'intelrealsense' }>>({});

    const getStatusForCamera = (cameraId: string): CameraStatusEntry => {
        return cameraStatus[cameraId] || { status: 'disconnected', error: '' };
    };

    const handleDeviceClick = (cam: CameraDevice, roleId: string) => {
        const current = selectedDeviceForRole[roleId];
        if (current && current.id === cam.id) {
            // Toggle off
            setSelectedDeviceForRole(prev => {
                const next = { ...prev };
                delete next[roleId];
                return next;
            });
            return;
        }
        setSelectedDeviceForRole(prev => ({ ...prev, [roleId]: { id: cam.id, type: cam.type } }));
        fetchCapabilities(cam.type, cam.id);
    };

    const getCapabilitiesKey = (deviceType: string, deviceId: string | number) => `${deviceType}:${deviceId}`;

    return (
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

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
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
                                const selectedDev = selectedDeviceForRole[roleId];

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
                                                {/* Show configured resolution */}
                                                <div className="text-[10px] text-neutral-400 dark:text-zinc-500 px-2">
                                                    {assigned.width}x{assigned.height} @ {assigned.fps}fps
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
                                                            onClick={() => handleDeviceClick(cam, roleId)}
                                                            className={`px-2 py-1 rounded text-xs transition-colors ${
                                                                selectedDev?.id === cam.id
                                                                    ? 'bg-black/15 dark:bg-white/15 text-black dark:text-white ring-1 ring-black/20 dark:ring-white/20'
                                                                    : 'bg-black/5 dark:bg-white/5 hover:bg-black/10 dark:hover:bg-white/10 text-black dark:text-white'
                                                            }`}
                                                        >
                                                            {typeof cam.id === 'string' && cam.id.includes('video') ? cam.id : (cam.name.length < 15 ? cam.name : cam.id)}
                                                            {cam.type === 'intelrealsense' && <span className="ml-1 text-blue-500">(D)</span>}
                                                        </button>
                                                    ))}
                                                </div>

                                                {/* Resolution selector panel */}
                                                {selectedDev && (() => {
                                                    const capKey = getCapabilitiesKey(selectedDev.type, selectedDev.id);
                                                    const caps = capabilitiesCache[capKey];
                                                    const isLoading = loadingCapabilities[capKey];
                                                    const selectedRes = selectedResolutions[capKey];
                                                    const resolutions = caps?.resolutions || [];

                                                    if (isLoading) {
                                                        return (
                                                            <div className="mt-3 p-3 bg-white/80 dark:bg-zinc-800/80 rounded-lg border border-black/10 dark:border-white/10 flex items-center justify-center gap-2">
                                                                <Loader2 className="w-3 h-3 animate-spin text-neutral-400" />
                                                                <span className="text-xs text-neutral-400 dark:text-zinc-500">Detecting resolutions...</span>
                                                            </div>
                                                        );
                                                    }

                                                    if (caps && resolutions.length === 0) {
                                                        return (
                                                            <div className="mt-3 p-3 bg-white/80 dark:bg-zinc-800/80 rounded-lg border border-black/10 dark:border-white/10 space-y-2">
                                                                <p className="text-[10px] text-neutral-400 dark:text-zinc-500">Resolution detection unavailable. Using default 1280x720@30fps.</p>
                                                                <button
                                                                    onClick={() => assignCamera(selectedDev.id, roleId, selectedDev.type)}
                                                                    className="w-full text-xs py-1.5 bg-black dark:bg-white text-white dark:text-black rounded-lg font-medium hover:opacity-80 transition-opacity"
                                                                >
                                                                    Assign with Defaults
                                                                </button>
                                                            </div>
                                                        );
                                                    }

                                                    if (!caps) return null;

                                                    const currentRes = resolutions.find(
                                                        r => selectedRes && r.width === selectedRes.width && r.height === selectedRes.height
                                                    ) || resolutions[0];
                                                    const currentFps = selectedRes?.fps || (currentRes?.fps.includes(30) ? 30 : currentRes?.fps[0]) || 30;

                                                    return (
                                                        <div className="mt-3 p-3 bg-white/80 dark:bg-zinc-800/80 rounded-lg border border-black/10 dark:border-white/10 space-y-2.5">
                                                            <div className="flex gap-2">
                                                                <div className="flex-1">
                                                                    <label className="text-[10px] text-neutral-500 dark:text-zinc-400 block mb-1">Resolution</label>
                                                                    <div className="relative">
                                                                        <select
                                                                            value={`${currentRes?.width}x${currentRes?.height}`}
                                                                            onChange={(e) => {
                                                                                const [w, h] = e.target.value.split('x').map(Number);
                                                                                const matchingRes = resolutions.find(r => r.width === w && r.height === h);
                                                                                const fps = matchingRes?.fps.includes(currentFps) ? currentFps : (matchingRes?.fps[0] || 30);
                                                                                setSelectedResolutions(prev => ({
                                                                                    ...prev,
                                                                                    [capKey]: { width: w, height: h, fps }
                                                                                }));
                                                                            }}
                                                                            className="w-full text-xs bg-white dark:bg-zinc-900 border border-black/10 dark:border-white/10 rounded-lg px-2 py-1.5 text-black dark:text-white appearance-none pr-6 outline-none focus:ring-1 focus:ring-black/20 dark:focus:ring-white/20"
                                                                        >
                                                                            {resolutions.map(r => {
                                                                                const isNative = caps.native && r.width === caps.native.width && r.height === caps.native.height;
                                                                                const isRecommended = r.width === 1280 && r.height === 720;
                                                                                let suffix = '';
                                                                                if (isNative && isRecommended) suffix = ' (Native, Recommended)';
                                                                                else if (isNative) suffix = ' (Native)';
                                                                                else if (isRecommended) suffix = ' (Recommended)';
                                                                                return (
                                                                                    <option key={`${r.width}x${r.height}`} value={`${r.width}x${r.height}`}>
                                                                                        {r.width}x{r.height}{r.label ? ` ${r.label}` : ''}{suffix}
                                                                                    </option>
                                                                                );
                                                                            })}
                                                                        </select>
                                                                        <ChevronDown className="w-3 h-3 absolute right-2 top-1/2 -translate-y-1/2 text-neutral-400 pointer-events-none" />
                                                                    </div>
                                                                </div>
                                                                <div className="w-20">
                                                                    <label className="text-[10px] text-neutral-500 dark:text-zinc-400 block mb-1">FPS</label>
                                                                    <div className="relative">
                                                                        <select
                                                                            value={currentFps}
                                                                            onChange={(e) => {
                                                                                setSelectedResolutions(prev => ({
                                                                                    ...prev,
                                                                                    [capKey]: { ...prev[capKey], fps: Number(e.target.value) }
                                                                                }));
                                                                            }}
                                                                            className="w-full text-xs bg-white dark:bg-zinc-900 border border-black/10 dark:border-white/10 rounded-lg px-2 py-1.5 text-black dark:text-white appearance-none pr-6 outline-none focus:ring-1 focus:ring-black/20 dark:focus:ring-white/20"
                                                                        >
                                                                            {(currentRes?.fps || [30]).map(f => (
                                                                                <option key={f} value={f}>{f} fps</option>
                                                                            ))}
                                                                        </select>
                                                                        <ChevronDown className="w-3 h-3 absolute right-2 top-1/2 -translate-y-1/2 text-neutral-400 pointer-events-none" />
                                                                    </div>
                                                                </div>
                                                            </div>
                                                            <p className="text-[10px] text-neutral-400 dark:text-zinc-500 leading-relaxed">
                                                                1280x720 recommended for policy training. Higher resolutions increase storage requirements.
                                                            </p>
                                                            {caps.connected && (
                                                                <p className="text-[10px] text-blue-500 dark:text-blue-400">
                                                                    Camera is currently connected. Showing active resolution only.
                                                                </p>
                                                            )}
                                                            <button
                                                                onClick={() => {
                                                                    const res = selectedRes || { width: currentRes!.width, height: currentRes!.height, fps: currentFps };
                                                                    assignCamera(selectedDev.id, roleId, selectedDev.type, res);
                                                                    setSelectedDeviceForRole(prev => {
                                                                        const next = { ...prev };
                                                                        delete next[roleId];
                                                                        return next;
                                                                    });
                                                                }}
                                                                className="w-full text-xs py-1.5 bg-black dark:bg-white text-white dark:text-black rounded-lg font-medium hover:opacity-80 transition-opacity"
                                                            >
                                                                Assign at {currentRes?.width}x{currentRes?.height} @ {currentFps}fps
                                                            </button>
                                                        </div>
                                                    );
                                                })()}
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
                                                    <p className="text-sm font-medium text-black dark:text-white">{typeof cam.id === 'string' && cam.id.includes('video') ? cam.id : cam.name}</p>
                                                    <p className="text-xs text-neutral-500 dark:text-zinc-400 truncate max-w-[150px]">{cam.type === 'intelrealsense' ? `RealSense ${cam.id}` : cam.name}</p>
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
                    <div className="grid gap-6" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))' }}>
                        {configs.map(conf => {
                            const status = getStatusForCamera(conf.id);
                            const isConnected = status.status === 'connected';
                            const isLoading = status.status === 'connecting' || status.status === 'disconnecting';
                            const isError = status.status === 'error';

                            const actualW = status.actual_width;
                            const actualH = status.actual_height;
                            const actualFps = status.actual_fps;
                            const hasActual = actualW != null && actualH != null;
                            const resMatches = hasActual && actualW === conf.width && actualH === conf.height;

                            return (
                                <div key={conf.id} className="bg-neutral-900 rounded-2xl overflow-hidden border border-white/10 shadow-2xl relative group">
                                    {/* Stream or placeholder */}
                                    {isConnected ? (
                                        <div className="aspect-video">
                                            <CameraFeed
                                                cameraId={conf.id}
                                                maxStreamWidth={640}
                                                mode="contain"
                                                badge={conf.type === 'intelrealsense' ? 'RS' : undefined}
                                                className="rounded-none border-0"
                                            />
                                        </div>
                                    ) : (
                                        <div className="w-full aspect-video bg-neutral-800 flex items-center justify-center">
                                            {isLoading ? (
                                                <div className="flex flex-col items-center gap-3">
                                                    <Loader2 className="w-8 h-8 text-white/30 animate-spin" />
                                                    <p className="text-white/30 text-sm">
                                                        {status.status === 'connecting' ? 'Connecting...' : 'Disconnecting...'}
                                                    </p>
                                                </div>
                                            ) : isError ? (
                                                <div className="flex flex-col items-center gap-3 px-6 text-center">
                                                    <AlertCircle className="w-8 h-8 text-red-400/60" />
                                                    <p className="text-red-400/80 text-sm">Connection Failed</p>
                                                    <p className="text-white/20 text-xs max-w-[200px] truncate">{status.error}</p>
                                                </div>
                                            ) : (
                                                <div className="flex flex-col items-center gap-3">
                                                    <Camera className="w-8 h-8 text-white/15" />
                                                    <p className="text-white/20 text-sm">Disconnected</p>
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {/* Resolution badges (bottom-left) */}
                                    {isConnected && (
                                        <div className="absolute bottom-3 left-3 z-20 flex flex-col gap-1 pointer-events-none">
                                            <span className="bg-black/50 backdrop-blur-md rounded-full px-2.5 py-0.5 text-[10px] text-white/50 border border-white/5">
                                                Config: {conf.width}x{conf.height}@{conf.fps}
                                            </span>
                                            {hasActual && (
                                                <span className={`backdrop-blur-md rounded-full px-2.5 py-0.5 text-[10px] border ${
                                                    resMatches
                                                        ? 'bg-green-500/20 text-green-300 border-green-500/20'
                                                        : 'bg-orange-500/20 text-orange-300 border-orange-500/20'
                                                }`}>
                                                    Actual: {actualW}x{actualH}{actualFps ? `@${actualFps}` : ''}
                                                    {!resMatches && ' \u2022 mismatch'}
                                                </span>
                                            )}
                                        </div>
                                    )}

                                    {/* Connect/Disconnect button (always visible, overlays CameraFeed when connected) */}
                                    {!isConnected && (
                                        <div className="absolute top-4 left-4">
                                            <span className="px-3 py-1 bg-black/50 backdrop-blur-md rounded-full text-xs text-white border border-white/20 font-medium flex items-center gap-2">
                                                <div className={`w-2 h-2 rounded-full ${
                                                    isLoading ? 'bg-yellow-500 animate-pulse' :
                                                    isError ? 'bg-red-500' :
                                                    'bg-neutral-500'
                                                }`} />
                                                {conf.id}
                                                {conf.type === 'intelrealsense' && (
                                                    <span className="text-blue-400 text-[10px]">RS</span>
                                                )}
                                            </span>
                                        </div>
                                    )}

                                    <div className="absolute top-4 right-4 z-30">
                                        {isConnected ? (
                                            <button
                                                onClick={() => disconnectCamera(conf.id)}
                                                className="px-3 py-1.5 bg-red-500/80 hover:bg-red-500 backdrop-blur-md rounded-full text-xs text-white font-medium transition-all flex items-center gap-1.5 opacity-0 group-hover:opacity-100"
                                            >
                                                <Power className="w-3 h-3" />
                                                Disconnect
                                            </button>
                                        ) : isLoading ? (
                                            <div className="px-3 py-1.5 bg-yellow-500/50 backdrop-blur-md rounded-full text-xs text-white font-medium flex items-center gap-1.5">
                                                <Loader2 className="w-3 h-3 animate-spin" />
                                                {status.status === 'connecting' ? 'Connecting' : 'Stopping'}
                                            </div>
                                        ) : (
                                            <button
                                                onClick={() => connectCamera(conf.id)}
                                                className="px-3 py-1.5 bg-green-500/80 hover:bg-green-500 backdrop-blur-md rounded-full text-xs text-white font-medium transition-all flex items-center gap-1.5"
                                            >
                                                <Power className="w-3 h-3" />
                                                Connect
                                            </button>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                        {configs.length === 0 && (
                            <div className="col-span-full flex flex-col items-center justify-center p-12 text-neutral-500 border-2 border-dashed border-neutral-800 rounded-3xl">
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
    );
}
