import React from 'react';
import { RefreshCw, Camera, Check, Plus, Pencil, Layers, Power, Loader2, AlertCircle, ArrowRight, Trash2, Image as ImageIcon } from 'lucide-react';

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

export type CameraConnectionStatus = 'connected' | 'disconnected' | 'connecting' | 'disconnecting' | 'error';

export interface CameraStatusEntry {
    status: CameraConnectionStatus;
    error: string;
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
    assignCamera: (deviceId: number | string, cameraName: string, deviceType: 'opencv' | 'intelrealsense') => void;
    removeAssignment: (cameraName: string) => void;
    toggleDepth: (cameraId: string) => void;

    // Preview-tab state & callbacks
    cameraStatus: Record<string, CameraStatusEntry>;
    connectCamera: (cameraKey: string) => void;
    disconnectCamera: (cameraKey: string) => void;
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
}: CameraGridProps) {

    const getStatusForCamera = (cameraId: string): CameraStatusEntry => {
        return cameraStatus[cameraId] || { status: 'disconnected', error: '' };
    };

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
                    <div className="grid grid-cols-2 gap-6">
                        {configs.map(conf => {
                            const status = getStatusForCamera(conf.id);
                            const isConnected = status.status === 'connected';
                            const isLoading = status.status === 'connecting' || status.status === 'disconnecting';
                            const isError = status.status === 'error';

                            return (
                                <div key={conf.id} className="bg-neutral-900 rounded-2xl overflow-hidden border border-white/10 shadow-2xl relative group">
                                    {/* Stream or placeholder */}
                                    {isConnected ? (
                                        <img
                                            src={`http://127.0.0.1:8000/video_feed/${conf.id}?t=${Date.now()}`}
                                            alt={conf.id}
                                            className="w-full aspect-video object-cover bg-neutral-800"
                                            onError={(e) => {
                                                // Hide broken image but don't remove — browser will retry MJPEG
                                                (e.target as HTMLImageElement).style.opacity = '0.3';
                                            }}
                                        />
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

                                    {/* Camera label */}
                                    <div className="absolute top-4 left-4">
                                        <span className="px-3 py-1 bg-black/50 backdrop-blur-md rounded-full text-xs text-white border border-white/20 font-medium flex items-center gap-2">
                                            <div className={`w-2 h-2 rounded-full ${
                                                isConnected ? 'bg-green-500 animate-pulse' :
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

                                    {/* Connect/Disconnect button */}
                                    <div className="absolute top-4 right-4">
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

                                    {/* Overlay Stats (on hover, only when connected) */}
                                    {isConnected && (
                                        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-sm pointer-events-none">
                                            <div className="text-center">
                                                <p className="text-white font-medium mb-2">{conf.width}x{conf.height} @ {conf.fps}fps</p>
                                                <p className="text-neutral-400 text-xs">
                                                    {conf.type === 'intelrealsense' ? `RealSense ${conf.video_device_id}` : conf.video_device_id}
                                                </p>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            );
                        })}
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
    );
}
