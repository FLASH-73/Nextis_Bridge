import { Camera, Check } from "lucide-react";
import CameraFeed from "../../ui/CameraFeed";
import type { CameraConfig, CameraStatusEntry } from "../../../lib/api/types";

interface CameraGridProps {
  cameras: CameraConfig[];
  statuses: Record<string, CameraStatusEntry>;
  selectedCameras: string[];
  onToggleCamera: (id: string) => void;
}

export default function CameraGrid({
  cameras,
  statuses,
  selectedCameras,
  onToggleCamera,
}: CameraGridProps) {
  if (cameras.length === 0) {
    return (
      <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700 p-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-zinc-400 flex items-center gap-1.5 mb-3">
          <Camera className="w-3.5 h-3.5" />
          Cameras
        </h3>
        <p className="text-xs text-neutral-400 dark:text-zinc-500 text-center py-4">
          No cameras configured
        </p>
      </div>
    );
  }

  return (
    <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700 p-4">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-zinc-400 flex items-center gap-1.5 mb-3">
        <Camera className="w-3.5 h-3.5" />
        Cameras
      </h3>

      <div className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-3">
        {cameras.map((cam) => {
          const isSelected = selectedCameras.includes(cam.id);
          const status = statuses?.[cam.id];
          const isConnected = status?.status === "connected";

          return (
            <div
              key={cam.id}
              onClick={() => onToggleCamera(cam.id)}
              className={`rounded-lg overflow-hidden border-2 cursor-pointer transition-all ${
                isSelected
                  ? "border-black dark:border-white ring-1 ring-black/10 dark:ring-white/10"
                  : "border-neutral-200 dark:border-zinc-700 opacity-50 hover:opacity-70"
              }`}
            >
              {/* Preview */}
              <div className="aspect-video bg-neutral-100 dark:bg-zinc-800 relative">
                {isConnected ? (
                  <CameraFeed
                    cameraId={cam.id}
                    showOverlay={false}
                    mode="cover"
                    className="rounded-none border-0"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <span className="text-[10px] text-neutral-400 dark:text-zinc-500">
                      No signal
                    </span>
                  </div>
                )}
                {isSelected && (
                  <div className="absolute top-1.5 right-1.5 w-5 h-5 bg-black dark:bg-white rounded-full flex items-center justify-center z-10">
                    <Check className="w-3 h-3 text-white dark:text-black" />
                  </div>
                )}
              </div>

              {/* Label */}
              <div className="px-2 py-1.5 flex items-center justify-between bg-white dark:bg-zinc-800">
                <span className="text-xs font-medium text-neutral-700 dark:text-zinc-300 truncate">
                  {cam.id.replace(/_/g, " ")}
                </span>
                <div
                  className={`w-2 h-2 rounded-full flex-shrink-0 ${
                    isConnected
                      ? "bg-emerald-500"
                      : "bg-neutral-300 dark:bg-zinc-600"
                  }`}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
