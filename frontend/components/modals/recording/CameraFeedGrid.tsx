import { useState, useEffect } from "react";
import CameraFeed from "../../ui/CameraFeed";
import { camerasApi } from "../../../lib/api";
import type { CameraConfig } from "../../../lib/api/types";

interface CameraFeedGridProps {
  compact?: boolean;
}

export default function CameraFeedGrid({ compact }: CameraFeedGridProps) {
  const [cameras, setCameras] = useState<CameraConfig[]>([]);

  useEffect(() => {
    let cancelled = false;
    camerasApi.config().then((cfg) => {
      if (!cancelled) setCameras(cfg);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, []);

  if (cameras.length === 0) {
    return (
      <p className="text-xs text-zinc-500 text-center py-3">
        No cameras configured.
      </p>
    );
  }

  const cols = compact ? "grid-cols-2" : cameras.length <= 2 ? "grid-cols-2" : "grid-cols-3";

  return (
    <div className={`grid ${cols} gap-1.5`}>
      {cameras.map((cam) => (
        <div key={cam.id} className="aspect-video rounded-lg overflow-hidden bg-black">
          <CameraFeed
            cameraId={cam.id}
            showOverlay={!compact}
            mode="contain"
            className="rounded-none border-0"
          />
        </div>
      ))}
    </div>
  );
}
