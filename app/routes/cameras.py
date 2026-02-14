import time

import cv2

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from app.dependencies import get_state

router = APIRouter(tags=["cameras"])


def generate_frames(camera_key: str):
    system = get_state()
    while True:
        frame = None

        # PRIORITY 1: CameraService managed cameras (independent of robot connection)
        if system.camera_service and camera_key in system.camera_service.cameras:
            cam = system.camera_service.cameras[camera_key]
            try:
                frame = cam.async_read(blocking=False)  # ZOH pattern: return cached frame immediately
            except Exception:
                pass

        # PRIORITY 1b: Robot cameras (legacy path, if cameras were attached directly)
        if frame is None and system.robot and hasattr(system.robot, 'cameras') and system.robot.cameras and camera_key in system.robot.cameras:
            cam = system.robot.cameras[camera_key]
            try:
                frame = cam.async_read(blocking=False)
            except Exception:
                pass

        # PRIORITY 2: Orchestrator Observation (What the Agent sees) - Fallback
        if frame is None and system.orchestrator and system.orchestrator.intervention_engine:
            obs = system.orchestrator.intervention_engine.latest_observation
            if obs:
                # Try explicit key then partial match
                full_key = f"observation.images.{camera_key}"
                if full_key in obs:
                    frame = obs[full_key]
                elif camera_key in obs:
                    frame = obs[camera_key]
                else:
                    for k in obs.keys():
                        if camera_key in k:
                            frame = obs[k]
                            break

        # PRIORITY 3: Snapshot Fallback (always try, even if robot has camera)
        # Note: If robot's camera thread has device open, snapshot might fail (device busy)
        # but we still try as a last resort before showing "Waiting..."
        if frame is None and system.camera_service:
            snapshot = system.camera_service.capture_snapshot(camera_key)
            if snapshot is not None:
                frame = snapshot

        if frame is None:
            # Yield placeholder
            import numpy as np
            blank_image = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_image, f"Waiting for {camera_key}...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame = blank_image
        else:
            # Convert PyTorch tensor to numpy if needed
            import torch
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).cpu().numpy() * 255
                frame = frame.astype("uint8")
                # RGB to BGR for OpenCV encoding
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.016)  # ~60 FPS streaming for lower latency


@router.get("/video_feed/{camera_key}")
def video_feed(camera_key: str):
    return StreamingResponse(generate_frames(camera_key), media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/cameras/{camera_key}/connect")
def connect_camera(camera_key: str):
    """Connect a single camera by its config key.
    Uses def (not async) so FastAPI runs it in a thread pool — avoids blocking the event loop
    during the 2-3s camera warmup (time.sleep + frame reads).
    """
    system = get_state()
    if not system.camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized")
    result = system.camera_service.connect_camera(camera_key)
    return result

@router.post("/cameras/{camera_key}/disconnect")
def disconnect_camera(camera_key: str):
    """Disconnect a single camera by its config key.
    Uses def (not async) so FastAPI runs it in a thread pool.
    """
    system = get_state()
    if not system.camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized")
    result = system.camera_service.disconnect_camera(camera_key)
    return result

@router.get("/cameras/status")
def get_camera_status():
    """Get connection status of all configured cameras."""
    system = get_state()
    if not system.camera_service:
        return {}
    return system.camera_service.get_status()

@router.get("/cameras/scan")
def scan_cameras():
    system = get_state()
    if not system.camera_service:
        return {"status": "error", "message": "Camera service not initialized"}

    # If robot is connected, we combine active cameras with a fresh scan for available ones.
    if system.robot and system.robot.is_connected:
        current_config = system.camera_service.get_camera_config()

        # 1. Get Active Cameras
        active_opencv = []
        active_realsense = []

        active_opencv_indices = set()
        active_realsense_serials = set()

        for key, conf in current_config.items():
            if conf.get("type") == "opencv":
                idx = conf.get("index_or_path")
                active_opencv.append({
                    "index": idx,
                    "name": f"Active Camera ({key})",
                    "port": idx,
                    "is_active": True
                })
                active_opencv_indices.add(str(idx))
            elif conf.get("type") == "intelrealsense":
                serial = conf.get("serial_number_or_name")
                active_realsense.append({
                    "name": f"Active Camera ({key})",
                    "serial_number": serial,
                    "is_active": True
                })
                active_realsense_serials.add(str(serial))

        # 2. Scan for Available Cameras (ignoring errors on busy ones)
        available = system.camera_service.scan_cameras(active_ids=list(active_opencv_indices))

        # 3. Merge (avoid duplicates)
        final_opencv = active_opencv[:]
        for cam in available.get("opencv", []):
            cam_idx = str(cam.get("port") or cam.get("id") or cam.get("index"))
            if cam_idx not in active_opencv_indices:
                cam["is_active"] = False
                final_opencv.append(cam)

        final_realsense = active_realsense[:]
        for cam in available.get("realsense", []):
            cam_serial = str(cam.get("serial_number"))
            if cam_serial not in active_realsense_serials:
                cam["is_active"] = False
                final_realsense.append(cam)
    else:
        # Robot not connected - just do a fresh scan
        available = system.camera_service.scan_cameras()
        final_opencv = available.get("opencv", [])
        final_realsense = available.get("realsense", [])
        for cam in final_opencv:
            cam["is_active"] = False
        for cam in final_realsense:
            cam["is_active"] = False

    # Standaradize keys for Frontend
    for cam in final_opencv:
         if "id" not in cam:
             cam["id"] = cam.get("port") or cam.get("index") or cam.get("index_or_path")
         if "index_or_path" not in cam:
             cam["index_or_path"] = cam.get("id")

    for cam in final_realsense:
         if "id" not in cam:
             cam["id"] = cam.get("serial_number") or cam.get("serial_number_or_name")
         if "serial_number_or_name" not in cam:
             cam["serial_number_or_name"] = cam.get("id")

    return {
        "opencv": final_opencv,
        "realsense": final_realsense,
        "note": "Merged active and available cameras."
    }

@router.get("/cameras/config")
def get_camera_config():
    system = get_state()
    if not system.camera_service:
        return []

    # Return List for Frontend (CameraModal.tsx expects array)
    config = system.camera_service.get_camera_config()
    export_list = []
    for key, val in config.items():
        # Inject key as 'id'
        item = val.copy()
        item["id"] = key # 'camera_1' etc
        export_list.append(item)
    return export_list

@router.post("/cameras/config")
async def update_camera_config(request: Request, background_tasks: BackgroundTasks):
    system = get_state()
    if not system.camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized")

    data = await request.json()

    # Get existing config to compare
    existing_config = system.camera_service.get_camera_config()

    # Convert array format from frontend to dict format for storage
    # Frontend sends: [{id: "cam1", video_device_id: ..., type: ..., use_depth: ...}, ...]
    # Backend expects: {"cam1": {video_device_id: ..., type: ..., use_depth: ...}, ...}
    if isinstance(data, list):
        config_dict = {}
        for item in data:
            cam_id = item.get("id", "unknown")
            # Remove 'id' from the stored config (it's the key)
            config_entry = {k: v for k, v in item.items() if k != "id"}
            vid = config_entry.get("video_device_id", "")

            # Ensure type is set based on video_device_id if not provided
            if "type" not in config_entry:
                if str(vid).startswith("/dev/video") or (str(vid).isdigit() and len(str(vid)) < 4):
                    config_entry["type"] = "opencv"
                else:
                    config_entry["type"] = "intelrealsense"

            # Synthesize index_or_path for opencv cameras
            if config_entry.get("type") == "opencv" and "index_or_path" not in config_entry:
                config_entry["index_or_path"] = vid

            # Synthesize serial_number_or_name for realsense cameras
            if config_entry.get("type") == "intelrealsense" and "serial_number_or_name" not in config_entry:
                config_entry["serial_number_or_name"] = vid

            config_dict[cam_id] = config_entry
        data = config_dict

    # Check if only use_depth changed (no need to reload for depth-only changes)
    # Compare camera assignments (video_device_id, type) - ignore use_depth
    needs_reload = False

    # Check for new/removed cameras or changed assignments
    if set(data.keys()) != set(existing_config.keys()):
        needs_reload = True
    else:
        for cam_id, new_cfg in data.items():
            old_cfg = existing_config.get(cam_id, {})
            # Compare assignment-critical fields (not use_depth)
            if (new_cfg.get("video_device_id") != old_cfg.get("video_device_id") or
                new_cfg.get("type") != old_cfg.get("type") or
                new_cfg.get("width") != old_cfg.get("width") or
                new_cfg.get("height") != old_cfg.get("height") or
                new_cfg.get("fps") != old_cfg.get("fps")):
                needs_reload = True
                break

    system.camera_service.update_camera_config(data)

    if needs_reload:
        # Trigger System Reload in Background (only for camera assignment changes)
        background_tasks.add_task(system.reload)
        return {"status": "success", "message": "Camera config updated. System reloading..."}
    else:
        # Just a depth toggle or other non-critical change - no reload needed
        return {"status": "success", "message": "Camera config updated (no reload needed)."}
