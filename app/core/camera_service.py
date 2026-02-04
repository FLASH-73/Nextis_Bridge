import logging
from typing import List, Dict, Any
from pathlib import Path
import yaml
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from app.core.config import load_config, save_config, CONFIG_PATH

logger = logging.getLogger(__name__)

class CameraService:
    def __init__(self):
        pass

    def scan_cameras(self, active_ids: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scans for available OpenCV and RealSense cameras.
        Returns a dict with 'opencv' and 'realsense' lists.
        Filters out devices that cannot be opened or read.
        """
        if active_ids is None:
            active_ids = []
            
        results = {
            "opencv": [],
            "realsense": []
        }
        
        # Scan OpenCV
        from app.core.camera_discovery import discover_cameras
        
        try:
            logger.info("Scanning for cameras using unified discovery...")
            discovered = discover_cameras(skip_devices=active_ids)
            
            # Filter out active cameras if needed, but the frontend/main.py logic handles merging.
            # Here we just return what discover_cameras found.
            
            results["opencv"] = discovered.get("opencv", [])
            results["realsense"] = discovered.get("realsense", [])
            
        except Exception as e:
            logger.error(f"Error during unified camera scan: {e}")
            
        return results

    def capture_snapshot(self, camera_key: str):
        config = self.get_camera_config()
        if camera_key not in config:
            logger.warning(f"Snapshot: {camera_key} not in config.")
            return None
            
        cam_cfg = config[camera_key]
        cam_type = cam_cfg.get("type", "opencv")
        
        try:
            import cv2
            if cam_type == "opencv":
                idx = cam_cfg.get("index_or_path")
                # logger.info(f"Snapshot: Opening {idx} for {camera_key}") # Commented out to reduce spam
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        return frame
                    else:
                        logger.warning(f"Snapshot: Read failed for {idx}")
                else:
                    # logger.warning(f"Snapshot: Failed to open {idx} (is it busy?)")
                    pass
            
        except Exception as e:
            logger.error(f"Snapshot failed for {camera_key}: {e}")
        
        return None

    def get_camera_config(self) -> Dict[str, Any]:
        """Returns the current camera configuration from settings.yaml."""
        config = load_config()
        raw_cameras = config.get("robot", {}).get("cameras", {})
        
        # Normalize List to Dict if needed
        if isinstance(raw_cameras, list):
            normalized = {}
            for c in raw_cameras:
                c_id = c.get("id", "unknown")
                vid = c.get("video_device_id")
                c_type = c.get("type", "opencv" if (str(vid).startswith("/dev/video") or (str(vid).isdigit() and len(str(vid)) < 4)) else "intelrealsense")
                normalized[c_id] = {
                    "type": c_type,
                    "index_or_path": vid,
                    "serial_number_or_name": vid,
                    **c
                }
            return normalized

        # Normalize dict format too (ensure index_or_path/serial_number_or_name exist)
        if isinstance(raw_cameras, dict):
            for cam_id, cam_cfg in raw_cameras.items():
                vid = cam_cfg.get("video_device_id")
                cam_type = cam_cfg.get("type", "opencv")

                # For opencv cameras: ensure index_or_path exists
                if cam_type == "opencv" and "index_or_path" not in cam_cfg:
                    cam_cfg["index_or_path"] = vid

                # For realsense cameras: ensure serial_number_or_name exists
                if cam_type == "intelrealsense" and "serial_number_or_name" not in cam_cfg:
                    cam_cfg["serial_number_or_name"] = vid

        return raw_cameras

    def update_camera_config(self, new_cameras_config: Dict[str, Any]):
        """
        Updates the camera configuration in settings.yaml.
        new_cameras_config: Dict mapping camera_key (e.g. 'camera_1') to config dict.
        """
        config = load_config()
        if "robot" not in config:
            config["robot"] = {}
        
        config["robot"]["cameras"] = new_cameras_config
        save_config(config)
        logger.info("Camera configuration updated.")

    def test_camera(self, camera_key: str) -> Dict[str, Any]:
        """
        Tests if a configured camera can be opened and read.
        """
        config = self.get_camera_config()
        if camera_key not in config:
            return {"status": "error", "message": f"Camera {camera_key} not found in config."}
        
        cam_cfg = config[camera_key]
        cam_type = cam_cfg.get("type")
        
        try:
            camera = None
            if cam_type == "opencv":
                from lerobot.cameras.opencv import OpenCVCameraConfig
                c_conf = OpenCVCameraConfig(
                    index_or_path=cam_cfg.get("index_or_path"),
                    fps=cam_cfg.get("fps", 30),
                    width=cam_cfg.get("width", 640),
                    height=cam_cfg.get("height", 480)
                )
                camera = OpenCVCamera(c_conf)
                
            elif cam_type == "intelrealsense":
                from lerobot.cameras.realsense import RealSenseCameraConfig
                c_conf = RealSenseCameraConfig(
                    serial_number_or_name=cam_cfg.get("serial_number_or_name"),
                    fps=cam_cfg.get("fps", 30),
                    width=cam_cfg.get("width", 640),
                    height=cam_cfg.get("height", 480)
                )
                camera = RealSenseCamera(c_conf)
            
            if camera:
                camera.connect()
                frame = camera.read()
                camera.disconnect()
                
                if frame is not None:
                    return {"status": "success", "message": "Camera connected and frame read successfully."}
                else:
                    return {"status": "error", "message": "Camera connected but returned empty frame."}
            else:
                return {"status": "error", "message": f"Unsupported camera type: {cam_type}"}

        except Exception as e:
            return {"status": "error", "message": f"Failed to connect: {str(e)}"}
