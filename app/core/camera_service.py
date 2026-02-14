import logging
import threading
from typing import List, Dict, Any
from pathlib import Path
import yaml
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from app.core.config import load_config, save_config, CONFIG_PATH

logger = logging.getLogger(__name__)


class CameraService:
    def __init__(self):
        self._cameras: Dict[str, Any] = {}          # Connected camera instances (Camera objects)
        self._camera_status: Dict[str, str] = {}    # "connected" | "disconnected" | "error"
        self._camera_errors: Dict[str, str] = {}    # Last error message per camera
        self._lock = threading.Lock()
        self._connect_lock = threading.Lock()        # Serializes connect attempts (USB contention)

    @property
    def cameras(self) -> Dict[str, Any]:
        """Returns dict of currently connected camera instances."""
        with self._lock:
            return dict(self._cameras)

    def connect_camera(self, camera_key: str) -> Dict[str, Any]:
        """
        Connect a single camera by its config key (e.g. 'camera_1').
        Creates the appropriate LeRobot camera instance, connects it,
        and starts the background read thread for streaming.
        Serialized via _connect_lock to prevent USB contention.
        """
        with self._connect_lock:
            config = self.get_camera_config()
            if camera_key not in config:
                return {"status": "error", "message": f"Camera '{camera_key}' not found in config."}

            # Disconnect existing instance if any
            if camera_key in self._cameras:
                try:
                    self._cameras[camera_key].disconnect()
                except Exception:
                    pass
                with self._lock:
                    del self._cameras[camera_key]

            cam_cfg = config[camera_key]
            cam_type = cam_cfg.get("type", "opencv")

            try:
                if cam_type == "opencv":
                    camera = self._connect_opencv_camera(camera_key, cam_cfg)
                elif cam_type == "intelrealsense":
                    camera = self._connect_realsense_camera(camera_key, cam_cfg)
                else:
                    self._camera_status[camera_key] = "error"
                    self._camera_errors[camera_key] = f"Unsupported camera type: {cam_type}"
                    return {"status": "error", "message": f"Unsupported camera type: {cam_type}"}

                with self._lock:
                    self._cameras[camera_key] = camera
                self._camera_status[camera_key] = "connected"
                self._camera_errors[camera_key] = ""
                logger.info(f"Camera '{camera_key}' ({cam_type}) connected successfully.")
                return {"status": "connected", "camera_key": camera_key}

            except Exception as e:
                self._camera_status[camera_key] = "error"
                self._camera_errors[camera_key] = str(e)
                logger.error(f"Failed to connect camera '{camera_key}': {e}")
                return {"status": "error", "message": str(e)}

    def _connect_opencv_camera(self, camera_key: str, cam_cfg: dict) -> OpenCVCamera:
        """Connect an OpenCV camera, with fps/resolution auto-detect and discovery fallbacks."""
        from lerobot.cameras.opencv import OpenCVCameraConfig
        import gc

        idx = cam_cfg.get("index_or_path")
        fps = cam_cfg.get("fps", 30)
        width = cam_cfg.get("width", 640)
        height = cam_cfg.get("height", 480)

        # Attempt 1: Try with configured fps/width/height
        try:
            c_conf = OpenCVCameraConfig(
                index_or_path=idx,
                fps=fps,
                width=width,
                height=height,
            )
            camera = OpenCVCamera(c_conf)
            camera.connect(warmup=True)
            return camera
        except RuntimeError as fps_err:
            # FPS or resolution validation failed — camera opened but can't match settings
            logger.warning(
                f"Camera '{camera_key}': configured {width}x{height}@{fps}fps failed "
                f"({fps_err}), retrying with auto-detect..."
            )
            try:
                del camera
            except NameError:
                pass
            gc.collect()
        except ConnectionError as conn_err:
            # Camera can't be opened at all — skip to discovery
            logger.warning(f"Camera '{camera_key}': path '{idx}' can't be opened ({conn_err}), trying discovery...")
            # Jump straight to discovery (auto-detect won't help if device can't open)
            return self._opencv_discovery_fallback(camera_key, idx, fps, width, height)

        # Attempt 2: Auto-detect fps/resolution (let camera pick native capabilities)
        try:
            c_conf = OpenCVCameraConfig(
                index_or_path=idx,
                fps=None,
                width=None,
                height=None,
            )
            camera = OpenCVCamera(c_conf)
            camera.connect(warmup=True)
            logger.info(
                f"Camera '{camera_key}': auto-detect connected at "
                f"{camera.width}x{camera.height}@{camera.fps}fps"
            )
            return camera
        except (ConnectionError, RuntimeError) as second_err:
            logger.warning(f"Camera '{camera_key}': auto-detect also failed ({second_err}), trying discovery...")
            try:
                del camera
            except NameError:
                pass
            gc.collect()

        # Attempt 3: Auto-discovery fallback
        return self._opencv_discovery_fallback(camera_key, idx, fps, width, height)

    def _opencv_discovery_fallback(self, camera_key: str, idx, fps, width, height) -> OpenCVCamera:
        """Find available OpenCV cameras via discovery and connect to the first one."""
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from app.core.camera_discovery import discover_cameras
        discovered = discover_cameras(opencv_only=True)
        opencv_devices = discovered.get("opencv", [])

        if not opencv_devices:
            raise ConnectionError(
                f"Configured path '{idx}' failed and no USB webcams were discovered. "
                "Check that a webcam is plugged in."
            )

        # Filter out devices already connected by this service
        connected_paths = set()
        for key, cam in self._cameras.items():
            if hasattr(cam, 'index_or_path'):
                connected_paths.add(str(cam.index_or_path))

        available = [d for d in opencv_devices if d.get("index_or_path") not in connected_paths]

        if not available:
            raise ConnectionError(
                f"Configured path '{idx}' failed and all discovered webcams are already in use."
            )

        # Use the first available device
        new_path = available[0]["index_or_path"]
        logger.info(f"Camera '{camera_key}': auto-discovered device at '{new_path}', connecting...")

        c_conf = OpenCVCameraConfig(
            index_or_path=new_path,
            fps=fps,
            width=width,
            height=height,
        )
        camera = OpenCVCamera(c_conf)
        camera.connect(warmup=True)

        # Update settings.yaml with the correct path so next time it works directly
        self._update_camera_path(camera_key, new_path)

        return camera

    def _connect_realsense_camera(self, camera_key: str, cam_cfg: dict) -> RealSenseCamera:
        """Connect a RealSense camera by serial number, with resolution auto-detect fallback."""
        from lerobot.cameras.realsense import RealSenseCameraConfig
        import gc

        serial = cam_cfg.get("serial_number_or_name")
        fps = cam_cfg.get("fps", 30)
        width = cam_cfg.get("width", 640)
        height = cam_cfg.get("height", 480)
        use_depth = cam_cfg.get("use_depth", False)

        # Attempt 1: Try with configured resolution
        try:
            c_conf = RealSenseCameraConfig(
                serial_number_or_name=serial,
                fps=fps,
                width=width,
                height=height,
                use_depth=use_depth,
            )
            camera = RealSenseCamera(c_conf)
            camera.connect(warmup=True)

            # Prime the frame cache
            try:
                camera.async_read(blocking=True, timeout_ms=3000)
            except Exception as e:
                logger.warning(f"Camera '{camera_key}': initial frame prime failed ({e}), stream may take a moment")

            return camera

        except (ConnectionError, RuntimeError) as first_err:
            cause = first_err.__cause__ if first_err.__cause__ else first_err
            logger.warning(
                f"Camera '{camera_key}': configured {width}x{height}@{fps}fps failed "
                f"(pyrealsense2: {cause}), retrying with auto-detect resolution..."
            )
            # Release failed camera's USB handles before retry
            try:
                del camera
            except NameError:
                pass
            gc.collect()

        # Attempt 2: Auto-detect resolution (let camera pick native defaults)
        c_conf = RealSenseCameraConfig(
            serial_number_or_name=serial,
            fps=None,
            width=None,
            height=None,
            use_depth=use_depth,
        )
        camera = RealSenseCamera(c_conf)
        camera.connect(warmup=True)

        logger.info(
            f"Camera '{camera_key}': auto-detect connected at "
            f"{camera.width}x{camera.height}@{camera.fps}fps"
        )

        # Prime the frame cache
        try:
            camera.async_read(blocking=True, timeout_ms=3000)
        except Exception as e:
            logger.warning(f"Camera '{camera_key}': initial frame prime failed ({e}), stream may take a moment")

        return camera

    def _update_camera_path(self, camera_key: str, new_path: str):
        """Update the device path for an OpenCV camera in settings.yaml."""
        try:
            full_config = load_config()
            cameras_cfg = full_config.get("robot", {}).get("cameras", {})
            if camera_key in cameras_cfg:
                cameras_cfg[camera_key]["index_or_path"] = new_path
                cameras_cfg[camera_key]["video_device_id"] = new_path
                save_config(full_config)
                logger.info(f"Updated settings.yaml: {camera_key} path → {new_path}")
        except Exception as e:
            logger.warning(f"Failed to update settings.yaml for {camera_key}: {e}")

    def disconnect_camera(self, camera_key: str) -> Dict[str, Any]:
        """Disconnect a single camera by key."""
        with self._lock:
            camera = self._cameras.pop(camera_key, None)

        if camera:
            try:
                camera.disconnect()
                logger.info(f"Camera '{camera_key}' disconnected.")
            except Exception as e:
                logger.warning(f"Error disconnecting camera '{camera_key}': {e}")

        self._camera_status[camera_key] = "disconnected"
        self._camera_errors[camera_key] = ""
        return {"status": "disconnected", "camera_key": camera_key}

    def disconnect_all(self):
        """Disconnect all managed cameras. Called during shutdown."""
        keys = list(self._cameras.keys())
        for key in keys:
            self.disconnect_camera(key)

    def get_status(self) -> Dict[str, Any]:
        """
        Returns connection status for all configured cameras.
        Detects if a connected camera's background thread has died.
        """
        config = self.get_camera_config()
        status = {}
        for camera_key in config:
            cam = self._cameras.get(camera_key)
            if cam and getattr(cam, 'is_connected', False):
                # Check if background thread is still alive
                thread_alive = hasattr(cam, 'thread') and cam.thread is not None and cam.thread.is_alive()
                if thread_alive:
                    status[camera_key] = {"status": "connected", "error": ""}
                else:
                    status[camera_key] = {"status": "error", "error": "Background read thread stopped"}
                    self._camera_status[camera_key] = "error"
                    self._camera_errors[camera_key] = "Background read thread stopped"
            else:
                status[camera_key] = {
                    "status": self._camera_status.get(camera_key, "disconnected"),
                    "error": self._camera_errors.get(camera_key, ""),
                }
        return status

    # ── Legacy methods (unchanged) ────────────────────────────────────────

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

        from app.core.camera_discovery import discover_cameras

        try:
            logger.info("Scanning for cameras using unified discovery...")
            discovered = discover_cameras(skip_devices=active_ids)
            results["opencv"] = discovered.get("opencv", [])
            results["realsense"] = discovered.get("realsense", [])
        except Exception as e:
            logger.error(f"Error during unified camera scan: {e}")

        return results

    def capture_snapshot(self, camera_key: str):
        """
        Capture a single frame from a camera.
        Prefers using a connected managed camera (fast path).
        Falls back to one-shot open/read/close for OpenCV cameras.
        """
        # Fast path: use managed connected camera
        cam = self._cameras.get(camera_key)
        if cam and getattr(cam, 'is_connected', False):
            try:
                frame = cam.async_read(blocking=False)
                if frame is not None:
                    return frame
            except Exception:
                pass

        # Slow path: one-shot capture (OpenCV only, RealSense needs persistent connection)
        config = self.get_camera_config()
        if camera_key not in config:
            return None

        cam_cfg = config[camera_key]
        cam_type = cam_cfg.get("type", "opencv")

        try:
            import cv2
            if cam_type == "opencv":
                idx = cam_cfg.get("index_or_path")
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        return frame
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

                if cam_type == "opencv" and "index_or_path" not in cam_cfg:
                    cam_cfg["index_or_path"] = vid

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
        """Tests if a configured camera can be opened and read."""
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
