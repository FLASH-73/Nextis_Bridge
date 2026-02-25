import glob
import logging
import re
import time

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

# Module-level discovery cache (avoids repeated hardware probes)
_discovery_cache: dict = {"result": None, "timestamp": 0.0}
_DISCOVERY_CACHE_TTL = 30.0  # seconds


def get_available_video_ports():
    """
    Scans /dev/video* and returns a list of available video device paths.
    """
    ports = glob.glob('/dev/video*')
    ports.sort()
    return ports

def is_camera_available(port):
    """
    Tries to open the camera port using OpenCV.
    Returns True if successful, False otherwise.
    """
    # v4l2 ioctl constants
    import fcntl
    import struct

    VIDIOC_QUERYCAP = 0x80685600
    V4L2_CAP_VIDEO_CAPTURE = 0x00000001

    try:
        # Extract index from /dev/videoX
        match = re.search(r'video(\d+)', port)
        if not match:
            return False
        index = int(match.group(1))

        # 1. IOCTL Capability Check (Filter Metadata nodes)
        try:
            with open(port, 'rb') as device_file:
                capability = fcntl.ioctl(device_file, VIDIOC_QUERYCAP, b'\0' * 104)
                caps = struct.unpack('I', capability[84:88])[0]

                if not (caps & V4L2_CAP_VIDEO_CAPTURE):
                     # Not a video capture device (likely metadata, radio, vbi, etc)
                     return False

        except Exception:
            # If ioctl fails (permission?), fall back to open check
            pass

        # 2. Sysfs Name Check (Filter Intel RealSense UVC nodes to avoid dupes)
        try:
            with open(f"/sys/class/video4linux/video{index}/name", "r") as f:
                name = f.read().strip()
                if "Intel(R) RealSense" in name:
                    return False
        except Exception:
            pass

        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return False

        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        return ret
    except Exception as e:
        logger.error(f"Error checking camera {port}: {e}")
        return False

def discover_cameras(skip_devices: list = None, opencv_only: bool = False, force: bool = False):
    """
    Scans for available cameras and returns a list of working camera configurations.
    Returns a dictionary with 'opencv' and 'realsense' lists.

    Results are cached for 30 seconds to avoid repeated hardware probes.

    Args:
        skip_devices: List of device paths to skip (e.g., ['/dev/video12']).
                      Used to avoid opening cameras that are already in use.
        opencv_only: If True, skip RealSense scanning. Used during OpenCV
                     fallback to avoid creating an rs.context that could
                     interfere with subsequent RealSense connections.
        force: If True, bypass cache and force a fresh scan.
    """
    global _discovery_cache

    if skip_devices is None:
        skip_devices = []

    # Return cached results if still fresh (unless forced or skip_devices changed)
    if (not force
            and _discovery_cache["result"] is not None
            and (time.time() - _discovery_cache["timestamp"]) < _DISCOVERY_CACHE_TTL):
        logger.debug("Returning cached discovery results")
        return _discovery_cache["result"]

    available_cameras = {
        "opencv": [],
        "realsense": []
    }

    # 1. Scan OpenCV Cameras (USB Webcams)
    ports = get_available_video_ports()
    logger.debug(f"Scanning {len(ports)} video ports")

    for port in ports:
        if port in skip_devices:
            logger.debug(f"Skipping {port} (already in use)")
            continue

        if is_camera_available(port):
            logger.info(f"Camera found at {port}")
            available_cameras["opencv"].append({
                "id": port,
                "index_or_path": port,
                "name": f"Camera {port}",
                "width": 640,
                "height": 480,
                "fps": 30
            })

    # 2. Scan RealSense Cameras (skip if caller only needs OpenCV)
    if not opencv_only:
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                serial = dev.get_info(rs.camera_info.serial_number)
                name = dev.get_info(rs.camera_info.name)
                logger.info(f"RealSense found: {name} ({serial})")
                available_cameras["realsense"].append({
                    "id": serial,
                    "serial_number_or_name": serial,
                    "name": name,
                    "width": 848,
                    "height": 480,
                    "fps": 30
                })
        except ImportError:
            logger.warning("pyrealsense2 not installed, skipping RealSense scan.")
        except Exception as e:
            logger.error(f"Error scanning RealSense: {e}")

    # Update cache
    _discovery_cache["result"] = available_cameras
    _discovery_cache["timestamp"] = time.time()

    return available_cameras
