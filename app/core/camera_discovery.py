import cv2
import glob
import logging
import re
import subprocess

logger = logging.getLogger(__name__)

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
                # struct v4l2_capability {
                #   u8 driver[16]; u8 card[32]; u8 bus_info[32];
                #   u32 version; u32 capabilities; u32 device_caps; u32 reserved[3];
                # }; Total ~104 bytes.
                # We need capabilities (offset 84?) -> 16+32+32+4 = 84.
                # Let's read full buffer.
                capability = fcntl.ioctl(device_file, VIDIOC_QUERYCAP, b'\0' * 104)
                # Unpack: 16s 32s 32s I I I 3I
                # Python struct: 16s 32s 32s I I I (reserved)
                # Capabilities is 5th element (index 4 in list after unpack?) or offset 84 (driver+card+bus+ver)
                # Correct unpack format: '16s32s32sIII' (rest doesn't matter much)
                
                # Using simple buffer slicing to be safe
                caps = struct.unpack('I', capability[84:88])[0]
                
                if not (caps & V4L2_CAP_VIDEO_CAPTURE):
                     # Not a video capture device (likely metadata, radio, vbi, etc)
                     return False
                     
        except Exception as e:
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

def discover_cameras(skip_devices: list = None, opencv_only: bool = False):
    """
    Scans for available cameras and returns a list of working camera configurations.
    Returns a dictionary with 'opencv' and 'realsense' lists.

    Args:
        skip_devices: List of device paths to skip (e.g., ['/dev/video12']).
                      Used to avoid opening cameras that are already in use.
        opencv_only: If True, skip RealSense scanning. Used during OpenCV
                     fallback to avoid creating an rs.context that could
                     interfere with subsequent RealSense connections.
    """
    if skip_devices is None:
        skip_devices = []

    available_cameras = {
        "opencv": [],
        "realsense": []
    }

    # 1. Scan OpenCV Cameras (USB Webcams)
    ports = get_available_video_ports()
    print(f"Scanning video ports: {ports}")

    for port in ports:
        # Skip devices that are already in use by the robot
        if port in skip_devices:
            print(f"Skipping {port} (already in use)")
            continue

        if is_camera_available(port):
            print(f"Camera found at {port}")
            available_cameras["opencv"].append({
                "id": port,
                "index_or_path": port,
                "name": f"Camera {port}",
                "width": 640, # Default
                "height": 480, # Default
                "fps": 30
            })
            
    # 2. Scan RealSense Cameras (skip if caller only needs OpenCV)
    if opencv_only:
        return available_cameras

    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            print(f"RealSense found: {name} ({serial})")
            available_cameras["realsense"].append({
                "id": serial,
                "serial_number_or_name": serial,
                "name": name,
                "width": 848, # Default for RealSense
                "height": 480,
                "fps": 30
            })
    except ImportError:
        logger.warning("pyrealsense2 not installed, skipping RealSense scan.")
    except Exception as e:
        logger.error(f"Error scanning RealSense: {e}")

    return available_cameras
