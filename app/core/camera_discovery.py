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
    try:
        # Extract index from /dev/videoX
        match = re.search(r'video(\d+)', port)
        if not match:
            return False
        index = int(match.group(1))
        
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

def discover_cameras():
    """
    Scans for available cameras and returns a list of working camera configurations.
    Returns a dictionary with 'opencv' and 'realsense' lists.
    """
    available_cameras = {
        "opencv": [],
        "realsense": []
    }
    
    # 1. Scan OpenCV Cameras (USB Webcams)
    ports = get_available_video_ports()
    print(f"Scanning video ports: {ports}")
    
    for port in ports:
        if is_camera_available(port):
            print(f"Camera found at {port}")
            available_cameras["opencv"].append({
                "index_or_path": port,
                "width": 640, # Default
                "height": 480, # Default
                "fps": 30
            })
            
    # 2. Scan RealSense Cameras
    # We can use rs-enumerate-devices or python library if available.
    # For now, let's assume if we find a device with specific name/id via lsusb or similar
    # Or just try to init RealSense if we have the library.
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            print(f"RealSense found: {name} ({serial})")
            available_cameras["realsense"].append({
                "serial_number_or_name": serial,
                "width": 848, # Default for RealSense
                "height": 480,
                "fps": 30
            })
    except ImportError:
        logger.warning("pyrealsense2 not installed, skipping RealSense scan.")
    except Exception as e:
        logger.error(f"Error scanning RealSense: {e}")

    return available_cameras
