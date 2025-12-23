import sys
import os
import logging
from unittest.mock import MagicMock

logging.basicConfig(level=logging.INFO)
sys.modules['app.core.config'] = MagicMock()

try:
    from app.core.camera_service import CameraService
    print("CameraService imported.")
    cs = CameraService()
    
    # Mock config
    cs.get_camera_config = MagicMock(return_value={
        "test_cam": {
            "type": "opencv",
            "index_or_path": "/dev/video4" # Pick one of the known working paths
        }
    })
    
    print("Testing snapshot for 'test_cam' (/dev/video4)...")
    frame = cs.capture_snapshot("test_cam")
    
    if frame is not None:
        print(f"Snapshot SUCCESS. Shape: {frame.shape}")
    else:
        print("Snapshot FAILED (None returned).")

except Exception as e:
    print(f"Error: {e}")
