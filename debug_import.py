import sys
import os
import logging

logging.basicConfig(level=logging.INFO)

# Mock config loading since we don't haven't set up the full app context
from unittest.mock import MagicMock
sys.modules['app.core.config'] = MagicMock()

try:
    from app.core.camera_service import CameraService
    print("SUCCESS: CameraService imported.")
    
    cs = CameraService()
    print("Scanning...")
    cams = cs.scan_cameras()
    print(f"Scan Result: {cams}")
    
except ImportError as e:
    print(f"FAIL: ImportError: {e}")
except Exception as e:
    print(f"FAIL: Other error: {e}")
