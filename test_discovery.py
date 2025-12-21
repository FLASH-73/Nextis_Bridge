from app.core.camera_discovery import discover_cameras
import logging

# Configure logging to see errors
logging.basicConfig(level=logging.INFO)

print("Testing Camera Discovery...")
cameras = discover_cameras()
print(f"Discovered Cameras: {cameras}")
