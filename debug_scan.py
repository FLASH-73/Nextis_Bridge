import cv2
import platform
from pathlib import Path
import os

print(f"Platform: {platform.system()}")
print(f"CWD: {os.getcwd()}")

def find_cameras():
    found_cameras_info = []
    targets_to_scan = []

    if platform.system() == "Linux":
        print("Scanning /dev/video*...")
        try:
            possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            targets_to_scan = [str(p) for p in possible_paths]
            print(f"Found paths: {targets_to_scan}")
        except Exception as e:
            print(f"Error globbing /dev: {e}")
    else:
        # Fallback for testing on other platforms (if relevant)
        targets_to_scan = [int(i) for i in range(10)]

    for target in targets_to_scan:
        print(f"Checking {target}...", end="")
        try:
            camera = cv2.VideoCapture(target)
            if camera.isOpened():
                print(" OPENED!")
                width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"  - Res: {width}x{height}")
                
                camera.release()
            else:
                print(" Failed to open.")
        except Exception as e:
            print(f" Exception: {e}")

if __name__ == "__main__":
    find_cameras()
