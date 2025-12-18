import sys
import os
from pathlib import Path

# Mimic what run_backend.py does (or should do)
root_path = Path(__file__).parent
sys.path.append(str(root_path))

# Also add lerobot path if needed, to test if that's the issue
lerobot_path = root_path / "lerobot" / "src"
sys.path.append(str(lerobot_path))

print(f"Sys Path: {sys.path}")

try:
    import app.main
    print("Successfully imported app.main")
except Exception as e:
    import traceback
    traceback.print_exc()
