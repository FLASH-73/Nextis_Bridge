import sys
import os
from pathlib import Path

# Add project root and lerobot/src to sys.path
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "lerobot" / "src"))

import uvicorn

if __name__ == "__main__":
    # Run the FastAPI app using uvicorn in a loop to support restarts
    # Exit Code 42 = Restart Requested
    import subprocess
    import time
    
    while True:
        print("\n🚀 Starting Backend Server...")
        # We invoke uvicorn via subprocess to verify clean process state on restart
        # We must use sys.executable to ensure we use the same python env
        cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"]
        
        try:
            # Run and wait
            p = subprocess.Popen(cmd)
            p.wait()
            ret = p.returncode
        except KeyboardInterrupt:
            p.terminate()
            print("\n🛑 Server Stopped by User.")
            break
            
        if ret == 42:
            print("\n🔄 SYSTEM RESTART REQUESTED. Rebooting in 1s...")
            time.sleep(1)
            continue
        else:
            print(f"\n🛑 Server exited with code {ret}. Stopping.")
            break
