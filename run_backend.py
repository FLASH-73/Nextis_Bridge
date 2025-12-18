import sys
import os
from pathlib import Path

# Add project root and lerobot/src to sys.path
root_path = Path(__file__).parent
sys.path.append(str(root_path))
sys.path.append(str(root_path / "lerobot" / "src"))

import uvicorn

if __name__ == "__main__":
    # Run the FastAPI app using uvicorn
    # reload=True enables auto-reload on code changes
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
