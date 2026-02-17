import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to sys.path so we can import 'app'
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))
# Add lerobot/src to sys.path
sys.path.insert(0, str(root_path / "lerobot" / "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.state import state
from app.routes import (
    system, arms, motors, calibration, teleop,
    cameras, recording, datasets, training,
    policies, hil, rl, chat, debug, tools,
)

app = FastAPI(title="Nextis Robotics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in [system, arms, motors, calibration, teleop, cameras,
          recording, datasets, training, policies, hil, rl, chat, debug, tools]:
    app.include_router(r.router)


@app.on_event("startup")
async def startup_event():
    import threading
    # Run initialization in background to allow server to start immediately
    t = threading.Thread(target=state.initialize, daemon=True)
    t.start()


@app.on_event("shutdown")
async def shutdown_event():
    state.shutdown()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
