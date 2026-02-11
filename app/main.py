import sys
import os
from dotenv import load_dotenv
from pathlib import Path
import json

# DIAGNOSTIC: Print immediately on import to verify stdout works
print("\n" + "="*60)
print("MAIN.PY LOADED - If you see this, stdout is working!")
print("="*60 + "\n")
sys.stdout.flush()

# Load environment variables from .env file
load_dotenv()

# DIAGNOSTIC: Set up file logging for recording debug
import logging
_log_file = Path(__file__).parent.parent / "recording_debug.log"
_file_handler = logging.FileHandler(_log_file, mode='a')
_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
_recording_logger = logging.getLogger("recording_debug")
_recording_logger.setLevel(logging.DEBUG)
_recording_logger.addHandler(_file_handler)
_recording_logger.info("=== Backend Started ===")
print(f"Recording debug log: {_log_file}")

# Add the project root to sys.path so we can import 'app'
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))
# Add lerobot/src to sys.path
sys.path.insert(0, str(root_path / "lerobot" / "src"))

from fastapi import FastAPI, WebSocket, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.core.config import load_config
from app.core.orchestrator import TaskOrchestrator
from app.core.recorder import DataRecorder
from app.core.calibration_service import CalibrationService
from app.core.teleop_service import TeleoperationService
from app.core.dataset_service import DatasetService
from app.core.training_service import TrainingService

app = FastAPI(title="Nextis Robotics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import threading
import time
import asyncio

from app.core.camera_service import CameraService

# Global State
class SystemState:
    def __init__(self):
        self.robot = None
        self.leader = None
        self.recorder = None
        self.orchestrator = None
        self.calibration_service = None
        self.camera_service = None
        self.teleop_service = None
        self.dataset_service = None
        self.training_service = None
        self.hil_service = None
        self.reward_classifier_service = None
        self.gvl_reward_service = None
        self.sarm_reward_service = None
        self.rl_service = None
        self.arm_registry = None  # Arm Manager Service
        self.leader_assists = {} # {arm_prefix: LeaderAssistService}
        self.lock = threading.Lock()

        self.is_initializing = False
        self.init_error = None


    def initialize(self):
        self.is_initializing = True
        self.init_error = None
        try:
             self._inner_initialize()
        except Exception as e:
             import traceback
             print(f"CRITICAL INIT ERROR: {e}")
             traceback.print_exc()
             self.init_error = str(e)
        finally:
             self.is_initializing = False
             print("System Initialization Complete.")

    def _inner_initialize(self):
        print("Initializing System (Fast Startup — no hardware)...")

        # 1. Lightweight services (no hardware deps)
        self.camera_service = CameraService()
        self.dataset_service = DatasetService()
        self.training_service = TrainingService()

        # 2. Load config
        config_data = load_config()

        # 3. Arm Registry (reads config only, no hardware)
        try:
            from app.core.arm_registry import ArmRegistryService
            self.arm_registry = ArmRegistryService(config_path="app/config/settings.yaml")
            print(f"Arm Registry initialized: {self.arm_registry.get_status_summary()}")
        except Exception as e:
            print(f"Warning: Arm Registry init failed: {e}")
            self.arm_registry = None

        # 4. Data Recorder
        self.recorder = DataRecorder(repo_id="roberto/nextis_data", robot_type="bi_umbra_follower")

        # 5. CalibrationService (works with robot=None, uses arm_registry)
        self.calibration_service = CalibrationService(
            robot=None, leader=None, robot_lock=self.lock,
            arm_registry=self.arm_registry
        )

        # 6. TeleoperationService (works with robot=None, uses arm_registry)
        self.teleop_service = TeleoperationService(
            robot=None, leader=None, robot_lock=self.lock,
            leader_assists={}, arm_registry=self.arm_registry
        )

        # 7. Orchestrator with minimal mock robot
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.is_connected = False
        mock.is_mock = True
        mock.is_calibrated = True
        mock.robot_type = "mock_robot"
        mock.observation_features = {}
        mock.action_features = {}
        self.robot = mock

        self.orchestrator = TaskOrchestrator(self.robot, self.recorder, robot_lock=self.lock)
        self.orchestrator.start()

        # 8. Reward / RL services (lightweight inits)
        from app.core.reward_classifier_service import RewardClassifierService
        self.reward_classifier_service = RewardClassifierService()
        from app.core.gvl_reward_service import GVLRewardService
        self.gvl_reward_service = GVLRewardService()
        from app.core.sarm_reward_service import SARMRewardService
        self.sarm_reward_service = SARMRewardService()

        # 9. HIL Service
        from app.core.hil_service import HILService
        self.hil_service = HILService(
            teleop_service=self.teleop_service,
            orchestrator=self.orchestrator,
            training_service=self.training_service,
            robot_lock=self.lock
        )

        # NOTE: Planner is lazy-loaded on first /chat request
        self.planner = None

        print("System ready (connect arms via UI when needed)")

    def reload(self):
        print("Reloading System...")
        # Stop Orchestrator first to stop using the robot
        if self.orchestrator:
            self.orchestrator.stop()
            
        with self.lock:
            # Disconnect Robot
            if self.robot:
                try:
                    if hasattr(self.robot, 'disconnect'):
                        self.robot.disconnect()
                except Exception as e:
                    print(f"Error disconnecting robot: {e}")
            
            # Disconnect Leader
            if self.leader:
                try:
                    if hasattr(self.leader, 'disconnect'):
                        self.leader.disconnect()
                except Exception as e:
                    print(f"Error disconnecting leader: {e}")
            
            # Re-initialize
            import time
            time.sleep(2)
            self.initialize()

    def shutdown(self):
        print("Shutting Down System State...")
        if self.orchestrator:
            self.orchestrator.stop()
            
        with self.lock:
            # Disconnect Robot
            if self.robot:
                try:
                    if hasattr(self.robot, 'disconnect'):
                        self.robot.disconnect()
                except Exception as e:
                    print(f"Error disconnecting robot: {e}")
            
            # Disconnect Leader
            if self.leader:
                try:
                    if hasattr(self.leader, 'disconnect'):
                        self.leader.disconnect()
                except Exception as e:
                    print(f"Error disconnecting leader: {e}")
        
        # Brief pause to ensure OS releases handles
        import time
        time.sleep(0.5)

    def restart(self):
        print("SYSTEM RESTART REQUESTED. RESPRAINING PROCESS...")
        import sys
        import time
        import os
        
        # Flush buffers
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Close hardware connections safely if possible
        try:
             self.shutdown() # Clean shutdown (no re-init)
        except Exception as e:
             print(f"Error during shutdown: {e}")
             
        # os._exit(42) forces exit without cleanup/exception handling, guaranteeing the return code
        import os
        os._exit(42)

system = SystemState()

@app.on_event("startup")
async def startup_event():
    import threading
    # Run initialization in background to allow server to start immediately
    t = threading.Thread(target=system.initialize, daemon=True)
    t.start()

@app.on_event("shutdown")
async def shutdown_event():
    if system.orchestrator:
        system.orchestrator.stop()



@app.post("/system/restart")
def restart_system(background_tasks: BackgroundTasks):
    # Queue restart to allow response to send first
    background_tasks.add_task(delayed_restart)
    return {"status": "restarting", "message": "System is restarting..."}
    
def delayed_restart():
    """Waits briefly then forces a system restart."""
    import time
    time.sleep(0.5)
    system.restart()

@app.post("/system/reconnect")
def reconnect_system(background_tasks: BackgroundTasks):
    """Reinitialize services (arms are connected via UI)."""
    if system.is_initializing:
        return {"status": "busy", "message": "Already initializing..."}
    print("Manual Reconnect Requested — reinitializing services...")
    background_tasks.add_task(system.initialize)
    return {"status": "initializing", "message": "Reinitializing services..."}

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "online", "service": "nextis-robotics"}

@app.get("/test/ping")
def test_ping():
    """Simple test endpoint to verify API connection."""
    print(">>> /test/ping called - API is working!")
    _recording_logger.info("PING received - API connection verified")
    sys.stdout.flush()
    return {"status": "pong", "message": "API connection verified"}


@app.get("/status")
def get_status():
    # 1. Connection State
    connection = "DISCONNECTED"
    if system.is_initializing:
        connection = "INITIALIZING"
    elif system.init_error:
        connection = "ERROR"
    elif system.robot:
        if getattr(system.robot, 'is_mock', False):
            connection = "MOCK"
        elif system.robot.is_connected:
            connection = "CONNECTED"
            
    # 2. Execution State
    execution = "IDLE"
    status_text = "READY" # Default text
    
    if system.orchestrator:
        if system.orchestrator.active_policy:
            execution = "EXECUTING"
            status_text = f"BUSY: {system.orchestrator.task_chain[system.orchestrator.current_task_index]}"
        elif system.orchestrator.intervention_engine.is_human_controlling:
            execution = "INTERVENTION"
            status_text = "RECORDING"
            
    # 3. Overall System Status Label (Legacy + UI)
    if connection == "INITIALIZING":
        status_text = "STARTING..."
    elif connection == "ERROR":
        status_text = "ERROR"
    elif connection == "DISCONNECTED":
        status_text = "OFFLINE"
    elif connection == "MOCK":
        status_text = "MOCK MODE"
    # Else if CONNECTED + IDLE -> READY
            
    return {
        "status": status_text,      # Legacy support for frontend simple check
        "connection": connection,   # CONNECTED, DISCONNECTED, MOCK, INITIALIZING, ERROR
        "execution": execution,     # IDLE, EXECUTING, INTERVENTION
        "error": system.init_error,
        "left_arm": connection,     # Simplification for now
        "right_arm": connection,
        "fps": 30.0 if connection == "CONNECTED" else 0.0
    }

@app.get("/config")
def get_config():
    return load_config()

@app.post("/execute")
async def execute_endpoint(request: Request, background_tasks: BackgroundTasks):
    if not system.orchestrator:
        return {"status": "error", "message": "Orchestrator not initialized"}
    data = await request.json()
    plan = data.get("plan")
    if not plan:
        return {"status": "error", "message": "No plan provided"}

    background_tasks.add_task(system.orchestrator.execute_plan, plan)
    return {"status": "success", "message": "Execution started"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_msg = data.get("message", "").lower()
    messages = data.get("messages", []) # Full conversation history

    # Lazy-load planner on first use
    if system.planner is None:
        try:
            from app.core.planner import LocalPlanner, GeminiPlanner
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key and gemini_key.strip():
                print("Lazy-loading GeminiPlanner...")
                system.planner = GeminiPlanner(api_key=gemini_key)
            else:
                print("Lazy-loading LocalPlanner (Qwen)...")
                system.planner = LocalPlanner("Qwen/Qwen2.5-7B-Instruct", device="cuda")
        except Exception as e:
            print(f"Warning: Planner load failed: {e}")

    # Simple keyword matching for MVP
    actions = []
    response = ""
    plan = []

    if "stop" in user_msg:
        actions = []
        response = "Stopping all tasks."
        if system.orchestrator:
            system.orchestrator.stop()
            system.orchestrator.task_chain = []
    else:
        if system.planner:
            print(f"Planning for: {user_msg}")
            
            # Use history if available, otherwise just the message
            input_data = messages if messages else user_msg
            
            plan = system.planner.plan(input_data)
            print(f"Task chain loaded: {json.dumps(plan, indent=2)}")
            
            # Orchestrator expects a list of strings for now, or we need to upgrade Orchestrator
            # For now, let's extract the task names or format them
            actions = [p.get("task") for p in plan]
            
            # CRITICAL: The response content MUST be the JSON string so it gets added to the chat history
            # this allows the planner to 'see' its previous plans in future turns.
            response = json.dumps(plan)
        else:
            # Fallback
            actions = ["move_to_bin", "pick_object", "place_in_box"]
            response = "Planner not ready. Using dummy plan."

    # Send tasks to orchestrator
    if actions and system.orchestrator:
        system.orchestrator.load_task_chain(actions)
    
    return {
        "reply": response, # Frontend expects 'reply'
        "response": response, # Legacy
        "actions": actions,
        "plan": plan if system.planner else [] # For visualizer
    }

from fastapi.responses import StreamingResponse
import cv2
import io
import time

def generate_frames(camera_key: str):
    while True:
        frame = None
        
        # PRIORITY 1: Direct Live Feed from Robot (Low Latency, Always Fresh)
        # Note: Don't check robot.is_connected - it fails if ANY camera has issues,
        # blocking ALL cameras. Let individual async_read handle connection gracefully.
        if system.robot and system.robot.cameras and camera_key in system.robot.cameras:
            cam = system.robot.cameras[camera_key]
            try:
                frame = cam.async_read(blocking=False)  # ZOH pattern: return cached frame immediately
                if frame is None:
                    # occasional log?
                    pass
            except Exception:
                pass

        # PRIORITY 2: Orchestrator Observation (What the Agent sees) - Fallback
        if frame is None and system.orchestrator and system.orchestrator.intervention_engine:
            obs = system.orchestrator.intervention_engine.latest_observation
            if obs:
                # Try explicit key then partial match
                full_key = f"observation.images.{camera_key}"
                if full_key in obs:
                    frame = obs[full_key]
                elif camera_key in obs:
                    frame = obs[camera_key]
                else:
                    for k in obs.keys():
                        if camera_key in k:
                            frame = obs[k]
                            break
                            
        # PRIORITY 3: Snapshot Fallback (always try, even if robot has camera)
        # Note: If robot's camera thread has device open, snapshot might fail (device busy)
        # but we still try as a last resort before showing "Waiting..."
        if frame is None and system.camera_service:
            snapshot = system.camera_service.capture_snapshot(camera_key)
            if snapshot is not None:
                frame = snapshot
        
        if frame is None:
            # Yield placeholder
            import numpy as np
            blank_image = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_image, f"Waiting for {camera_key}...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame = blank_image
        else:
            # Convert PyTorch tensor to numpy if needed
            import torch
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).cpu().numpy() * 255
                frame = frame.astype("uint8")
                # RGB to BGR for OpenCV encoding
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.016)  # ~60 FPS streaming for lower latency

@app.get("/video_feed/{camera_key}")
def video_feed(camera_key: str):
    return StreamingResponse(generate_frames(camera_key), media_type="multipart/x-mixed-replace; boundary=frame")

# --- Calibration Endpoints ---

@app.get("/calibration/arms")
def get_calibration_arms():
    # Stop Orchestrator to free up the robot bus for calibration
    if system.orchestrator and system.orchestrator.is_running:
        print("Stopping Orchestrator for Calibration...")
        system.orchestrator.stop()
        
    if not system.calibration_service:
        return {"arms": []}
    return {"arms": system.calibration_service.get_arms()}

@app.get("/calibration/{arm_id}/state")
def get_calibration_state(arm_id: str):
    if not system.calibration_service:
        return {"state": []}
    return {"state": system.calibration_service.get_calibration_state(arm_id)}

@app.post("/teleop/start")
async def start_teleop(request: Request):
    try:
        data = await request.json()
    except:
        data = {}
        
    force = data.get("force", False)
    active_arms = data.get("active_arms", None) 
    
    # Lazy Initialize Teleop Service if needed
    if not system.teleop_service:
        if system.robot:
             from app.core.teleop_service import TeleoperationService
             system.teleop_service = TeleoperationService(system.robot, system.leader, system.lock, leader_assists=system.leader_assists, arm_registry=system.arm_registry)
        else:
             return {"status": "error", "message": "Teleop Service not initialized: Robot not connected"}

    if system.teleop_service:
        try:
            system.teleop_service.start(force=force, active_arms=active_arms)
        except Exception as e:
            print(f"Teleop Start Failed: {e}")
            return {"status": "error", "message": str(e)}
            
    return {"status": "started"}

@app.post("/teleop/stop")
def stop_teleop():
    if system.teleop_service:
        system.teleop_service.stop()
    return {"status": "stopped"}
    
@app.get("/teleop/data")
def get_teleop_data():
    if system.teleop_service:
        return system.teleop_service.get_data()
    return {"history": [], "torque": {}}

# --- Recording Endpoints ---

@app.get("/recording/options")
def get_recording_options():
    """Returns available cameras and arm pairs for recording selection."""
    cameras = []
    arms = []

    # Get cameras from robot or camera config
    if system.robot and system.robot.is_connected:
        if hasattr(system.robot, 'cameras') and system.robot.cameras:
            for cam_key in sorted(system.robot.cameras.keys()):
                cameras.append({
                    "id": cam_key,
                    "name": cam_key.replace("_", " ").title()
                })

        # Get arm pairs - check for bi-arm setup
        if hasattr(system.robot, 'left_arm') or hasattr(system.robot, 'right_arm'):
            if hasattr(system.robot, 'left_arm') and system.robot.left_arm:
                arms.append({"id": "left", "name": "Left Arm", "joints": 7})
            if hasattr(system.robot, 'right_arm') and system.robot.right_arm:
                arms.append({"id": "right", "name": "Right Arm", "joints": 7})
        else:
            # Single arm or default setup
            arms.append({"id": "default", "name": "Robot Arm", "joints": 7})
    else:
        # Fallback: try to get cameras from config
        try:
            cam_configs = camera_service.get_camera_config()
            for cam in cam_configs:
                cameras.append({
                    "id": cam.get("id", "unknown"),
                    "name": cam.get("id", "unknown").replace("_", " ").title()
                })
        except:
            pass

    return {"cameras": cameras, "arms": arms}

@app.post("/recording/session/start")
async def start_recording_session(request: Request):
    print("\n>>> API: /recording/session/start called")
    _recording_logger.info("API: /recording/session/start called")
    sys.stdout.flush()

    data = await request.json()
    repo_id = data.get("repo_id")
    task = data.get("task")
    selected_cameras = data.get("selected_cameras")  # list of camera IDs or None (all)
    selected_arms = data.get("selected_arms")        # list of arm IDs ("left", "right") or None (all)
    print(f"    repo_id={repo_id}, task={task}, cameras={selected_cameras}, arms={selected_arms}")
    _recording_logger.info(f"  repo_id={repo_id}, task={task}, cameras={selected_cameras}, arms={selected_arms}")

    if not repo_id or not task:
        print("    ERROR: Missing repo_id or task")
        _recording_logger.error("Missing repo_id or task")
        return {"status": "error", "message": "Missing repo_id or task"}

    if not system.teleop_service:
        print("    ERROR: Teleop Service not active")
        _recording_logger.error("Teleop Service not active")
        return {"status": "error", "message": "Teleop Service not active"}

    try:
        _recording_logger.info("Calling teleop_service.start_recording_session...")
        system.teleop_service.start_recording_session(
            repo_id, task,
            selected_cameras=selected_cameras,
            selected_arms=selected_arms
        )
        episode_count = system.teleop_service.episode_count
        print(f"    SUCCESS: Session started (episode_count={episode_count})")
        _recording_logger.info(f"SUCCESS: Session started (episode_count={episode_count})")
    except Exception as e:
        import traceback
        print(f"    ERROR: {e}")
        _recording_logger.error(f"ERROR: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

    return {"status": "success", "message": "Recording Session Started", "episode_count": episode_count}

@app.post("/recording/session/stop")
def stop_recording_session():
    print("\n>>> API: /recording/session/stop called")
    _recording_logger.info("API: /recording/session/stop called")
    sys.stdout.flush()

    if not system.teleop_service:
        print("    ERROR: Teleop Service not active")
        _recording_logger.error("Teleop Service not active")
        return {"status": "error", "message": "Teleop Service not active"}

    try:
        system.teleop_service.stop_recording_session()
        print("    SUCCESS: Session stopped")
        _recording_logger.info("SUCCESS: Session stopped")
    except Exception as e:
        import traceback
        _recording_logger.error(f"ERROR: {e}\n{traceback.format_exc()}")

    return {"status": "success", "message": "Recording Session Finalized"}

@app.post("/recording/episode/start")
def start_episode():
    print("\n>>> API: /recording/episode/start called")
    _recording_logger.info("API: /recording/episode/start called")
    sys.stdout.flush()

    if not system.teleop_service:
        print("    ERROR: Teleop Service not active")
        _recording_logger.error("Teleop Service not active")
        return {"status": "error", "message": "Teleop Service not active"}

    try:
        system.teleop_service.start_episode()
        print("    SUCCESS: Episode started")
        _recording_logger.info("SUCCESS: Episode started")
    except Exception as e:
        import traceback
        print(f"    ERROR: {e}")
        _recording_logger.error(f"ERROR: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

    return {"status": "success", "message": "Episode Started"}

@app.post("/recording/episode/stop")
def stop_episode():
    print("\n>>> API: /recording/episode/stop called")
    _recording_logger.info("API: /recording/episode/stop called")
    sys.stdout.flush()

    if not system.teleop_service:
        print("    ERROR: Teleop Service not active")
        _recording_logger.error("Teleop Service not active")
        return {"status": "error", "message": "Teleop Service not active"}

    try:
        system.teleop_service.stop_episode()
        episode_count = system.teleop_service.episode_count
        print(f"    SUCCESS: Episode stopped (total: {episode_count})")
        _recording_logger.info(f"SUCCESS: Episode stopped (total: {episode_count})")
        return {"status": "success", "message": "Episode Saved", "episode_count": episode_count}
    except Exception as e:
        import traceback
        _recording_logger.error(f"ERROR: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

@app.delete("/recording/episode/last")
def delete_last_episode():
    if not system.teleop_service:
         return {"status": "error", "message": "Teleop Service not active"}

    if not system.teleop_service.session_active:
         return {"status": "error", "message": "No recording session active"}

    if not system.teleop_service.dataset:
         return {"status": "error", "message": "No dataset loaded"}

    try:
        repo_id = system.teleop_service.dataset.repo_id
        current_count = system.teleop_service.episode_count

        if current_count <= 0:
            return {"status": "error", "message": "No episodes to delete"}

        # Delete the last episode (index = count - 1)
        last_index = current_count - 1

        print(f"[DELETE_LAST] Starting delete for episode {last_index}")
        print(f"[DELETE_LAST] BEFORE: episode_count={current_count}, meta.total_episodes={system.teleop_service.dataset.meta.total_episodes}")

        # CRITICAL: Flush pending episode data to disk BEFORE deletion
        # Without this, the metadata_buffer may have unflushed episode data
        # that won't be found on disk by delete_episode(), causing ghost episodes
        system.teleop_service.sync_to_disk()

        result = system.dataset_service.delete_episode(repo_id, last_index)
        print(f"[DELETE_LAST] delete_episode returned: {result}")

        # Refresh metadata from disk AFTER deletion to reload clean state
        system.teleop_service.refresh_metadata_from_disk()

        print(f"[DELETE_LAST] AFTER: episode_count={system.teleop_service.episode_count}, meta.total_episodes={system.teleop_service.dataset.meta.total_episodes}")

        return {"status": "success", "message": "Last Episode Deleted", "episode_count": system.teleop_service.episode_count}
    except Exception as e:
        import traceback
        print(f"[DELETE_LAST] ERROR: {e}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

# --- Dataset Viewer Endpoints ---

@app.get("/datasets")
def list_datasets():
    if not system.dataset_service:
         return []
    return system.dataset_service.list_datasets()

@app.get("/datasets/{repo_id:path}/episodes")
def get_dataset_episodes(repo_id: str):
    if not system.dataset_service:
         return []
    try:
        dataset = system.dataset_service.get_dataset(repo_id)
        episodes = dataset.meta.episodes
        if hasattr(episodes, "to_pydict"):
             d = episodes.to_pydict()
             keys = list(d.keys())
             length = len(d[keys[0]])
             return [{k: d[k][i] for k in keys} for i in range(length)]
        elif isinstance(episodes, list): 
             return episodes
        else:
             return [{"index": i} for i in range(dataset.num_episodes)]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/datasets/{repo_id:path}/episode/{index}")
def get_episode_detail(repo_id: str, index: int):
    if not system.dataset_service:
         return {}
    try:
        return system.dataset_service.get_episode_data(repo_id, index)
    except Exception as e:
         return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/datasets/{repo_id:path}/episode/{index}")
def delete_episode_endpoint(repo_id: str, index: int):
    if not system.dataset_service:
         return {"status": "error", "message": "Dataset service not ready"}
    try:
        print(f"[DELETE_EP] Deleting episode {index} from {repo_id}")

        # Check if session is active on this dataset
        session_matches = (system.teleop_service and
            system.teleop_service.session_active and
            system.teleop_service.dataset and
            system.teleop_service.dataset.repo_id == repo_id)

        print(f"[DELETE_EP] Session active on this dataset: {session_matches}")
        if session_matches:
            print(f"[DELETE_EP] BEFORE: episode_count={system.teleop_service.episode_count}, meta.total_episodes={system.teleop_service.dataset.meta.total_episodes}")

        # CRITICAL: Flush pending data BEFORE deletion if session is active on this dataset
        # Without this, the metadata_buffer may have unflushed episode data
        if session_matches:
            system.teleop_service.sync_to_disk()

        result = system.dataset_service.delete_episode(repo_id, index)
        print(f"[DELETE_EP] delete_episode returned: {result}")

        # Refresh metadata from disk AFTER deletion if session is active
        if session_matches:
            system.teleop_service.refresh_metadata_from_disk()
            print(f"[DELETE_EP] AFTER: episode_count={system.teleop_service.episode_count}, meta.total_episodes={system.teleop_service.dataset.meta.total_episodes}")

        return result
    except Exception as e:
         return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/datasets/{repo_id:path}")
def delete_dataset_endpoint(repo_id: str):
    """Delete an entire dataset repository."""
    if not system.dataset_service:
         return {"status": "error", "message": "Dataset service not ready"}
    try:
        result = system.dataset_service.delete_dataset(repo_id)
        return result
    except FileNotFoundError as e:
         return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
         return JSONResponse(status_code=500, content={"error": str(e)})

# --- Dataset Merge Endpoints ---

class MergeValidateRequest(BaseModel):
    repo_ids: list[str]

class MergeStartRequest(BaseModel):
    repo_ids: list[str]
    output_repo_id: str

@app.post("/datasets/merge/validate")
async def validate_merge(request: MergeValidateRequest):
    """Validate that datasets can be merged (same fps, robot_type, features)."""
    if not system.dataset_service:
        return JSONResponse(status_code=503, content={"error": "Dataset service not ready"})

    try:
        result = system.dataset_service.validate_merge(request.repo_ids)
        return {
            "compatible": result.compatible,
            "datasets": result.datasets,
            "merged_info": result.merged_info,
            "errors": result.errors,
            "warnings": result.warnings
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/datasets/merge/start")
async def start_merge(request: MergeStartRequest):
    """Start a background merge job."""
    if not system.dataset_service:
        return JSONResponse(status_code=503, content={"error": "Dataset service not ready"})

    # Validate first
    validation = system.dataset_service.validate_merge(request.repo_ids)
    if not validation.compatible:
        return JSONResponse(status_code=400, content={
            "error": "Datasets are not compatible for merge",
            "details": validation.errors
        })

    # Check output name doesn't exist
    output_path = system.dataset_service.base_path / request.output_repo_id
    if output_path.exists():
        return JSONResponse(status_code=400, content={
            "error": f"Dataset '{request.output_repo_id}' already exists"
        })

    try:
        job = system.dataset_service.start_merge_job(request.repo_ids, request.output_repo_id)
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "message": "Merge job started"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/datasets/merge/status/{job_id}")
def get_merge_status(job_id: str):
    """Get status of a merge job."""
    if not system.dataset_service:
        return JSONResponse(status_code=503, content={"error": "Dataset service not ready"})

    job = system.dataset_service.get_merge_job_status(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "error": job.error,
        "output_repo_id": job.output_repo_id,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None
    }

from fastapi.responses import FileResponse, Response
from fastapi import HTTPException

# --- Training Endpoints ---

@app.post("/training/validate")
async def validate_training(request: Request):
    """Validate a dataset for compatibility with a policy type."""
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    data = await request.json()
    dataset_repo_id = data.get("dataset_repo_id")
    policy_type = data.get("policy_type", "smolvla")

    if not dataset_repo_id:
        return JSONResponse(status_code=400, content={"error": "dataset_repo_id is required"})

    result = system.training_service.validate_dataset(dataset_repo_id, policy_type)
    return result.to_dict()

@app.post("/training/start")
async def start_training(request: Request, background_tasks: BackgroundTasks):
    """Start a new training job."""
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    data = await request.json()
    dataset_repo_id = data.get("dataset_repo_id")
    policy_type = data.get("policy_type", "smolvla")
    config = data.get("config", {})

    if not dataset_repo_id:
        return JSONResponse(status_code=400, content={"error": "dataset_repo_id is required"})

    try:
        # Create the job
        job = system.training_service.create_job(dataset_repo_id, policy_type, config)

        # Start training
        system.training_service.start_job(job.id)

        return {"status": "started", "job_id": job.id, "message": f"Training job {job.id} started"}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/training/jobs")
def list_training_jobs():
    """List all training jobs."""
    if not system.training_service:
        return []
    return system.training_service.list_jobs()

@app.get("/training/jobs/{job_id}")
def get_training_job(job_id: str):
    """Get status and progress of a specific training job."""
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    try:
        return system.training_service.get_job_status(job_id)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

@app.get("/training/jobs/{job_id}/logs")
def get_training_logs(job_id: str, offset: int = 0, limit: int = 100):
    """Get logs for a training job."""
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    try:
        return system.training_service.get_job_logs(job_id, offset, limit)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

@app.post("/training/jobs/{job_id}/cancel")
def cancel_training_job(job_id: str):
    """Cancel a running training job."""
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    try:
        system.training_service.cancel_job(job_id)
        return {"status": "cancelled", "job_id": job_id}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/training/presets")
def get_training_presets(policy_type: str = "smolvla"):
    """Get available training presets for a policy type."""
    if not system.training_service:
        return {}
    return system.training_service.get_presets(policy_type)

@app.get("/training/hardware")
def get_training_hardware():
    """Detect available training hardware (CUDA, MPS, CPU)."""
    if not system.training_service:
        return {
            "devices": [{"id": "cpu", "type": "cpu", "name": "CPU", "memory_gb": None, "recommended": True}],
            "default": "cpu"
        }
    return system.training_service.detect_hardware()


@app.get("/training/dataset/{repo_id:path}/quantiles")
def check_dataset_quantiles(repo_id: str):
    """Check if a dataset has quantile statistics needed for Pi0.5 training."""
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    return system.training_service.has_quantile_stats(repo_id)


@app.post("/training/dataset/{repo_id:path}/compute-quantiles")
def compute_dataset_quantiles(repo_id: str):
    """Compute quantile statistics for a dataset (required for Pi0.5 with default normalization).

    This can take several minutes for large datasets.
    """
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    # Run synchronously for now (could be made async with job tracking)
    result = system.training_service.compute_quantile_stats(repo_id)
    return result


# --- Policy Management Endpoints ---

@app.get("/policies")
def list_policies():
    """List all trained policies."""
    if not system.training_service:
        return []
    policies = system.training_service.list_policies()
    return [p.to_dict() for p in policies]


@app.get("/policies/{policy_id}")
def get_policy(policy_id: str):
    """Get details of a specific policy."""
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    policy = system.training_service.get_policy(policy_id)
    if not policy:
        return JSONResponse(status_code=404, content={"error": f"Policy {policy_id} not found"})

    return policy.to_dict()


@app.delete("/policies/{policy_id}")
def delete_policy(policy_id: str):
    """Delete a policy and its output directory."""
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    try:
        system.training_service.delete_policy(policy_id)
        return {"status": "deleted", "policy_id": policy_id}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.patch("/policies/{policy_id}")
async def rename_policy(policy_id: str, request: Request):
    """Rename a policy."""
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    try:
        data = await request.json()
        new_name = data.get("name")
        if not new_name:
            return JSONResponse(status_code=400, content={"error": "Missing 'name' in request body"})

        system.training_service.rename_policy(policy_id, new_name)
        return {"status": "updated", "policy_id": policy_id, "name": new_name}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/policies/{policy_id}/deploy")
def deploy_policy(policy_id: str):
    """Deploy a policy for autonomous execution."""
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    policy = system.training_service.get_policy(policy_id)
    if not policy:
        return JSONResponse(status_code=404, content={"error": f"Policy {policy_id} not found"})

    if not policy.checkpoint_path:
        return JSONResponse(status_code=400, content={"error": "Policy has no checkpoint to deploy"})

    # Deploy via orchestrator if available
    if system.orchestrator:
        try:
            system.orchestrator.deploy_policy(policy.checkpoint_path)
            return {"status": "deployed", "policy_id": policy_id, "checkpoint_path": policy.checkpoint_path}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Failed to deploy: {str(e)}"})
    else:
        # Just return success - policy path can be used later
        return {"status": "ready", "policy_id": policy_id, "checkpoint_path": policy.checkpoint_path}


@app.get("/policies/{policy_id}/config")
def get_policy_config(policy_id: str):
    """Get the input/output configuration of a trained policy.

    Returns which cameras and arms the policy was trained on.
    Useful for configuring HIL deployment to show only relevant cameras.
    """
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    try:
        config = system.training_service.get_policy_config(policy_id)
        if not config:
            return JSONResponse(status_code=404, content={"error": f"Policy {policy_id} not found or no config available"})
        return config.to_dict()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/policies/{policy_id}/resume")
async def resume_policy_training(policy_id: str, request: Request):
    """Resume training a policy from its last checkpoint."""
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    try:
        data = await request.json()
        additional_steps = data.get("additional_steps", 10000)

        job = system.training_service.resume_training(policy_id, additional_steps)
        return {"status": "started", "job_id": job.id, "message": f"Resumed training for {additional_steps} additional steps"}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- HIL (Human-in-the-Loop) Endpoints ---

@app.post("/hil/session/start")
async def start_hil_session(request: Request):
    """Start a HIL deployment session with policy and intervention dataset."""
    if not system.hil_service:
        return JSONResponse(status_code=503, content={"error": "HIL service not initialized"})

    try:
        data = await request.json()
        policy_id = data.get("policy_id")
        intervention_dataset = data.get("intervention_dataset")
        task = data.get("task", "HIL intervention correction")
        movement_scale = data.get("movement_scale", 1.0)

        # Validate movement_scale
        try:
            movement_scale = float(movement_scale)
            movement_scale = max(0.1, min(1.0, movement_scale))
        except (ValueError, TypeError):
            movement_scale = 1.0

        if not policy_id or not intervention_dataset:
            return JSONResponse(status_code=400, content={"error": "policy_id and intervention_dataset required"})

        result = system.hil_service.start_session(policy_id, intervention_dataset, task, movement_scale)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/hil/session/stop")
def stop_hil_session():
    """Stop the current HIL session and finalize recording."""
    if not system.hil_service:
        return JSONResponse(status_code=503, content={"error": "HIL service not initialized"})

    try:
        return system.hil_service.stop_session()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/hil/episode/start")
def start_hil_episode():
    """Start a new HIL episode (begin recording)."""
    if not system.hil_service:
        return JSONResponse(status_code=503, content={"error": "HIL service not initialized"})

    try:
        return system.hil_service.start_episode()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/hil/episode/stop")
def stop_hil_episode():
    """Stop current HIL episode and save data."""
    if not system.hil_service:
        return JSONResponse(status_code=503, content={"error": "HIL service not initialized"})

    try:
        return system.hil_service.stop_episode()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/hil/episode/next")
def next_hil_episode():
    """
    Stop current episode and immediately start next one.

    Used when human finishes intervention and wants robot to try again.
    Saves the current episode data, then starts a new autonomous episode.
    """
    if not system.hil_service:
        return JSONResponse(status_code=503, content={"error": "HIL service not initialized"})

    try:
        # 1. Stop and save current episode
        stop_result = system.hil_service.stop_episode()

        # 2. Start new episode
        start_result = system.hil_service.start_episode()

        return {
            "status": "next_episode_started",
            "previous_episode": stop_result,
            "new_episode": start_result
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/hil/resume")
def resume_hil_autonomous():
    """
    Explicitly resume autonomous mode after intervention pause.

    Called when user clicks "Resume Autonomous" button after intervention
    ends and system is in PAUSED state.
    """
    if not system.hil_service:
        return JSONResponse(status_code=503, content={"error": "HIL service not initialized"})

    try:
        return system.hil_service.resume_autonomous()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/hil/status")
def get_hil_status():
    """Get current HIL session status."""
    if not system.hil_service:
        return {"active": False}

    return system.hil_service.get_status()


@app.patch("/hil/settings")
async def update_hil_settings(request: Request):
    """Update HIL settings during an active session (e.g., movement_scale)."""
    if not system.hil_service:
        return JSONResponse(status_code=503, content={"error": "HIL service not initialized"})

    if not system.hil_service.state.active:
        return JSONResponse(status_code=400, content={"error": "No active HIL session"})

    try:
        data = await request.json()

        # Update movement_scale if provided
        if "movement_scale" in data:
            try:
                scale = float(data["movement_scale"])
                scale = max(0.1, min(1.0, scale))
                system.hil_service.state.movement_scale = scale
            except (ValueError, TypeError):
                pass

        return {
            "status": "updated",
            "movement_scale": system.hil_service.state.movement_scale
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/hil/retrain")
async def trigger_hil_retrain(request: Request):
    """Trigger retraining on intervention data."""
    if not system.hil_service:
        return JSONResponse(status_code=503, content={"error": "HIL service not initialized"})

    try:
        data = await request.json()
    except:
        data = {}

    try:
        return system.hil_service.trigger_retrain(config=data.get("config"))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/datasets/{repo_id:path}/video/{index}/{key}")
def stream_video(repo_id: str, index: int, key: str):
    """Stream video for an episode, handling LeRobot's concatenated video files."""
    if not system.dataset_service:
         return Response(status_code=404)

    # CORS headers for cross-origin video canvas access
    cors_headers = {"Access-Control-Allow-Origin": "*"}

    try:
        import pandas as pd
        dataset_root = system.dataset_service.base_path / repo_id
        video_root = dataset_root / "videos" / key

        # Try to load episode metadata to get correct file_index
        # LeRobot concatenates all episodes into a single file (file-000.mp4)
        # and tracks each episode's position via from_timestamp/to_timestamp
        file_index = index  # Default: assume episode_index == file_index
        chunk_index = 0

        episodes_path = dataset_root / "meta" / "episodes"
        if episodes_path.exists():
            try:
                episodes_df = pd.read_parquet(episodes_path)
                episode_row = episodes_df[episodes_df["episode_index"] == index]
                if not episode_row.empty:
                    # Get the actual file_index from metadata (usually 0 for concatenated videos)
                    file_index_col = f"videos/{key}/file_index"
                    chunk_index_col = f"videos/{key}/chunk_index"
                    if file_index_col in episode_row.columns:
                        file_index = int(episode_row[file_index_col].iloc[0])
                    if chunk_index_col in episode_row.columns:
                        chunk_index = int(episode_row[chunk_index_col].iloc[0])
            except Exception as e:
                pass  # Fall back to using episode_index

        # Standard LeRobot v3: videos/{key}/episode_{index}.mp4 (or inside chunks)
        # Check direct first
        direct_path = video_root / f"episode_{index:06d}.mp4"
        if direct_path.exists():
             return FileResponse(direct_path, media_type="video/mp4", headers=cors_headers)

        # LeRobot v3 chunked format: chunk-XXX/file-YYY.mp4
        # Use file_index from metadata (not episode_index!)
        chunk_path = video_root / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
        if chunk_path.exists():
             return FileResponse(chunk_path, media_type="video/mp4", headers=cors_headers)

        # Try with 6-digit file index too
        chunk_path_6 = video_root / f"chunk-{chunk_index:03d}" / f"file-{file_index:06d}.mp4"
        if chunk_path_6.exists():
             return FileResponse(chunk_path_6, media_type="video/mp4", headers=cors_headers)

        # Fallback: try with episode_index as file_index (for older datasets)
        if file_index != index:
            chunk_path_fallback = video_root / "chunk-000" / f"file-{index:03d}.mp4"
            if chunk_path_fallback.exists():
                return FileResponse(chunk_path_fallback, media_type="video/mp4", headers=cors_headers)

        # Last resort: glob for any matching file
        matches = list(video_root.rglob(f"*file-{file_index:03d}.mp4"))
        if not matches:
            matches = list(video_root.rglob(f"*file-{index:03d}.mp4"))
        if matches:
             return FileResponse(matches[0], media_type="video/mp4", headers=cors_headers)

        return Response(content=f"Video not found for episode {index} (file_index={file_index}) in {video_root}", status_code=404)

    except Exception as e:
         return Response(content=str(e), status_code=500)

# --- Cloud Upload Support Endpoints ---

@app.get("/datasets/{repo_id:path}/files")
def list_dataset_files(repo_id: str):
    """List all files in a dataset with their relative paths and sizes for cloud upload."""
    if not system.dataset_service:
        return JSONResponse(status_code=503, content={"error": "Dataset service not ready"})

    try:
        dataset_root = system.dataset_service.base_path / repo_id
        if not dataset_root.exists():
            return JSONResponse(status_code=404, content={"error": f"Dataset {repo_id} not found"})

        files = []
        for dirpath, _, filenames in os.walk(dataset_root):
            for filename in filenames:
                if filename.startswith('.'):
                    continue  # Skip hidden files
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, dataset_root)
                files.append({
                    "path": rel_path,
                    "size": os.path.getsize(full_path)
                })

        return files
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/datasets/{repo_id:path}/file/{file_path:path}")
def get_dataset_file(repo_id: str, file_path: str):
    """Serve a specific file from a dataset for cloud upload."""
    if not system.dataset_service:
        return Response(status_code=503, content="Dataset service not ready")

    try:
        dataset_root = system.dataset_service.base_path / repo_id
        full_path = dataset_root / file_path

        # Security check - ensure path is within dataset root
        if not full_path.resolve().is_relative_to(dataset_root.resolve()):
            return Response(status_code=403, content="Access denied")

        if not full_path.exists():
            return Response(status_code=404, content=f"File not found: {file_path}")

        return FileResponse(full_path)
    except Exception as e:
        return Response(status_code=500, content=str(e))


@app.get("/recording/status")
def get_recording_status():
    if not system.teleop_service:
         return {"active": False, "episode_count": 0}
         
    # TeleopService.get_data() already returns recording info, but explicit endpoint helps too
    return system.teleop_service.get_data().get("recording", {})

@app.post("/emergency/stop")
def emergency_stop():
    print("🚨 EMERGENCY STOP TRIGGERED 🚨")
    errors = []
    
    # 1. Stop Higher Level Logic
    try:
        if system.orchestrator:
            system.orchestrator.stop()
    except Exception as e:
        errors.append(f"Orchestrator: {e}")

    try:
        if system.teleop_service:
            system.teleop_service.stop()
    except Exception as e:
        errors.append(f"Teleop: {e}")

    def disable_bus_robust(bus, name="Bus"):
        try:
             # Try new Broadcast method if available
             if hasattr(bus, "emergency_stop_broadcast"):
                 bus.emergency_stop_broadcast()
             else:
                 # Fallback
                 bus.disable_torque(None, num_retry=5)
        except Exception as e:
             print(f"Emergency Disable {name} Failed: {e}")
             errors.append(f"{name}: {e}")

    # 2. Force Disable Torque (Hardware Level) - ROBOT
    if system.robot:
        try:
            if hasattr(system.robot, "left_arm"): # BiUmbra
                disable_bus_robust(system.robot.left_arm.bus, "Robot_Left")
                disable_bus_robust(system.robot.right_arm.bus, "Robot_Right")
            elif hasattr(system.robot, "bus"):
                disable_bus_robust(system.robot.bus, "Robot")
        except Exception as e:
            errors.append(f"Robot_Outer: {e}")
            
    # 3. Force Disable Torque (Hardware Level) - LEADER
    if system.leader:
        try:
            if hasattr(system.leader, "left_arm"): # BiUmbra
                disable_bus_robust(system.leader.left_arm.bus, "Leader_Left")
                disable_bus_robust(system.leader.right_arm.bus, "Leader_Right")
            elif hasattr(system.leader, "bus"):
                disable_bus_robust(system.leader.bus, "Leader")
        except Exception as e:
             errors.append(f"Leader_Outer: {e}")

    # 4. Force Disable Torque - ARM REGISTRY ARMS
    if system.arm_registry and hasattr(system.arm_registry, 'arm_instances'):
        for arm_id, instance in list(system.arm_registry.arm_instances.items()):
            try:
                if hasattr(instance, 'bus'):
                    disable_bus_robust(instance.bus, f"Registry_{arm_id}")
                elif hasattr(instance, 'left_arm'):
                    disable_bus_robust(instance.left_arm.bus, f"Registry_{arm_id}_Left")
                    disable_bus_robust(instance.right_arm.bus, f"Registry_{arm_id}_Right")
            except Exception as e:
                errors.append(f"Registry_{arm_id}: {e}")

    if errors:
        return {"status": "partial_success", "errors": errors}
    return {"status": "success", "message": "EMERGENCY STOP EXECUTED"}

@app.post("/teleop/tune")
async def tune_teleop(request: Request):
    data = await request.json()
    # data: {k_gravity, k_assist, k_haptic, v_threshold}
    
    if system.teleop_service and system.teleop_service.leader_assists:
        count = 0
        for arm_id, service in system.teleop_service.leader_assists.items():
            service.update_gains(
                k_gravity=data.get("k_gravity"), 
                k_assist=data.get("k_assist"),
                k_haptic=data.get("k_haptic"),
                v_threshold=data.get("v_threshold"),
                k_damping=data.get("k_damping") # New Damping Parameter
            )
            count += 1
        return {"status": "success", "message": f"Updated gains for {count} arms"}
    return {"status": "error", "message": "Teleop service not active"}

@app.post("/calibration/{arm_id}/torque")
async def set_torque(arm_id: str, request: Request):
    data = await request.json()
    enable = data.get("enable", True)
    if system.calibration_service:
        if enable:
            system.calibration_service.enable_torque(arm_id)
        else:
            system.calibration_service.disable_torque(arm_id)
    return {"status": "success"}

@app.post("/calibration/{arm_id}/limit")
async def set_limit(arm_id: str, request: Request):
    data = await request.json()
    motor = data.get("motor")
    limit_type = data.get("type") # min or max
    value = data.get("value")

@app.post("/teleop/assist/set")
async def set_teleop_assist(request: Request):
    data = await request.json()
    enabled = data.get("enabled", True)
    if system.teleop_service:
        system.teleop_service.set_assist_enabled(enabled)
        return {"status": "success", "enabled": enabled}
    return {"status": "error", "message": "Teleop Service not running"}

# --- Gravity Calibration (Wizard) ---

def _get_calibration_target(arm_key: str):
    """
    Helper to resolve (Service, ArmObject) from arm_key (e.g. 'left_leader', 'left_follower').
    Returns (service, arm) or raises Exception.
    """
    service = None
    arm = None
    
    # 1. Resolve Service
    if "follower" in arm_key:
        # Follower Service
        side = "left" if "left" in arm_key else "right" if "right" in arm_key else "default"
        if side in system.teleop_service.follower_gravity_models:
             service = system.teleop_service.follower_gravity_models[side]
    else:
        # Leader Service
        side = "left" if "left" in arm_key else "right" if "right" in arm_key else "default"
        # Check system.leader_assists first (populated by TeleopService)
        if hasattr(system.teleop_service, "leader_assists") and side in system.teleop_service.leader_assists:
             service = system.teleop_service.leader_assists[side]
             
    if not service:
        raise Exception(f"Calibration Service not found for {arm_key}")

    # 2. Resolve Physical Arm (for sampling)
    # If we are calibrating a FOLLOWER, we need to read from the ROBOT (follower).
    # If LEADER, read from LEADER.
    
    is_follower = "follower" in arm_key
    
    if is_follower:
        if not system.robot: raise Exception("Follower Robot not connected")
        # Match side
        if "left" in arm_key and hasattr(system.robot, "left_arm"): arm = system.robot.left_arm
        elif "right" in arm_key and hasattr(system.robot, "right_arm"): arm = system.robot.right_arm
        elif hasattr(system.robot, "bus") and not hasattr(system.robot, "left_arm"): arm = system.robot # Mono robot
    else:
        if not system.leader: raise Exception("Leader Arm not connected")
        if "left" in arm_key and hasattr(system.leader, "left_arm"): arm = system.leader.left_arm
        elif "right" in arm_key and hasattr(system.leader, "right_arm"): arm = system.leader.right_arm
        elif hasattr(system.leader, "bus") and not hasattr(system.leader, "left_arm"): arm = system.leader # Mono leader
        
    if not arm:
        raise Exception(f"Physical Arm interface not found for {arm_key}")
        
    return service, arm

@app.post("/calibration/{arm_key}/gravity/start")
def start_gravity_calibration(arm_key: str): # arm_key: left_leader, left_follower, etc.
    try:
        service, _ = _get_calibration_target(arm_key)
        service.start_calibration()
        return {"status": "success", "message": f"Calibration Started for {arm_key}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/calibration/{arm_key}/gravity/sample")
def sample_gravity_calibration(arm_key: str):
    try:
        service, arm = _get_calibration_target(arm_key)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    try:
        # Perform Hold & Measure Routine
        # 1. Read Pos
        pos_dict = arm.bus.sync_read("Present_Position")
        
        # 2. Hold (Torque ON)
        # Assuming we need to set Goal to Current to hold
        # Note: write() is usually single motor, use write_goal_positions for batch or loop
        # But 'Goal_Position' expects dict for sync_write? No, bus.write checks if motor is list.
        # Let's simple loop to hold current pos
        
        # arm.bus.write("Goal_Position", pos_dict) # This might fail if pos_dict keys aren't handled right in one go if not implemented
        # Let's use simple loop to be safe and robust
        for name, pos in pos_dict.items():
            arm.bus.write("Goal_Position", name, pos)
            
        arm.bus.enable_torque()
        
        time.sleep(1.0) # Stabilize
        
        # 3. Read Load (Avg)
        loads_list = []
        for _ in range(10):
            # sync_read returns dict {name: value}
            loads_list.append(arm.bus.sync_read("Present_Load"))
            time.sleep(0.05)
            
        # 4. Release
        arm.bus.disable_torque()
        
        # Process Data
        avg_load = {}
        # Get list of joints from bus motors
        names = arm.bus.motors.keys()
        
        for name in names:
            # Filter valid reads
            vals = [sample[name] for sample in loads_list if name in sample]
            if vals:
                 avg_load[name] = sum(vals) / len(vals)
            else:
                 avg_load[name] = 0.0
            
        # Convert to arrays for Service
        q_vec = []
        tau_vec = []
        
        # Use template names if possible or sorted names?
        # LeaderAssistService logic uses internal index based on how they are passed.
        # It doesn't strictly enforce name order but logic uses "joint_names_template" in teleop.
        # Ideally we follow the standard order: base, link1...gripper
        
        standard_order = ["base", "link1", "link2", "link3", "link4", "link5", "gripper"]
        
        # Filter only existing names
        target_names = [n for n in standard_order if n in pos_dict]
        
        for name in target_names:
            raw_pos = pos_dict[name]
            raw_load = avg_load[name]
            deg = (raw_pos - 2048.0) * (360.0/4096.0)
            q_vec.append(deg)
            tau_vec.append(raw_load)
            
        service.record_sample(q_vec, tau_vec)
        
        return {"status": "success", "samples": len(service.calibration_data)}
        
    except Exception as e:
        print(f"Sample Error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/calibration/{arm_key}/gravity/compute")
def compute_gravity_calibration(arm_key: str):
    try:
        service, _ = _get_calibration_target(arm_key)
        service.compute_weights()
        return {"status": "success", "message": "Calibration Computed and Saved"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    if system.calibration_service:
        system.calibration_service.set_calibration_limit(arm_id, motor, limit_type, value)
    return {"status": "success"}

@app.post("/calibration/{arm_id}/save")
async def save_calibration(arm_id: str):
    if not system.robot:
        raise HTTPException(status_code=503, detail="Robot not connected")
    
    if system.calibration_service:
        system.calibration_service.save_calibration(arm_id)
    return {"status": "saved"}

@app.post("/calibration/{arm_id}/homing")
async def perform_homing(arm_id: str):
    if not system.calibration_service:
        raise HTTPException(status_code=503, detail="Calibration service not available")

    result = system.calibration_service.perform_homing(arm_id)
    if isinstance(result, dict):
        return result
    if not result:
        return {"status": "error", "message": "Homing failed"}
    return {"status": "success"}

@app.get("/calibration/{arm_id}/files")
def list_calibration_files(arm_id: str):
    if not system.calibration_service:
        return {"files": []}
    return {"files": system.calibration_service.list_calibration_files(arm_id)}

@app.post("/calibration/{arm_id}/load")
async def load_calibration_file(arm_id: str, request: Request):
    data = await request.json()
    filename = data.get("filename")
    if not system.calibration_service:
        raise HTTPException(status_code=503, detail="Service not ready")
        
    success = system.calibration_service.load_calibration_file(arm_id, filename)
    if not success:
         return {"status": "error", "message": "Failed to load file"}
    return {"status": "success"}

@app.post("/calibration/{arm_id}/delete")
async def delete_calibration_file(arm_id: str, request: Request):
    data = await request.json()
    filename = data.get("filename")
    if system.calibration_service:
        success = system.calibration_service.delete_calibration_file(arm_id, filename)
        return {"status": "success" if success else "error"}
    return {"status": "error"}

@app.get("/calibration/{arm_id}/inversions")
def get_inversions(arm_id: str):
    if not system.calibration_service:
        return {"inversions": {}, "motors": []}
    inversions = system.calibration_service.get_inversions(arm_id)
    _, motors = system.calibration_service._get_arm_context(arm_id)
    return {"inversions": inversions, "motors": motors}

@app.post("/calibration/{arm_id}/inversions")
async def set_inversion(arm_id: str, payload: dict):
    system.calibration_service.set_inversion(arm_id, payload["motor"], payload["inverted"])
    return {"status": "success"}

@app.post("/calibration/{arm_id}/set-zero")
async def set_zero_pose(arm_id: str):
    """Step 1: Capture Zero Pose"""
    try:
        success = system.calibration_service.set_zero_pose(arm_id)
        if success:
             return {"status": "success"}
        else:
             return {"status": "error", "message": "Arm not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/calibration/{arm_id}/auto-align")
async def auto_align(arm_id: str):
    """Step 2: Compute Inversions based on movement from Zero"""
    try:
        result = system.calibration_service.compute_auto_alignment(arm_id)
        return result
    except Exception as e:
        logger.error(f"Auto-Align failed: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/calibration/{arm_id}/save_named")
async def save_named_calibration(arm_id: str, request: Request):
    data = await request.json()
    name = data.get("name")
    if not system.robot:
         raise HTTPException(status_code=503, detail="Robot not connected")
    
    if system.calibration_service:
        system.calibration_service.save_calibration(arm_id, name=name)
    return {"status": "saved"}

@app.post("/calibration/{arm_id}/discovery/start")
def start_discovery(arm_id: str):
    """Start range discovery. Runs in threadpool (def not async) to avoid blocking
    the event loop during serial I/O and lock acquisition."""
    if not system.calibration_service:
        return {"status": "error", "message": "Calibration service not initialized"}
    try:
        system.calibration_service.start_discovery(arm_id)
        return {"status": "started"}
    except Exception as e:
        logger.error(f"start_discovery failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/calibration/{arm_id}/discovery/stop")
def stop_discovery(arm_id: str):
    """Stop range discovery. Runs in threadpool (def not async) to avoid blocking
    the event loop during serial I/O and lock acquisition."""
    if not system.calibration_service:
        return {"status": "error", "message": "Calibration service not initialized"}
    try:
        result = system.calibration_service.stop_discovery(arm_id)
        return result
    except Exception as e:
        logger.error(f"stop_discovery failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# --- Damiao Velocity Limiter Endpoints ---

@app.get("/robot/velocity-limit")
async def get_velocity_limit():
    """Get current global velocity limit for Damiao robots.

    Returns velocity_limit as float (0.0-1.0) where 1.0 = 100% max velocity.
    """
    # Check if we have a Damiao robot
    if system.robot and hasattr(system.robot, 'velocity_limit'):
        return {"velocity_limit": system.robot.velocity_limit, "has_velocity_limit": True}

    # Check teleop service's active robot (set during teleop start from arm registry)
    if system.teleop_service:
        active = getattr(system.teleop_service, '_active_robot', None)
        if active and hasattr(active, 'velocity_limit'):
            return {"velocity_limit": active.velocity_limit, "has_velocity_limit": True}
        robot = getattr(system.teleop_service, 'robot', None)
        if robot and hasattr(robot, 'velocity_limit'):
            return {"velocity_limit": robot.velocity_limit, "has_velocity_limit": True}

    # Check arm registry for any connected Damiao follower
    if system.arm_registry:
        for arm_id, instance in system.arm_registry.arm_instances.items():
            if hasattr(instance, 'velocity_limit'):
                return {"velocity_limit": instance.velocity_limit, "has_velocity_limit": True}

    return {"velocity_limit": 1.0, "has_velocity_limit": False}

@app.post("/robot/velocity-limit")
async def set_velocity_limit(request: Request):
    """Set global velocity limit for Damiao robots (0.0-1.0).

    SAFETY: This limits the maximum velocity of ALL motor commands.
    Default for Damiao is 0.2 (20%) for safety with high-torque motors.
    """
    data = await request.json()
    limit = float(data.get("limit", 1.0))
    limit = max(0.0, min(1.0, limit))

    updated = False

    # Update main robot if it's a Damiao
    if system.robot and hasattr(system.robot, 'velocity_limit'):
        system.robot.velocity_limit = limit
        updated = True
        logger.info(f"Set velocity_limit to {limit:.2f} on main robot")

    # Update teleop service's active robot and legacy robot
    if system.teleop_service:
        active = getattr(system.teleop_service, '_active_robot', None)
        if active and hasattr(active, 'velocity_limit'):
            active.velocity_limit = limit
            updated = True
            logger.info(f"Set velocity_limit to {limit:.2f} on teleop active robot")
        robot = getattr(system.teleop_service, 'robot', None)
        if robot and hasattr(robot, 'velocity_limit') and robot is not active:
            robot.velocity_limit = limit
            updated = True
            logger.info(f"Set velocity_limit to {limit:.2f} on teleop robot")

    # Update any Damiao followers in arm registry
    if system.arm_registry:
        for arm_id, instance in system.arm_registry.arm_instances.items():
            if hasattr(instance, 'velocity_limit'):
                instance.velocity_limit = limit
                updated = True
                logger.info(f"Set velocity_limit to {limit:.2f} on arm_registry/{arm_id}")

    if updated:
        return {"status": "ok", "velocity_limit": limit}
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "No robot with velocity limit found (Damiao required)"}
        )

@app.get("/robot/damiao/status")
async def get_damiao_status():
    """Get status of Damiao robot (if connected).

    Returns connection status, velocity limit, and torque readings.
    """
    result = {
        "connected": False,
        "velocity_limit": 1.0,
        "motor_type": None,
    }

    # Check main robot
    robot = None
    if system.robot and hasattr(system.robot, 'velocity_limit'):
        robot = system.robot
    elif system.teleop_service and hasattr(system.teleop_service, 'robot'):
        r = system.teleop_service.robot
        if hasattr(r, 'velocity_limit'):
            robot = r

    if robot:
        result["connected"] = robot.is_connected
        result["velocity_limit"] = robot.velocity_limit
        result["motor_type"] = "damiao"

        # Try to get torque readings for safety monitoring
        if hasattr(robot, 'get_torques'):
            try:
                result["torques"] = robot.get_torques()
                result["torque_limits"] = robot.get_torque_limits()
            except Exception as e:
                logger.warning(f"Failed to read Damiao torques: {e}")

    return result

# --- Arm Registry Endpoints ---
# NOTE: Static routes (/arms/pairings, /arms/scan-ports) MUST come before
# parameterized routes (/arms/{arm_id}) to avoid routing conflicts in FastAPI.

@app.get("/arms")
async def get_all_arms():
    """Get all registered arms with their status."""
    if not system.arm_registry:
        return {"arms": [], "summary": {}}
    return {
        "arms": system.arm_registry.get_all_arms(),
        "summary": system.arm_registry.get_status_summary()
    }

@app.post("/arms")
async def add_arm(request: Request):
    """Add a new arm to the registry."""
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    data = await request.json()
    result = system.arm_registry.add_arm(data)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result

# Static routes - must be defined before /{arm_id} routes
@app.get("/arms/pairings")
async def get_pairings():
    """Get all leader-follower pairings."""
    if not system.arm_registry:
        return {"pairings": []}
    return {"pairings": system.arm_registry.get_pairings()}

@app.post("/arms/pairings")
async def create_pairing(request: Request):
    """Create a new leader-follower pairing."""
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    data = await request.json()
    result = system.arm_registry.create_pairing(
        leader_id=data.get("leader_id"),
        follower_id=data.get("follower_id"),
        name=data.get("name")
    )
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result

@app.delete("/arms/pairings")
async def remove_pairing(request: Request):
    """Remove a leader-follower pairing."""
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    data = await request.json()
    result = system.arm_registry.remove_pairing(
        leader_id=data.get("leader_id"),
        follower_id=data.get("follower_id")
    )
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result

@app.get("/arms/scan-ports")
async def scan_ports():
    """Scan for available serial ports."""
    if not system.arm_registry:
        return {"ports": []}
    return {"ports": system.arm_registry.scan_ports()}

# Parameterized routes - must come after static routes
@app.get("/arms/{arm_id}")
async def get_arm(arm_id: str):
    """Get details of a specific arm."""
    if not system.arm_registry:
        return JSONResponse(status_code=404, content={"error": "Arm registry not initialized"})
    arm = system.arm_registry.get_arm(arm_id)
    if not arm:
        return JSONResponse(status_code=404, content={"error": f"Arm '{arm_id}' not found"})
    return arm

@app.put("/arms/{arm_id}")
async def update_arm(arm_id: str, request: Request):
    """Update an existing arm."""
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    data = await request.json()
    result = system.arm_registry.update_arm(arm_id, **data)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result

@app.delete("/arms/{arm_id}")
async def remove_arm(arm_id: str):
    """Remove an arm from the registry."""
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    result = system.arm_registry.remove_arm(arm_id)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result

@app.post("/arms/{arm_id}/connect")
def connect_arm(arm_id: str):
    """Connect a specific arm. Runs in threadpool (def not async) to avoid blocking
    the event loop during serial port connection and motor configuration."""
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    result = system.arm_registry.connect_arm(arm_id)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result

@app.post("/arms/{arm_id}/disconnect")
def disconnect_arm(arm_id: str):
    """Disconnect a specific arm. Runs in threadpool (def not async) to avoid blocking
    the event loop during serial port disconnection."""
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    result = system.arm_registry.disconnect_arm(arm_id)
    return result

@app.post("/arms/{arm_id}/set-home")
def set_home_position(arm_id: str):
    """Capture current motor positions as home position for this arm."""
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})

    instance = system.arm_registry.arm_instances.get(arm_id)
    if not instance or not getattr(instance, 'is_connected', False):
        return JSONResponse(status_code=400, content={"error": f"Arm '{arm_id}' not connected"})

    from lerobot.motors.damiao.damiao import DamiaoMotorsBus
    bus = getattr(instance, 'bus', None)
    if not bus or not isinstance(bus, DamiaoMotorsBus):
        return JSONResponse(status_code=400, content={"error": "Not a Damiao arm"})

    positions = dict(bus._last_positions)
    if not positions:
        return JSONResponse(status_code=400, content={"error": "No position data available"})

    result = system.arm_registry.update_arm(arm_id, config={"home_position": positions})
    if result.get("success"):
        return {"success": True, "home_position": {k: round(v, 4) for k, v in positions.items()}}
    return JSONResponse(status_code=400, content=result)

@app.delete("/arms/{arm_id}/set-home")
async def clear_home_position(arm_id: str):
    """Clear saved home position for this arm."""
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})
    arm = system.arm_registry.arms.get(arm_id)
    if not arm:
        return JSONResponse(status_code=404, content={"error": f"Arm '{arm_id}' not found"})
    arm.config.pop("home_position", None)
    system.arm_registry._save_config()
    return {"success": True}

@app.get("/arms/{arm_id}/motors/diagnostics")
def get_motor_diagnostics(arm_id: str):
    """Read live motor telemetry (position, temperature, voltage, current, errors) for a connected arm."""
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})

    arm_def = system.arm_registry.arms.get(arm_id)
    if not arm_def:
        return JSONResponse(status_code=404, content={"error": f"Arm '{arm_id}' not found"})

    instance = system.arm_registry.arm_instances.get(arm_id)
    if not instance or not getattr(instance, 'is_connected', False):
        return JSONResponse(status_code=400, content={"error": f"Arm '{arm_id}' not connected"})

    bus = getattr(instance, 'bus', None)
    if not bus:
        return JSONResponse(status_code=400, content={"error": "No motor bus available"})

    motor_type = arm_def.motor_type.value
    motors_info = []

    # Determine which registers to read based on motor type
    try:
        from lerobot.motors.dynamixel.dynamixel import DynamixelMotorsBus
        from lerobot.motors.feetech.feetech import FeetechMotorsBus
        is_dynamixel = isinstance(bus, DynamixelMotorsBus)
        is_feetech = isinstance(bus, FeetechMotorsBus)
    except ImportError:
        is_dynamixel = False
        is_feetech = False

    # Check for Damiao (CAN-based, different API)
    is_damiao = False
    try:
        from lerobot.motors.damiao.damiao import DamiaoMotorsBus
        is_damiao = isinstance(bus, DamiaoMotorsBus)
    except ImportError:
        pass

    if is_damiao:
        # Damiao: read from CAN state cache (no sync_read for these)
        for name, motor in bus.motors.items():
            motor_data = {
                "name": name,
                "id": motor.id if hasattr(motor, 'id') else getattr(motor, 'slave_id', 0),
                "model": getattr(motor, 'model', 'damiao'),
                "position": None,
                "velocity": None,
                "current": None,
                "temperature": None,
                "voltage": None,
                "load": None,
                "error": 0,
                "error_names": [],
            }
            # Damiao stores last state in _last_positions, _last_velocities, _last_torques
            if hasattr(bus, '_last_positions'):
                motor_data["position"] = round(bus._last_positions.get(name, 0), 3)
            if hasattr(bus, '_last_velocities'):
                motor_data["velocity"] = round(bus._last_velocities.get(name, 0), 3)
            if hasattr(bus, '_last_torques'):
                motor_data["load"] = round(bus._last_torques.get(name, 0), 3)
            motors_info.append(motor_data)
    else:
        # Dynamixel / Feetech: use sync_read for each register
        motor_names = list(bus.motors.keys())

        # Build motor list with IDs
        motor_map = {}
        for name in motor_names:
            m = bus.motors[name]
            motor_map[name] = {
                "name": name,
                "id": m.id,
                "model": getattr(m, 'model', motor_type),
                "position": None,
                "velocity": None,
                "current": None,
                "temperature": None,
                "voltage": None,
                "load": None,
                "error": 0,
                "error_names": [],
            }

        def safe_read(data_name):
            try:
                return bus.sync_read(data_name, normalize=False)
            except Exception:
                return {}

        positions = safe_read("Present_Position")
        for name, val in positions.items():
            if name in motor_map:
                motor_map[name]["position"] = int(val) if val is not None else None

        velocities = safe_read("Present_Velocity")
        for name, val in velocities.items():
            if name in motor_map:
                motor_map[name]["velocity"] = int(val) if val is not None else None

        temperatures = safe_read("Present_Temperature")
        for name, val in temperatures.items():
            if name in motor_map:
                motor_map[name]["temperature"] = int(val) if val is not None else None

        # Current (Dynamixel) or Present_Current (Feetech)
        currents = safe_read("Present_Current")
        for name, val in currents.items():
            if name in motor_map:
                motor_map[name]["current"] = int(val) if val is not None else None

        # Voltage — different register names per type
        if is_dynamixel:
            voltages = safe_read("Present_Input_Voltage")
        else:
            voltages = safe_read("Present_Voltage")
        for name, val in voltages.items():
            if name in motor_map:
                motor_map[name]["voltage"] = round(int(val) * 0.1, 1) if val is not None else None

        # Load (Feetech only)
        if is_feetech:
            loads = safe_read("Present_Load")
            for name, val in loads.items():
                if name in motor_map:
                    motor_map[name]["load"] = int(val) if val is not None else None

        # Hardware errors (Dynamixel only)
        if is_dynamixel:
            errors = safe_read("Hardware_Error_Status")
            for name, val in errors.items():
                if name in motor_map and val:
                    err = int(val)
                    motor_map[name]["error"] = err
                    names = []
                    if err & 0x01: names.append("Voltage")
                    if err & 0x04: names.append("Overheat")
                    if err & 0x08: names.append("Encoder")
                    if err & 0x10: names.append("Shock")
                    if err & 0x20: names.append("Overload")
                    motor_map[name]["error_names"] = names

        motors_info = list(motor_map.values())

    return {
        "arm_id": arm_id,
        "motor_type": motor_type,
        "motors": motors_info,
    }


@app.get("/arms/{leader_id}/compatible-followers")
async def get_compatible_followers(leader_id: str):
    """Get followers compatible with a leader arm."""
    if not system.arm_registry:
        return {"followers": []}
    return {"followers": system.arm_registry.get_compatible_followers(leader_id)}

# --- Motor Configuration Endpoints ---

@app.post("/motors/scan")
async def scan_motors(request: Request):
    """Scan a port for connected motors.

    IMPORTANT: For reliable results, connect only ONE motor at a time.

    Request body:
        port: Serial port path (e.g., /dev/ttyACM0)
        motor_type: Motor type (dynamixel_xl330, dynamixel_xl430, sts3215)

    Returns:
        found_ids: List of motor IDs responding on the bus
    """
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})

    data = await request.json()
    port = data.get("port")
    motor_type = data.get("motor_type")

    if not port or not motor_type:
        return JSONResponse(status_code=400, content={"error": "port and motor_type are required"})

    result = system.arm_registry.scan_motors(port, motor_type)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


@app.post("/motors/set-id")
async def set_motor_id(request: Request):
    """Change a motor's ID.

    IMPORTANT: Connect ONLY ONE motor at a time when using this endpoint!

    Request body:
        port: Serial port path
        motor_type: Motor type (dynamixel_xl330, dynamixel_xl430, sts3215)
        current_id: Current motor ID (often 1 for factory default)
        new_id: New ID to assign (1-253)

    Returns:
        success: Boolean
        new_id: The new motor ID if successful
    """
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})

    data = await request.json()
    port = data.get("port")
    motor_type = data.get("motor_type")
    current_id = data.get("current_id")
    new_id = data.get("new_id")

    if not all([port, motor_type, current_id is not None, new_id is not None]):
        return JSONResponse(status_code=400, content={"error": "port, motor_type, current_id, and new_id are required"})

    result = system.arm_registry.set_motor_id(port, motor_type, int(current_id), int(new_id))
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


@app.post("/motors/recover")
async def recover_motor(request: Request):
    """Attempt to recover an unresponsive or error-state Dynamixel motor.

    Recovery steps:
    1. Scan all baud rates with broadcast ping
    2. If found with errors: reboot to clear
    3. If not found: try reboot at ID=1 (factory default)
    4. If still not found: try factory reset
    5. Final verification scan

    Request body:
        port: Serial port path
        motor_type: Motor type (dynamixel_xl330, dynamixel_xl430)

    Returns:
        recovered: Boolean - whether motor was recovered
        motor: Motor info if found
        log: Step-by-step recovery log
    """
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})

    data = await request.json()
    port = data.get("port")
    motor_type = data.get("motor_type")

    if not all([port, motor_type]):
        return JSONResponse(status_code=400, content={"error": "port and motor_type are required"})

    result = system.arm_registry.recover_motor(port, motor_type)
    return result


@app.post("/motors/ping")
async def ping_motor(request: Request):
    """Ping a specific motor ID to verify connection.

    Request body:
        port: Serial port path
        motor_type: Motor type
        motor_id: ID to ping

    Returns:
        success: Boolean
        responding: Boolean - whether the motor responded
    """
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})

    data = await request.json()
    port = data.get("port")
    motor_type = data.get("motor_type")
    motor_id = data.get("motor_id")

    if not all([port, motor_type, motor_id is not None]):
        return JSONResponse(status_code=400, content={"error": "port, motor_type, and motor_id are required"})

    try:
        if motor_type in ["dynamixel_xl330", "dynamixel_xl430"]:
            from lerobot.motors.dynamixel import DynamixelMotorsBus
            from lerobot.motors import Motor, MotorNormMode

            bus = DynamixelMotorsBus(port=port, motors={})
            bus.connect()
            responding = bus.ping(int(motor_id))
            bus.disconnect()
            return {"success": True, "responding": responding, "motor_id": motor_id}

        elif motor_type == "sts3215":
            from lerobot.motors.feetech import FeetechMotorsBus

            bus = FeetechMotorsBus(port=port, motors={})
            bus.connect()
            responding = bus.ping(int(motor_id))
            bus.disconnect()
            return {"success": True, "responding": responding, "motor_id": motor_id}

        else:
            return JSONResponse(status_code=400, content={"error": f"Unsupported motor type: {motor_type}"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- Teleoperation Endpoints ---
from app.core.teleop_service import TeleoperationService





@app.post("/teleop/stop")
async def stop_teleop():
    if system.teleop_service:
        system.teleop_service.stop()
    return {"status": "stopped"}

@app.get("/teleop/status")
def get_teleop_status():
    running = False
    if system.teleop_service:
        running = system.teleop_service.is_running
    return {"running": running}

@app.get("/teleop/data")
def get_teleop_data():
    if system.teleop_service:
        return {"data": system.teleop_service.get_data()}
    return {"data": []}


@app.post("/system/reset")
async def reset_system(background_tasks: BackgroundTasks):
    """Soft reset attempts to re-initialize hardware without killing the process."""
    try:
        # Run reload in background to avoid blocking return
        background_tasks.add_task(system.reload)
        return {"status": "success", "message": "System reset initiated..."}
    except Exception as e:
        logger.error(f"Failed to reset system: {e}")
        return {"status": "error", "message": str(e)}

# --- Camera Endpoints ---

@app.get("/cameras/scan")
def scan_cameras():
    if not system.camera_service:
        return {"status": "error", "message": "Camera service not initialized"}
    
    # If robot is connected, we combine active cameras with a fresh scan for available ones.
    if system.robot and system.robot.is_connected:
        current_config = system.camera_service.get_camera_config()
        
        # 1. Get Active Cameras
        active_opencv = []
        active_realsense = []
        
        active_opencv_indices = set()
        active_realsense_serials = set()
        
        for key, conf in current_config.items():
            if conf.get("type") == "opencv":
                idx = conf.get("index_or_path")
                active_opencv.append({
                    "index": idx,
                    "name": f"Active Camera ({key})",
                    "port": idx,
                    "is_active": True
                })
                active_opencv_indices.add(str(idx))
            elif conf.get("type") == "intelrealsense":
                serial = conf.get("serial_number_or_name")
                active_realsense.append({
                    "name": f"Active Camera ({key})",
                    "serial_number": serial,
                    "is_active": True
                })
                active_realsense_serials.add(str(serial))
        
        # 2. Scan for Available Cameras (ignoring errors on busy ones)
        available = system.camera_service.scan_cameras(active_ids=list(active_opencv_indices))
        
        # 3. Merge (avoid duplicates)
        final_opencv = active_opencv[:]
        for cam in available.get("opencv", []):
            # OpenCVCamera.find_cameras returns 'port' or 'index' depending on implementation
            # Let's assume it returns 'port' or 'id'
            cam_idx = str(cam.get("port") or cam.get("id") or cam.get("index"))
            if cam_idx not in active_opencv_indices:
                cam["is_active"] = False
                final_opencv.append(cam)
                
        final_realsense = active_realsense[:]
        for cam in available.get("realsense", []):
            cam_serial = str(cam.get("serial_number"))
            if cam_serial not in active_realsense_serials:
                cam["is_active"] = False
                final_realsense.append(cam)
    else:
        # Robot not connected - just do a fresh scan
        available = system.camera_service.scan_cameras()
        final_opencv = available.get("opencv", [])
        final_realsense = available.get("realsense", [])
        for cam in final_opencv:
            cam["is_active"] = False
        for cam in final_realsense:
            cam["is_active"] = False

    # Standaradize keys for Frontend
    for cam in final_opencv:
         if "id" not in cam:
             cam["id"] = cam.get("port") or cam.get("index") or cam.get("index_or_path")
         if "index_or_path" not in cam:
             cam["index_or_path"] = cam.get("id")

    for cam in final_realsense:
         if "id" not in cam:
             cam["id"] = cam.get("serial_number") or cam.get("serial_number_or_name")
         if "serial_number_or_name" not in cam:
             cam["serial_number_or_name"] = cam.get("id")
    
    return {
        "opencv": final_opencv,
        "realsense": final_realsense,
        "note": "Merged active and available cameras."
    }

@app.get("/cameras/config")
def get_camera_config():
    if not system.camera_service:
        return []
    
    # Return List for Frontend (CameraModal.tsx expects array)
    config = system.camera_service.get_camera_config()
    export_list = []
    for key, val in config.items():
        # Inject key as 'id'
        item = val.copy()
        item["id"] = key # 'camera_1' etc
        export_list.append(item)
    return export_list

@app.post("/cameras/config")
async def update_camera_config(request: Request, background_tasks: BackgroundTasks):
    if not system.camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized")

    data = await request.json()

    # Get existing config to compare
    existing_config = system.camera_service.get_camera_config()

    # Convert array format from frontend to dict format for storage
    # Frontend sends: [{id: "cam1", video_device_id: ..., type: ..., use_depth: ...}, ...]
    # Backend expects: {"cam1": {video_device_id: ..., type: ..., use_depth: ...}, ...}
    if isinstance(data, list):
        config_dict = {}
        for item in data:
            cam_id = item.get("id", "unknown")
            # Remove 'id' from the stored config (it's the key)
            config_entry = {k: v for k, v in item.items() if k != "id"}
            vid = config_entry.get("video_device_id", "")

            # Ensure type is set based on video_device_id if not provided
            if "type" not in config_entry:
                if str(vid).startswith("/dev/video") or (str(vid).isdigit() and len(str(vid)) < 4):
                    config_entry["type"] = "opencv"
                else:
                    config_entry["type"] = "intelrealsense"

            # Synthesize index_or_path for opencv cameras
            if config_entry.get("type") == "opencv" and "index_or_path" not in config_entry:
                config_entry["index_or_path"] = vid

            # Synthesize serial_number_or_name for realsense cameras
            if config_entry.get("type") == "intelrealsense" and "serial_number_or_name" not in config_entry:
                config_entry["serial_number_or_name"] = vid

            config_dict[cam_id] = config_entry
        data = config_dict

    # Check if only use_depth changed (no need to reload for depth-only changes)
    # Compare camera assignments (video_device_id, type) - ignore use_depth
    needs_reload = False

    # Check for new/removed cameras or changed assignments
    if set(data.keys()) != set(existing_config.keys()):
        needs_reload = True
    else:
        for cam_id, new_cfg in data.items():
            old_cfg = existing_config.get(cam_id, {})
            # Compare assignment-critical fields (not use_depth)
            if (new_cfg.get("video_device_id") != old_cfg.get("video_device_id") or
                new_cfg.get("type") != old_cfg.get("type") or
                new_cfg.get("width") != old_cfg.get("width") or
                new_cfg.get("height") != old_cfg.get("height") or
                new_cfg.get("fps") != old_cfg.get("fps")):
                needs_reload = True
                break

    system.camera_service.update_camera_config(data)

    if needs_reload:
        # Trigger System Reload in Background (only for camera assignment changes)
        background_tasks.add_task(system.reload)
        return {"status": "success", "message": "Camera config updated. System reloading..."}
    else:
        # Just a depth toggle or other non-critical change - no reload needed
        return {"status": "success", "message": "Camera config updated (no reload needed)."}

@app.get("/debug/observation")
def debug_observation():
    if not system.orchestrator or not system.orchestrator.intervention_engine:
        return {"status": "error", "message": "Orchestrator/InterventionEngine not initialized"}
    
    obs = system.orchestrator.intervention_engine.latest_observation
    if obs is None:
        return {"status": "empty", "keys": [], "message": "Observation is None"}
    
    # Summarize the observation (don't send full tensors)
    summary = {}
    for k, v in obs.items():
        if hasattr(v, 'shape'):
            summary[k] = f"Tensor shape: {v.shape}"
        else:
            summary[k] = str(v)
            
    return {
        "status": "ok",
        "keys": list(obs.keys()),
        "summary": summary,
        "robot_connected": system.robot.is_connected if system.robot else False,
        "is_running": system.orchestrator.is_running
    }

# ============================================================================
# RL Training (HIL-SERL) API Endpoints
# ============================================================================

@app.post("/rl/training/start")
async def start_rl_training(request: Request):
    """Start HIL-SERL RL training session."""
    if not system.rl_service:
        return JSONResponse(status_code=503, content={"error": "RL service not initialized"})

    try:
        data = await request.json()
        result = system.rl_service.start_training(data)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/rl/training/stop")
async def stop_rl_training():
    """Stop RL training and save policy."""
    if not system.rl_service:
        return JSONResponse(status_code=503, content={"error": "RL service not initialized"})

    return system.rl_service.stop_training()


@app.post("/rl/training/pause")
async def pause_rl_training():
    """Pause RL training."""
    if not system.rl_service:
        return JSONResponse(status_code=503, content={"error": "RL service not initialized"})

    return system.rl_service.pause_training()


@app.post("/rl/training/resume")
async def resume_rl_training():
    """Resume paused RL training."""
    if not system.rl_service:
        return JSONResponse(status_code=503, content={"error": "RL service not initialized"})

    return system.rl_service.resume_training()


@app.get("/rl/training/status")
def get_rl_training_status():
    """Get current RL training status and metrics."""
    if not system.rl_service:
        return {"status": "unavailable"}

    return system.rl_service.get_status()


@app.patch("/rl/training/settings")
async def update_rl_settings(request: Request):
    """Update RL training settings mid-training."""
    if not system.rl_service:
        return JSONResponse(status_code=503, content={"error": "RL service not initialized"})

    try:
        data = await request.json()
        return system.rl_service.update_settings(data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ============================================================================
# Reward Classifier API Endpoints
# ============================================================================

@app.post("/rl/reward-classifier/train")
async def train_reward_classifier(request: Request):
    """Train a reward classifier from demonstration dataset."""
    if not system.reward_classifier_service:
        return JSONResponse(status_code=503, content={"error": "Reward classifier service not initialized"})

    try:
        data = await request.json()
        result = system.reward_classifier_service.train_classifier(
            dataset_repo_id=data.get("dataset_repo_id", ""),
            name=data.get("name", ""),
            success_frames_per_episode=data.get("success_frames_per_episode", 5),
            failure_frames_per_episode=data.get("failure_frames_per_episode", 10),
            epochs=data.get("epochs", 50),
            batch_size=data.get("batch_size", 32),
            learning_rate=data.get("learning_rate", 1e-4),
        )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/rl/reward-classifier/list")
def list_reward_classifiers():
    """List available trained reward classifiers."""
    if not system.reward_classifier_service:
        return []

    classifiers = system.reward_classifier_service.list_classifiers()
    return [
        {
            "name": c.name,
            "dataset_repo_id": c.dataset_repo_id,
            "num_cameras": c.num_cameras,
            "accuracy": c.accuracy,
            "created_at": c.created_at,
        }
        for c in classifiers
    ]


@app.get("/rl/reward-classifier/training-status")
def reward_classifier_training_status():
    """Get reward classifier training status."""
    if not system.reward_classifier_service:
        return {"status": "unavailable"}

    return system.reward_classifier_service.get_training_status()


@app.delete("/rl/reward-classifier/{name}")
def delete_reward_classifier(name: str):
    """Delete a trained reward classifier."""
    if not system.reward_classifier_service:
        return JSONResponse(status_code=503, content={"error": "Service not initialized"})

    success = system.reward_classifier_service.delete_classifier(name)
    if success:
        return {"status": "deleted", "name": name}
    return JSONResponse(status_code=404, content={"error": f"Classifier '{name}' not found"})


# ============================================================================
# GVL Reward Service API Endpoints
# ============================================================================

@app.get("/rl/gvl/status")
def get_gvl_status():
    """Get GVL reward service status."""
    if not system.gvl_reward_service:
        return {"status": "unavailable"}

    return system.gvl_reward_service.get_status()


@app.patch("/rl/gvl/config")
async def update_gvl_config(request: Request):
    """Update GVL reward service configuration."""
    if not system.gvl_reward_service:
        return JSONResponse(status_code=503, content={"error": "GVL service not initialized"})

    try:
        data = await request.json()
        system.gvl_reward_service.update_config(**data)
        return {"status": "updated"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ============================================================================
# SARM Reward Service API Endpoints
# ============================================================================

@app.post("/rl/sarm/train")
async def train_sarm(request: Request):
    """Train SARM reward model on demo dataset."""
    if not system.sarm_reward_service:
        return JSONResponse(status_code=503, content={"error": "SARM service not initialized"})

    try:
        data = await request.json()
        dataset_repo_id = data.get("dataset_repo_id")
        name = data.get("name")
        config = data.get("config", {})

        if not dataset_repo_id:
            return JSONResponse(status_code=400, content={"error": "dataset_repo_id required"})
        if not name:
            return JSONResponse(status_code=400, content={"error": "name required"})

        result = system.sarm_reward_service.train_sarm(dataset_repo_id, name, config)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/rl/sarm/list")
def list_sarm_models():
    """List trained SARM models."""
    if not system.sarm_reward_service:
        return []

    return system.sarm_reward_service.list_sarm_models()


@app.get("/rl/sarm/training-status")
def get_sarm_training_status():
    """Get SARM training progress."""
    if not system.sarm_reward_service:
        return {"status": "unavailable"}

    return system.sarm_reward_service.get_training_status()


@app.post("/rl/sarm/stop-training")
def stop_sarm_training():
    """Stop SARM training in progress."""
    if not system.sarm_reward_service:
        return JSONResponse(status_code=503, content={"error": "SARM service not initialized"})

    return system.sarm_reward_service.stop_training()


@app.delete("/rl/sarm/{name}")
def delete_sarm_model(name: str):
    """Delete a SARM model."""
    if not system.sarm_reward_service:
        return JSONResponse(status_code=503, content={"error": "SARM service not initialized"})

    return system.sarm_reward_service.delete_sarm(name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
