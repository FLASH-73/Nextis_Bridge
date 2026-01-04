import sys
import os
from dotenv import load_dotenv
from pathlib import Path
import json

# Load environment variables from .env file
load_dotenv()

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
from app.core.leader_assist import LeaderAssistService
from app.core.teleop_service import TeleoperationService
from lerobot.robots.utils import make_robot_from_config
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.robots.bi_umbra_follower.config_bi_umbra_follower import BiUmbraFollowerConfig
from app.core.config import load_config

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
        print("Initializing System (Async Internal)...")
        import lerobot
        # print(f"DEBUG: lerobot path: {lerobot.__file__}")
        
        # Initialize Camera Service
        self.camera_service = CameraService()
        
        # Load Config
        config_data = load_config()
        robot_cfg = config_data.get("robot", {})
        teleop_cfg = config_data.get("teleop", {})
        
        try:
            print(f"Attempting to connect to robot: {robot_cfg.get('type')}...")
            
            # Import Config Classes dynamically to avoid top-level import errors if deps missing
            from lerobot.robots.bi_umbra_follower.config_bi_umbra_follower import BiUmbraFollowerConfig
            from lerobot.cameras.opencv import OpenCVCameraConfig
            from lerobot.cameras.realsense import RealSenseCameraConfig
            
            from app.core.camera_discovery import discover_cameras
            
            # Discover working cameras
            # Discover working cameras
            print("Scanning for available cameras...")
            discovered = discover_cameras()
            # discovered = {"opencv": [], "realsense": []}
            working_opencv_ports = []
            working_realsense_serials = []
            
            # working_opencv_ports = [c["index_or_path"] for c in discovered["opencv"]]
            # working_realsense_serials = [c["serial_number_or_name"] for c in discovered["realsense"]]
            
            print(f"Discovered OpenCV: {working_opencv_ports}")
            print(f"Discovered RealSense: {working_realsense_serials}")

            # Construct Camera Configs from Discovery
            cameras = {}

            # --- Fix Config Format Mismatch (List vs Dict) ---
            raw_cameras = robot_cfg.get("cameras", [])
            configured_cameras = {}
            if isinstance(raw_cameras, list):
                for c in raw_cameras:
                    c_id = c.get("id", "unknown")
                    vid = c.get("video_device_id")
                    # Heuristic for type if missing
                    c_type = c.get("type")
                    if not c_type:
                        if str(vid).startswith("/dev/video") or (str(vid).isdigit() and len(str(vid)) < 4):
                             c_type = "opencv"
                        else:
                             c_type = "intelrealsense"
                    
                    configured_cameras[c_id] = {
                        "type": c_type,
                        "index_or_path": vid,
                        "serial_number_or_name": vid,
                        **c
                    }
            elif isinstance(raw_cameras, dict):
                configured_cameras = raw_cameras
            # -------------------------------------------------
            
            # Add OpenCV Cameras
            for i, cam in enumerate(discovered["opencv"]):
                # Use a generic name or try to match with config if possible
                # User wants "check which one actually work and only add them"
                # We can try to map back to config names if ports match, otherwise generate new names
                
                cam_id = cam.get("id")
                
                # Critical Fix: Skip if ID is missing to prevent OpenCVCamera(None) crash
                if cam_id is None:
                    print(f"⚠️ Skipping OpenCV camera {i} due to missing ID.")
                    continue
                
                # Try to find if this port was in config
                config_name = None
                print(f"DEBUG: Checking OpenCV Camera Discovered ID: '{cam_id}' (type: {type(cam_id)})")
                
                for name, cfg in configured_cameras.items():
                    cfg_type = cfg.get("type")
                    cfg_idx = cfg.get("index_or_path")
                    # print(f"  - Comparing against Config '{name}': type={cfg_type}, idx='{cfg_idx}'")
                    
                    if cfg_type == "opencv" and str(cfg_idx) == str(cam_id):
                        config_name = name
                        print(f"  -> MATCH FOUND: '{name}' matches device '{cam_id}'")
                        break
                
                if not config_name:
                    print(f"  -> NO MATCH found for device '{cam_id}'.")
                    # STRICT MODE: If we have ANY configured cameras, ignore unconfigured ones to preventing polluting the system.
                    # Unless it's a fresh setup (no config).
                    if configured_cameras:
                        print(f"  -> Skipping unconfigured device '{cam_id}' because valid config exists.")
                        continue
                    
                    print(f"  -> Will use auto-generated name for '{cam_id}'.")

                cam_name = config_name if config_name else f"camera_{i+1}_opencv"
                
                cameras[cam_name] = OpenCVCameraConfig(
                    fps=cam.get("fps", 30),
                    width=cam.get("width", 848),
                    height=cam.get("height", 480),
                    index_or_path=cam_id
                )
                print(f"Added {cam_name} at {cam_id}")

            # --- Explicitly Add Configured Cameras if Missed by Discovery ---
            # e.g. /dev/video16 might be a loopback or not found by simple glob
            for name, cfg in configured_cameras.items():
                if name not in cameras and cfg.get("type") == "opencv":
                    idx = cfg.get("index_or_path")
                    if idx:
                        print(f"Adding configured camera '{name}' explicitly (was not in discovery scan) at {idx}")
                        cameras[name] = OpenCVCameraConfig(
                            fps=cfg.get("fps", 30),
                            width=cfg.get("width", 640),
                            height=cfg.get("height", 480),
                            index_or_path=idx
                        )
            # ----------------------------------------------------------------

            # Add RealSense Cameras
            for i, cam in enumerate(discovered["realsense"]):
                serial = cam.get("id")
                if not serial and "serial_number" in cam:
                    serial = cam["serial_number"]

                if not serial:
                    print(f"⚠️ Skipping RealSense camera {i} due to missing serial number.")
                    continue
                    
                # Ensure serial is a string
                serial = str(serial)

                # Try to find if this serial was in config
                config_name = None
                print(f"DEBUG: Checking RealSense Camera Discovered Serial: '{serial}'")

                for name, cfg in configured_cameras.items():
                    cfg_type = cfg.get("type")
                    cfg_serial = cfg.get("serial_number_or_name")
                    
                    if cfg_type == "intelrealsense" and str(cfg_serial) == serial:
                        config_name = name
                        print(f"  -> MATCH FOUND: '{name}' matches serial '{serial}'")
                        break
                
                cam_name = config_name if config_name else f"camera_{i+1}_realsense"
                
                cameras[cam_name] = RealSenseCameraConfig(
                    fps=cam.get("fps", 30),
                    width=cam.get("width", 848),
                    height=cam.get("height", 480),
                    serial_number_or_name=serial
                )
                print(f"Added {cam_name} (RealSense {serial})")
            
            if not cameras:
                print("⚠️ No cameras found! Robot might fail to initialize if it requires cameras.")

            # --- ROBUSTNESS: Verify cameras can actually connect before passing to Robot ---
            print("Verifying camera connections...")
            verified_cameras = {}
            for name, cfg in cameras.items():
                try:
                    # Instantiate and test connection temporarily
                    if isinstance(cfg, OpenCVCameraConfig):
                        test_cam = OpenCVCamera(cfg)
                    elif isinstance(cfg, RealSenseCameraConfig):
                        # skip verification for RealSense to avoid USB reset/busy issues
                        print(f"Skipping stress verification for RealSense {name} to avoid USB race conditions.")
                        verified_cameras[name] = cfg
                        continue
                    else:
                        print(f"Skipping verification for unknown type {type(cfg)}")
                        verified_cameras[name] = cfg
                        continue

                    # Attempt connection with warmup to verify READ works
                    test_cam.connect(warmup=False) 
                    
                    # Stress test: Read multiple frames to ensure stability (filter out flaky video4)
                    stress_passed = True
                    for _ in range(10):
                        if test_cam.read() is None:
                            stress_passed = False
                            break
                            
                    if stress_passed and test_cam.is_connected:
                        print(f"  [OK] Camera {name} connected and passed stress test.")
                        test_cam.disconnect()
                        verified_cameras[name] = cfg
                    else:
                         print(f"  [FAIL] Camera {name} failed check (unstable/null frames). Ignoring.")
                except Exception as e:
                    print(f"  [FAIL] Camera {name} failed to connect: {e}. Ignoring.")
                finally:
                    # Clean up if connected or thread started
                    try:
                        if 'test_cam' in locals() and test_cam is not None:
                            # Check if connected or if explicit disconnect needed
                            # OpenCVCamera.disconnect() is safe to call if not connected? 
                            # It raises DeviceNotConnectedError if not connected.
                            if test_cam.is_connected or (hasattr(test_cam, 'thread') and test_cam.thread is not None):
                                test_cam.disconnect()
                    except Exception as cleanup_err:
                        # Ignore cleanup errors (e.g. already disconnected)
                        pass
                
                # Tiny sleep to ensure OS releases the device handle
                import time
                time.sleep(0.2)
            
            cameras = verified_cameras
            print(f"Proceeding with {len(cameras)} verified cameras.")
            # -------------------------------------------------------------------------------
            
            # Construct Robot Config
            # We assume bi_umbra_follower for now as per user request
            if robot_cfg.get("type") == "bi_umbra_follower":
                r_config = BiUmbraFollowerConfig(
                    left_arm_port=robot_cfg["left_arm_port"],
                    right_arm_port=robot_cfg["right_arm_port"],
                    cameras=cameras
                )
                
                try:
                    self.robot = make_robot_from_config(r_config)
                    self.robot.connect(calibrate=False)
                    print("✅ Real Robot Connected Successfully!")
                    
                    # Initialize Leader if configured
                    if teleop_cfg and teleop_cfg.get("type") == "bi_umbra_leader":
                        print("Initializing BiUmbraLeader...")
                        from lerobot.teleoperators.bi_umbra_leader.bi_umbra_leader import BiUmbraLeader
                        from lerobot.teleoperators.bi_umbra_leader.config_bi_umbra_leader import BiUmbraLeaderConfig
                        
                        l_config = BiUmbraLeaderConfig(
                            id="bi_umbra_leader_main",  # Explicit ID to avoid 'None.json'
                            left_arm_port=teleop_cfg["left_arm_port"],
                            right_arm_port=teleop_cfg["right_arm_port"],
                            calibration_dir=Path(teleop_cfg.get("calibration_dir", ".cache/calibration"))
                        )
                        self.leader = BiUmbraLeader(l_config)
                        self.leader.connect(calibrate=False)
                        print("✅ Leader Arms Connected Successfully!")
                        
                        # Initialize Leader Assist Services
                        self.leader_assists = {}
                        if hasattr(self.leader, "left_arm") and hasattr(self.leader, "right_arm"):
                             self.leader_assists["left"] = LeaderAssistService(arm_id="left_leader")
                             self.leader_assists["right"] = LeaderAssistService(arm_id="right_leader")
                        else:
                             self.leader_assists["default"] = LeaderAssistService(arm_id="leader")
                             
                    else:
                        self.leader = None
                        
                except Exception as e:
                    print(f"⚠️ Robot Connection Failed (will fallback to Mock): {e}")
                    import traceback
                    traceback.print_exc()
                    self.robot = None
                    self.leader = None

            else:
                print(f"Unknown robot type: {robot_cfg.get('type')}. Skipping connection.")
                self.robot = None
                self.leader = None

        except Exception as e:
            import traceback
            err_msg = f"⚠️ Failed to connect robot: {e}\n{traceback.format_exc()}"
            print(err_msg)
            with open("startup_error.log", "w") as f:
                f.write(err_msg)
                
            print("Falling back to Mock Robot for MVP.")
            self.robot = None
            self.leader = None

        self.recorder = DataRecorder(repo_id="roberto/nextis_data", robot_type="bi_umbra_follower")
        
        # We need a mock robot for the orchestrator if real one isn't there
        if self.robot is None:
            from unittest.mock import MagicMock
            self.robot = MagicMock()
            self.robot.is_connected = False
            self.robot.is_mock = True # Flag as Mock
            self.robot.capture_observation.return_value = {}

        self.orchestrator = TaskOrchestrator(self.robot, self.recorder, robot_lock=self.lock)
        self.orchestrator.start() # Start the orchestrator and intervention engine
        


        # Initialize Planner
        try:
            from app.core.planner import LocalPlanner, GeminiPlanner
            
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key and gemini_key.strip():
                print("✨ GEMINI_API_KEY found! Using GeminiPlanner.")
                self.planner = GeminiPlanner(api_key=gemini_key)
            else:
                print("⚠️ GEMINI_API_KEY not found. Using LocalPlanner (Qwen).")
                self.planner = LocalPlanner("Qwen/Qwen2.5-7B-Instruct", device="cuda")
                
        except Exception as e:
            print(f"⚠️ Failed to load Planner: {e}")
            self.planner = None
            
        if self.robot or self.leader:
            self.calibration_service = CalibrationService(self.robot, self.leader, robot_lock=self.lock)
            self.calibration_service.restore_active_profiles()
            
            # Initialize Teleoperation
            self.teleop_service = TeleoperationService(self.robot, self.leader, self.lock, leader_assists=getattr(self, 'leader_assists', {}))



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
    """Attempts to re-initialize hardware without killing the server."""
    if system.is_initializing:
        return {"status": "busy", "message": "Already initializing..."}
        
    print("Manual Reconnect Requested.")
    
    def async_init():
        max_retries = 3
        for attempt in range(max_retries):
            print(f"Reconnect Attempt {attempt+1}/{max_retries}...")
            
            # 1. Force Disconnect with Safety
            with system.lock:
                # Disconnect Robot
                if system.robot:
                    try: 
                        print("Disconnecting Robot...")
                        system.robot.disconnect()
                    except Exception as e: 
                        print(f"Warning: Robot disconnect failed: {e}")
                    system.robot = None
                    
                # Disconnect Leader
                if system.leader:
                    try: 
                        print("Disconnecting Leader...")
                        system.leader.disconnect()
                    except Exception as e: 
                        print(f"Warning: Leader disconnect failed: {e}")
                    system.leader = None
            
            # 2. Wait for OS to release ports (Critical)
            import time
            print("Waiting for ports to release...")
            time.sleep(2.0)
            
            # 3. Try Initialize
            print("Initializing System...")
            system.initialize()
            
            # 4. Check Success
            if not system.init_error and system.robot and system.robot.is_connected:
                print("Reconnect SUCCESS!")
                return
            
            # 5. If failed, wait before retry
            print(f"Reconnect Failed (Error: {system.init_error}). Retrying in 3s...")
            time.sleep(3.0)
            
        print("All Reconnect Attempts Failed.")

    background_tasks.add_task(async_init)
    return {"status": "initializing", "message": "Reconnecting hardware..."}

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "online", "service": "nextis-robotics"}


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
        if system.robot and system.robot.is_connected and system.robot.cameras and camera_key in system.robot.cameras:
            cam = system.robot.cameras[camera_key]
            try:
                frame = cam.async_read()
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
                            
        # PRIORITY 3: Snapshot Fallback
        if frame is None and system.camera_service:
            robot_has_cam = (system.robot and system.robot.cameras and camera_key in system.robot.cameras)
            if not robot_has_cam:
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
        
        time.sleep(0.03) # Limit to ~30 FPS

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
    active_arms = data.get("active_arms", None) # List[str] e.g. ["left_leader", "left_follower"]
    
    # Lazy Initialize Teleop Service if needed
    if not system.teleop_service:
        if system.robot:
             from app.core.teleop_service import TeleoperationService
             # Ensure leader is available (BiUmbraLeader or similar)
             system.teleop_service = TeleoperationService(system.robot, system.leader, system.lock, leader_assists=system.leader_assists)
        else:
             return {"status": "error", "message": "Teleop Service not initialized: Robot not connected"}

    if system.teleop_service:
        try:
            system.teleop_service.start(force=force, active_arms=active_arms)
        except Exception as e:
            print(f"Teleop Start Failed: {e}")
            return {"status": "error", "message": str(e)}
            
    return {"status": "started"}

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
    if not system.robot:
        raise HTTPException(status_code=503, detail="Robot not connected")
    
    if not system.calibration_service:
        raise HTTPException(status_code=503, detail="Calibration service not available")

    success = system.calibration_service.perform_homing(arm_id)
    if not success:
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
        return {"inversions": {}}
    return {"inversions": system.calibration_service.get_inversions(arm_id)}

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
async def start_discovery(arm_id: str):
    if system.calibration_service:
        system.calibration_service.start_discovery(arm_id)
    return {"status": "started"}

@app.post("/calibration/{arm_id}/discovery/stop")
async def stop_discovery(arm_id: str):
    if system.calibration_service:
        system.calibration_service.stop_discovery(arm_id)
    return {"status": "stopped"}

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
    system.camera_service.update_camera_config(data)
    
    # Trigger System Reload in Background
    background_tasks.add_task(system.reload)
    
    return {"status": "success", "message": "Camera config updated. System reloading..."}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
