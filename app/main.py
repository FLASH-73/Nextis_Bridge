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

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.core.config import load_config
from app.core.orchestrator import TaskOrchestrator
from app.core.recorder import DataRecorder
from app.core.calibration_service import CalibrationService
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
        self.lock = threading.Lock()


    def initialize(self):
        print("Initializing System...")
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
            print("Scanning for available cameras... [SKIPPED FOR DEBUG]")
            # discovered = discover_cameras()
            discovered = {"opencv": [], "realsense": []}
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
                    print(f"  -> NO MATCH found for device '{cam_id}'. Will use auto-generated name.")

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
                        test_cam = RealSenseCamera(cfg)
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
                            left_arm_port=teleop_cfg["left_arm_port"],
                            right_arm_port=teleop_cfg["right_arm_port"],
                            calibration_dir=Path(teleop_cfg.get("calibration_dir", ".cache/calibration"))
                        )
                        self.leader = BiUmbraLeader(l_config)
                        self.leader.connect(calibrate=False)
                        print("✅ Leader Arms Connected Successfully!")
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
            self.robot.capture_observation.return_value = {}

        self.orchestrator = TaskOrchestrator(self.robot, self.recorder, robot_lock=self.lock)
        self.orchestrator.start() # Start the orchestrator and intervention engine
        
        # Initialize Calibration Service
        if self.robot or self.leader:
            self.calibration_service = CalibrationService(self.robot, self.leader, robot_lock=self.lock)

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
            
        # Initialize Calibration Service
        if self.robot or self.leader:
            self.calibration_service = CalibrationService(self.robot, self.leader, robot_lock=self.lock)

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
             self.reload() # Reuse reload to stop threads/connections
        except:
             pass
             
        # os._exit(42) forces exit without cleanup/exception handling, guaranteeing the return code
        import os
        os._exit(42)

system = SystemState()

@app.on_event("startup")
async def startup_event():
    system.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    if system.orchestrator:
        system.orchestrator.stop()



@app.post("/system/restart")
def restart_system(background_tasks: BackgroundTasks):
    # Schedule restart slightly in future to allow response to return
    def _restart():
        import time
        time.sleep(1) # Give time for response to flush
        if system:
             system.restart()
        else:
             # Fallback if system not init
             import sys
             import os
             os._exit(42)
             
    background_tasks.add_task(_restart)
    return {"status": "restarting", "message": "System is restarting. Please wait 10 seconds."}

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "online", "service": "nextis-robotics"}

@app.get("/status")
def get_status():
    status = "IDLE"
    if system.orchestrator:
        if system.orchestrator.active_policy:
            status = f"EXECUTING: {system.orchestrator.task_chain[system.orchestrator.current_task_index]}"
        elif system.orchestrator.intervention_engine.is_human_controlling:
            status = "INTERVENTION (RECORDING)"
    
    return {
        "system_status": status,
        "left_arm": "CONNECTED" if system.robot else "MOCK",
        "right_arm": "CONNECTED" if system.robot else "MOCK",
        "fps": 30.0
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
        # Get frame from orchestrator -> intervention_engine -> latest_observation
        frame = None
        if system.orchestrator and system.orchestrator.intervention_engine:
            obs = system.orchestrator.intervention_engine.latest_observation
            
            # Observation keys are usually "observation.images.camera_1"
            # But the user config has "camera_1", "camera_2".
            # LeRobot usually formats keys as f"observation.images.{name}"
            
            if obs:
                # Try to find the key
                full_key = f"observation.images.{camera_key}"
                if full_key in obs:
                    frame = obs[full_key]
                elif camera_key in obs: # Fallback
                    frame = obs[camera_key]
                else:
                    # Try partial match
                    for k in obs.keys():
                        if camera_key in k:
                            frame = obs[k]
                            break
        
        if frame is None:
            # Fallback 1: If Robot is connected, ask IT for the frame (reuses existing connection)
            if system.robot and system.robot.is_connected and system.robot.cameras and camera_key in system.robot.cameras:
                cam = system.robot.cameras[camera_key]
                try:
                    # Using async_read because Orchestrator might be running background thread
                    frame = cam.async_read()
                except Exception as e:
                    # print(f"Robot cam read failed: {e}")
                    pass

            # Fallback 2: Try to get a direct snapshot if the robot isn't streaming it
            # This is useful for the settings preview.
            if frame is None and system.camera_service:
                snapshot = system.camera_service.capture_snapshot(camera_key)
                if snapshot is not None:
                    frame = snapshot
            
            if frame is None:
                # Yield a placeholder (black frame with text)
                import numpy as np
                blank_image = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(blank_image, f"Waiting for {camera_key}...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frame = blank_image
            else:
                # Ensure snapshot is BGR (it comes from cv2 so it should be)
                pass
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
             system.teleop_service = TeleoperationService(system.robot, system.leader, system.lock)
        else:
             return {"status": "error", "message": "Teleop Service not initialized: Robot not connected"}

    if system.teleop_service:
        try:
            system.teleop_service.start(force=force, active_arms=active_arms)
        except Exception as e:
            logger.error(f"Teleop Start Failed: {e}")
            return {"status": "error", "message": str(e)}
            
    return {"status": "started"}

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
        
        return {
            "opencv": final_opencv,
            "realsense": final_realsense,
            "note": "Merged active and available cameras."
        }

    return system.camera_service.scan_cameras()

@app.get("/cameras/config")
def get_camera_config():
    if not system.camera_service:
        return {}
    return system.camera_service.get_camera_config()

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
