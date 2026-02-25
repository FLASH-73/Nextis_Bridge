import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.dependencies import get_state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["calibration"])


def _get_calibration_target(arm_key: str):
    """
    Helper to resolve (Service, ArmObject) from arm_key (e.g. 'left_leader', 'left_follower').
    Returns (service, arm) or raises Exception.
    """
    system = get_state()
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
        if not system.robot:
            raise Exception("Follower Robot not connected")
        # Match side
        if "left" in arm_key and hasattr(system.robot, "left_arm"):
            arm = system.robot.left_arm
        elif "right" in arm_key and hasattr(system.robot, "right_arm"):
            arm = system.robot.right_arm
        elif hasattr(system.robot, "bus") and not hasattr(system.robot, "left_arm"):
            arm = system.robot  # Mono robot
    else:
        if not system.leader:
            raise Exception("Leader Arm not connected")
        if "left" in arm_key and hasattr(system.leader, "left_arm"):
            arm = system.leader.left_arm
        elif "right" in arm_key and hasattr(system.leader, "right_arm"):
            arm = system.leader.right_arm
        elif hasattr(system.leader, "bus") and not hasattr(system.leader, "left_arm"):
            arm = system.leader  # Mono leader

    if not arm:
        raise Exception(f"Physical Arm interface not found for {arm_key}")

    return service, arm


@router.get("/calibration/arms")
def get_calibration_arms():
    system = get_state()
    if not system.calibration_service:
        return {"arms": []}
    return {"arms": system.calibration_service.get_arms()}

@router.get("/calibration/{arm_id}/state")
def get_calibration_state(arm_id: str):
    system = get_state()
    if not system.calibration_service:
        return {"motors": []}
    return {"motors": system.calibration_service.get_calibration_state(arm_id)}

@router.post("/calibration/{arm_id}/torque")
async def set_torque(arm_id: str, request: Request):
    system = get_state()
    data = await request.json()
    enable = data.get("enable", True)
    if system.calibration_service:
        if enable:
            system.calibration_service.enable_torque(arm_id)
        else:
            system.calibration_service.disable_torque(arm_id)
    return {"status": "success"}

@router.post("/calibration/{arm_id}/limit")
async def set_limit(arm_id: str, request: Request):
    data = await request.json()
    _motor = data.get("motor")
    _limit_type = data.get("type")  # min or max
    _value = data.get("value")


# --- Gravity Calibration (Wizard) ---

@router.post("/calibration/{arm_key}/gravity/start")
def start_gravity_calibration(arm_key: str): # arm_key: left_leader, left_follower, etc.
    try:
        service, _ = _get_calibration_target(arm_key)
        service.start_calibration()
        return {"status": "success", "message": f"Calibration Started for {arm_key}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/calibration/{arm_key}/gravity/sample")
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

@router.post("/calibration/{arm_key}/gravity/compute")
def compute_gravity_calibration(arm_key: str):
    try:
        service, _ = _get_calibration_target(arm_key)
        service.compute_weights()
        return {"status": "success", "message": "Calibration Computed and Saved"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/calibration/{arm_id}/save")
async def save_calibration(arm_id: str):
    system = get_state()
    if not system.robot:
        raise HTTPException(status_code=503, detail="Robot not connected")

    if system.calibration_service:
        system.calibration_service.save_calibration(arm_id)
    return {"status": "saved"}

@router.post("/calibration/{arm_id}/homing")
async def perform_homing(arm_id: str):
    system = get_state()
    if not system.calibration_service:
        raise HTTPException(status_code=503, detail="Calibration service not available")

    result = system.calibration_service.perform_homing(arm_id)
    if isinstance(result, dict):
        return result
    if not result:
        return {"status": "error", "message": "Homing failed"}
    return {"status": "success"}

@router.get("/calibration/{arm_id}/files")
def list_calibration_files(arm_id: str):
    system = get_state()
    if not system.calibration_service:
        return {"files": []}
    return {"files": system.calibration_service.list_calibration_files(arm_id)}

@router.post("/calibration/{arm_id}/load")
async def load_calibration_file(arm_id: str, request: Request):
    system = get_state()
    data = await request.json()
    filename = data.get("filename")
    if not system.calibration_service:
        raise HTTPException(status_code=503, detail="Service not ready")

    success = system.calibration_service.load_calibration_file(arm_id, filename)
    if not success:
         return {"status": "error", "message": "Failed to load file"}
    return {"status": "success"}

@router.post("/calibration/{arm_id}/delete")
async def delete_calibration_file(arm_id: str, request: Request):
    system = get_state()
    data = await request.json()
    filename = data.get("filename")
    if system.calibration_service:
        success = system.calibration_service.delete_calibration_file(arm_id, filename)
        return {"status": "success" if success else "error"}
    return {"status": "error"}

@router.get("/calibration/{arm_id}/inversions")
def get_inversions(arm_id: str):
    system = get_state()
    if not system.calibration_service:
        return {"inversions": {}, "motors": []}
    inversions = system.calibration_service.get_inversions(arm_id)
    _, motors = system.calibration_service.get_arm_context(arm_id)
    return {"inversions": inversions, "motors": motors}

@router.post("/calibration/{arm_id}/inversions")
async def set_inversion(arm_id: str, payload: dict):
    system = get_state()
    system.calibration_service.set_inversion(arm_id, payload["motor"], payload["inverted"])
    return {"status": "success"}

@router.post("/calibration/{arm_id}/set-zero")
async def set_zero_pose(arm_id: str):
    """Step 1: Capture Zero Pose"""
    system = get_state()
    try:
        success = system.calibration_service.set_zero_pose(arm_id)
        if success:
             return {"status": "success"}
        else:
             return {"status": "error", "message": "Arm not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/calibration/{arm_id}/auto-align")
async def auto_align(arm_id: str):
    """Step 2: Compute Inversions based on movement from Zero"""
    system = get_state()
    try:
        result = system.calibration_service.compute_auto_alignment(arm_id)
        return result
    except Exception as e:
        logger.error(f"Auto-Align failed: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/calibration/{arm_id}/save_named")
async def save_named_calibration(arm_id: str, request: Request):
    system = get_state()
    data = await request.json()
    name = data.get("name")
    if not system.robot:
         raise HTTPException(status_code=503, detail="Robot not connected")

    if system.calibration_service:
        system.calibration_service.save_calibration(arm_id, name=name)
    return {"status": "saved"}

@router.post("/calibration/{arm_id}/discovery/start")
def start_discovery(arm_id: str):
    """Start range discovery. Runs in threadpool (def not async) to avoid blocking
    the event loop during serial I/O and lock acquisition."""
    system = get_state()
    if not system.calibration_service:
        return {"status": "error", "message": "Calibration service not initialized"}
    try:
        system.calibration_service.start_discovery(arm_id)
        return {"status": "started"}
    except Exception as e:
        logger.error(f"start_discovery failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@router.post("/calibration/{arm_id}/discovery/stop")
def stop_discovery(arm_id: str):
    """Stop range discovery. Runs in threadpool (def not async) to avoid blocking
    the event loop during serial I/O and lock acquisition."""
    system = get_state()
    if not system.calibration_service:
        return {"status": "error", "message": "Calibration service not initialized"}
    try:
        result = system.calibration_service.stop_discovery(arm_id)
        return result
    except Exception as e:
        logger.error(f"stop_discovery failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
