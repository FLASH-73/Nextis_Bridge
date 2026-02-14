import sys
import time
import logging

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from app.core.config import load_config
from app.dependencies import get_state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


def delayed_restart():
    """Waits briefly then forces a system restart."""
    time.sleep(0.5)
    get_state().restart()


@router.post("/system/restart")
def restart_system(background_tasks: BackgroundTasks):
    # Queue restart to allow response to send first
    background_tasks.add_task(delayed_restart)
    return {"status": "restarting", "message": "System is restarting..."}


@router.post("/system/reconnect")
def reconnect_system(background_tasks: BackgroundTasks):
    """Reinitialize services (arms are connected via UI)."""
    system = get_state()
    if system.is_initializing:
        return {"status": "busy", "message": "Already initializing..."}
    print("Manual Reconnect Requested — reinitializing services...")
    background_tasks.add_task(system.initialize)
    return {"status": "initializing", "message": "Reinitializing services..."}


@router.post("/system/reset")
async def reset_system(background_tasks: BackgroundTasks):
    """Soft reset attempts to re-initialize hardware without killing the process."""
    system = get_state()
    try:
        # Run reload in background to avoid blocking return
        background_tasks.add_task(system.reload)
        return {"status": "success", "message": "System reset initiated..."}
    except Exception as e:
        print(f"Failed to reset system: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/")
def read_root():
    return {"status": "online", "service": "nextis-robotics"}


@router.get("/test/ping")
def test_ping():
    """Simple test endpoint to verify API connection."""
    print(">>> /test/ping called - API is working!")
    sys.stdout.flush()
    return {"status": "pong", "message": "API connection verified"}


@router.get("/status")
def get_status():
    system = get_state()
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


@router.get("/config")
def get_config():
    return load_config()


@router.post("/emergency/stop")
def emergency_stop():
    system = get_state()
    print("\U0001f6a8 EMERGENCY STOP TRIGGERED \U0001f6a8")
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
