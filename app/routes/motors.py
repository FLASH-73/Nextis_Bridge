from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.dependencies import get_state

router = APIRouter(tags=["motors"])


@router.post("/motors/scan")
async def scan_motors(request: Request):
    """Scan a port for connected motors.

    IMPORTANT: For reliable results, connect only ONE motor at a time.

    Request body:
        port: Serial port path (e.g., /dev/ttyACM0)
        motor_type: Motor type (dynamixel_xl330, dynamixel_xl430, sts3215)

    Returns:
        found_ids: List of motor IDs responding on the bus
    """
    system = get_state()
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


@router.post("/motors/set-id")
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
    system = get_state()
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


@router.post("/motors/recover")
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
    system = get_state()
    if not system.arm_registry:
        return JSONResponse(status_code=400, content={"error": "Arm registry not initialized"})

    data = await request.json()
    port = data.get("port")
    motor_type = data.get("motor_type")

    if not all([port, motor_type]):
        return JSONResponse(status_code=400, content={"error": "port and motor_type are required"})

    result = system.arm_registry.recover_motor(port, motor_type)
    return result


@router.post("/motors/ping")
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
    system = get_state()
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
