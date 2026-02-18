import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.dependencies import get_state

logger = logging.getLogger(__name__)
router = APIRouter(tags=["tools"])


def _no_registry():
    return JSONResponse(status_code=503, content={"error": "Tool registry not initialized"})


def _no_listener():
    return JSONResponse(status_code=503, content={"error": "Trigger listener not initialized"})


# ── Tools (static routes first) ──────────────────────────────────────


@router.get("/tools")
async def list_tools():
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    return system.tool_registry.get_all_tools()


@router.post("/tools")
async def add_tool(request: Request):
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    data = await request.json()
    result = system.tool_registry.add_tool(data)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


# Static: /tools/scan must come before /tools/{tool_id}
@router.post("/tools/scan")
def scan_tool_motors(request: Request):
    """Scan a serial port for tool motors (blocks on serial I/O)."""
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    import asyncio
    data = asyncio.get_event_loop().run_until_complete(request.json())
    result = system.tool_registry.scan_tool_motors(data["port"], data["motor_type"])
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


# ── Tools (parameterized routes) ─────────────────────────────────────


@router.put("/tools/{tool_id}")
async def update_tool(tool_id: str, request: Request):
    """Update a tool's settings (name, config, etc.)."""
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    data = await request.json()
    result = system.tool_registry.update_tool(tool_id, **data)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


@router.delete("/tools/{tool_id}")
async def remove_tool(tool_id: str):
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    result = system.tool_registry.remove_tool(tool_id)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


@router.post("/tools/{tool_id}/connect")
def connect_tool(tool_id: str):
    """Connect a tool motor (blocks on serial I/O)."""
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    result = system.tool_registry.connect_tool(tool_id)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


@router.post("/tools/{tool_id}/disconnect")
def disconnect_tool(tool_id: str):
    """Disconnect a tool motor (blocks on serial I/O)."""
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    result = system.tool_registry.disconnect_tool(tool_id)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


@router.post("/tools/{tool_id}/activate")
def activate_tool(tool_id: str, request: Request):
    """Activate a tool with optional speed/direction (blocks on serial I/O)."""
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    import asyncio
    try:
        data = asyncio.get_event_loop().run_until_complete(request.json())
    except Exception:
        data = {}
    kwargs = {}
    if "speed" in data:
        kwargs["speed"] = data["speed"]
    if "direction" in data:
        kwargs["direction"] = data["direction"]
    result = system.tool_registry.activate_tool(tool_id, **kwargs)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


@router.post("/tools/{tool_id}/deactivate")
def deactivate_tool(tool_id: str):
    """Deactivate a tool (blocks on serial I/O)."""
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    result = system.tool_registry.deactivate_tool(tool_id)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


# ── Triggers (static routes first) ───────────────────────────────────


@router.get("/triggers")
async def list_triggers():
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    return system.tool_registry.get_all_triggers()


@router.post("/triggers")
async def add_trigger(request: Request):
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    data = await request.json()
    result = system.tool_registry.add_trigger(data)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


# Static: /triggers/identify must come before /triggers/{trigger_id}
@router.post("/triggers/identify")
def identify_trigger(request: Request):
    """Identify a trigger device on a serial port (blocks on serial I/O)."""
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    import asyncio
    data = asyncio.get_event_loop().run_until_complete(request.json())
    result = system.tool_registry.identify_trigger_device(data["port"])
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


# ── Triggers (parameterized routes) ──────────────────────────────────


@router.delete("/triggers/{trigger_id}")
async def remove_trigger(trigger_id: str):
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    result = system.tool_registry.remove_trigger(trigger_id)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


# ── Tool Pairings (all static) ───────────────────────────────────────


@router.get("/tool-pairings")
async def list_tool_pairings():
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    return system.tool_registry.get_all_tool_pairings()


@router.post("/tool-pairings")
async def create_tool_pairing(request: Request):
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    data = await request.json()
    result = system.tool_registry.create_tool_pairing(**data)
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


@router.delete("/tool-pairings")
async def remove_tool_pairing(request: Request):
    system = get_state()
    if not system.tool_registry:
        return _no_registry()
    data = await request.json()
    result = system.tool_registry.remove_tool_pairing(data["trigger_id"], data["tool_id"])
    if not result.get("success"):
        return JSONResponse(status_code=400, content=result)
    return result


# ── Trigger Listener Control ─────────────────────────────────────────


@router.post("/tool-pairings/listener/start")
def start_listener():
    """Start the trigger listener (spawns serial reader threads)."""
    system = get_state()
    if not system.trigger_listener:
        return _no_listener()
    result = system.trigger_listener.start()
    if not result.get("success"):
        return JSONResponse(
            status_code=400,
            content={**result, "detail": result.get("error", "Failed to start listener")},
        )
    return result


@router.post("/tool-pairings/listener/stop")
def stop_listener():
    """Stop the trigger listener (joins serial reader threads)."""
    system = get_state()
    if not system.trigger_listener:
        return _no_listener()
    system.trigger_listener.stop()
    return {"success": True, "message": "Trigger listener stopped"}


@router.get("/tool-pairings/listener/status")
async def listener_status():
    system = get_state()
    if not system.trigger_listener:
        return {
            "running": False,
            "trigger_states": {},
            "tool_states": {},
            "trigger_count": 0,
            "tool_pairing_count": 0,
            "ports": [],
        }
    listener = system.trigger_listener
    registry = system.tool_registry
    running = any(t.is_alive() for t in listener._port_threads.values())
    return {
        "running": running,
        "trigger_states": listener.get_trigger_states(),
        "tool_states": listener.get_tool_states(),
        "trigger_count": len(registry.triggers) if registry else 0,
        "tool_pairing_count": len(registry.tool_pairings) if registry else 0,
        "ports": list(listener._port_threads.keys()),
    }
