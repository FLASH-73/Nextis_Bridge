from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.dependencies import get_state

router = APIRouter(tags=["hil"])


@router.post("/hil/session/start")
async def start_hil_session(request: Request):
    """Start a HIL deployment session with policy and intervention dataset."""
    system = get_state()
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


@router.post("/hil/session/stop")
def stop_hil_session():
    """Stop the current HIL session and finalize recording."""
    system = get_state()
    if not system.hil_service:
        return JSONResponse(status_code=503, content={"error": "HIL service not initialized"})

    try:
        return system.hil_service.stop_session()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/hil/episode/start")
def start_hil_episode():
    """Start a new HIL episode (begin recording)."""
    system = get_state()
    if not system.hil_service:
        return JSONResponse(status_code=503, content={"error": "HIL service not initialized"})

    try:
        return system.hil_service.start_episode()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/hil/episode/stop")
def stop_hil_episode():
    """Stop current HIL episode and save data."""
    system = get_state()
    if not system.hil_service:
        return JSONResponse(status_code=503, content={"error": "HIL service not initialized"})

    try:
        return system.hil_service.stop_episode()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/hil/episode/next")
def next_hil_episode():
    """
    Stop current episode and immediately start next one.

    Used when human finishes intervention and wants robot to try again.
    Saves the current episode data, then starts a new autonomous episode.
    """
    system = get_state()
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


@router.post("/hil/resume")
def resume_hil_autonomous():
    """
    Explicitly resume autonomous mode after intervention pause.

    Called when user clicks "Resume Autonomous" button after intervention
    ends and system is in PAUSED state.
    """
    system = get_state()
    if not system.hil_service:
        return JSONResponse(status_code=503, content={"error": "HIL service not initialized"})

    try:
        return system.hil_service.resume_autonomous()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/hil/status")
def get_hil_status():
    """Get current HIL session status."""
    system = get_state()
    if not system.hil_service:
        return {"active": False}

    return system.hil_service.get_status()


@router.patch("/hil/settings")
async def update_hil_settings(request: Request):
    """Update HIL settings during an active session (e.g., movement_scale)."""
    system = get_state()
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


@router.post("/hil/retrain")
async def trigger_hil_retrain(request: Request):
    """Trigger retraining on intervention data."""
    system = get_state()
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
