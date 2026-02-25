from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.dependencies import get_state

router = APIRouter(tags=["rl"])


# ============================================================================
# RL Training (HIL-SERL) API Endpoints
# ============================================================================

@router.post("/rl/training/start")
async def start_rl_training(request: Request):
    """Start HIL-SERL RL training session."""
    system = get_state()
    if not system.rl_service:
        return JSONResponse(status_code=503, content={"error": "RL service not initialized"})

    try:
        data = await request.json()
        result = system.rl_service.start_training(data)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/rl/training/stop")
async def stop_rl_training():
    """Stop RL training and save policy."""
    system = get_state()
    if not system.rl_service:
        return JSONResponse(status_code=503, content={"error": "RL service not initialized"})

    return system.rl_service.stop_training()


@router.post("/rl/training/pause")
async def pause_rl_training():
    """Pause RL training."""
    system = get_state()
    if not system.rl_service:
        return JSONResponse(status_code=503, content={"error": "RL service not initialized"})

    return system.rl_service.pause_training()


@router.post("/rl/training/resume")
async def resume_rl_training():
    """Resume paused RL training."""
    system = get_state()
    if not system.rl_service:
        return JSONResponse(status_code=503, content={"error": "RL service not initialized"})

    return system.rl_service.resume_training()


@router.get("/rl/training/status")
def get_rl_training_status():
    """Get current RL training status and metrics."""
    system = get_state()
    if not system.rl_service:
        return {"status": "unavailable"}

    return system.rl_service.get_status()


@router.patch("/rl/training/settings")
async def update_rl_settings(request: Request):
    """Update RL training settings mid-training."""
    system = get_state()
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

@router.post("/rl/reward-classifier/train")
async def train_reward_classifier(request: Request):
    """Train a reward classifier from demonstration dataset."""
    system = get_state()
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


@router.get("/rl/reward-classifier/list")
def list_reward_classifiers():
    """List available trained reward classifiers."""
    system = get_state()
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


@router.get("/rl/reward-classifier/training-status")
def reward_classifier_training_status():
    """Get reward classifier training status."""
    system = get_state()
    if not system.reward_classifier_service:
        return {"status": "unavailable"}

    return system.reward_classifier_service.get_training_status()


@router.delete("/rl/reward-classifier/{name}")
def delete_reward_classifier(name: str):
    """Delete a trained reward classifier."""
    system = get_state()
    if not system.reward_classifier_service:
        return JSONResponse(status_code=503, content={"error": "Service not initialized"})

    success = system.reward_classifier_service.delete_classifier(name)
    if success:
        return {"status": "deleted", "name": name}
    return JSONResponse(status_code=404, content={"error": f"Classifier '{name}' not found"})


# ============================================================================
# GVL Reward Service API Endpoints
# ============================================================================

@router.get("/rl/gvl/status")
def get_gvl_status():
    """Get GVL reward service status."""
    system = get_state()
    if not system.gvl_reward_service:
        return {"status": "unavailable"}

    return system.gvl_reward_service.get_status()


@router.patch("/rl/gvl/config")
async def update_gvl_config(request: Request):
    """Update GVL reward service configuration."""
    system = get_state()
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

@router.post("/rl/sarm/train")
async def train_sarm(request: Request):
    """Train SARM reward model on demo dataset."""
    system = get_state()
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


@router.get("/rl/sarm/list")
def list_sarm_models():
    """List trained SARM models."""
    system = get_state()
    if not system.sarm_reward_service:
        return []

    return system.sarm_reward_service.list_sarm_models()


@router.get("/rl/sarm/training-status")
def get_sarm_training_status():
    """Get SARM training progress."""
    system = get_state()
    if not system.sarm_reward_service:
        return {"status": "unavailable"}

    return system.sarm_reward_service.get_training_status()


@router.post("/rl/sarm/stop-training")
def stop_sarm_training():
    """Stop SARM training in progress."""
    system = get_state()
    if not system.sarm_reward_service:
        return JSONResponse(status_code=503, content={"error": "SARM service not initialized"})

    return system.sarm_reward_service.stop_training()


@router.delete("/rl/sarm/{name}")
def delete_sarm_model(name: str):
    """Delete a SARM model."""
    system = get_state()
    if not system.sarm_reward_service:
        return JSONResponse(status_code=503, content={"error": "SARM service not initialized"})

    return system.sarm_reward_service.delete_sarm(name)
