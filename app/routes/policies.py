from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.dependencies import get_state

router = APIRouter(tags=["policies"])


@router.get("/policies")
def list_policies():
    """List all trained policies."""
    system = get_state()
    if not system.training_service:
        return []
    policies = system.training_service.list_policies()
    return [p.to_dict() for p in policies]


@router.get("/policies/{policy_id}")
def get_policy(policy_id: str):
    """Get details of a specific policy."""
    system = get_state()
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    policy = system.training_service.get_policy(policy_id)
    if not policy:
        return JSONResponse(status_code=404, content={"error": f"Policy {policy_id} not found"})

    return policy.to_dict()


@router.delete("/policies/{policy_id}")
def delete_policy(policy_id: str):
    """Delete a policy and its output directory."""
    system = get_state()
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    try:
        system.training_service.delete_policy(policy_id)
        return {"status": "deleted", "policy_id": policy_id}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.patch("/policies/{policy_id}")
async def rename_policy(policy_id: str, request: Request):
    """Rename a policy."""
    system = get_state()
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


@router.post("/policies/{policy_id}/deploy")
def deploy_policy(policy_id: str):
    """Deploy a policy for autonomous execution."""
    system = get_state()
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


@router.get("/policies/{policy_id}/config")
def get_policy_config(policy_id: str):
    """Get the input/output configuration of a trained policy.

    Returns which cameras and arms the policy was trained on.
    Useful for configuring HIL deployment to show only relevant cameras.
    """
    system = get_state()
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    try:
        config = system.training_service.get_policy_config(policy_id)
        if not config:
            return JSONResponse(status_code=404, content={"error": f"Policy {policy_id} not found or no config available"})
        return config.to_dict()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/policies/{policy_id}/resume")
async def resume_policy_training(policy_id: str, request: Request):
    """Resume training a policy from its last checkpoint."""
    system = get_state()
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
