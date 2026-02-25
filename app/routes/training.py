from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from app.dependencies import get_state

router = APIRouter(tags=["training"])


@router.post("/training/validate")
async def validate_training(request: Request):
    """Validate a dataset for compatibility with a policy type."""
    system = get_state()
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    data = await request.json()
    dataset_repo_id = data.get("dataset_repo_id")
    policy_type = data.get("policy_type", "smolvla")

    if not dataset_repo_id:
        return JSONResponse(status_code=400, content={"error": "dataset_repo_id is required"})

    result = system.training_service.validate_dataset(dataset_repo_id, policy_type)
    return result.to_dict()

@router.post("/training/start")
async def start_training(request: Request, background_tasks: BackgroundTasks):
    """Start a new training job."""
    system = get_state()
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

@router.get("/training/jobs")
def list_training_jobs():
    """List all training jobs."""
    system = get_state()
    if not system.training_service:
        return []
    return system.training_service.list_jobs()

@router.get("/training/jobs/{job_id}")
def get_training_job(job_id: str):
    """Get status and progress of a specific training job."""
    system = get_state()
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    try:
        return system.training_service.get_job_status(job_id)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

@router.get("/training/jobs/{job_id}/logs")
def get_training_logs(job_id: str, offset: int = 0, limit: int = 100):
    """Get logs for a training job."""
    system = get_state()
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    try:
        return system.training_service.get_job_logs(job_id, offset, limit)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

@router.post("/training/jobs/{job_id}/cancel")
def cancel_training_job(job_id: str):
    """Cancel a running training job."""
    system = get_state()
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    try:
        system.training_service.cancel_job(job_id)
        return {"status": "cancelled", "job_id": job_id}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@router.get("/training/presets")
def get_training_presets(policy_type: str = "smolvla"):
    """Get available training presets for a policy type."""
    system = get_state()
    if not system.training_service:
        return {}
    return system.training_service.get_presets(policy_type)

@router.get("/training/hardware")
def get_training_hardware():
    """Detect available training hardware (CUDA, MPS, CPU)."""
    system = get_state()
    if not system.training_service:
        return {
            "devices": [{"id": "cpu", "type": "cpu", "name": "CPU", "memory_gb": None, "recommended": True}],
            "default": "cpu"
        }
    return system.training_service.detect_hardware()


@router.get("/training/dataset/{repo_id:path}/quantiles")
def check_dataset_quantiles(repo_id: str):
    """Check if a dataset has quantile statistics needed for Pi0.5 training."""
    system = get_state()
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    return system.training_service.has_quantile_stats(repo_id)


@router.post("/training/dataset/{repo_id:path}/compute-quantiles")
def compute_dataset_quantiles(repo_id: str):
    """Compute quantile statistics for a dataset (required for Pi0.5 with default normalization).

    This can take several minutes for large datasets.
    """
    system = get_state()
    if not system.training_service:
        return JSONResponse(status_code=503, content={"error": "Training service not initialized"})

    # Run synchronously for now (could be made async with job tracking)
    result = system.training_service.compute_quantile_stats(repo_id)
    return result
