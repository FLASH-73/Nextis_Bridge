import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel
from app.dependencies import get_state

router = APIRouter(tags=["datasets"])


class MergeValidateRequest(BaseModel):
    repo_ids: list[str]

class MergeStartRequest(BaseModel):
    repo_ids: list[str]
    output_repo_id: str


@router.get("/datasets")
def list_datasets():
    system = get_state()
    if not system.dataset_service:
         return []
    return system.dataset_service.list_datasets()

@router.get("/datasets/{repo_id:path}/episodes")
def get_dataset_episodes(repo_id: str):
    system = get_state()
    if not system.dataset_service:
         return []
    try:
        dataset = system.dataset_service.get_dataset(repo_id)
        episodes = dataset.meta.episodes
        if hasattr(episodes, "to_pydict"):
             d = episodes.to_pydict()
             keys = list(d.keys())
             length = len(d[keys[0]])
             return [{k: d[k][i] for k in keys} for i in range(length)]
        elif isinstance(episodes, list):
             return episodes
        else:
             return [{"index": i} for i in range(dataset.num_episodes)]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/datasets/{repo_id:path}/episode/{index}")
def get_episode_detail(repo_id: str, index: int):
    system = get_state()
    if not system.dataset_service:
         return {}
    try:
        return system.dataset_service.get_episode_data(repo_id, index)
    except Exception as e:
         return JSONResponse(status_code=500, content={"error": str(e)})

@router.delete("/datasets/{repo_id:path}/episode/{index}")
def delete_episode_endpoint(repo_id: str, index: int):
    system = get_state()
    if not system.dataset_service:
         return {"status": "error", "message": "Dataset service not ready"}
    try:
        print(f"[DELETE_EP] Deleting episode {index} from {repo_id}")

        # Check if session is active on this dataset
        session_matches = (system.teleop_service and
            system.teleop_service.session_active and
            system.teleop_service.dataset and
            system.teleop_service.dataset.repo_id == repo_id)

        print(f"[DELETE_EP] Session active on this dataset: {session_matches}")
        if session_matches:
            print(f"[DELETE_EP] BEFORE: episode_count={system.teleop_service.episode_count}, meta.total_episodes={system.teleop_service.dataset.meta.total_episodes}")

        # CRITICAL: Flush pending data BEFORE deletion if session is active on this dataset
        # Without this, the metadata_buffer may have unflushed episode data
        if session_matches:
            system.teleop_service.sync_to_disk()

        result = system.dataset_service.delete_episode(repo_id, index)
        print(f"[DELETE_EP] delete_episode returned: {result}")

        # Refresh metadata from disk AFTER deletion if session is active
        if session_matches:
            system.teleop_service.refresh_metadata_from_disk()
            print(f"[DELETE_EP] AFTER: episode_count={system.teleop_service.episode_count}, meta.total_episodes={system.teleop_service.dataset.meta.total_episodes}")

        return result
    except Exception as e:
         return JSONResponse(status_code=500, content={"error": str(e)})

@router.delete("/datasets/{repo_id:path}")
def delete_dataset_endpoint(repo_id: str):
    """Delete an entire dataset repository."""
    system = get_state()
    if not system.dataset_service:
         return {"status": "error", "message": "Dataset service not ready"}
    try:
        result = system.dataset_service.delete_dataset(repo_id)
        return result
    except FileNotFoundError as e:
         return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
         return JSONResponse(status_code=500, content={"error": str(e)})


# --- Dataset Merge Endpoints ---

@router.post("/datasets/merge/validate")
async def validate_merge(request: MergeValidateRequest):
    """Validate that datasets can be merged (same fps, robot_type, features)."""
    system = get_state()
    if not system.dataset_service:
        return JSONResponse(status_code=503, content={"error": "Dataset service not ready"})

    try:
        result = system.dataset_service.validate_merge(request.repo_ids)
        return {
            "compatible": result.compatible,
            "datasets": result.datasets,
            "merged_info": result.merged_info,
            "errors": result.errors,
            "warnings": result.warnings
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/datasets/merge/start")
async def start_merge(request: MergeStartRequest):
    """Start a background merge job."""
    system = get_state()
    if not system.dataset_service:
        return JSONResponse(status_code=503, content={"error": "Dataset service not ready"})

    # Validate first
    validation = system.dataset_service.validate_merge(request.repo_ids)
    if not validation.compatible:
        return JSONResponse(status_code=400, content={
            "error": "Datasets are not compatible for merge",
            "details": validation.errors
        })

    # Check output name doesn't exist
    output_path = system.dataset_service.base_path / request.output_repo_id
    if output_path.exists():
        return JSONResponse(status_code=400, content={
            "error": f"Dataset '{request.output_repo_id}' already exists"
        })

    try:
        job = system.dataset_service.start_merge_job(request.repo_ids, request.output_repo_id)
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "message": "Merge job started"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/datasets/merge/status/{job_id}")
def get_merge_status(job_id: str):
    """Get status of a merge job."""
    system = get_state()
    if not system.dataset_service:
        return JSONResponse(status_code=503, content={"error": "Dataset service not ready"})

    job = system.dataset_service.get_merge_job_status(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "error": job.error,
        "output_repo_id": job.output_repo_id,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None
    }


# --- Video Streaming ---

@router.get("/api/datasets/{repo_id:path}/video/{index}/{key}")
def stream_video(repo_id: str, index: int, key: str):
    """Stream video for an episode, handling LeRobot's concatenated video files."""
    system = get_state()
    if not system.dataset_service:
         return Response(status_code=404)

    # CORS headers for cross-origin video canvas access
    cors_headers = {"Access-Control-Allow-Origin": "*"}

    try:
        import pandas as pd
        dataset_root = system.dataset_service.base_path / repo_id
        video_root = dataset_root / "videos" / key

        # Try to load episode metadata to get correct file_index
        # LeRobot concatenates all episodes into a single file (file-000.mp4)
        # and tracks each episode's position via from_timestamp/to_timestamp
        file_index = index  # Default: assume episode_index == file_index
        chunk_index = 0

        episodes_path = dataset_root / "meta" / "episodes"
        if episodes_path.exists():
            try:
                episodes_df = pd.read_parquet(episodes_path)
                episode_row = episodes_df[episodes_df["episode_index"] == index]
                if not episode_row.empty:
                    # Get the actual file_index from metadata (usually 0 for concatenated videos)
                    file_index_col = f"videos/{key}/file_index"
                    chunk_index_col = f"videos/{key}/chunk_index"
                    if file_index_col in episode_row.columns:
                        file_index = int(episode_row[file_index_col].iloc[0])
                    if chunk_index_col in episode_row.columns:
                        chunk_index = int(episode_row[chunk_index_col].iloc[0])
            except Exception as e:
                pass  # Fall back to using episode_index

        # Standard LeRobot v3: videos/{key}/episode_{index}.mp4 (or inside chunks)
        # Check direct first
        direct_path = video_root / f"episode_{index:06d}.mp4"
        if direct_path.exists():
             return FileResponse(direct_path, media_type="video/mp4", headers=cors_headers)

        # LeRobot v3 chunked format: chunk-XXX/file-YYY.mp4
        # Use file_index from metadata (not episode_index!)
        chunk_path = video_root / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
        if chunk_path.exists():
             return FileResponse(chunk_path, media_type="video/mp4", headers=cors_headers)

        # Try with 6-digit file index too
        chunk_path_6 = video_root / f"chunk-{chunk_index:03d}" / f"file-{file_index:06d}.mp4"
        if chunk_path_6.exists():
             return FileResponse(chunk_path_6, media_type="video/mp4", headers=cors_headers)

        # Fallback: try with episode_index as file_index (for older datasets)
        if file_index != index:
            chunk_path_fallback = video_root / "chunk-000" / f"file-{index:03d}.mp4"
            if chunk_path_fallback.exists():
                return FileResponse(chunk_path_fallback, media_type="video/mp4", headers=cors_headers)

        # Last resort: glob for any matching file
        matches = list(video_root.rglob(f"*file-{file_index:03d}.mp4"))
        if not matches:
            matches = list(video_root.rglob(f"*file-{index:03d}.mp4"))
        if matches:
             return FileResponse(matches[0], media_type="video/mp4", headers=cors_headers)

        return Response(content=f"Video not found for episode {index} (file_index={file_index}) in {video_root}", status_code=404)

    except Exception as e:
         return Response(content=str(e), status_code=500)


# --- Cloud Upload Support Endpoints ---

@router.get("/datasets/{repo_id:path}/files")
def list_dataset_files(repo_id: str):
    """List all files in a dataset with their relative paths and sizes for cloud upload."""
    system = get_state()
    if not system.dataset_service:
        return JSONResponse(status_code=503, content={"error": "Dataset service not ready"})

    try:
        dataset_root = system.dataset_service.base_path / repo_id
        if not dataset_root.exists():
            return JSONResponse(status_code=404, content={"error": f"Dataset {repo_id} not found"})

        files = []
        for dirpath, _, filenames in os.walk(dataset_root):
            for filename in filenames:
                if filename.startswith('.'):
                    continue  # Skip hidden files
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, dataset_root)
                files.append({
                    "path": rel_path,
                    "size": os.path.getsize(full_path)
                })

        return files
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/datasets/{repo_id:path}/file/{file_path:path}")
def get_dataset_file(repo_id: str, file_path: str):
    """Serve a specific file from a dataset for cloud upload."""
    system = get_state()
    if not system.dataset_service:
        return Response(status_code=503, content="Dataset service not ready")

    try:
        dataset_root = system.dataset_service.base_path / repo_id
        full_path = dataset_root / file_path

        # Security check - ensure path is within dataset root
        if not full_path.resolve().is_relative_to(dataset_root.resolve()):
            return Response(status_code=403, content="Access denied")

        if not full_path.exists():
            return Response(status_code=404, content=f"File not found: {file_path}")

        return FileResponse(full_path)
    except Exception as e:
        return Response(status_code=500, content=str(e))
