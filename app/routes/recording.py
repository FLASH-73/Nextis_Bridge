import sys
import logging

from fastapi import APIRouter, Request
from app.dependencies import get_state

_recording_logger = logging.getLogger("recording_debug")

router = APIRouter(tags=["recording"])


@router.get("/recording/options")
def get_recording_options():
    """Returns available cameras and arm pairs for recording selection."""
    system = get_state()
    cameras = []
    arms = []

    # Get cameras from robot or camera config
    if system.robot and system.robot.is_connected:
        if hasattr(system.robot, 'cameras') and system.robot.cameras:
            for cam_key in sorted(system.robot.cameras.keys()):
                cameras.append({
                    "id": cam_key,
                    "name": cam_key.replace("_", " ").title()
                })

        # Get arm pairs - check for bi-arm setup
        if hasattr(system.robot, 'left_arm') or hasattr(system.robot, 'right_arm'):
            if hasattr(system.robot, 'left_arm') and system.robot.left_arm:
                arms.append({"id": "left", "name": "Left Arm", "joints": 7})
            if hasattr(system.robot, 'right_arm') and system.robot.right_arm:
                arms.append({"id": "right", "name": "Right Arm", "joints": 7})
        else:
            # Single arm or default setup
            arms.append({"id": "default", "name": "Robot Arm", "joints": 7})
    else:
        # Fallback: try to get cameras from config
        try:
            camera_service = system.camera_service
            cam_configs = camera_service.get_camera_config()
            for cam in cam_configs:
                cameras.append({
                    "id": cam.get("id", "unknown"),
                    "name": cam.get("id", "unknown").replace("_", " ").title()
                })
        except:
            pass

    return {"cameras": cameras, "arms": arms}

@router.post("/recording/session/start")
async def start_recording_session(request: Request):
    system = get_state()
    print("\n>>> API: /recording/session/start called")
    _recording_logger.info("API: /recording/session/start called")
    sys.stdout.flush()

    data = await request.json()
    repo_id = data.get("repo_id")
    task = data.get("task")
    selected_cameras = data.get("selected_cameras")  # list of camera IDs or None (all)
    selected_arms = data.get("selected_arms")        # list of arm IDs ("left", "right") or None (all)
    print(f"    repo_id={repo_id}, task={task}, cameras={selected_cameras}, arms={selected_arms}")
    _recording_logger.info(f"  repo_id={repo_id}, task={task}, cameras={selected_cameras}, arms={selected_arms}")

    if not repo_id or not task:
        print("    ERROR: Missing repo_id or task")
        _recording_logger.error("Missing repo_id or task")
        return {"status": "error", "message": "Missing repo_id or task"}

    if not system.teleop_service:
        print("    ERROR: Teleop Service not active")
        _recording_logger.error("Teleop Service not active")
        return {"status": "error", "message": "Teleop Service not active"}

    try:
        _recording_logger.info("Calling teleop_service.start_recording_session...")
        system.teleop_service.start_recording_session(
            repo_id, task,
            selected_cameras=selected_cameras,
            selected_arms=selected_arms
        )
        episode_count = system.teleop_service.episode_count
        print(f"    SUCCESS: Session started (episode_count={episode_count})")
        _recording_logger.info(f"SUCCESS: Session started (episode_count={episode_count})")
    except Exception as e:
        import traceback
        print(f"    ERROR: {e}")
        _recording_logger.error(f"ERROR: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

    return {"status": "success", "message": "Recording Session Started", "episode_count": episode_count}

@router.post("/recording/session/stop")
def stop_recording_session():
    system = get_state()
    print("\n>>> API: /recording/session/stop called")
    _recording_logger.info("API: /recording/session/stop called")
    sys.stdout.flush()

    if not system.teleop_service:
        print("    ERROR: Teleop Service not active")
        _recording_logger.error("Teleop Service not active")
        return {"status": "error", "message": "Teleop Service not active"}

    try:
        system.teleop_service.stop_recording_session()
        print("    SUCCESS: Session stopped")
        _recording_logger.info("SUCCESS: Session stopped")
    except Exception as e:
        import traceback
        _recording_logger.error(f"ERROR: {e}\n{traceback.format_exc()}")

    return {"status": "success", "message": "Recording Session Finalized"}

@router.post("/recording/episode/start")
def start_episode():
    system = get_state()
    print("\n>>> API: /recording/episode/start called")
    _recording_logger.info("API: /recording/episode/start called")
    sys.stdout.flush()

    if not system.teleop_service:
        print("    ERROR: Teleop Service not active")
        _recording_logger.error("Teleop Service not active")
        return {"status": "error", "message": "Teleop Service not active"}

    try:
        system.teleop_service.start_episode()
        print("    SUCCESS: Episode started")
        _recording_logger.info("SUCCESS: Episode started")
    except Exception as e:
        import traceback
        print(f"    ERROR: {e}")
        _recording_logger.error(f"ERROR: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

    return {"status": "success", "message": "Episode Started"}

@router.post("/recording/episode/stop")
def stop_episode():
    system = get_state()
    print("\n>>> API: /recording/episode/stop called")
    _recording_logger.info("API: /recording/episode/stop called")
    sys.stdout.flush()

    if not system.teleop_service:
        print("    ERROR: Teleop Service not active")
        _recording_logger.error("Teleop Service not active")
        return {"status": "error", "message": "Teleop Service not active"}

    try:
        system.teleop_service.stop_episode()
        episode_count = system.teleop_service.episode_count
        print(f"    SUCCESS: Episode stopped (total: {episode_count})")
        _recording_logger.info(f"SUCCESS: Episode stopped (total: {episode_count})")
        return {"status": "success", "message": "Episode Saved", "episode_count": episode_count}
    except Exception as e:
        import traceback
        _recording_logger.error(f"ERROR: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

@router.delete("/recording/episode/last")
def delete_last_episode():
    system = get_state()
    if not system.teleop_service:
         return {"status": "error", "message": "Teleop Service not active"}

    if not system.teleop_service.session_active:
         return {"status": "error", "message": "No recording session active"}

    if not system.teleop_service.dataset:
         return {"status": "error", "message": "No dataset loaded"}

    try:
        repo_id = system.teleop_service.dataset.repo_id
        current_count = system.teleop_service.episode_count

        if current_count <= 0:
            return {"status": "error", "message": "No episodes to delete"}

        # Delete the last episode (index = count - 1)
        last_index = current_count - 1

        print(f"[DELETE_LAST] Starting delete for episode {last_index}")
        print(f"[DELETE_LAST] BEFORE: episode_count={current_count}, meta.total_episodes={system.teleop_service.dataset.meta.total_episodes}")

        # CRITICAL: Flush pending episode data to disk BEFORE deletion
        # Without this, the metadata_buffer may have unflushed episode data
        # that won't be found on disk by delete_episode(), causing ghost episodes
        system.teleop_service.sync_to_disk()

        result = system.dataset_service.delete_episode(repo_id, last_index)
        print(f"[DELETE_LAST] delete_episode returned: {result}")

        # Refresh metadata from disk AFTER deletion to reload clean state
        system.teleop_service.refresh_metadata_from_disk()

        print(f"[DELETE_LAST] AFTER: episode_count={system.teleop_service.episode_count}, meta.total_episodes={system.teleop_service.dataset.meta.total_episodes}")

        return {"status": "success", "message": "Last Episode Deleted", "episode_count": system.teleop_service.episode_count}
    except Exception as e:
        import traceback
        print(f"[DELETE_LAST] ERROR: {e}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

@router.get("/recording/status")
def get_recording_status():
    system = get_state()
    if not system.teleop_service:
         return {"active": False, "episode_count": 0}

    # TeleopService.get_data() already returns recording info, but explicit endpoint helps too
    return system.teleop_service.get_data().get("recording", {})
