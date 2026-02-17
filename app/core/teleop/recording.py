import time
import threading
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

from app.core.config import DATASETS_DIR

_DEFAULT_DATASETS_PATH = DATASETS_DIR

# Conditional lerobot imports
try:
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import VideoEncodingManager
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.utils.constants import OBS_STR, ACTION
except ImportError:
    def precise_sleep(dt):
        time.sleep(max(0, dt))


def filter_observation_features(
    obs_features: dict,
    selected_cameras: list = None,
    selected_arms: list = None
) -> dict:
    """Filter observation features based on camera and arm selections.

    Args:
        obs_features: Full robot observation features dict
        selected_cameras: List of camera IDs to include (None = all)
        selected_arms: List of arm IDs ("left", "right") to include (None = all)

    Returns:
        Filtered observation features dict
    """
    filtered = {}

    for key, feat in obs_features.items():
        # Always include tool/trigger features
        if key.startswith("tool.") or key.startswith("trigger."):
            filtered[key] = feat
            continue

        # Check if this is a camera feature (tuple shape like (H, W, 3))
        is_camera = isinstance(feat, tuple) and len(feat) == 3

        if is_camera:
            # Camera filtering: key is the camera name (e.g., "camera_1")
            if selected_cameras is None or key in selected_cameras:
                filtered[key] = feat
        else:
            # Motor position feature: key like "left_base.pos", "right_link1.pos"
            if selected_arms is None:
                filtered[key] = feat
            elif key.startswith("left_") and "left" in selected_arms:
                filtered[key] = feat
            elif key.startswith("right_") and "right" in selected_arms:
                filtered[key] = feat
            elif not key.startswith("left_") and not key.startswith("right_"):
                # Non-arm-specific features (e.g., single-arm robot) - always include
                filtered[key] = feat

    return filtered


def filter_action_features(
    action_features: dict,
    selected_arms: list = None
) -> dict:
    """Filter action features based on arm selections.

    Args:
        action_features: Full robot action features dict
        selected_arms: List of arm IDs ("left", "right") to include (None = all)

    Returns:
        Filtered action features dict
    """
    filtered = {}

    for key, feat in action_features.items():
        # Always include tool/trigger features
        if key.startswith("tool.") or key.startswith("trigger."):
            filtered[key] = feat
            continue

        if selected_arms is None:
            filtered[key] = feat
        elif key.startswith("left_") and "left" in selected_arms:
            filtered[key] = feat
        elif key.startswith("right_") and "right" in selected_arms:
            filtered[key] = feat
        elif not key.startswith("left_") and not key.startswith("right_"):
            # Non-arm-specific features - always include
            filtered[key] = feat

    return filtered


def frame_writer_loop(svc):
    """Background thread for writing frames to dataset without blocking teleop loop."""
    print("[FRAME WRITER] Thread Started!")
    written_count = 0

    while not svc._frame_writer_stop.is_set():
        try:
            if svc._frame_queue and svc.dataset is not None:
                try:
                    frame = svc._frame_queue.popleft()
                    svc.dataset.add_frame(frame)
                    written_count += 1
                    if written_count == 1:
                        print(f"[FRAME WRITER] FIRST FRAME added to dataset buffer!")
                        # Log buffer size to confirm it's working
                        if hasattr(svc.dataset, 'episode_buffer'):
                            buf_size = svc.dataset.episode_buffer.get('size', 0)
                            print(f"[FRAME WRITER] Episode buffer size: {buf_size}")
                    elif written_count % 30 == 0:
                        print(f"[FRAME WRITER] Written {written_count} frames")
                except IndexError:
                    time.sleep(0.005)  # Queue empty, short sleep
            else:
                time.sleep(0.01)
        except Exception as e:
            import traceback
            print(f"[FRAME WRITER] ERROR adding frame: {e}")
            print(traceback.format_exc())
            time.sleep(0.01)

    # Drain remaining frames before stopping
    while svc._frame_queue and svc.dataset is not None:
        try:
            frame = svc._frame_queue.popleft()
            svc.dataset.add_frame(frame)
            written_count += 1
        except IndexError:
            break

    print(f"[FRAME WRITER] Thread Stopped (written={written_count})")


def start_frame_writer(svc):
    """Starts the background frame writer thread."""
    if svc._frame_writer_thread is not None and svc._frame_writer_thread.is_alive():
        return
    svc._frame_writer_stop.clear()
    svc._frame_writer_thread = threading.Thread(target=frame_writer_loop, args=(svc,), daemon=True, name="FrameWriter")
    svc._frame_writer_thread.start()
    logger.info("Frame Writer Thread Started")


def stop_frame_writer(svc):
    """Stops the background frame writer thread."""
    svc._frame_writer_stop.set()
    if svc._frame_writer_thread is not None:
        svc._frame_writer_thread.join(timeout=2.0)
        svc._frame_writer_thread = None
    logger.info("Frame Writer Thread Stopped")


def recording_capture_loop(svc):
    """Background thread that captures observations at recording fps.

    OPTIMIZED: Uses cached teleop data for motors (no slow hardware reads).
    Only cameras use async_read which is fast (Zero-Order Hold pattern).
    This allows reliable 30fps capture without blocking.
    """
    print(f"[REC CAPTURE] Thread started at {svc.recording_fps}fps")
    target_dt = 1.0 / svc.recording_fps

    # Wall-clock tracking for actual fps measurement
    episode_start_time = None

    while not svc._recording_stop_event.is_set():
        if svc.recording_active and svc.robot and svc.dataset is not None:
            start = time.perf_counter()

            # Track when recording actually starts
            if episode_start_time is None:
                episode_start_time = time.perf_counter()
                print(f"[REC CAPTURE] Episode recording started at wall-clock t=0")
                print(f"[REC CAPTURE] Target: {svc.recording_fps}fps ({target_dt*1000:.1f}ms per frame)")

            try:
                # FAST STRATEGY: Use cached data from teleop loop (no hardware reads!)
                # The teleop loop runs at 60Hz and caches motor positions in _latest_leader_action
                action = {}
                obs = {}

                # SOURCE 1: Get motor positions from teleop cache (FAST - no hardware read)
                # Filter based on selected arms
                with svc._action_lock:
                    if svc._latest_leader_action:
                        for key, val in svc._latest_leader_action.items():
                            # Filter by selected arms
                            if svc._selected_arms is None:
                                action[key] = val
                                obs[key] = val
                            elif key.startswith("left_") and "left" in svc._selected_arms:
                                action[key] = val
                                obs[key] = val
                            elif key.startswith("right_") and "right" in svc._selected_arms:
                                action[key] = val
                                obs[key] = val
                            elif not key.startswith("left_") and not key.startswith("right_"):
                                # Non-arm-specific features - always include
                                action[key] = val
                                obs[key] = val

                # SOURCE 2: Capture camera images with async_read (FAST - ZOH pattern)
                # Only capture selected cameras
                # Priority: CameraService (standalone) > robot.cameras (legacy)
                cameras_dict = None
                if svc.camera_service and svc.camera_service.cameras:
                    cameras_dict = svc.camera_service.cameras
                elif hasattr(svc, '_active_robot') and svc._active_robot and hasattr(svc._active_robot, 'cameras') and svc._active_robot.cameras:
                    cameras_dict = svc._active_robot.cameras
                elif hasattr(svc.robot, 'cameras') and svc.robot.cameras:
                    cameras_dict = svc.robot.cameras

                if cameras_dict:
                    cameras_to_capture = svc._selected_cameras if svc._selected_cameras else list(cameras_dict.keys())

                    for cam_key in cameras_to_capture:
                        if cam_key not in cameras_dict:
                            continue
                        cam = cameras_dict[cam_key]
                        try:
                            # async_read(blocking=False) returns last cached frame instantly
                            if hasattr(cam, 'async_read'):
                                frame = cam.async_read(blocking=False)
                                if frame is not None:
                                    obs[cam_key] = frame
                                    if svc._recording_frame_counter == 0:
                                        print(f"[REC CAPTURE] Camera {cam_key}: shape={frame.shape}")
                            # Also capture depth if enabled
                            if hasattr(cam, 'async_read_depth') and hasattr(cam.config, 'use_depth') and cam.config.use_depth:
                                depth_frame = cam.async_read_depth(blocking=False)
                                if depth_frame is not None:
                                    # Expand depth from (H, W) to (H, W, 1) for dataset compatibility
                                    if depth_frame.ndim == 2:
                                        depth_frame = depth_frame[..., np.newaxis]
                                    obs[f"{cam_key}_depth"] = depth_frame
                                    if svc._recording_frame_counter == 0:
                                        print(f"[REC CAPTURE] Depth {cam_key}_depth: shape={depth_frame.shape}")
                        except Exception as cam_err:
                            if svc._recording_frame_counter == 0:
                                print(f"[REC CAPTURE] Camera {cam_key} error: {cam_err}")

                # SOURCE 3: Tool/trigger state
                from app.core.hardware.tool_state import get_tool_observations
                from app.state import state
                tool_obs = get_tool_observations(state.tool_registry, state.trigger_listener)
                obs.update(tool_obs)

                # Check data availability
                has_motor_data = any('.pos' in k for k in obs.keys())
                has_camera_data = any(hasattr(obs.get(k), 'shape') for k in obs.keys())

                if svc._recording_frame_counter == 0:
                    print(f"[REC CAPTURE] Data: motors={has_motor_data}, cameras={has_camera_data}")
                    print(f"[REC CAPTURE] obs keys: {list(obs.keys())}")
                    print(f"[REC CAPTURE] action keys: {list(action.keys())}")

                # Need at least motor data OR camera data
                if not has_motor_data and not has_camera_data:
                    if svc._recording_frame_counter == 0:
                        print(f"[REC CAPTURE] WARNING: No data! Teleop running: {svc.is_running}")
                    time.sleep(0.005)
                    continue

                # Build frame using LeRobot helpers
                try:
                    obs_frame = build_dataset_frame(svc.dataset.features, obs, prefix=OBS_STR)
                    action_frame = build_dataset_frame(svc.dataset.features, action, prefix=ACTION)

                    frame = {
                        **obs_frame,
                        **action_frame,
                        "task": svc.dataset_config.get("task", ""),
                    }

                    if svc._recording_frame_counter == 0:
                        print(f"[REC CAPTURE] Built frame with keys: {list(frame.keys())}")

                    # Queue for async writing
                    svc._frame_queue.append(frame)
                    svc._recording_frame_counter += 1

                    if svc._recording_frame_counter == 1:
                        print(f"[REC CAPTURE] FIRST FRAME captured and queued!")
                    elif svc._recording_frame_counter % 30 == 0:
                        wall_elapsed = time.perf_counter() - episode_start_time
                        actual_fps = svc._recording_frame_counter / wall_elapsed if wall_elapsed > 0 else 0
                        queue_size = len(svc._frame_queue)
                        print(f"[REC CAPTURE] {svc._recording_frame_counter} frames ({actual_fps:.1f}fps), queue: {queue_size}")

                except Exception as frame_err:
                    if svc._recording_frame_counter == 0:
                        import traceback
                        print(f"[REC CAPTURE] Frame build error: {frame_err}")
                        print(traceback.format_exc())

            except Exception as e:
                import traceback
                if svc._recording_frame_counter == 0 or svc._recording_frame_counter % 30 == 0:
                    print(f"[REC CAPTURE] Error: {e}")
                    print(traceback.format_exc())

            # Maintain target fps with precise timing
            elapsed = time.perf_counter() - start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                precise_sleep(sleep_time)
        else:
            # Not recording - log why (only once per state change)
            if not hasattr(svc, '_last_idle_reason') or svc._last_idle_reason != (svc.recording_active, svc.robot is not None, svc.dataset is not None):
                svc._last_idle_reason = (svc.recording_active, svc.robot is not None, svc.dataset is not None)
                if not svc.recording_active:
                    pass  # Normal idle state, don't spam logs
                else:
                    print(f"[REC CAPTURE] IDLE - recording_active:{svc.recording_active}, robot:{svc.robot is not None}, dataset:{svc.dataset is not None}")

            # Reset episode timing when not actively recording
            if episode_start_time is not None:
                wall_elapsed = time.perf_counter() - episode_start_time
                actual_fps = svc._recording_frame_counter / wall_elapsed if wall_elapsed > 0 else 0
                print(f"[REC CAPTURE] Episode ended: {svc._recording_frame_counter} frames in {wall_elapsed:.1f}s = {actual_fps:.1f}fps")
                episode_start_time = None
            time.sleep(0.01)  # Idle when not recording

    print(f"[REC CAPTURE] Thread stopped ({svc._recording_frame_counter} total frames)")


def start_recording_capture(svc):
    """Starts the recording capture thread."""
    if svc._recording_capture_thread is not None and svc._recording_capture_thread.is_alive():
        return
    svc._recording_stop_event.clear()
    svc._recording_capture_thread = threading.Thread(
        target=recording_capture_loop,
        args=(svc,),
        daemon=True,
        name="RecCapture"
    )
    svc._recording_capture_thread.start()
    logger.info("Recording Capture Thread Started")


def stop_recording_capture(svc):
    """Stops the recording capture thread."""
    svc._recording_stop_event.set()
    if svc._recording_capture_thread is not None:
        svc._recording_capture_thread.join(timeout=2.0)
        svc._recording_capture_thread = None
    logger.info("Recording Capture Thread Stopped")


def start_recording_session(
    svc,
    repo_id: str,
    task: str,
    fps: int = 30,
    root: str = None,
    selected_cameras: list = None,
    selected_arms: list = None
):
    """Initializes a new LeRobotDataset for recording.

    Args:
        svc: TeleoperationService instance
        repo_id: Dataset repository ID
        task: Task description
        fps: Recording frames per second
        root: Custom dataset root path
        selected_cameras: List of camera IDs to record (None = all cameras)
        selected_arms: List of arm IDs ("left", "right") to record (None = all arms)
    """
    print("=" * 60)
    print(f"[START_SESSION] Called with repo_id='{repo_id}', task='{task}'")
    print(f"  selected_cameras: {selected_cameras}")
    print(f"  selected_arms: {selected_arms}")
    print(f"  session_active: {svc.session_active}")
    print(f"  robot: {svc.robot is not None}")
    print("=" * 60)

    if svc.session_active:
        print("[START_SESSION] ERROR: Session already active!")
        raise Exception("Session already active")

    # Store selections for use during recording
    svc._selected_cameras = selected_cameras  # None means all cameras
    svc._selected_arms = selected_arms        # None means all arms

    # Set default root to app datasets directory
    if root is None:
        base_dir = _DEFAULT_DATASETS_PATH
    else:
        base_dir = Path(root)

    # Target Path
    dataset_dir = base_dir / repo_id

    print(f"[START_SESSION] Dataset dir: {dataset_dir}")

    try:
        if not svc.robot:
             raise Exception("Robot not connected")

        # 1. Define Features
        if not hasattr(svc.robot, "observation_features") or not hasattr(svc.robot, "action_features"):
             raise RuntimeError("Robot does not have feature definitions ready (observation_features/action_features).")

        # Use LeRobot Helpers to construct correct feature dicts
        from lerobot.datasets.utils import combine_feature_dicts, hw_to_dataset_features

        # Filter observation features based on selections
        filtered_obs_features = filter_observation_features(
            svc.robot.observation_features,
            selected_cameras,
            selected_arms
        )

        # Filter action features based on arm selections
        filtered_action_features = filter_action_features(
            svc.robot.action_features,
            selected_arms
        )

        # Add tool/trigger features to observation schema
        from app.core.hardware.tool_state import get_tool_action_features
        from app.state import state
        tool_features = get_tool_action_features(state.tool_registry)
        if tool_features:
            filtered_obs_features.update(tool_features)

        print(f"[START_SESSION] Filtered obs features: {list(filtered_obs_features.keys())}")
        print(f"[START_SESSION] Filtered action features: {list(filtered_action_features.keys())}")

        features = combine_feature_dicts(
            hw_to_dataset_features(filtered_obs_features, prefix=OBS_STR, use_video=True),
            hw_to_dataset_features(filtered_action_features, prefix=ACTION, use_video=True)
        )

        # 2. Open or Create Dataset (In-Process)
        # Check for VALID dataset (must have meta/info.json)
        is_valid_dataset = (dataset_dir / "meta/info.json").exists()

        if dataset_dir.exists() and not is_valid_dataset:
             logger.warning(f"Found existing folder '{dataset_dir}' but it is not a valid dataset (missing info.json). Backing up...")
             import datetime
             import shutil
             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
             backup_dir = base_dir / f"{repo_id}_backup_{timestamp}"
             shutil.move(str(dataset_dir), str(backup_dir))
             logger.info(f"Moved invalid folder to {backup_dir}")

        if dataset_dir.exists():
            logger.info(f"Valid Dataset exists at {dataset_dir}. Resuming...")
            svc.dataset = LeRobotDataset(
                repo_id=repo_id,
                root=dataset_dir,
                local_files_only=True,
            )
        else:
            logger.info("Creating new Dataset...")
            svc.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                root=dataset_dir,
                robot_type=svc.robot.robot_type,
                features=features,
                use_videos=True,
            )

        svc.dataset.meta.metadata_buffer_size = 1
        print(f"[START_SESSION] Dataset created/loaded successfully")

        # CRITICAL: Set episode_count from actual dataset state
        # For new dataset: total_episodes = 0
        # For existing dataset: total_episodes = actual count
        svc.episode_count = svc.dataset.meta.total_episodes
        print(f"[START_SESSION] Episode count set to {svc.episode_count} (from dataset)")

        # 3. Start Video Encoding
        if not svc.video_manager:
            svc.video_manager = VideoEncodingManager(svc.dataset)
            svc.video_manager.__enter__()
        print("[START_SESSION] Video manager started")

        # 4. Start Image Writer (Threads)
        svc.dataset.start_image_writer(num_processes=0, num_threads=4)
        print("[START_SESSION] Image writer started")

        svc.dataset_config = {"repo_id": repo_id, "task": task}
        svc.session_active = True

        # Update recording fps to match dataset
        svc.recording_fps = fps
        svc._recording_frame_counter = 0
        print(f"[START_SESSION] Recording at {svc.recording_fps}fps (teleop={svc.frequency}Hz)")

        # Start async frame writer thread
        start_frame_writer(svc)

        # Start recording capture thread (separate from teleop loop for smooth control)
        start_recording_capture(svc)

        print("[START_SESSION] SUCCESS! Session is now active")
        print("=" * 60)

    except Exception as e:
        import traceback
        print(f"[START_SESSION] ERROR: {e}")
        print(traceback.format_exc())
        svc.session_active = False
        svc.dataset = None
        raise


def stop_recording_session(svc):
    """Finalizes the LeRobotDataset."""
    print("=" * 60)
    print(f"[STOP_SESSION] Called!")
    print(f"  session_active: {svc.session_active}")
    print(f"  dataset: {svc.dataset is not None}")
    print(f"  episode_saving: {svc._episode_saving}")
    print("=" * 60)

    if not svc.session_active:
        print("[STOP_SESSION] Not active, returning")
        return

    # AUTO-SAVE: If an episode is actively recording, save it first
    if svc.recording_active:
        print("[STOP_SESSION] Episode still recording - auto-saving before finalize...")
        try:
            stop_episode(svc)
            print("[STOP_SESSION] Auto-save completed successfully")
        except Exception as e:
            print(f"[STOP_SESSION] WARNING: Auto-save failed: {e}")

    print("[STOP_SESSION] Stopping Recording Session...")
    svc.session_active = False

    # Stop recording capture thread first (it produces frames)
    print("[STOP_SESSION] Stopping recording capture thread...")
    stop_recording_capture(svc)

    # Stop async frame writer (drains remaining frames)
    print(f"[STOP_SESSION] Stopping frame writer (queue size: {len(svc._frame_queue)})")
    stop_frame_writer(svc)

    # CRITICAL: Wait for any pending episode save to complete
    # This prevents finalize() from closing writers while save_episode() is still writing
    print("[STOP_SESSION] Acquiring episode save lock (waiting for pending save)...")
    with svc._episode_save_lock:
        print("[STOP_SESSION] Episode save lock acquired, safe to finalize")

        try:
            if svc.dataset:
                print("[STOP_SESSION] Finalizing Dataset...")

                # Check if writer exists before finalize
                has_writer = hasattr(svc.dataset, 'writer') and svc.dataset.writer is not None
                has_meta_writer = (hasattr(svc.dataset, 'meta') and
                                   hasattr(svc.dataset.meta, 'writer') and
                                   svc.dataset.meta.writer is not None)
                print(f"[STOP_SESSION] Before finalize - data writer: {has_writer}, meta writer: {has_meta_writer}")

                # Call finalize to close parquet writers
                svc.dataset.finalize()

                # Verify writers are closed
                has_writer_after = hasattr(svc.dataset, 'writer') and svc.dataset.writer is not None
                has_meta_writer_after = (hasattr(svc.dataset, 'meta') and
                                         hasattr(svc.dataset.meta, 'writer') and
                                         svc.dataset.meta.writer is not None)
                print(f"[STOP_SESSION] After finalize - data writer: {has_writer_after}, meta writer: {has_meta_writer_after}")

                if has_writer_after or has_meta_writer_after:
                    print("[STOP_SESSION] WARNING: Writers not fully closed, forcing close...")
                    # Force close if still open
                    if hasattr(svc.dataset, '_close_writer'):
                        svc.dataset._close_writer()
                    if hasattr(svc.dataset, 'meta') and hasattr(svc.dataset.meta, '_close_writer'):
                        svc.dataset.meta._close_writer()

                print("[STOP_SESSION] Dataset finalized")

                if svc.video_manager:
                    svc.video_manager.__exit__(None, None, None)
                    svc.video_manager = None
                    print("[STOP_SESSION] Video manager closed")

                svc.dataset = None
                print("[STOP_SESSION] SUCCESS! Session Stopped and Saved!")
        except Exception as e:
            import traceback
            print(f"[STOP_SESSION] ERROR: {e}")
            print(traceback.format_exc())
            # Ensure cleanup even on error
            svc.dataset = None


def sync_to_disk(svc):
    """
    Flush all pending episode data to disk and close writers.
    MUST be called BEFORE any external deletion operation.

    This ensures:
    1. Metadata buffer is flushed to parquet (episodes saved to disk)
    2. Parquet writers are closed (prevents appending to wrong files)
    3. Disk state is consistent for external modifications

    Without this, the metadata_buffer may contain episode data that hasn't
    been written to disk yet. If deletion runs, it won't find the episode
    on disk, but the buffer still has it. When recording resumes, both
    the old buffered episode and new episode get saved = 2 episodes!
    """
    if not svc.dataset or not svc.session_active:
        print("[SYNC_TO_DISK] Skipped (no dataset or session not active)")
        return

    print(f"[SYNC_TO_DISK] BEFORE: meta.total_episodes={svc.dataset.meta.total_episodes}, episode_count={svc.episode_count}")

    try:
        # 1. Flush and close metadata writer (this flushes metadata_buffer to disk)
        if hasattr(svc.dataset, 'meta') and hasattr(svc.dataset.meta, '_close_writer'):
            svc.dataset.meta._close_writer()
            print("[SYNC_TO_DISK] Flushed metadata buffer and closed metadata writer")

        # 2. Close data parquet writer
        if hasattr(svc.dataset, '_close_writer'):
            svc.dataset._close_writer()
            print("[SYNC_TO_DISK] Closed data writer")

    except Exception as e:
        import traceback
        print(f"[SYNC_TO_DISK] Error: {e}")
        print(traceback.format_exc())


def refresh_metadata_from_disk(svc):
    """
    Re-read episode metadata from disk to sync after external modifications.
    Called AFTER external operations like delete_episode().

    Assumes sync_to_disk() was called BEFORE the external operation.

    Resets ALL stale state:
    - latest_episode: Used by _save_episode_metadata() to compute frame indices
    - _current_file_start_frame: Tracks current parquet file position
    - episodes DataFrame: Cached episode metadata
    - metadata_buffer: Cleared to prevent ghost episodes
    """
    if not svc.dataset or not svc.session_active:
        print("[REFRESH] Skipped (no dataset or session not active)")
        return

    import json

    info_path = svc.dataset.meta.root / "meta" / "info.json"
    print(f"[REFRESH] Reading from: {info_path}")

    if info_path.exists():
        try:
            with open(info_path, "r") as f:
                disk_info = json.load(f)

            old_memory_count = svc.dataset.meta.info.get("total_episodes", 0)
            disk_count = disk_info.get("total_episodes", 0)

            print(f"[REFRESH] Disk: {disk_count}, Memory: {old_memory_count}")

            # 1. Update info dict
            svc.dataset.meta.info["total_episodes"] = disk_count
            svc.dataset.meta.info["total_frames"] = disk_info.get("total_frames", 0)

            # Verify the update worked
            verify_count = svc.dataset.meta.total_episodes
            print(f"[REFRESH] After update: meta.total_episodes = {verify_count}")
            if verify_count != disk_count:
                print(f"[REFRESH] ERROR: Update failed! Expected {disk_count}, got {verify_count}")

            # 2. Reset latest_episode to force fresh index calculation
            if hasattr(svc.dataset, 'meta'):
                svc.dataset.meta.latest_episode = None
                # Clear metadata buffer (should be empty after sync_to_disk, but ensure it)
                if hasattr(svc.dataset.meta, 'metadata_buffer'):
                    svc.dataset.meta.metadata_buffer = []
            if hasattr(svc.dataset, 'latest_episode'):
                svc.dataset.latest_episode = None

            # 3. Reset current file tracking
            if hasattr(svc.dataset, '_current_file_start_frame'):
                svc.dataset._current_file_start_frame = None

            # 4. Reset episodes to None - DON'T reload from parquet
            # LeRobot expects episodes to be a specific internal structure (not a raw DataFrame)
            # Setting to None forces LeRobot to start fresh when latest_episode is also None
            svc.dataset.meta.episodes = None

            # 5. CRITICAL: Clear episode_buffer to force fresh creation on next start_episode()
            # Without this, stale buffer with old episode_index causes validation failure
            if hasattr(svc.dataset, 'episode_buffer') and svc.dataset.episode_buffer is not None:
                svc.dataset.episode_buffer = None
                print("[REFRESH] Cleared stale episode_buffer")

            # 6. Close and reset data writer to prevent stale frame counting
            if hasattr(svc.dataset, '_close_writer'):
                try:
                    svc.dataset._close_writer()
                    print("[REFRESH] Closed data writer")
                except:
                    pass
            if hasattr(svc.dataset, 'writer'):
                svc.dataset.writer = None

            # 7. Sync local episode counter
            svc.episode_count = disk_count

            print(f"[REFRESH] Complete: episode_count={svc.episode_count}, meta.total_episodes={svc.dataset.meta.total_episodes}")

        except Exception as e:
            import traceback
            print(f"[REFRESH] Error: {e}")
            print(traceback.format_exc())
    else:
        print(f"[REFRESH] ERROR: info.json not found at {info_path}")


def start_episode(svc):
    """Starts recording a new episode."""
    print("=" * 60)
    print("[START_EPISODE] Called!")
    print(f"  session_active: {svc.session_active}")
    print(f"  recording_active: {svc.recording_active}")
    print(f"  _episode_saving: {svc._episode_saving}")
    print(f"  dataset: {svc.dataset is not None}")
    print("=" * 60)

    if not svc.session_active:
        print("[START_EPISODE] ERROR: No active session!")
        raise Exception("No active recording session")

    if svc.recording_active:
        print("[START_EPISODE] Already recording, skipping")
        return

    # Wait for any ongoing episode save to complete before starting new episode
    if svc._episode_saving:
        print("[START_EPISODE] Waiting for previous episode to finish saving...")
        wait_start = time.time()
        max_wait = 10.0  # Maximum 10 seconds wait
        while svc._episode_saving and (time.time() - wait_start) < max_wait:
            time.sleep(0.1)
        if svc._episode_saving:
            raise Exception("Previous episode save timed out. Please try again.")
        print(f"[START_EPISODE] Previous save completed after {time.time() - wait_start:.1f}s")

    # Warn if teleop isn't running - recording needs teleop for actions
    if not svc.is_running:
        print("[START_EPISODE] WARNING: Teleop is NOT running!")
        print("[START_EPISODE] Recording requires teleop to be active for action/state data.")
        print("[START_EPISODE] Will use robot state fallback if available.")

    print("[START_EPISODE] Starting Episode Recording...")

    if svc.dataset:
        # Log current state BEFORE buffer creation (critical for debugging)
        meta_total = svc.dataset.meta.total_episodes
        print(f"[START_EPISODE] BEFORE buffer: meta.total_episodes={meta_total}, episode_count={svc.episode_count}")

        # Check for count mismatch and warn
        if svc.episode_count != meta_total:
            print(f"[START_EPISODE] WARNING: Count mismatch detected!")
            print(f"[START_EPISODE]   episode_count={svc.episode_count} != meta.total_episodes={meta_total}")
            # Don't auto-sync here - the mismatch indicates a bug we need to find

        # Initialize episode buffer - handle case where buffer is None on first use
        try:
            if svc.dataset.episode_buffer is None:
                print("[START_EPISODE] Creating new episode buffer (first episode)")
                svc.dataset.episode_buffer = svc.dataset.create_episode_buffer()
            else:
                print("[START_EPISODE] Clearing existing episode buffer")
                svc.dataset.clear_episode_buffer()
        except Exception as e:
            print(f"[START_EPISODE] Error with buffer, recreating: {e}")
            svc.dataset.episode_buffer = svc.dataset.create_episode_buffer()

        # Log the episode index that will be used
        buffer_ep_idx = svc.dataset.episode_buffer.get("episode_index", "N/A")
        print(f"[START_EPISODE] Episode buffer ready, episode_index={buffer_ep_idx}")
        print(f"[START_EPISODE] Dataset features: {list(svc.dataset.features.keys())[:5]}...")

    # Reset frame counter for new episode
    svc._recording_frame_counter = 0

    svc.recording_active = True
    print("[START_EPISODE] recording_active = True, ready to capture frames!")


def stop_episode(svc):
    """Stops current episode and saves it."""
    print("=" * 60)
    print("[STOP_EPISODE] Called!")
    print(f"  recording_active: {svc.recording_active}")
    print(f"  session_active: {svc.session_active}")
    print(f"  dataset: {svc.dataset is not None}")
    if svc.dataset:
        print(f"  dataset type: {type(svc.dataset)}")
    print("=" * 60)

    if not svc.recording_active:
        print("[STOP_EPISODE] WARNING: recording_active is False, returning")
        return

    # Acquire lock to prevent race with stop_session
    # This ensures save_episode() completes before finalize() can be called
    print("[STOP_EPISODE] Acquiring episode save lock...")
    with svc._episode_save_lock:
        svc._episode_saving = True
        print("[STOP_EPISODE] Lock acquired, proceeding with save")

        # Capture dataset reference BEFORE changing state
        # This prevents race conditions where dataset could be set to None
        current_dataset = svc.dataset

        print("[STOP_EPISODE] Stopping Episode Recording...")
        svc.recording_active = False

        # Wait for frame queue to drain before saving episode
        print(f"[STOP_EPISODE] Waiting for frame queue to drain ({len(svc._frame_queue)} frames)...")
        drain_timeout = 5.0  # seconds
        drain_start = time.time()
        while len(svc._frame_queue) > 0 and (time.time() - drain_start) < drain_timeout:
            time.sleep(0.05)
        if len(svc._frame_queue) > 0:
            print(f"[STOP_EPISODE] WARNING: Queue not fully drained ({len(svc._frame_queue)} remaining)")
        else:
            print(f"[STOP_EPISODE] Queue drained successfully")

        # Reset first frame logging flag for next episode
        if hasattr(svc, '_first_frame_logged'):
            delattr(svc, '_first_frame_logged')
        if hasattr(svc, '_last_rec_error'):
            delattr(svc, '_last_rec_error')

        if current_dataset is not None:
            try:
                # Check episode buffer before saving
                if hasattr(current_dataset, 'episode_buffer') and current_dataset.episode_buffer:
                    buffer_size = current_dataset.episode_buffer.get('size', 0)
                    print(f"[STOP_EPISODE] Episode buffer has {buffer_size} frames (captured {svc._recording_frame_counter})")

                    if buffer_size == 0:
                        print("[STOP_EPISODE] WARNING: Buffer size is 0, no frames were recorded!")
                        print("[STOP_EPISODE] This means observations weren't captured during recording.")
                        svc._episode_saving = False
                        return
                else:
                    print("[STOP_EPISODE] WARNING: Episode buffer is empty or missing!")
                    print(f"  has episode_buffer attr: {hasattr(current_dataset, 'episode_buffer')}")
                    if hasattr(current_dataset, 'episode_buffer'):
                        print(f"  episode_buffer value: {current_dataset.episode_buffer}")
                    svc._episode_saving = False
                    return

                # Log state BEFORE save
                print(f"[STOP_EPISODE] BEFORE save: meta.total_episodes={current_dataset.meta.total_episodes}, episode_count={svc.episode_count}")

                # Diagnostic: Check image writer status before save
                if hasattr(current_dataset, 'image_writer') and current_dataset.image_writer:
                    try:
                        queue_size = current_dataset.image_writer.queue.qsize()
                        print(f"[STOP_EPISODE] Image writer queue size: {queue_size}")
                    except Exception:
                        print("[STOP_EPISODE] Image writer queue size: (unable to check)")

                print(f"[STOP_EPISODE] Calling save_episode()...")
                save_start = time.time()
                # Note: task is already included in each frame, no need to pass to save_episode
                current_dataset.save_episode()
                save_duration = time.time() - save_start
                print(f"[STOP_EPISODE] save_episode() completed in {save_duration:.1f}s")
                # Log state AFTER save
                print(f"[STOP_EPISODE] AFTER save: meta.total_episodes={current_dataset.meta.total_episodes}")
                svc.episode_count += 1
                print(f"[STOP_EPISODE] SUCCESS! Episode {svc.episode_count} Saved! ({svc._recording_frame_counter} frames)")
            except Exception as e:
                import traceback
                print(f"[STOP_EPISODE] ERROR saving episode: {e}")
                print(traceback.format_exc())
        else:
            print("[STOP_EPISODE] WARNING: No dataset available (current_dataset is None)!")

        svc._episode_saving = False
        print("[STOP_EPISODE] Episode save lock released")


def delete_last_episode(svc):
    """Deletes the last recorded episode (if possible/implemented)."""
    logger.warning("Delete Last Episode not fully supported yet.")


def manual_finalize_dataset(svc, repo_id):
    """Emergency fix to generate episode metadata if LeRobotDataset fails."""
    if not repo_id: return
    pass  # Not implemented fully as per original file
