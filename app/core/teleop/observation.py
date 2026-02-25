import logging
import threading
import time

logger = logging.getLogger(__name__)


def obs_capture_loop(svc):
    """Background thread for continuous observation capture during recording.

    This decouples observation reading from the control loop, ensuring
    motor commands are sent immediately without waiting for camera capture.
    Uses Zero-Order Hold (ZOH) pattern - always provides latest available observation.
    """
    print("[OBS CAPTURE] Thread Started!")
    capture_count = 0
    lock_fail_count = 0

    while not svc._obs_stop_event.is_set():
        try:
            if svc.robot and svc.recording_active:
                # Try to acquire robot lock with short timeout to avoid USB serial conflicts
                lock_acquired = False
                if svc.robot_lock:
                    lock_acquired = svc.robot_lock.acquire(timeout=0.005)

                try:
                    if lock_acquired or not svc.robot_lock:
                        # Capture observation
                        obs = svc.robot.get_observation(include_images=True)

                        if obs:
                            camera_keys = [k for k in obs.keys() if 'camera' in k.lower()]
                            motor_keys = [k for k in obs.keys() if '.pos' in k]

                            if capture_count == 0:
                                print(f"[OBS CAPTURE] FIRST observation: {len(obs)} keys")
                                print(f"  motors={len(motor_keys)}, cameras={len(camera_keys)}")
                                print(f"  Keys: {list(obs.keys())[:10]}...")

                            with svc._latest_obs_lock:
                                svc._latest_obs = obs
                            svc._obs_ready_event.set()

                            capture_count += 1
                            if capture_count % 60 == 0:
                                print(f"[OBS CAPTURE] Running: {capture_count} frames")
                        else:
                            print("[OBS CAPTURE] WARNING: Empty observation!")
                    else:
                        lock_fail_count += 1
                        if lock_fail_count % 100 == 0:
                            print(f"[OBS CAPTURE] Lock failed {lock_fail_count} times")
                finally:
                    if lock_acquired and svc.robot_lock:
                        svc.robot_lock.release()
            else:
                time.sleep(0.01)
        except Exception as e:
            import traceback
            print(f"[OBS CAPTURE] ERROR: {e}")
            print(traceback.format_exc())
            time.sleep(0.01)

    print(f"[OBS CAPTURE] Thread Stopped (captured={capture_count}, lock_fails={lock_fail_count})")


def start_obs_thread(svc):
    """Starts the background observation capture thread."""
    if svc._obs_thread is not None and svc._obs_thread.is_alive():
        return  # Already running

    svc._obs_stop_event.clear()
    svc._latest_obs = None
    svc._obs_ready_event.clear()
    svc._obs_thread = threading.Thread(target=obs_capture_loop, args=(svc,), daemon=True, name="ObsCapture")
    svc._obs_thread.start()
    logger.info("Observation Capture Thread Initialized")


def stop_obs_thread(svc):
    """Stops the background observation capture thread (legacy, kept for compatibility)."""
    if hasattr(svc, '_obs_stop_event') and svc._obs_stop_event:
        svc._obs_stop_event.set()
    if hasattr(svc, '_obs_thread') and svc._obs_thread is not None:
        svc._obs_thread.join(timeout=1.0)
        svc._obs_thread = None
    if hasattr(svc, '_latest_obs'):
        svc._latest_obs = None
    logger.info("Observation Capture Thread Terminated")


def update_history(svc, action_dict):
    """Append action data to history for graph display."""
    # Convert dictionary to simple list of values for graph
    timestamp = time.time()

    data_point = {"time": timestamp}

    for k, v in action_dict.items():
        # Simplify key name for UI
        short_key = k.replace(".pos", "").replace("follower", "").strip("_")
        data_point[short_key] = float(v)

    with svc.history_lock:
        svc.action_history.append(data_point)


def get_data(svc):
    """Returns the current data history and latest status."""
    history = []
    with svc.history_lock:
        history = list(svc.action_history)

    return {
        "history": history,
        "torque": svc.safety.latest_loads,
        "recording": {
            "session_active": svc.session_active,
            "episode_active": svc.recording_active,
            "episode_count": svc.episode_count
        }
    }
