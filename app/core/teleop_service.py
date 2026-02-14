import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from app.core.safety_layer import SafetyLayer
from app.core.leader_assist import LeaderAssistService
from lerobot.motors.feetech.feetech import OperatingMode


@dataclass
class PairingContext:
    """Per-pairing state for independent teleop loops.

    Each leader→follower pair gets its own context so that multiple pairs
    can run simultaneously without cross-contaminating mapping, scaling,
    or value-mode state.
    """
    pairing_id: str            # e.g. "aira_zero_leader→aira_zero"
    active_leader: object      # Leader arm instance
    active_robot: object       # Follower arm instance
    joint_mapping: dict        # {leader_key: follower_key}
    follower_value_mode: str   # "float" (Damiao rad), "rad_to_percent" (Dyn→Feetech), "int" (legacy)
    has_damiao_follower: bool
    leader_cal_ranges: dict    # {follower_key: (range_min, range_max)} from leader calibration
    # Mutable per-loop state (reset each start):
    follower_start_pos: dict = field(default_factory=dict)
    leader_start_rad: dict = field(default_factory=dict)
    rad_to_percent_scale: dict = field(default_factory=dict)
    blend_start_time: float | None = None
    filtered_gripper_torque: float = 0.0

# Compute project root relative to this file (app/core/teleop_service.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASETS_PATH = _PROJECT_ROOT / "datasets"

# Try to import precise_sleep, fallback to time.sleep if not found (though it should be there)
try:
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import VideoEncodingManager
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.utils.constants import OBS_STR, ACTION
except ImportError:
    def precise_sleep(dt):
        time.sleep(max(0, dt))

logger = logging.getLogger(__name__)

class TeleoperationService:
    def __init__(self, robot, leader, robot_lock, leader_assists=None, arm_registry=None, camera_service=None):
        self.robot = robot
        self.leader = leader
        self.robot_lock = robot_lock
        self.arm_registry = arm_registry  # For pairing-based mapping
        self.camera_service = camera_service  # Standalone camera manager

        self.safety = SafetyLayer(robot_lock) # Initialize Safety Layer
        
        # Initialize Leader Assist (Leader Only)
        # Passed from SystemState to share calibration state with API
        self.leader_assists = leader_assists if leader_assists else {}
        
        # Fallback local init if not passed (Legacy/Standalone)
        if self.leader and not self.leader_assists:
            # Detect BiUmbra
            if hasattr(self.leader, "left_arm") and hasattr(self.leader, "right_arm"):
                 logger.info("Initializing Leader Assist for Bi-Manual Leader (Local)")
                 self.leader_assists["left"] = LeaderAssistService(arm_id="left_leader")
                 self.leader_assists["right"] = LeaderAssistService(arm_id="right_leader")
            else:
                 # Mono or Generic
                 logger.info("Initializing Leader Assist for Single Leader (Local)")
                 self.leader_assists["default"] = LeaderAssistService(arm_id="leader")

        # Initialize Calibration Models for Followers (For Haptics & Transparency)
        self.follower_gravity_models = {}
        if self.robot:
            logger.info("Initializing Follower Gravity Models...")
            if hasattr(self.robot, "left_arm") and hasattr(self.robot, "right_arm"):
                 self.follower_gravity_models["left"] = LeaderAssistService(arm_id="left_follower")
                 self.follower_gravity_models["right"] = LeaderAssistService(arm_id="right_follower")
            else:
                 self.follower_gravity_models["default"] = LeaderAssistService(arm_id="follower")

        self.joint_names_template = ["base", "link1", "link2", "link3", "link4", "link5", "gripper"]
        self.last_leader_pos = {} # Stores {full_joint_name: deg}
        self.last_loop_time = None
        
        # Velocity Smoothing (EMA)
        self.leader_vel_kf = {} # Stores {full_joint_name: last_filtered_vel}
        self.alpha_vel = 0.2     # Smoothing factor (0.2 = heavy smoothing, 0.8 = light)

        if self.leader and self.leader_assists:
             # Default to disabled as per user request
             self.assist_enabled = False
        else:
             self.assist_enabled = False
             
        self.is_running = False
        
        # Data storage for Graph
        self.max_history = 100
        self.history_lock = threading.Lock()
        self.action_history = deque(maxlen=self.max_history)
        
        # Optimization: Pre-computed mappings (legacy single-pair; kept for backward compat)
        self.joint_mapping = {} # {leader_key: follower_key}
        self.assist_groups = {} # {arm_key: [joint_name, ...]}
        self._leader_cal_ranges = {}  # {follower_key: (range_min, range_max)} from leader calibration
        self._active_leader = None  # Resolved in start() from arm registry or legacy
        self._active_robot = None   # Resolved in start() from arm registry or legacy

        # Multi-pair support: per-pairing contexts and threads
        self._pairing_contexts: list[PairingContext] = []
        self._teleop_threads: list[threading.Thread] = []

        # Gripper Force Feedback (follower torque → leader current ceiling)
        self._force_feedback_enabled = True
        self._filtered_gripper_torque = 0.0   # EMA-filtered absolute torque (Nm)
        self._ff_alpha = 0.3                  # EMA smoothing (τ ≈ 55ms at 60Hz)
        self._ff_baseline_current = 60        # mA — light spring (perceptible, not fatiguing)
        self._ff_max_current = 1750           # mA — full XL330 range (must be <= Current_Limit)
        self._ff_torque_threshold = 0.2       # Nm — dead zone (friction/gravity noise)
        self._ff_torque_saturation = 2.0      # Nm — torque at which max current is reached

        # Joint Force Feedback: CURRENT_POSITION mode (same mechanism as gripper)
        # Goal_Position = follower position, Goal_Current = error magnitude
        self._joint_ff_enabled = True
        self._joint_ff_k_spring = 15000.0   # mA/rad — gentler ramp for better tactile gradient
        self._joint_ff_deadzone = 0.10      # rad (~6°) — covers normal tracking lag (0.03-0.06 rad)
        self._joint_ff_max_current = 1750   # mA — full XL330 range
        self._joint_ff_min_force = 100      # mA — lower entry threshold (larger deadzone means bigger initial error)

        # Teleop Configuration
        # Lowered to 60Hz to match lerobot default and reduce USB congestion
        self.frequency = 60
        self.dt = 1.0 / self.frequency

        # Recording State (Preserved from original)
        self.dataset = None
        self.dataset_config = None
        self.recording_active = False # Episode Level
        self.session_active = False   # Dataset Level
        self.episode_count = 0
        self.video_manager = None
        self.data_queue = deque(maxlen=1) # For UI data streaming if needed, though get_data uses history

        # Recording selections (which cameras/arms to record)
        self._selected_cameras = None  # None = all cameras
        self._selected_arms = None     # None = all arms

        # Recording frame rate control
        # Record at 30fps to match dataset fps, not teleop rate (60Hz)
        self.recording_fps = 30
        self._recording_frame_counter = 0
        self._recording_skip_frames = max(1, self.frequency // self.recording_fps)  # Skip every 2nd frame

        # Async frame writing queue
        self._frame_queue = deque(maxlen=100)
        self._frame_writer_thread = None
        self._frame_writer_stop = threading.Event()

        # Shared action state for recording thread
        self._latest_leader_action = {}
        self._action_lock = threading.Lock()

        # Recording capture thread (separate from teleop loop)
        self._recording_capture_thread = None
        self._recording_stop_event = threading.Event()

        # Lock to prevent race between stop_episode and stop_session
        # Ensures save_episode() completes before finalize() is called
        self._episode_save_lock = threading.Lock()
        self._episode_saving = False  # Flag to track if save is in progress

        # Observation capture thread (for recording without teleop)
        self._obs_thread = None
        self._obs_stop_event = threading.Event()
        self._latest_obs = None
        self._latest_obs_lock = threading.Lock()
        self._obs_ready_event = threading.Event()

    def _obs_capture_loop(self):
        """Background thread for continuous observation capture during recording.

        This decouples observation reading from the control loop, ensuring
        motor commands are sent immediately without waiting for camera capture.
        Uses Zero-Order Hold (ZOH) pattern - always provides latest available observation.
        """
        print("[OBS CAPTURE] Thread Started!")
        capture_count = 0
        lock_fail_count = 0

        while not self._obs_stop_event.is_set():
            try:
                if self.robot and self.recording_active:
                    # Try to acquire robot lock with short timeout to avoid USB serial conflicts
                    lock_acquired = False
                    if self.robot_lock:
                        lock_acquired = self.robot_lock.acquire(timeout=0.005)

                    try:
                        if lock_acquired or not self.robot_lock:
                            # Capture observation
                            obs = self.robot.get_observation(include_images=True)

                            if obs:
                                camera_keys = [k for k in obs.keys() if 'camera' in k.lower()]
                                motor_keys = [k for k in obs.keys() if '.pos' in k]

                                if capture_count == 0:
                                    print(f"[OBS CAPTURE] FIRST observation: {len(obs)} keys")
                                    print(f"  motors={len(motor_keys)}, cameras={len(camera_keys)}")
                                    print(f"  Keys: {list(obs.keys())[:10]}...")

                                with self._latest_obs_lock:
                                    self._latest_obs = obs
                                self._obs_ready_event.set()

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
                        if lock_acquired and self.robot_lock:
                            self.robot_lock.release()
                else:
                    time.sleep(0.01)
            except Exception as e:
                import traceback
                print(f"[OBS CAPTURE] ERROR: {e}")
                print(traceback.format_exc())
                time.sleep(0.01)

        print(f"[OBS CAPTURE] Thread Stopped (captured={capture_count}, lock_fails={lock_fail_count})")

    def _start_obs_thread(self):
        """Starts the background observation capture thread."""
        if self._obs_thread is not None and self._obs_thread.is_alive():
            return  # Already running

        self._obs_stop_event.clear()
        self._latest_obs = None
        self._obs_ready_event.clear()
        self._obs_thread = threading.Thread(target=self._obs_capture_loop, daemon=True, name="ObsCapture")
        self._obs_thread.start()
        logger.info("Observation Capture Thread Initialized")

    def _stop_obs_thread(self):
        """Stops the background observation capture thread (legacy, kept for compatibility)."""
        if hasattr(self, '_obs_stop_event') and self._obs_stop_event:
            self._obs_stop_event.set()
        if hasattr(self, '_obs_thread') and self._obs_thread is not None:
            self._obs_thread.join(timeout=1.0)
            self._obs_thread = None
        if hasattr(self, '_latest_obs'):
            self._latest_obs = None
        logger.info("Observation Capture Thread Terminated")

    def _frame_writer_loop(self):
        """Background thread for writing frames to dataset without blocking teleop loop."""
        print("[FRAME WRITER] Thread Started!")
        written_count = 0

        while not self._frame_writer_stop.is_set():
            try:
                if self._frame_queue and self.dataset is not None:
                    try:
                        frame = self._frame_queue.popleft()
                        self.dataset.add_frame(frame)
                        written_count += 1
                        if written_count == 1:
                            print(f"[FRAME WRITER] FIRST FRAME added to dataset buffer!")
                            # Log buffer size to confirm it's working
                            if hasattr(self.dataset, 'episode_buffer'):
                                buf_size = self.dataset.episode_buffer.get('size', 0)
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
        while self._frame_queue and self.dataset is not None:
            try:
                frame = self._frame_queue.popleft()
                self.dataset.add_frame(frame)
                written_count += 1
            except IndexError:
                break

        print(f"[FRAME WRITER] Thread Stopped (written={written_count})")

    def _start_frame_writer(self):
        """Starts the background frame writer thread."""
        if self._frame_writer_thread is not None and self._frame_writer_thread.is_alive():
            return
        self._frame_writer_stop.clear()
        self._frame_writer_thread = threading.Thread(target=self._frame_writer_loop, daemon=True, name="FrameWriter")
        self._frame_writer_thread.start()
        logger.info("Frame Writer Thread Started")

    def _stop_frame_writer(self):
        """Stops the background frame writer thread."""
        self._frame_writer_stop.set()
        if self._frame_writer_thread is not None:
            self._frame_writer_thread.join(timeout=2.0)
            self._frame_writer_thread = None
        logger.info("Frame Writer Thread Stopped")

    def _recording_capture_loop(self):
        """Background thread that captures observations at recording fps.

        OPTIMIZED: Uses cached teleop data for motors (no slow hardware reads).
        Only cameras use async_read which is fast (Zero-Order Hold pattern).
        This allows reliable 30fps capture without blocking.
        """
        print(f"[REC CAPTURE] Thread started at {self.recording_fps}fps")
        target_dt = 1.0 / self.recording_fps

        # Wall-clock tracking for actual fps measurement
        episode_start_time = None

        while not self._recording_stop_event.is_set():
            if self.recording_active and self.robot and self.dataset is not None:
                start = time.perf_counter()

                # Track when recording actually starts
                if episode_start_time is None:
                    episode_start_time = time.perf_counter()
                    print(f"[REC CAPTURE] Episode recording started at wall-clock t=0")
                    print(f"[REC CAPTURE] Target: {self.recording_fps}fps ({target_dt*1000:.1f}ms per frame)")

                try:
                    # FAST STRATEGY: Use cached data from teleop loop (no hardware reads!)
                    # The teleop loop runs at 60Hz and caches motor positions in _latest_leader_action
                    action = {}
                    obs = {}

                    # SOURCE 1: Get motor positions from teleop cache (FAST - no hardware read)
                    # Filter based on selected arms
                    with self._action_lock:
                        if self._latest_leader_action:
                            for key, val in self._latest_leader_action.items():
                                # Filter by selected arms
                                if self._selected_arms is None:
                                    action[key] = val
                                    obs[key] = val
                                elif key.startswith("left_") and "left" in self._selected_arms:
                                    action[key] = val
                                    obs[key] = val
                                elif key.startswith("right_") and "right" in self._selected_arms:
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
                    if self.camera_service and self.camera_service.cameras:
                        cameras_dict = self.camera_service.cameras
                    elif hasattr(self, '_active_robot') and self._active_robot and hasattr(self._active_robot, 'cameras') and self._active_robot.cameras:
                        cameras_dict = self._active_robot.cameras
                    elif hasattr(self.robot, 'cameras') and self.robot.cameras:
                        cameras_dict = self.robot.cameras

                    if cameras_dict:
                        cameras_to_capture = self._selected_cameras if self._selected_cameras else list(cameras_dict.keys())

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
                                        if self._recording_frame_counter == 0:
                                            print(f"[REC CAPTURE] Camera {cam_key}: shape={frame.shape}")
                                # Also capture depth if enabled
                                if hasattr(cam, 'async_read_depth') and hasattr(cam.config, 'use_depth') and cam.config.use_depth:
                                    depth_frame = cam.async_read_depth(blocking=False)
                                    if depth_frame is not None:
                                        # Expand depth from (H, W) to (H, W, 1) for dataset compatibility
                                        if depth_frame.ndim == 2:
                                            depth_frame = depth_frame[..., np.newaxis]
                                        obs[f"{cam_key}_depth"] = depth_frame
                                        if self._recording_frame_counter == 0:
                                            print(f"[REC CAPTURE] Depth {cam_key}_depth: shape={depth_frame.shape}")
                            except Exception as cam_err:
                                if self._recording_frame_counter == 0:
                                    print(f"[REC CAPTURE] Camera {cam_key} error: {cam_err}")

                    # Check data availability
                    has_motor_data = any('.pos' in k for k in obs.keys())
                    has_camera_data = any(hasattr(obs.get(k), 'shape') for k in obs.keys())

                    if self._recording_frame_counter == 0:
                        print(f"[REC CAPTURE] Data: motors={has_motor_data}, cameras={has_camera_data}")
                        print(f"[REC CAPTURE] obs keys: {list(obs.keys())}")
                        print(f"[REC CAPTURE] action keys: {list(action.keys())}")

                    # Need at least motor data OR camera data
                    if not has_motor_data and not has_camera_data:
                        if self._recording_frame_counter == 0:
                            print(f"[REC CAPTURE] WARNING: No data! Teleop running: {self.is_running}")
                        time.sleep(0.005)
                        continue

                    # Build frame using LeRobot helpers
                    try:
                        obs_frame = build_dataset_frame(self.dataset.features, obs, prefix=OBS_STR)
                        action_frame = build_dataset_frame(self.dataset.features, action, prefix=ACTION)

                        frame = {
                            **obs_frame,
                            **action_frame,
                            "task": self.dataset_config.get("task", ""),
                        }

                        if self._recording_frame_counter == 0:
                            print(f"[REC CAPTURE] Built frame with keys: {list(frame.keys())}")

                        # Queue for async writing
                        self._frame_queue.append(frame)
                        self._recording_frame_counter += 1

                        if self._recording_frame_counter == 1:
                            print(f"[REC CAPTURE] FIRST FRAME captured and queued!")
                        elif self._recording_frame_counter % 30 == 0:
                            wall_elapsed = time.perf_counter() - episode_start_time
                            actual_fps = self._recording_frame_counter / wall_elapsed if wall_elapsed > 0 else 0
                            queue_size = len(self._frame_queue)
                            print(f"[REC CAPTURE] {self._recording_frame_counter} frames ({actual_fps:.1f}fps), queue: {queue_size}")

                    except Exception as frame_err:
                        if self._recording_frame_counter == 0:
                            import traceback
                            print(f"[REC CAPTURE] Frame build error: {frame_err}")
                            print(traceback.format_exc())

                except Exception as e:
                    import traceback
                    if self._recording_frame_counter == 0 or self._recording_frame_counter % 30 == 0:
                        print(f"[REC CAPTURE] Error: {e}")
                        print(traceback.format_exc())

                # Maintain target fps with precise timing
                elapsed = time.perf_counter() - start
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    precise_sleep(sleep_time)
            else:
                # Not recording - log why (only once per state change)
                if not hasattr(self, '_last_idle_reason') or self._last_idle_reason != (self.recording_active, self.robot is not None, self.dataset is not None):
                    self._last_idle_reason = (self.recording_active, self.robot is not None, self.dataset is not None)
                    if not self.recording_active:
                        pass  # Normal idle state, don't spam logs
                    else:
                        print(f"[REC CAPTURE] IDLE - recording_active:{self.recording_active}, robot:{self.robot is not None}, dataset:{self.dataset is not None}")

                # Reset episode timing when not actively recording
                if episode_start_time is not None:
                    wall_elapsed = time.perf_counter() - episode_start_time
                    actual_fps = self._recording_frame_counter / wall_elapsed if wall_elapsed > 0 else 0
                    print(f"[REC CAPTURE] Episode ended: {self._recording_frame_counter} frames in {wall_elapsed:.1f}s = {actual_fps:.1f}fps")
                    episode_start_time = None
                time.sleep(0.01)  # Idle when not recording

        print(f"[REC CAPTURE] Thread stopped ({self._recording_frame_counter} total frames)")

    def _start_recording_capture(self):
        """Starts the recording capture thread."""
        if self._recording_capture_thread is not None and self._recording_capture_thread.is_alive():
            return
        self._recording_stop_event.clear()
        self._recording_capture_thread = threading.Thread(
            target=self._recording_capture_loop,
            daemon=True,
            name="RecCapture"
        )
        self._recording_capture_thread.start()
        logger.info("Recording Capture Thread Started")

    def _stop_recording_capture(self):
        """Stops the recording capture thread."""
        self._recording_stop_event.set()
        if self._recording_capture_thread is not None:
            self._recording_capture_thread.join(timeout=2.0)
            self._recording_capture_thread = None
        logger.info("Recording Capture Thread Stopped")

    def set_assist_enabled(self, enabled: bool):
        self.assist_enabled = enabled
        logger.info(f"Leader Assist Enabled: {self.assist_enabled}")
        
        # Apply Hardware Change Immediately
        if self.is_running and self.leader and self.leader_assists:
            try:
                if self.assist_enabled:
                    logger.info("Enabling Leader Torque (PWM Mode)...")
                    if "left" in self.leader_assists:
                         self.leader.left_arm.bus.set_operating_mode(OperatingMode.PWM)
                         self.leader.left_arm.bus.enable_torque()
                         self.leader.right_arm.bus.set_operating_mode(OperatingMode.PWM)
                         self.leader.right_arm.bus.enable_torque()
                    else:
                         self.leader.bus.set_operating_mode(OperatingMode.PWM)
                         self.leader.bus.enable_torque()
                else:
                    logger.info("Disabling Leader Torque...")
                    if "left" in self.leader_assists:
                         self.leader.left_arm.bus.disable_torque()
                         self.leader.right_arm.bus.disable_torque()
                    else:
                         self.leader.bus.disable_torque()
            except Exception as e:
                logger.error(f"Failed to toggle Leader Assist State: {e}")

    def get_force_feedback_state(self) -> dict:
        """Return current force feedback toggle states."""
        return {
            "gripper": self._force_feedback_enabled,
            "joint": self._joint_ff_enabled,
        }

    def set_force_feedback(self, gripper: bool | None = None, joint: bool | None = None):
        """Toggle force feedback at runtime. When disabling, zero the current immediately."""
        if gripper is not None:
            self._force_feedback_enabled = gripper
            logger.info(f"Gripper force feedback: {'enabled' if gripper else 'disabled'}")
            leader = getattr(self, '_active_leader', None)
            if not gripper and self._has_damiao_follower and leader:
                try:
                    leader.bus.write(
                        "Goal_Current", "gripper", self._ff_baseline_current, normalize=False
                    )
                except Exception:
                    pass

        if joint is not None:
            self._joint_ff_enabled = joint
            logger.info(f"Joint force feedback: {'enabled' if joint else 'disabled'}")
            leader = getattr(self, '_active_leader', None)
            if not joint and leader:
                try:
                    leader.bus.write(
                        "Goal_Current", "joint_4", 0, normalize=False
                    )
                except Exception:
                    pass

    def check_calibration(self):
        # Check Leader Calibration
        if self.leader:
            if not getattr(self.leader, "is_calibrated", False):
                logger.error(f"Leader {self.leader} is NOT calibrated.")
                return False
        
        # Check Follower Calibration
        if self.robot:
            if not getattr(self.robot, "is_calibrated", False):
                logger.error(f"Robot {self.robot} is NOT calibrated.")
                return False
                
        return True

    def _precompute_mappings(self):
        """Pre-computes active joint mappings to avoid string ops in the loop."""
        self.joint_mapping = {}
        self.assist_groups = {}

        # Detect if any follower arm is Damiao (uses float radians, not int ticks)
        self._has_damiao_follower = False
        # Value conversion mode: "float" (Damiao rad), "rad_to_percent" (Dyn→Feetech), "int" (legacy)
        self._follower_value_mode = "int"
        if self.robot:
            try:
                from lerobot.robots.damiao_follower.damiao_follower import DamiaoFollowerRobot as DamiaoFollower
                if isinstance(self.robot, DamiaoFollower):
                    self._has_damiao_follower = True
                elif hasattr(self.robot, 'left_arm') and isinstance(self.robot.left_arm, DamiaoFollower):
                    self._has_damiao_follower = True
                elif hasattr(self.robot, 'right_arm') and isinstance(self.robot.right_arm, DamiaoFollower):
                    self._has_damiao_follower = True
            except ImportError:
                pass

        # Check arm_registry for Damiao follower arms
        if not self._has_damiao_follower and self.arm_registry and self.active_arms:
            for arm_id in self.active_arms:
                arm = self.arm_registry.arms.get(arm_id)
                if arm and arm.motor_type == 'damiao' and arm.role.value == 'follower':
                    self._has_damiao_follower = True
                    break

        if not self.leader and not self.arm_registry:
             return

        # Try pairing-based mapping first (new arm registry system)
        if self.arm_registry:
            self._precompute_mappings_from_pairings()
        else:
            # Fallback to legacy side-based mapping
            self._precompute_mappings_legacy()

    # Dynamixel leader uses joint_N names, Damiao follower uses base/linkN names
    DYNAMIXEL_TO_DAMIAO_JOINT_MAP = {
        "joint_1": "base",
        "joint_2": "link1",
        "joint_3": "link2",
        "joint_4": "link3",
        "joint_5": "link4",
        "joint_6": "link5",
        "gripper": "gripper",
    }

    def _precompute_mappings_from_pairings(self):
        """Use explicit pairings from arm registry for joint mapping."""
        pairings = self.arm_registry.get_active_pairings(self.active_arms)

        for pairing in pairings:
            leader_id = pairing['leader_id']
            follower_id = pairing['follower_id']

            # Only map if both are in active selection (or no selection = all active)
            if self.active_arms is not None:
                if leader_id not in self.active_arms or follower_id not in self.active_arms:
                    continue

            # Check if this is a Dynamixel→Damiao pairing (different joint naming)
            # Use .arms dict directly to get ArmDefinition objects (not .get_arm() which returns dicts)
            leader_arm = self.arm_registry.arms.get(leader_id) if self.arm_registry else None
            follower_arm = self.arm_registry.arms.get(follower_id) if self.arm_registry else None

            is_dynamixel_leader = leader_arm and leader_arm.motor_type in ('dynamixel_xl330', 'dynamixel_xl430')
            is_damiao_follower = follower_arm and follower_arm.motor_type == 'damiao'
            is_feetech_follower = follower_arm and follower_arm.motor_type == 'sts3215'

            if is_dynamixel_leader and is_damiao_follower:
                # Dynamixel→Damiao: direct mapping, float radians passthrough
                for dyn_name, dam_name in self.DYNAMIXEL_TO_DAMIAO_JOINT_MAP.items():
                    self.joint_mapping[f"{dyn_name}.pos"] = f"{dam_name}.pos"
                self._has_damiao_follower = True
                self._follower_value_mode = "float"
            elif is_dynamixel_leader and is_feetech_follower:
                # Dynamixel→Feetech: direct mapping, rad→percent conversion
                for dyn_name, dam_name in self.DYNAMIXEL_TO_DAMIAO_JOINT_MAP.items():
                    self.joint_mapping[f"{dyn_name}.pos"] = f"{dam_name}.pos"
                self._follower_value_mode = "rad_to_percent"
                # Precompute leader calibration ranges for absolute rad→percent mapping
                if self._active_leader and hasattr(self._active_leader, 'calibration') and self._active_leader.calibration:
                    for dyn_name, dam_name in self.DYNAMIXEL_TO_DAMIAO_JOINT_MAP.items():
                        if dyn_name == "gripper":
                            continue
                        l_cal = self._active_leader.calibration.get(dyn_name)
                        if l_cal:
                            f_key = f"{dam_name}.pos"
                            self._leader_cal_ranges[f_key] = (l_cal.range_min, l_cal.range_max)
                    if self._leader_cal_ranges:
                        print(f"[Teleop] Absolute mapping: {len(self._leader_cal_ranges)} joints from leader calibration", flush=True)
                    else:
                        print("[Teleop] WARNING: Leader calibration missing — falling back to relative delta tracking", flush=True)
            else:
                # Legacy prefix-based mapping for Feetech/same-type arms
                leader_prefix = self._get_arm_prefix(leader_id)
                follower_prefix = self._get_arm_prefix(follower_id)
                for name in self.joint_names_template:
                    leader_key = f"{leader_prefix}{name}.pos"
                    follower_key = f"{follower_prefix}{name}.pos"
                    self.joint_mapping[leader_key] = follower_key

        logger.info(f"Pairing-based Mapping: {len(self.joint_mapping)} joints mapped from {len(pairings)} pairings.")
        if self.joint_mapping:
            for lk, fk in self.joint_mapping.items():
                logger.info(f"  {lk} → {fk}")

    def _build_pairing_context(self, pairing: dict, leader_inst, follower_inst) -> PairingContext:
        """Build a fully independent PairingContext for one leader→follower pair.

        This ensures each pair's mapping, value mode, and scaling are isolated
        from other pairs — preventing the cross-contamination that caused the
        Damiao crash when running simultaneously with Feetech pairs.
        """
        leader_id = pairing['leader_id']
        follower_id = pairing['follower_id']
        pairing_id = f"{leader_id}→{follower_id}"

        joint_mapping = {}
        follower_value_mode = "int"
        has_damiao_follower = False
        leader_cal_ranges = {}

        # Determine arm types
        leader_arm = self.arm_registry.arms.get(leader_id) if self.arm_registry else None
        follower_arm = self.arm_registry.arms.get(follower_id) if self.arm_registry else None

        is_dynamixel_leader = leader_arm and leader_arm.motor_type in ('dynamixel_xl330', 'dynamixel_xl430')
        is_damiao_follower_arm = follower_arm and follower_arm.motor_type == 'damiao'
        is_feetech_follower = follower_arm and follower_arm.motor_type == 'sts3215'

        if is_dynamixel_leader and is_damiao_follower_arm:
            # Dynamixel→Damiao: direct mapping, float radians passthrough
            for dyn_name, dam_name in self.DYNAMIXEL_TO_DAMIAO_JOINT_MAP.items():
                joint_mapping[f"{dyn_name}.pos"] = f"{dam_name}.pos"
            has_damiao_follower = True
            follower_value_mode = "float"
        elif is_dynamixel_leader and is_feetech_follower:
            # Dynamixel→Feetech: direct mapping, rad→percent conversion
            for dyn_name, dam_name in self.DYNAMIXEL_TO_DAMIAO_JOINT_MAP.items():
                joint_mapping[f"{dyn_name}.pos"] = f"{dam_name}.pos"
            follower_value_mode = "rad_to_percent"
            # Precompute leader calibration ranges for absolute rad→percent mapping
            if leader_inst and hasattr(leader_inst, 'calibration') and leader_inst.calibration:
                for dyn_name, dam_name in self.DYNAMIXEL_TO_DAMIAO_JOINT_MAP.items():
                    if dyn_name == "gripper":
                        continue
                    l_cal = leader_inst.calibration.get(dyn_name)
                    if l_cal:
                        f_key = f"{dam_name}.pos"
                        leader_cal_ranges[f_key] = (l_cal.range_min, l_cal.range_max)
                if leader_cal_ranges:
                    print(f"[Teleop] [{pairing_id}] Absolute mapping: {len(leader_cal_ranges)} joints from leader calibration", flush=True)
                else:
                    print(f"[Teleop] [{pairing_id}] WARNING: Leader calibration missing — falling back to relative delta tracking", flush=True)
        else:
            # Legacy prefix-based mapping for Feetech/same-type arms
            leader_prefix = self._get_arm_prefix(leader_id)
            follower_prefix = self._get_arm_prefix(follower_id)
            for name in self.joint_names_template:
                leader_key = f"{leader_prefix}{name}.pos"
                follower_key = f"{follower_prefix}{name}.pos"
                joint_mapping[leader_key] = follower_key

        logger.info(f"[{pairing_id}] Built context: mode={follower_value_mode}, damiao={has_damiao_follower}, {len(joint_mapping)} joints")
        for lk, fk in joint_mapping.items():
            logger.info(f"  [{pairing_id}] {lk} → {fk}")

        return PairingContext(
            pairing_id=pairing_id,
            active_leader=leader_inst,
            active_robot=follower_inst,
            joint_mapping=joint_mapping,
            follower_value_mode=follower_value_mode,
            has_damiao_follower=has_damiao_follower,
            leader_cal_ranges=leader_cal_ranges,
        )

    def _get_arm_prefix(self, arm_id: str) -> str:
        """Get the joint name prefix for an arm ID."""
        # For legacy IDs like "left_follower", "right_leader" -> extract side
        if arm_id.startswith("left_"):
            return "left_"
        elif arm_id.startswith("right_"):
            return "right_"
        elif arm_id == "damiao_follower" or arm_id == "damiao_leader":
            return ""  # Damiao uses unprefixed joint names
        else:
            # For custom arm IDs, check the arm registry
            if self.arm_registry:
                arm = self.arm_registry.get_arm(arm_id)
                if arm:
                    # Use the arm ID as prefix for custom arms
                    return f"{arm_id}_"
            return ""  # Default to no prefix

    def _precompute_mappings_legacy(self):
        """Legacy side-based mapping (left_leader -> left_follower, etc.)"""
        # Helper to check if active
        def is_active(side, group):
            if self.active_arms is None: return True
            id_str = f"{side}_{group}"
            return id_str in self.active_arms

        # 1. Teleop Mapping (Leader -> Follower)
        # We iterate potential keys and check if they are active
        for side in ["left", "right", "default"]:
            if side == "default":
                 prefix = ""
            else:
                 prefix = f"{side}_"

            # Check if this side is active for BOTH leader and follower
            leader_active = is_active(side, "leader")
            follower_active = is_active(side, "follower")

            if leader_active and follower_active:
                for name in self.joint_names_template:
                    leader_key = f"{prefix}{name}.pos"
                    follower_key = f"{prefix}{name}.pos" # Robot action key (must end in .pos for Umbra)
                    self.joint_mapping[leader_key] = follower_key

        logger.info(f"Legacy Teleop Mapping: {len(self.joint_mapping)} joints mapped.")

        # 2. Assist Groups (Leader Keys for Assist Calculation)
        if self.leader_assists:
            for arm_key in self.leader_assists.keys():
                prefix = f"{arm_key}_" if arm_key != "default" else ""
                
                # Pre-generate the list of full joint names for this arm
                # This avoids f-string creation in the loop
                arm_joint_names = [f"{prefix}{name}" for name in self.joint_names_template]
                self.assist_groups[arm_key] = arm_joint_names
                
        logger.info(f"Assist Groups Optimized: {list(self.assist_groups.keys())}")


    def start(self, force=False, active_arms=None):
        # Cancel any ongoing homing before starting new teleop
        if getattr(self, '_homing_thread', None) and self._homing_thread.is_alive():
            self._homing_cancel = True
            self._homing_thread.join(timeout=2.0)

        if self.is_running:
            logger.info("Teleop already running, stopping first before restart...")
            self.stop()

        if not self.robot and not (self.arm_registry and active_arms):
             raise Exception("Robot not connected and no arm registry arms selected")
        
        # Store active arms (if provided, else None means All)
        self.active_arms = active_arms
        logger.info(f"Teleoperation Request: Active Arms = {self.active_arms}")
        
        # Validate selection if provided
        if self.active_arms is not None:
             leaders = []
             followers = []
             for a in self.active_arms:
                 if "leader" in a:
                     leaders.append(a)
                 elif "follower" in a:
                     followers.append(a)
                 elif self.arm_registry:
                     arm_info = self.arm_registry.get_arm(a)
                     if arm_info and arm_info.get("role") == "leader":
                         leaders.append(a)
                     elif arm_info and arm_info.get("role") == "follower":
                         followers.append(a)
             if not force and (not leaders or not followers):
                  logger.error(f"Selection Validation Failed: leaders={leaders}, followers={followers}, active_arms={self.active_arms}")
                  raise Exception("Invalid Selection: Must select at least one Leader and one Follower.")
        
        if not self.check_calibration():
             msg = "System not fully calibrated."
             if not force:
                 logger.warning(f"IGNORING CALIBRATION CHECK: {msg}")
             else:
                 logger.warning(f"FORCE START: {msg}")
        
        # Resolve active robot/leader from arm registry pairings
        # Build per-pairing contexts for independent teleop loops
        self._active_robot = self.robot  # default to legacy (used by recording)
        self._active_leader = self.leader  # default to legacy
        self._pairing_contexts = []
        self._teleop_threads = []
        print(f"[TELEOP] arm_registry={self.arm_registry is not None}, active_arms={self.active_arms}", flush=True)

        if self.arm_registry and self.active_arms:
            pairings = self.arm_registry.get_active_pairings(self.active_arms)
            print(f"[TELEOP] Found {len(pairings)} pairings for active_arms={self.active_arms}", flush=True)

            for pairing in pairings:
                leader_id = pairing['leader_id']
                follower_id = pairing['follower_id']
                print(f"[TELEOP] Resolving arm instances for pairing: {leader_id} → {follower_id}", flush=True)

                # Auto-connect if not already connected
                if leader_id not in self.arm_registry.arm_instances:
                    print(f"[TELEOP] Auto-connecting leader: {leader_id}", flush=True)
                    try:
                        result = self.arm_registry.connect_arm(leader_id)
                        print(f"[TELEOP] Leader connect result: {result}", flush=True)
                    except Exception as e:
                        import traceback
                        print(f"[TELEOP] Leader connect EXCEPTION: {e}", flush=True)
                        traceback.print_exc()
                        continue  # Skip this pairing if leader can't connect
                else:
                    print(f"[TELEOP] Leader {leader_id} already connected", flush=True)
                if follower_id not in self.arm_registry.arm_instances:
                    print(f"[TELEOP] Auto-connecting follower: {follower_id}", flush=True)
                    try:
                        result = self.arm_registry.connect_arm(follower_id)
                        print(f"[TELEOP] Follower connect result: {result}", flush=True)
                    except Exception as e:
                        import traceback
                        print(f"[TELEOP] Follower connect EXCEPTION: {e}", flush=True)
                        traceback.print_exc()
                        continue  # Skip this pairing if follower can't connect
                else:
                    print(f"[TELEOP] Follower {follower_id} already connected", flush=True)

                leader_inst = self.arm_registry.arm_instances.get(leader_id)
                follower_inst = self.arm_registry.arm_instances.get(follower_id)

                if leader_inst:
                    print(f"[TELEOP] Using arm-registry leader: {leader_id} ({type(leader_inst).__name__})", flush=True)
                else:
                    print(f"[TELEOP] WARNING: No instance for leader {leader_id}, skipping pairing", flush=True)
                    continue
                if follower_inst:
                    print(f"[TELEOP] Using arm-registry follower: {follower_id} ({type(follower_inst).__name__})", flush=True)
                else:
                    print(f"[TELEOP] WARNING: No instance for follower {follower_id}, skipping pairing", flush=True)
                    continue

                # Build isolated per-pairing context (prevents cross-contamination)
                ctx = self._build_pairing_context(pairing, leader_inst, follower_inst)
                self._pairing_contexts.append(ctx)

                # Reload inversions per follower
                if hasattr(follower_inst, "reload_inversions"):
                    try:
                        follower_inst.reload_inversions()
                    except Exception as e:
                        logger.warning(f"Failed to reload inversions for {follower_id}: {e}")

                # Enable torque per follower
                self._enable_torque_for_robot(follower_inst)

            # Set _active_robot/_active_leader to first pair for recording/legacy compatibility
            if self._pairing_contexts:
                self._active_robot = self._pairing_contexts[0].active_robot
                self._active_leader = self._pairing_contexts[0].active_leader
                # Also set legacy joint_mapping/value_mode from first pair (for any code
                # that still reads self.joint_mapping directly)
                self.joint_mapping = self._pairing_contexts[0].joint_mapping
                self._follower_value_mode = self._pairing_contexts[0].follower_value_mode
                self._has_damiao_follower = self._pairing_contexts[0].has_damiao_follower
                self._leader_cal_ranges = self._pairing_contexts[0].leader_cal_ranges
        else:
            print(f"[TELEOP] No arm_registry or no active_arms — using legacy robot/leader", flush=True)
            # Legacy single-pair: build context from self.robot/self.leader
            self._precompute_mappings()

        # Switch Leader to PWM Mode for Active Assist
        if self.leader and self.leader_assists:
            try:
                if self.assist_enabled:
                    logger.info("Switching Leader(s) to PWM Mode for Assist...")
                    if "left" in self.leader_assists:
                         self.leader.left_arm.bus.set_operating_mode(OperatingMode.PWM)
                         self.leader.left_arm.bus.enable_torque()
                         self.leader.right_arm.bus.set_operating_mode(OperatingMode.PWM)
                         self.leader.right_arm.bus.enable_torque()
                    else:
                         self.leader.bus.set_operating_mode(OperatingMode.PWM)
                         self.leader.bus.enable_torque()
                else:
                    logger.info("Assist Disabled: Ensuring Leader Torque is OFF.")
                    if "left" in self.leader_assists:
                         self.leader.left_arm.bus.disable_torque()
                         self.leader.right_arm.bus.disable_torque()
                    else:
                         self.leader.bus.disable_torque()

            except Exception as e:
                logger.error(f"Failed to switch Leader Mode: {e}")

        # Startup blend config
        self._blend_duration = 0.5  # seconds — rate limiter provides additional smooth ramping
        self._filtered_gripper_torque = 0.0  # Reset force feedback filter

        self.is_running = True

        # Start per-pairing teleop loop threads
        if self._pairing_contexts:
            for ctx in self._pairing_contexts:
                t = threading.Thread(
                    target=self._teleop_loop,
                    args=(ctx,),
                    daemon=True,
                    name=f"teleop-{ctx.pairing_id}",
                )
                self._teleop_threads.append(t)
                t.start()
                print(f"[TELEOP] Started loop thread for {ctx.pairing_id}", flush=True)
        else:
            # Legacy single-pair fallback: build a context from self state
            ctx = PairingContext(
                pairing_id="legacy",
                active_leader=self._active_leader,
                active_robot=self._active_robot,
                joint_mapping=self.joint_mapping,
                follower_value_mode=getattr(self, '_follower_value_mode', 'int'),
                has_damiao_follower=getattr(self, '_has_damiao_follower', False),
                leader_cal_ranges=self._leader_cal_ranges,
            )
            self._pairing_contexts = [ctx]
            t = threading.Thread(target=self._teleop_loop, args=(ctx,), daemon=True, name="teleop-legacy")
            self._teleop_threads = [t]
            t.start()

    def _enable_torque_for_active_arms(self):
        """Legacy helper — delegates to _enable_torque_for_robot with self._active_robot."""
        active_robot = getattr(self, '_active_robot', None) or self.robot
        self._enable_torque_for_robot(active_robot)

    def _enable_torque_for_robot(self, active_robot):
        """Enable torque on a specific follower robot instance."""
        if not active_robot:
            print("[TELEOP] WARNING: No active robot for torque enable", flush=True)
            return

        # Skip MagicMock (fallback mock robot)
        from unittest.mock import MagicMock
        if isinstance(active_robot, MagicMock):
            print("[TELEOP] Skipping torque enable on MagicMock robot", flush=True)
            return

        try:
            print(f"[TELEOP] Enabling torque on {type(active_robot).__name__}...", flush=True)

            # For Damiao arms: re-configure ensures correct control mode is set.
            # Motors ARE enabled at end of configure() and ready for position commands.
            # MIT mode is used by default (stable), POS_VEL available via config.
            from lerobot.robots.damiao_follower.damiao_follower import DamiaoFollowerRobot
            if isinstance(active_robot, DamiaoFollowerRobot):
                mode_name = "MIT" if active_robot.bus.config.use_mit_mode else "POS_VEL"
                print(f"[TELEOP] Damiao detected — running configure() ({mode_name} mode)...", flush=True)
                active_robot.bus.configure()
                print(f"[TELEOP] Damiao configure() complete (motors enabled in {mode_name} mode)", flush=True)
            else:
                # Single-arm robot
                if hasattr(active_robot, "bus"):
                     active_robot.bus.enable_torque()
                # Dual-arm robot (e.g. BiUmbraFollower)
                if hasattr(active_robot, "left_arm"):
                     active_robot.left_arm.bus.enable_torque()
                if hasattr(active_robot, "right_arm"):
                     active_robot.right_arm.bus.enable_torque()

            print("[TELEOP] Torque enabled successfully", flush=True)

        except Exception as e:
            import traceback
            print(f"[TELEOP] Failed to enable torque: {e}", flush=True)
            traceback.print_exc()
            logger.error(f"Failed to enable torque: {e}")

    def stop(self):
        if not self.is_running:
            return
        logger.info("Stopping teleoperation...")
        self.is_running = False

        # Join all per-pairing teleop threads
        current = threading.current_thread()
        for t in self._teleop_threads:
            if t and t != current and t.is_alive():
                t.join(timeout=2.0)
        # Legacy fallback: join self.thread if it exists
        if hasattr(self, 'thread') and self.thread and self.thread != current:
            try:
                self.thread.join(timeout=2.0)
            except Exception:
                pass

        # Stop observation capture thread if running
        self._stop_obs_thread()

        # Stop Recording if active
        if self.session_active:
            try:
                self.stop_recording_session()
            except Exception as e:
                logger.error(f"Failed to auto-stop recording session: {e}")

        # Switch Leader back to Position Mode (Safety)
        if self.leader:
            try:
                logger.info("Restoring Leader to Position Mode...")
                if "left" in self.leader_assists:
                     self.leader.left_arm.bus.set_operating_mode(OperatingMode.POSITION)
                     self.leader.left_arm.bus.disable_torque()
                     self.leader.right_arm.bus.set_operating_mode(OperatingMode.POSITION)
                     self.leader.right_arm.bus.disable_torque()
                else:
                     self.leader.bus.set_operating_mode(OperatingMode.POSITION)
                     self.leader.bus.disable_torque()
            except Exception as e:
                logger.error(f"Failed to restore Leader Mode: {e}")

        # Per-pairing cleanup: reset force feedback and home each follower
        for ctx in self._pairing_contexts:
            # Reset force feedback on leader gripper
            if ctx.has_damiao_follower and ctx.active_leader:
                try:
                    ctx.active_leader.bus.write(
                        "Goal_Current", "gripper", self._ff_baseline_current, normalize=False
                    )
                except Exception as e:
                    logger.warning(f"[{ctx.pairing_id}] Failed to reset gripper Goal_Current: {e}")

                # Zero joint force feedback current (joint_4 → limp)
                if self._joint_ff_enabled:
                    try:
                        ctx.active_leader.bus.write(
                            "Goal_Current", "joint_4", 0, normalize=False
                        )
                    except Exception:
                        pass

            # Home follower arm (or disable immediately if no home position)
            if ctx.active_robot:
                home_pos = self._get_home_position(ctx.active_robot)
                if home_pos:
                    logger.info(f"[{ctx.pairing_id}] Homing follower to saved position ({len(home_pos)} joints)...")
                    self._homing_cancel = False
                    self._homing_thread = threading.Thread(
                        target=self._homing_loop,
                        args=(ctx.active_robot, home_pos),
                        daemon=True
                    )
                    self._homing_thread.start()
                else:
                    self._disable_follower_motors(ctx.active_robot)

        # Legacy fallback if no pairing contexts (shouldn't happen, but just in case)
        if not self._pairing_contexts:
            active_robot = getattr(self, '_active_robot', self.robot)
            if active_robot:
                home_pos = self._get_home_position(active_robot)
                if home_pos:
                    self._homing_cancel = False
                    self._homing_thread = threading.Thread(
                        target=self._homing_loop,
                        args=(active_robot, home_pos),
                        daemon=True
                    )
                    self._homing_thread.start()
                else:
                    self._disable_follower_motors(active_robot)

        # Reset active instances
        self._active_robot = self.robot
        self._active_leader = self.leader
        self._pairing_contexts = []
        self._teleop_threads = []

        logger.info("Teleoperation stopped.")

        if hasattr(self, '_debug_handler'):
             logger.removeHandler(self._debug_handler)
             self._debug_handler.close()

    def _get_home_position(self, robot) -> dict | None:
        """Get saved home position from arm registry config for the active follower."""
        if not self.arm_registry:
            return None
        for arm_id, instance in self.arm_registry.arm_instances.items():
            if instance is robot:
                arm_def = self.arm_registry.arms.get(arm_id)
                if arm_def and arm_def.config.get("home_position"):
                    return arm_def.config["home_position"]
        return None

    def _disable_follower_motors(self, robot):
        """Immediately disable all Damiao follower motors."""
        try:
            from lerobot.motors.damiao.damiao import DamiaoMotorsBus
            bus = getattr(robot, 'bus', None)
            if bus and isinstance(bus, DamiaoMotorsBus):
                logger.info("Disabling Damiao follower torque...")
                for motor in bus._motors.values():
                    bus._control.disable(motor)
        except Exception as e:
            logger.warning(f"Failed to disable Damiao follower torque: {e}")

    def _homing_loop(self, robot, home_pos, duration=10.0, homing_vel=0.05):
        """Move robot to home position over duration, then disable motors.

        Uses the existing MIT rate limiter in sync_write() for smooth movement.
        At homing_vel=0.15: J8009P ~1 rad/s, J4340P ~0.8 rad/s.
        """
        try:
            from lerobot.motors.damiao.damiao import DamiaoMotorsBus
            bus = getattr(robot, 'bus', None)
            if not bus or not isinstance(bus, DamiaoMotorsBus):
                return

            old_vel = bus.velocity_limit
            bus.velocity_limit = homing_vel
            print(f"[Teleop] Homing started (vel={homing_vel}, duration={duration}s)", flush=True)

            start = time.time()
            while time.time() - start < duration:
                if getattr(self, '_homing_cancel', False):
                    print("[Teleop] Homing cancelled", flush=True)
                    break
                bus.sync_write("Goal_Position", home_pos)
                time.sleep(1.0 / 30)  # 30Hz

            bus.velocity_limit = old_vel
        except Exception as e:
            print(f"[Teleop] Homing error: {e}", flush=True)
        finally:
            # Always disable motors when done
            try:
                from lerobot.motors.damiao.damiao import DamiaoMotorsBus
                bus = getattr(robot, 'bus', None)
                if bus and isinstance(bus, DamiaoMotorsBus):
                    for motor in bus._motors.values():
                        bus._control.disable(motor)
                    print("[Teleop] Homing complete — motors disabled", flush=True)
            except Exception:
                pass

    def _teleop_loop(self, ctx: PairingContext):
        """Main teleop control loop for one leader→follower pairing.

        Each pairing runs in its own thread with fully isolated state via ctx,
        preventing cross-contamination between different arm types.
        """
        pid = ctx.pairing_id
        logger.info(f"[{pid}] Teleoperation Control Loop Running at {self.frequency}Hz")
        print(f"[TELEOP] [{pid}] Control loop started at {self.frequency}Hz", flush=True)

        # Start blend timer NOW — when the loop actually begins executing
        ctx.blend_start_time = time.time()

        loop_count = 0
        self.last_loop_time = time.perf_counter()
        
        # Performance Monitoring
        perf_interval = self.frequency # Log every 1s
        perf_start = time.time()
        
        try:
            while self.is_running:
                loop_start = time.perf_counter()
                
                # 1. Read Leader State
                leader_action = {}
                # Capture Follower Obs (Needed for Recording)
                follower_obs = {}

                if ctx.active_leader:
                    obs = None

                    # Retry loop for transient serial communication errors on leader
                    for attempt in range(3):
                        try:
                            obs = ctx.active_leader.get_action()
                            break  # Success
                        except (OSError, ConnectionError) as e:
                            error_str = str(e)
                            if "Incorrect status packet" in error_str or "Port is in use" in error_str:
                                if attempt < 2:
                                    time.sleep(0.005)  # 5ms backoff
                                    continue
                                else:
                                    if loop_count % 60 == 0:
                                        logger.warning(f"Leader read failed after 3 attempts: {e}")
                            else:
                                logger.error(f"Leader read error: {e}")
                                break

                    if not obs:
                        # Skip this loop iteration if leader read completely failed
                        continue

                    # 1a. Leader Assist (Gravity/Transparency)
                    if self.leader_assists and self.assist_enabled:
                         # Iterate pre-computed groups
                         for arm_key, arm_joint_names in self.assist_groups.items():
                             service = self.leader_assists[arm_key]
                             
                             positions = []
                             velocities = [] 
                             valid = True
                             
                             # Extract positions/velocities for this arm
                             for fullname in arm_joint_names:
                                 pos_key = f"{fullname}.pos"
                                 if pos_key in obs:
                                     deg = obs[pos_key]
                                     positions.append(deg)
                                     
                                     # Smooth Velocity Estimate (EMA)
                                     raw_vel = 0.0
                                     if fullname in self.last_leader_pos:
                                         delta = deg - self.last_leader_pos[fullname]
                                         # Handle wrapping? Usually robot driver handles this or returns absolute.
                                         # Assuming absolute degrees.
                                         raw_vel = delta / self.dt 
                                     
                                     # Apply EMA
                                     prev_vel = self.leader_vel_kf.get(fullname, 0.0)
                                     filtered_vel = self.alpha_vel * raw_vel + (1 - self.alpha_vel) * prev_vel
                                     self.leader_vel_kf[fullname] = filtered_vel
                                     
                                     velocities.append(filtered_vel)
                                     
                                     # Update Cache
                                     self.last_leader_pos[fullname] = deg
                                 else:
                                     valid = False
                                     break
                             
                             if valid:
                                 try:
                                     follower_loads = self.safety.latest_loads
                                     haptic_forces = {}
                                     
                                     # Haptics: Compute External Force on Follower
                                     if arm_key in self.follower_gravity_models:
                                          follower_model = self.follower_gravity_models[arm_key]
                                          predicted_gravity = follower_model.predict_gravity(positions)
                                          
                                          for i, name in enumerate(arm_joint_names):
                                              # Heuristic mapping for load lookup
                                              # In pre-compute we could optimize this too, but it's okay for now
                                              follower_name = name.replace("leader", "follower")
                                              measured_load = float(follower_loads.get(follower_name, 0))
                                              expected_load = predicted_gravity[i]
                                              haptic_forces[name] = measured_load - expected_load
                                     
                                     # Compute PWM
                                     pwm_dict = service.compute_assist_torque(
                                         arm_joint_names, positions, velocities, follower_torques=haptic_forces
                                     )
                                     
                                     # Write PWM
                                     if pwm_dict:
                                         if arm_key == "left":
                                              local_pwm = {k.replace("left_", ""): v for k, v in pwm_dict.items()}
                                              self.leader.left_arm.bus.write_pwm(local_pwm)
                                         elif arm_key == "right":
                                              local_pwm = {k.replace("right_", ""): v for k, v in pwm_dict.items()}
                                              self.leader.right_arm.bus.write_pwm(local_pwm)
                                         else:
                                              self.leader.bus.write_pwm(pwm_dict)
                                 except Exception as e:
                                     logger.error(f"Assist Error: {e}") # Enable logging for debug

                    # Debug: log on first frame to diagnose mapping issues
                    if loop_count == 0:
                        print(f"[TELEOP DEBUG] [{pid}] _active_leader type: {type(ctx.active_leader).__name__}", flush=True)
                        print(f"[TELEOP DEBUG] [{pid}] _active_robot type: {type(ctx.active_robot).__name__}", flush=True)
                        print(f"[TELEOP DEBUG] [{pid}] joint_mapping ({len(ctx.joint_mapping)} entries): {ctx.joint_mapping}", flush=True)
                        print(f"[TELEOP DEBUG] obs keys from leader: {list(obs.keys()) if obs else 'None'}", flush=True)

                    # 1b. Map to Follower Action (Optimized)
                    # Use per-pairing mapping from ctx (prevents cross-contamination)
                    for l_key, f_key in ctx.joint_mapping.items():
                        if l_key in obs:
                            val = obs[l_key]
                            if ctx.follower_value_mode == "float":
                                leader_action[f_key] = val  # Damiao: radians
                            elif ctx.follower_value_mode == "rad_to_percent":
                                if 'gripper' in f_key:
                                    # Gripper: leader 0-1 → follower 0-100 (already absolute)
                                    leader_action[f_key] = val * 100.0
                                elif f_key in ctx.leader_cal_ranges:
                                    # Absolute mapping: leader radians → leader's ±100% → follower's ±100%
                                    rmin, rmax = ctx.leader_cal_ranges[f_key]
                                    leader_ticks = (val + np.pi) * 4096.0 / (2 * np.pi)
                                    # Unwrap: if homed ticks crossed the 0/4096 encoder boundary,
                                    # bring them back into the calibration range
                                    center = (rmin + rmax) * 0.5
                                    while leader_ticks < center - 2048:
                                        leader_ticks += 4096
                                    while leader_ticks > center + 2048:
                                        leader_ticks -= 4096
                                    leader_action[f_key] = ((leader_ticks - rmin) / (rmax - rmin)) * 200 - 100
                                else:
                                    # Fallback: delta-based (no leader calibration available)
                                    if l_key in ctx.leader_start_rad and f_key in ctx.follower_start_pos:
                                        delta = val - ctx.leader_start_rad[l_key]
                                        scale = ctx.rad_to_percent_scale.get(f_key, 100.0 / np.pi)
                                        leader_action[f_key] = ctx.follower_start_pos[f_key] + delta * scale
                                    else:
                                        scale = ctx.rad_to_percent_scale.get(f_key, 100.0 / np.pi)
                                        leader_action[f_key] = val * scale
                            else:
                                leader_action[f_key] = int(val)  # Legacy Feetech→Feetech

                    if loop_count == 0:
                        print(f"[TELEOP DEBUG] [{pid}] leader_action after mapping ({len(leader_action)} entries): {leader_action}", flush=True)

                    # 1c. Startup blend: ramp from follower's current position
                    if ctx.blend_start_time and leader_action:
                        # First frame: capture leader start for delta-based tracking (Dynamixel→Feetech)
                        if not ctx.leader_start_rad and ctx.follower_value_mode == "rad_to_percent" and obs:
                            ctx.leader_start_rad = {l_key: obs[l_key] for l_key in ctx.joint_mapping if l_key in obs}
                            print(f"[Teleop] [{pid}] Delta tracking: captured {len(ctx.leader_start_rad)} leader start positions", flush=True)

                        # First frame: capture follower's actual position
                        if not ctx.follower_start_pos and ctx.active_robot:
                            try:
                                fobs = ctx.active_robot.get_observation()
                                ctx.follower_start_pos = {
                                    k: v for k, v in fobs.items() if k.endswith('.pos')
                                }
                                print(f"[Teleop] [{pid}] Startup blend: captured {len(ctx.follower_start_pos)} follower positions", flush=True)
                            except Exception as e:
                                print(f"[Teleop] [{pid}] Startup blend: get_observation() FAILED: {e}", flush=True)
                                # Fallback: use last known positions from bus (seeded during configure)
                                if hasattr(ctx.active_robot, 'bus') and hasattr(ctx.active_robot.bus, '_last_positions'):
                                    lp = ctx.active_robot.bus._last_positions
                                    ctx.follower_start_pos = {f"{k}.pos": v for k, v in lp.items()}
                                    print(f"[Teleop] [{pid}] Startup blend: using bus fallback ({len(ctx.follower_start_pos)} joints)", flush=True)

                        # Compute per-joint rad→percent scale factors from follower calibration
                        if not ctx.rad_to_percent_scale and ctx.follower_value_mode == "rad_to_percent":
                            if hasattr(ctx.active_robot, 'calibration') and ctx.active_robot.calibration:
                                for l_key, f_key in ctx.joint_mapping.items():
                                    motor_name = f_key.replace('.pos', '')
                                    if 'gripper' in motor_name:
                                        continue  # Gripper uses its own scaling
                                    cal = ctx.active_robot.calibration.get(motor_name)
                                    if cal:
                                        tick_range = cal.range_max - cal.range_min
                                        if tick_range > 0:
                                            ctx.rad_to_percent_scale[f_key] = 4096.0 * 100.0 / (np.pi * tick_range)
                                print(f"[Teleop] [{pid}] Per-joint scales: { {k: f'{v:.1f}' for k, v in ctx.rad_to_percent_scale.items()} }", flush=True)

                        elapsed = time.time() - ctx.blend_start_time
                        alpha = min(1.0, elapsed / self._blend_duration)

                        if alpha < 1.0 and ctx.follower_start_pos:
                            for key in list(leader_action.keys()):
                                if key in ctx.follower_start_pos:
                                    start = ctx.follower_start_pos[key]
                                    target = leader_action[key]
                                    leader_action[key] = start + alpha * (target - start)

                            # First-frame diagnostic: log large position deltas
                            if loop_count == 0:
                                for key in list(leader_action.keys()):
                                    if key in ctx.follower_start_pos:
                                        orig_target = leader_action.get(key, 0)  # Already blended
                                        follower_pos = ctx.follower_start_pos[key]
                                        raw_delta = orig_target - follower_pos
                                        if abs(raw_delta) > 0.01:
                                            print(f"[Teleop] [{pid}] Blend frame 0: {key} "
                                                  f"follower={follower_pos:.3f}, blended={orig_target:.3f}, "
                                                  f"delta={raw_delta:+.4f} rad", flush=True)

                # Debug: trace link3 position through teleop pipeline (first 5 frames)
                if loop_count < 5 and leader_action and "link3.pos" in leader_action:
                    _alpha = alpha if ctx.blend_start_time else -1.0
                    blend_active = (ctx.blend_start_time is not None and _alpha < 1.0)
                    fstart = ctx.follower_start_pos.get("link3.pos", "N/A") if ctx.follower_start_pos else "N/A"
                    print(f"[Teleop] Frame {loop_count}: link3.pos={leader_action['link3.pos']:.4f} "
                          f"(alpha={_alpha:.4f}, blend_active={blend_active}, "
                          f"follower_start={fstart})", flush=True)

                # 2. Send Action IMMEDIATELY (Low Latency Control)
                # Key insight from LeRobot: send_action and get_observation are independent.
                # Motor writes use sync_write (no response wait), cameras have background threads.
                # NO LOCK NEEDED - LeRobot's architecture handles thread safety at bus level.
                if leader_action and ctx.active_robot:
                    try:
                        ctx.active_robot.send_action(leader_action)
                        if loop_count == 0:
                            print(f"[TELEOP DEBUG] [{pid}] send_action SUCCESS, sent {len(leader_action)} values to {type(ctx.active_robot).__name__}", flush=True)
                    except Exception as e:
                        if loop_count % 60 == 0:
                            print(f"[TELEOP] Send Action Failed: {e}", flush=True)
                            logger.error(f"Send Action Failed: {e}")

                    # SAFETY: Check if CAN bus died (emergency shutdown triggered in driver)
                    if hasattr(ctx.active_robot, 'bus') and getattr(ctx.active_robot.bus, '_can_bus_dead', False):
                        print(f"[TELEOP] [{pid}] CAN BUS DEAD — stopping teleop for safety", flush=True)
                        logger.error(f"[TELEOP] [{pid}] CAN bus failure detected — emergency stop")
                        self.is_running = False
                        break

                    # SAFETY: Check Damiao torque limits (every 6th frame = ~10Hz)
                    # read_torques() costs ~14ms (7 motors x 2ms), so throttle to avoid
                    # consuming too much of the 16.7ms frame budget at 60Hz.
                    # 3-violation debounce in SafetyLayer triggers e-stop within ~300ms.
                    if ctx.has_damiao_follower and loop_count % 6 == 3:
                        try:
                            if not self.safety.check_damiao_limits(ctx.active_robot):
                                print(f"[TELEOP] [{pid}] SAFETY: Torque limit exceeded — EMERGENCY STOP", flush=True)
                                logger.error(f"[TELEOP] [{pid}] Damiao torque limit exceeded — emergency stop")
                                self.is_running = False
                                break
                        except Exception as e:
                            if loop_count % 60 == 0:
                                logger.warning(f"[TELEOP] Safety check error (non-fatal): {e}")

                    # 2a. Gripper Force Feedback (follower torque → leader current ceiling)
                    if (self._force_feedback_enabled
                            and ctx.has_damiao_follower
                            and ctx.active_leader
                            and loop_count > 0):
                        try:
                            torques = ctx.active_robot.get_torques()
                            raw_torque = torques.get("gripper", 0.0)

                            # Use absolute value — direction doesn't matter for grip feel
                            torque_mag = abs(raw_torque)

                            # EMA filter
                            ctx.filtered_gripper_torque = (
                                self._ff_alpha * torque_mag
                                + (1 - self._ff_alpha) * ctx.filtered_gripper_torque
                            )

                            # Map to Goal_Current: dead zone → linear ramp → saturation
                            if ctx.filtered_gripper_torque <= self._ff_torque_threshold:
                                goal_current = self._ff_baseline_current
                            elif ctx.filtered_gripper_torque >= self._ff_torque_saturation:
                                goal_current = self._ff_max_current
                            else:
                                t = (ctx.filtered_gripper_torque - self._ff_torque_threshold) / (
                                    self._ff_torque_saturation - self._ff_torque_threshold
                                )
                                goal_current = int(
                                    self._ff_baseline_current
                                    + t * (self._ff_max_current - self._ff_baseline_current)
                                )

                            goal_current = max(self._ff_baseline_current, min(self._ff_max_current, goal_current))

                            ctx.active_leader.bus.write(
                                "Goal_Current", "gripper", goal_current, normalize=False
                            )

                            # Debug log once per second
                            if loop_count % 60 == 0:
                                print(
                                    f"[FORCE_FB] [{pid}] torque={raw_torque:.2f}Nm "
                                    f"filtered={ctx.filtered_gripper_torque:.2f}Nm "
                                    f"goal_current={goal_current}mA",
                                    flush=True,
                                )
                        except Exception as e:
                            if loop_count % 60 == 0:
                                logger.warning(f"[FORCE_FB] Error: {e}")

                        # 2a-ii. Joint force feedback: CURRENT_POSITION mode (virtual spring)
                        # Goal_Position = follower's actual position (spring target)
                        # Goal_Current = position error magnitude (how firmly to hold)
                        if self._joint_ff_enabled:
                            try:
                                cached = ctx.active_robot.get_cached_positions()
                                follower_pos = cached.get("link3", None)
                                leader_pos = obs.get("joint_4.pos", None)

                                if follower_pos is not None and leader_pos is not None:
                                    pos_error = abs(leader_pos - follower_pos)

                                    # Goal_Current: how firmly the motor holds at Goal_Position
                                    if pos_error > self._joint_ff_deadzone:
                                        excess = pos_error - self._joint_ff_deadzone
                                        goal_current = min(
                                            int(max(self._joint_ff_k_spring * excess, self._joint_ff_min_force)),
                                            self._joint_ff_max_current,
                                        )
                                    else:
                                        goal_current = 0  # Completely limp during normal tracking

                                    # Goal_Position: follower's actual position (spring target)
                                    # Convert radians → Dynamixel raw ticks
                                    homed_ticks = int((follower_pos + np.pi) / (2 * np.pi) * 4096)
                                    j4_id = ctx.active_leader.bus.motors["joint_4"].id
                                    raw_ticks = homed_ticks - ctx.active_leader.bus._software_homing_offsets.get(j4_id, 0)

                                    ctx.active_leader.bus.write(
                                        "Goal_Position", "joint_4", int(raw_ticks), normalize=False
                                    )
                                    ctx.active_leader.bus.write(
                                        "Goal_Current", "joint_4", goal_current, normalize=False
                                    )

                                    if loop_count % 60 == 0:
                                        print(
                                            f"[JOINT_FB] leader={leader_pos:.3f} follower={follower_pos:.3f} "
                                            f"error={pos_error:.3f}rad current={goal_current}mA",
                                            flush=True,
                                        )
                            except Exception as e:
                                if loop_count % 60 == 0:
                                    logger.warning(f"[JOINT_FB] Error: {e}")

                elif ctx.active_robot and loop_count % 60 == 0:
                    logger.warning(f"[{pid}] No Leader Action generated (Mapping issue or Empty Obs)")

                # 2b. Share motor positions with recording thread
                # IMPORTANT: Use FOLLOWER robot positions, not leader positions!
                # This ensures recorded data matches what HIL reads at inference time.
                # Leader and follower may have different calibrations, so same physical
                # position can give different encoder values.
                if ctx.active_robot:
                    if self.recording_active:
                        # Only read follower positions when recording — needed for
                        # accurate dataset frames. For Damiao (CAN), this costs ~14ms
                        # per loop (7 motors × 2ms each), so skip when not recording.
                        follower_obs = None

                        # Retry loop for transient serial communication errors
                        for attempt in range(3):  # Up to 3 attempts
                            try:
                                follower_obs = ctx.active_robot.get_observation()
                                break  # Success, exit retry loop
                            except (OSError, ConnectionError) as e:
                                error_str = str(e)
                                if "Incorrect status packet" in error_str or "Port is in use" in error_str:
                                    if attempt < 2:  # Not the last attempt
                                        time.sleep(0.005)  # 5ms backoff
                                        continue
                                    else:
                                        # Log only on final failure (rate-limited)
                                        if loop_count % 60 == 0:
                                            logger.warning(f"Motor read failed after 3 attempts: {e}")
                                else:
                                    # Non-transient error, don't retry
                                    logger.error(f"Motor read error: {e}")
                                    break

                        # Process observation (whether from retry success or previous cache)
                        if follower_obs:
                            follower_motors = {k: v for k, v in follower_obs.items() if '.pos' in k}
                            if follower_motors:
                                with self._action_lock:
                                    self._latest_leader_action = follower_motors.copy()
                        elif leader_action:
                            # Fallback to leader action if all retries failed
                            with self._action_lock:
                                self._latest_leader_action = leader_action.copy()
                    elif leader_action:
                        # Not recording: use leader action directly as cached action
                        # (no need to read follower positions over CAN)
                        with self._action_lock:
                            self._latest_leader_action = leader_action.copy()

                # Debug heartbeat (disabled - too spammy)
                # if loop_count % 60 == 0:
                #     print(f"Teleop Heartbeat: {loop_count} (Active: {self.is_running}, Recording: {self.recording_active})")

                # 4. Store Data (for UI)
                if loop_count % 5 == 0:
                    self._update_history(leader_action)
                
                # 5. Performance Logging
                loop_count += 1
                if loop_count % perf_interval == 0:
                     now = time.time()
                     real_hz = perf_interval / (now - perf_start)
                     logger.info(f"Teleop Loop Rate: {real_hz:.1f} Hz")
                     perf_start = now

                # 6. Sleep
                dt_s = time.perf_counter() - loop_start
                precise_sleep(self.dt - dt_s)

            # Normal exit - log why we stopped
            logger.info(f"[{pid}] Teleop loop exited normally (is_running=False)")
            print(f"[TELEOP] [{pid}] Loop exited normally")

        except OSError as e:
            if e.errno == 5:
                logger.error(f"[{pid}] TELEOP STOPPED: Hardware Disconnected: {e}")
                print(f"[TELEOP ERROR] [{pid}] Hardware Disconnected: {e}")
            else:
                logger.error(f"[{pid}] TELEOP STOPPED: OSError {e.errno}: {e}")
                print(f"[TELEOP ERROR] [{pid}] OSError: {e}")
        except Exception as e:
            logger.error(f"[{pid}] TELEOP STOPPED: {e}")
            print(f"[TELEOP ERROR] [{pid}] Loop Failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Signal all loops to stop (idempotent), then trigger cleanup
            logger.info(f"[{pid}] TELEOP CLEANUP: Calling stop()")
            print(f"[TELEOP] [{pid}] Cleanup - calling stop()")
            self.stop()

    def _update_history(self, action_dict):
        # Convert dictionary to simple list of values for graph
        timestamp = time.time()
        
        data_point = {"time": timestamp}
        
        for k, v in action_dict.items():
            # Simplify key name for UI
            short_key = k.replace(".pos", "").replace("follower", "").strip("_")
            data_point[short_key] = float(v)
            
        with self.history_lock:
            self.action_history.append(data_point)
            
    def get_data(self):
        """Returns the current data history and latest status."""
        history = []
        with self.history_lock:
            history = list(self.action_history)
            
        return {
            "history": history,
            "torque": self.safety.latest_loads,
            "recording": {
                "session_active": self.session_active,
                "episode_active": self.recording_active,
                "episode_count": self.episode_count
            }
        }

    # --- Recording Methods ---

    def _filter_observation_features(
        self,
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

    def _filter_action_features(
        self,
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

    def start_recording_session(
        self,
        repo_id: str,
        task: str,
        fps: int = 30,
        root: str = None,
        selected_cameras: list = None,
        selected_arms: list = None
    ):
        """Initializes a new LeRobotDataset for recording.

        Args:
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
        print(f"  session_active: {self.session_active}")
        print(f"  robot: {self.robot is not None}")
        print("=" * 60)

        if self.session_active:
            print("[START_SESSION] ERROR: Session already active!")
            raise Exception("Session already active")

        # Store selections for use during recording
        self._selected_cameras = selected_cameras  # None means all cameras
        self._selected_arms = selected_arms        # None means all arms

        # Set default root to app datasets directory
        if root is None:
            base_dir = _DEFAULT_DATASETS_PATH
        else:
            base_dir = Path(root)

        # Target Path
        dataset_dir = base_dir / repo_id

        print(f"[START_SESSION] Dataset dir: {dataset_dir}")

        try:
            if not self.robot:
                 raise Exception("Robot not connected")

            # 1. Define Features
            if not hasattr(self.robot, "observation_features") or not hasattr(self.robot, "action_features"):
                 raise RuntimeError("Robot does not have feature definitions ready (observation_features/action_features).")

            # Use LeRobot Helpers to construct correct feature dicts
            from lerobot.datasets.utils import combine_feature_dicts, hw_to_dataset_features

            # Filter observation features based on selections
            filtered_obs_features = self._filter_observation_features(
                self.robot.observation_features,
                selected_cameras,
                selected_arms
            )

            # Filter action features based on arm selections
            filtered_action_features = self._filter_action_features(
                self.robot.action_features,
                selected_arms
            )

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
                self.dataset = LeRobotDataset(
                    repo_id=repo_id,
                    root=dataset_dir,
                    local_files_only=True,
                )
            else:
                logger.info("Creating new Dataset...")
                self.dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=fps,
                    root=dataset_dir,
                    robot_type=self.robot.robot_type,
                    features=features,
                    use_videos=True,
                )
            
            self.dataset.meta.metadata_buffer_size = 1
            print(f"[START_SESSION] Dataset created/loaded successfully")

            # CRITICAL: Set episode_count from actual dataset state
            # For new dataset: total_episodes = 0
            # For existing dataset: total_episodes = actual count
            self.episode_count = self.dataset.meta.total_episodes
            print(f"[START_SESSION] Episode count set to {self.episode_count} (from dataset)")

            # 3. Start Video Encoding
            if not self.video_manager:
                self.video_manager = VideoEncodingManager(self.dataset)
                self.video_manager.__enter__()
            print("[START_SESSION] Video manager started")

            # 4. Start Image Writer (Threads)
            self.dataset.start_image_writer(num_processes=0, num_threads=4)
            print("[START_SESSION] Image writer started")

            self.dataset_config = {"repo_id": repo_id, "task": task}
            self.session_active = True

            # Update recording fps to match dataset
            self.recording_fps = fps
            self._recording_frame_counter = 0
            print(f"[START_SESSION] Recording at {self.recording_fps}fps (teleop={self.frequency}Hz)")

            # Start async frame writer thread
            self._start_frame_writer()

            # Start recording capture thread (separate from teleop loop for smooth control)
            self._start_recording_capture()

            print("[START_SESSION] SUCCESS! Session is now active")
            print("=" * 60)

        except Exception as e:
            import traceback
            print(f"[START_SESSION] ERROR: {e}")
            print(traceback.format_exc())
            self.session_active = False
            self.dataset = None
            raise

    def stop_recording_session(self):
        """Finalizes the LeRobotDataset."""
        print("=" * 60)
        print(f"[STOP_SESSION] Called!")
        print(f"  session_active: {self.session_active}")
        print(f"  dataset: {self.dataset is not None}")
        print(f"  episode_saving: {self._episode_saving}")
        print("=" * 60)

        if not self.session_active:
            print("[STOP_SESSION] Not active, returning")
            return

        # AUTO-SAVE: If an episode is actively recording, save it first
        if self.recording_active:
            print("[STOP_SESSION] Episode still recording - auto-saving before finalize...")
            try:
                self.stop_episode()
                print("[STOP_SESSION] Auto-save completed successfully")
            except Exception as e:
                print(f"[STOP_SESSION] WARNING: Auto-save failed: {e}")

        print("[STOP_SESSION] Stopping Recording Session...")
        self.session_active = False

        # Stop recording capture thread first (it produces frames)
        print("[STOP_SESSION] Stopping recording capture thread...")
        self._stop_recording_capture()

        # Stop async frame writer (drains remaining frames)
        print(f"[STOP_SESSION] Stopping frame writer (queue size: {len(self._frame_queue)})")
        self._stop_frame_writer()

        # CRITICAL: Wait for any pending episode save to complete
        # This prevents finalize() from closing writers while save_episode() is still writing
        print("[STOP_SESSION] Acquiring episode save lock (waiting for pending save)...")
        with self._episode_save_lock:
            print("[STOP_SESSION] Episode save lock acquired, safe to finalize")

            try:
                if self.dataset:
                    print("[STOP_SESSION] Finalizing Dataset...")

                    # Check if writer exists before finalize
                    has_writer = hasattr(self.dataset, 'writer') and self.dataset.writer is not None
                    has_meta_writer = (hasattr(self.dataset, 'meta') and
                                       hasattr(self.dataset.meta, 'writer') and
                                       self.dataset.meta.writer is not None)
                    print(f"[STOP_SESSION] Before finalize - data writer: {has_writer}, meta writer: {has_meta_writer}")

                    # Call finalize to close parquet writers
                    self.dataset.finalize()

                    # Verify writers are closed
                    has_writer_after = hasattr(self.dataset, 'writer') and self.dataset.writer is not None
                    has_meta_writer_after = (hasattr(self.dataset, 'meta') and
                                             hasattr(self.dataset.meta, 'writer') and
                                             self.dataset.meta.writer is not None)
                    print(f"[STOP_SESSION] After finalize - data writer: {has_writer_after}, meta writer: {has_meta_writer_after}")

                    if has_writer_after or has_meta_writer_after:
                        print("[STOP_SESSION] WARNING: Writers not fully closed, forcing close...")
                        # Force close if still open
                        if hasattr(self.dataset, '_close_writer'):
                            self.dataset._close_writer()
                        if hasattr(self.dataset, 'meta') and hasattr(self.dataset.meta, '_close_writer'):
                            self.dataset.meta._close_writer()

                    print("[STOP_SESSION] Dataset finalized")

                    if self.video_manager:
                        self.video_manager.__exit__(None, None, None)
                        self.video_manager = None
                        print("[STOP_SESSION] Video manager closed")

                    self.dataset = None
                    print("[STOP_SESSION] SUCCESS! Session Stopped and Saved!")
            except Exception as e:
                import traceback
                print(f"[STOP_SESSION] ERROR: {e}")
                print(traceback.format_exc())
                # Ensure cleanup even on error
                self.dataset = None

    def sync_to_disk(self):
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
        if not self.dataset or not self.session_active:
            print("[SYNC_TO_DISK] Skipped (no dataset or session not active)")
            return

        print(f"[SYNC_TO_DISK] BEFORE: meta.total_episodes={self.dataset.meta.total_episodes}, episode_count={self.episode_count}")

        try:
            # 1. Flush and close metadata writer (this flushes metadata_buffer to disk)
            if hasattr(self.dataset, 'meta') and hasattr(self.dataset.meta, '_close_writer'):
                self.dataset.meta._close_writer()
                print("[SYNC_TO_DISK] Flushed metadata buffer and closed metadata writer")

            # 2. Close data parquet writer
            if hasattr(self.dataset, '_close_writer'):
                self.dataset._close_writer()
                print("[SYNC_TO_DISK] Closed data writer")

        except Exception as e:
            import traceback
            print(f"[SYNC_TO_DISK] Error: {e}")
            print(traceback.format_exc())

    def refresh_metadata_from_disk(self):
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
        if not self.dataset or not self.session_active:
            print("[REFRESH] Skipped (no dataset or session not active)")
            return

        import json

        info_path = self.dataset.meta.root / "meta" / "info.json"
        print(f"[REFRESH] Reading from: {info_path}")

        if info_path.exists():
            try:
                with open(info_path, "r") as f:
                    disk_info = json.load(f)

                old_memory_count = self.dataset.meta.info.get("total_episodes", 0)
                disk_count = disk_info.get("total_episodes", 0)

                print(f"[REFRESH] Disk: {disk_count}, Memory: {old_memory_count}")

                # 1. Update info dict
                self.dataset.meta.info["total_episodes"] = disk_count
                self.dataset.meta.info["total_frames"] = disk_info.get("total_frames", 0)

                # Verify the update worked
                verify_count = self.dataset.meta.total_episodes
                print(f"[REFRESH] After update: meta.total_episodes = {verify_count}")
                if verify_count != disk_count:
                    print(f"[REFRESH] ERROR: Update failed! Expected {disk_count}, got {verify_count}")

                # 2. Reset latest_episode to force fresh index calculation
                if hasattr(self.dataset, 'meta'):
                    self.dataset.meta.latest_episode = None
                    # Clear metadata buffer (should be empty after sync_to_disk, but ensure it)
                    if hasattr(self.dataset.meta, 'metadata_buffer'):
                        self.dataset.meta.metadata_buffer = []
                if hasattr(self.dataset, 'latest_episode'):
                    self.dataset.latest_episode = None

                # 3. Reset current file tracking
                if hasattr(self.dataset, '_current_file_start_frame'):
                    self.dataset._current_file_start_frame = None

                # 4. Reset episodes to None - DON'T reload from parquet
                # LeRobot expects episodes to be a specific internal structure (not a raw DataFrame)
                # Setting to None forces LeRobot to start fresh when latest_episode is also None
                self.dataset.meta.episodes = None

                # 5. CRITICAL: Clear episode_buffer to force fresh creation on next start_episode()
                # Without this, stale buffer with old episode_index causes validation failure
                if hasattr(self.dataset, 'episode_buffer') and self.dataset.episode_buffer is not None:
                    self.dataset.episode_buffer = None
                    print("[REFRESH] Cleared stale episode_buffer")

                # 6. Close and reset data writer to prevent stale frame counting
                if hasattr(self.dataset, '_close_writer'):
                    try:
                        self.dataset._close_writer()
                        print("[REFRESH] Closed data writer")
                    except:
                        pass
                if hasattr(self.dataset, 'writer'):
                    self.dataset.writer = None

                # 7. Sync local episode counter
                self.episode_count = disk_count

                print(f"[REFRESH] Complete: episode_count={self.episode_count}, meta.total_episodes={self.dataset.meta.total_episodes}")

            except Exception as e:
                import traceback
                print(f"[REFRESH] Error: {e}")
                print(traceback.format_exc())
        else:
            print(f"[REFRESH] ERROR: info.json not found at {info_path}")

    def start_episode(self):
        """Starts recording a new episode."""
        print("=" * 60)
        print("[START_EPISODE] Called!")
        print(f"  session_active: {self.session_active}")
        print(f"  recording_active: {self.recording_active}")
        print(f"  _episode_saving: {self._episode_saving}")
        print(f"  dataset: {self.dataset is not None}")
        print("=" * 60)

        if not self.session_active:
            print("[START_EPISODE] ERROR: No active session!")
            raise Exception("No active recording session")

        if self.recording_active:
            print("[START_EPISODE] Already recording, skipping")
            return

        # Wait for any ongoing episode save to complete before starting new episode
        if self._episode_saving:
            print("[START_EPISODE] Waiting for previous episode to finish saving...")
            wait_start = time.time()
            max_wait = 10.0  # Maximum 10 seconds wait
            while self._episode_saving and (time.time() - wait_start) < max_wait:
                time.sleep(0.1)
            if self._episode_saving:
                raise Exception("Previous episode save timed out. Please try again.")
            print(f"[START_EPISODE] Previous save completed after {time.time() - wait_start:.1f}s")

        # Warn if teleop isn't running - recording needs teleop for actions
        if not self.is_running:
            print("[START_EPISODE] WARNING: Teleop is NOT running!")
            print("[START_EPISODE] Recording requires teleop to be active for action/state data.")
            print("[START_EPISODE] Will use robot state fallback if available.")

        print("[START_EPISODE] Starting Episode Recording...")

        if self.dataset:
            # Log current state BEFORE buffer creation (critical for debugging)
            meta_total = self.dataset.meta.total_episodes
            print(f"[START_EPISODE] BEFORE buffer: meta.total_episodes={meta_total}, episode_count={self.episode_count}")

            # Check for count mismatch and warn
            if self.episode_count != meta_total:
                print(f"[START_EPISODE] WARNING: Count mismatch detected!")
                print(f"[START_EPISODE]   episode_count={self.episode_count} != meta.total_episodes={meta_total}")
                # Don't auto-sync here - the mismatch indicates a bug we need to find

            # Initialize episode buffer - handle case where buffer is None on first use
            try:
                if self.dataset.episode_buffer is None:
                    print("[START_EPISODE] Creating new episode buffer (first episode)")
                    self.dataset.episode_buffer = self.dataset.create_episode_buffer()
                else:
                    print("[START_EPISODE] Clearing existing episode buffer")
                    self.dataset.clear_episode_buffer()
            except Exception as e:
                print(f"[START_EPISODE] Error with buffer, recreating: {e}")
                self.dataset.episode_buffer = self.dataset.create_episode_buffer()

            # Log the episode index that will be used
            buffer_ep_idx = self.dataset.episode_buffer.get("episode_index", "N/A")
            print(f"[START_EPISODE] Episode buffer ready, episode_index={buffer_ep_idx}")
            print(f"[START_EPISODE] Dataset features: {list(self.dataset.features.keys())[:5]}...")

        # Reset frame counter for new episode
        self._recording_frame_counter = 0

        self.recording_active = True
        print("[START_EPISODE] recording_active = True, ready to capture frames!")

    def stop_episode(self):
        """Stops current episode and saves it."""
        print("=" * 60)
        print("[STOP_EPISODE] Called!")
        print(f"  recording_active: {self.recording_active}")
        print(f"  session_active: {self.session_active}")
        print(f"  dataset: {self.dataset is not None}")
        if self.dataset:
            print(f"  dataset type: {type(self.dataset)}")
        print("=" * 60)

        if not self.recording_active:
            print("[STOP_EPISODE] WARNING: recording_active is False, returning")
            return

        # Acquire lock to prevent race with stop_session
        # This ensures save_episode() completes before finalize() can be called
        print("[STOP_EPISODE] Acquiring episode save lock...")
        with self._episode_save_lock:
            self._episode_saving = True
            print("[STOP_EPISODE] Lock acquired, proceeding with save")

            # Capture dataset reference BEFORE changing state
            # This prevents race conditions where dataset could be set to None
            current_dataset = self.dataset

            print("[STOP_EPISODE] Stopping Episode Recording...")
            self.recording_active = False

            # Wait for frame queue to drain before saving episode
            print(f"[STOP_EPISODE] Waiting for frame queue to drain ({len(self._frame_queue)} frames)...")
            drain_timeout = 5.0  # seconds
            drain_start = time.time()
            while len(self._frame_queue) > 0 and (time.time() - drain_start) < drain_timeout:
                time.sleep(0.05)
            if len(self._frame_queue) > 0:
                print(f"[STOP_EPISODE] WARNING: Queue not fully drained ({len(self._frame_queue)} remaining)")
            else:
                print(f"[STOP_EPISODE] Queue drained successfully")

            # Reset first frame logging flag for next episode
            if hasattr(self, '_first_frame_logged'):
                delattr(self, '_first_frame_logged')
            if hasattr(self, '_last_rec_error'):
                delattr(self, '_last_rec_error')

            if current_dataset is not None:
                try:
                    # Check episode buffer before saving
                    if hasattr(current_dataset, 'episode_buffer') and current_dataset.episode_buffer:
                        buffer_size = current_dataset.episode_buffer.get('size', 0)
                        print(f"[STOP_EPISODE] Episode buffer has {buffer_size} frames (captured {self._recording_frame_counter})")

                        if buffer_size == 0:
                            print("[STOP_EPISODE] WARNING: Buffer size is 0, no frames were recorded!")
                            print("[STOP_EPISODE] This means observations weren't captured during recording.")
                            self._episode_saving = False
                            return
                    else:
                        print("[STOP_EPISODE] WARNING: Episode buffer is empty or missing!")
                        print(f"  has episode_buffer attr: {hasattr(current_dataset, 'episode_buffer')}")
                        if hasattr(current_dataset, 'episode_buffer'):
                            print(f"  episode_buffer value: {current_dataset.episode_buffer}")
                        self._episode_saving = False
                        return

                    # Log state BEFORE save
                    print(f"[STOP_EPISODE] BEFORE save: meta.total_episodes={current_dataset.meta.total_episodes}, episode_count={self.episode_count}")

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
                    self.episode_count += 1
                    print(f"[STOP_EPISODE] SUCCESS! Episode {self.episode_count} Saved! ({self._recording_frame_counter} frames)")
                except Exception as e:
                    import traceback
                    print(f"[STOP_EPISODE] ERROR saving episode: {e}")
                    print(traceback.format_exc())
            else:
                print("[STOP_EPISODE] WARNING: No dataset available (current_dataset is None)!")

            self._episode_saving = False
            print("[STOP_EPISODE] Episode save lock released")
             
    def delete_last_episode(self):
         """Deletes the last recorded episode (if possible/implemented)."""
         logger.warning("Delete Last Episode not fully supported yet.")

    def _manual_finalize_dataset(self, repo_id):
        """Emergency fix to generate episode metadata if LeRobotDataset fails."""
        if not repo_id: return
        pass # Not implemented fully as per original file
