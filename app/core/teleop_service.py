import time
import threading
import logging
from collections import deque
from pathlib import Path
import numpy as np

from app.core.safety_layer import SafetyLayer
from app.core.leader_assist import LeaderAssistService
from lerobot.motors.feetech.feetech import OperatingMode

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
    def __init__(self, robot, leader, robot_lock, leader_assists=None):
        self.robot = robot
        self.leader = leader
        self.robot_lock = robot_lock
        
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
        
        # Optimization: Pre-computed mappings
        self.joint_mapping = {} # {leader_key: follower_key}
        self.assist_groups = {} # {arm_key: [joint_name, ...]}

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
                    with self._action_lock:
                        if self._latest_leader_action:
                            action = self._latest_leader_action.copy()
                            # Use same positions for observation.state (follower = action target)
                            for key, val in action.items():
                                obs[key] = val

                    # SOURCE 2: Capture camera images with async_read (FAST - ZOH pattern)
                    if hasattr(self.robot, 'cameras') and self.robot.cameras:
                        for cam_key, cam in self.robot.cameras.items():
                            try:
                                # async_read(blocking=False) returns last cached frame instantly
                                if hasattr(cam, 'async_read'):
                                    frame = cam.async_read(blocking=False)
                                    if frame is not None:
                                        obs[cam_key] = frame
                                        if self._recording_frame_counter == 0:
                                            print(f"[REC CAPTURE] Camera {cam_key}: shape={frame.shape}")
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
        
        if not self.leader:
             return

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
        
        logger.info(f"Teleop Mapping Optimized: {len(self.joint_mapping)} joints mapped.")

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
        if self.is_running:
            return
            
        if not self.robot:
             raise Exception("Robot not connected")
        
        # Store active arms (if provided, else None means All)
        self.active_arms = active_arms
        logger.info(f"Teleoperation Request: Active Arms = {self.active_arms}")
        
        # Validate selection if provided
        if self.active_arms is not None:
             leaders = [a for a in self.active_arms if "leader" in a]
             followers = [a for a in self.active_arms if "follower" in a]
             if not force and (not leaders or not followers):
                  logger.error("Selection Validation Failed")
                  raise Exception("Invalid Selection: Must select at least one Leader and one Follower.")
        
        if not self.check_calibration():
             msg = "System not fully calibrated."
             if not force:
                 logger.warning(f"IGNORING CALIBRATION CHECK: {msg}")
             else:
                 logger.warning(f"FORCE START: {msg}")
        
        # Optimize Mappings
        self._precompute_mappings()
        
        # Reload Inversions (Ensure latest config from disk is applied)
        if hasattr(self.robot, "reload_inversions"):
            try:
                 self.robot.reload_inversions()
            except Exception as e:
                 logger.warning(f"Failed to reload inversions on start: {e}")

        # Enable Torque for Follower Arms
        self._enable_torque_for_active_arms()

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
                 
        self.is_running = True
        
        # Start Control Loop Thread
        self.thread = threading.Thread(target=self._teleop_loop, daemon=True)
        self.thread.start()

    def _enable_torque_for_active_arms(self):
        """Helper to enable torque on follower arms involved in teleop."""
        if not self.robot: return
        
        try:
            logger.info("Enabling Torque for Teleoperation...")
            
            # Robust Enable: Try to enable everything found
            if hasattr(self.robot, "left_arm"):
                 self.robot.left_arm.bus.enable_torque()
            if hasattr(self.robot, "right_arm"):
                 self.robot.right_arm.bus.enable_torque()
            if hasattr(self.robot, "bus"):
                 self.robot.bus.enable_torque()
                 
        except Exception as e:
            logger.error(f"Failed to enable torque: {e}")

    def stop(self):
        if not self.is_running:
            return
        logger.info("Stopping teleoperation...")
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)

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
            
        logger.info("Teleoperation stopped.")
        
        if hasattr(self, '_debug_handler'):
             logger.removeHandler(self._debug_handler)
             self._debug_handler.close()

    def _teleop_loop(self):
        logger.info(f"Teleoperation Control Loop Running at {self.frequency}Hz (Optimization Enabled)")
        
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

                if self.leader:
                    obs = self.leader.get_action()
                    
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

                    # 1b. Map to Follower Action (Optimized)
                    # Use pre-computed mapping
                    for l_key, f_key in self.joint_mapping.items():
                        if l_key in obs:
                            # Direct assignment, no EMA
                            leader_action[f_key] = int(obs[l_key])

                # 2. Send Action IMMEDIATELY (Low Latency Control)
                # Key insight from LeRobot: send_action and get_observation are independent.
                # Motor writes use sync_write (no response wait), cameras have background threads.
                # NO LOCK NEEDED - LeRobot's architecture handles thread safety at bus level.
                if leader_action and self.robot:
                    try:
                        self.robot.send_action(leader_action)
                    except Exception as e:
                        if loop_count % 60 == 0:
                            logger.error(f"Send Action Failed: {e}")
                elif self.robot and loop_count % 60 == 0:
                    logger.warning("No Leader Action generated (Mapping issue or Empty Obs)")

                # 2b. Share action with recording thread
                # Recording happens in a separate thread (_recording_capture_loop) at 30fps
                # This keeps the main teleop loop clean and fast at 60Hz
                if leader_action:
                    with self._action_lock:
                        self._latest_leader_action = leader_action.copy()

                # Debug heartbeat
                if loop_count % 60 == 0:
                    print(f"Teleop Heartbeat: {loop_count} (Active: {self.is_running}, Recording: {self.recording_active})")

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
                
        except OSError as e:
            if e.errno == 5: 
                 logger.error(f"Hardware Disconnected: {e}")
        except Exception as e:
             logger.error(f"Teleop Loop Failed: {e}")
             import traceback
             traceback.print_exc()
        finally:
            self.stop() # Ensure Cleanup

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

    def start_recording_session(self, repo_id: str, task: str, fps: int = 30, root: str = None):
        """Initializes a new LeRobotDataset for recording."""
        print("=" * 60)
        print(f"[START_SESSION] Called with repo_id='{repo_id}', task='{task}'")
        print(f"  session_active: {self.session_active}")
        print(f"  robot: {self.robot is not None}")
        print("=" * 60)

        if self.session_active:
            print("[START_SESSION] ERROR: Session already active!")
            raise Exception("Session already active")

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
            
            features = combine_feature_dicts(
                hw_to_dataset_features(self.robot.observation_features, prefix=OBS_STR, use_video=True),
                hw_to_dataset_features(self.robot.action_features, prefix=ACTION, use_video=True)
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

    def start_episode(self):
        """Starts recording a new episode."""
        print("=" * 60)
        print("[START_EPISODE] Called!")
        print(f"  session_active: {self.session_active}")
        print(f"  recording_active: {self.recording_active}")
        print(f"  dataset: {self.dataset is not None}")
        print("=" * 60)

        if not self.session_active:
            print("[START_EPISODE] ERROR: No active session!")
            raise Exception("No active recording session")

        if self.recording_active:
            print("[START_EPISODE] Already recording, skipping")
            return

        # Warn if teleop isn't running - recording needs teleop for actions
        if not self.is_running:
            print("[START_EPISODE] WARNING: Teleop is NOT running!")
            print("[START_EPISODE] Recording requires teleop to be active for action/state data.")
            print("[START_EPISODE] Will use robot state fallback if available.")

        print("[START_EPISODE] Starting Episode Recording...")

        if self.dataset:
            self.dataset.clear_episode_buffer()
            print("[START_EPISODE] Episode buffer cleared")
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

                    print(f"[STOP_EPISODE] Calling save_episode()")
                    # Note: task is already included in each frame, no need to pass to save_episode
                    current_dataset.save_episode()
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
