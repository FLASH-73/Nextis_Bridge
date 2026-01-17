import time
import threading
import logging
from collections import deque
import numpy as np

from app.core.safety_layer import SafetyLayer
from app.core.leader_assist import LeaderAssistService
from lerobot.motors.feetech.feetech import OperatingMode

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

                # 2. Send to Follower (Modified for Recording Support Check)
                if leader_action and self.robot:
                    try:
                        # If recording, we MIGHT need to read obs?
                        # The user provided loop is BLIND. 
                        # But recording requires observation.
                        if self.recording_active:
                            # We must read observation if recording
                             if self.robot_lock:
                                  with self.robot_lock:
                                      follower_obs = self.robot.get_observation(include_images=True)
                                      self.robot.send_action(leader_action)
                             else:
                                  follower_obs = self.robot.get_observation(include_images=True)
                                  self.robot.send_action(leader_action)
                        else:
                             # Blind Teleop (Reference)
                             if self.robot_lock:
                                  with self.robot_lock:
                                      self.robot.send_action(leader_action)
                             else:
                                  self.robot.send_action(leader_action)
                                  
                    except Exception as e:
                         if loop_count % 60 == 0:
                             logger.error(f"Send Action Failed: {e}")
                elif self.robot and loop_count % 60 == 0:
                     logger.warning("No Leader Action generated (Mapping issue or Empty Obs)")
                
                # Debug freeze
                if loop_count % 60 == 0:
                    print(f"Teleop Heartbeat: {loop_count} (Active: {self.is_running})")
                
                # 3. Recording (In-Process)
                if self.is_running and self.recording_active and self.dataset and follower_obs:
                     try:
                         # Build Frames using LeRobot Helper
                         obs_frame = build_dataset_frame(self.dataset.features, follower_obs, prefix=OBS_STR)
                         action_frame = build_dataset_frame(self.dataset.features, leader_action, prefix=ACTION)
                         
                         # Combine
                         frame = {**obs_frame, **action_frame, "task": self.dataset_config.get("task", "")}
                         
                         # Add to Dataset (Non-Blocking)
                         self.dataset.add_frame(frame)
                     except Exception as e:
                          if loop_count % 60 == 0:
                               logger.error(f"Recording Capture Error: {e}")

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
        if self.session_active:
             raise Exception("Session already active")
             
        # Set default root to app datasets directory
        from pathlib import Path
        if root is None:
            base_dir = Path("/home/roberto/nextis_app/datasets")
        else:
            base_dir = Path(root)
            
        # Target Path
        dataset_dir = base_dir / repo_id
        
        logger.info(f"Starting Recording Session: {repo_id} at {dataset_dir}")

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

            # 3. Start Video Encoding
            if not self.video_manager:
                 self.video_manager = VideoEncodingManager(self.dataset)
                 self.video_manager.__enter__()
            
            # 4. Start Image Writer (Threads)
            self.dataset.start_image_writer(num_processes=0, num_threads=4)
            
            self.dataset_config = {"repo_id": repo_id, "task": task}
            self.session_active = True
            logger.info("Session Started Successfully.")
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to start recording session: {e}\n{traceback.format_exc()}")
            self.session_active = False
            self.dataset = None
            raise

    def stop_recording_session(self):
        """Finalizes the LeRobotDataset."""
        if not self.session_active:
            return
            
        logger.info("Stopping Recording Session...")
        self.session_active = False
        
        try:
            if self.dataset:
                logger.info("Finalizing Dataset...")
                self.dataset.finalize()
                
                if self.video_manager:
                    self.video_manager.__exit__(None, None, None)
                    self.video_manager = None
                
                self.dataset = None
                logger.info("Session Stopped and Saved.")
        except Exception as e:
            logger.error(f"Error stopping session: {e}")

    def start_episode(self):
        """Starts recording a new episode."""
        if not self.session_active:
             raise Exception("No active recording session")
        
        if self.recording_active:
             return # Already recording
        
        logger.info("Starting Episode Recording...")
        if self.dataset:
             # Ensure buffer is clear
             self.dataset.clear_episode_buffer()
             
        self.recording_active = True

    def stop_episode(self):
        """Stops current episode and saves it."""
        if not self.recording_active:
             return
             
        logger.info("Stopping Episode Recording. Saving...")
        self.recording_active = False
        
        if self.dataset:
             self.dataset.save_episode(task=self.dataset_config.get("task", ""))
             self.episode_count += 1
             logger.info(f"Episode {self.episode_count} Saved.")
             
    def delete_last_episode(self):
         """Deletes the last recorded episode (if possible/implemented)."""
         logger.warning("Delete Last Episode not fully supported yet.")

    def _manual_finalize_dataset(self, repo_id):
        """Emergency fix to generate episode metadata if LeRobotDataset fails."""
        if not repo_id: return
        pass # Not implemented fully as per original file
