import logging
from typing import Dict, List, Optional, Any
from lerobot.robots.bi_umbra_follower.bi_umbra_follower import BiUmbraFollower
from lerobot.robots.umbra_follower.umbra_follower import UmbraFollowerRobot
from lerobot.motors.feetech import OperatingMode

logger = logging.getLogger(__name__)

class CalibrationService:
    def __init__(self, robot, leader=None, robot_lock=None):
        self.robot = robot
        self.leader = leader
        self.robot_lock = robot_lock
        self.active_arm = None
        
        # Range Discovery State
        self.is_discovering = False
        self.session_ranges = {} # {motor_name: {"min": val, "max": val}}

    def get_arms(self) -> List[Dict[str, Any]]:
        arms = []
        logger.info(f"get_arms: self.robot type is {type(self.robot)}")
        logger.info(f"get_arms: BiUmbraFollower class is {BiUmbraFollower}")

        
        def add_arm_groups(prefix, robot_inst, include_leader=True):
            # Check Leader Status
            if include_leader:
                leader_motors = ["link1", "link2"]
                leader_calibrated = all(m in robot_inst.calibration for m in leader_motors)
                arms.append({
                    "id": f"{prefix}_leader",
                    "name": f"{prefix.capitalize()} Leader",
                    "calibrated": leader_calibrated,
                    "type": "leader"
                })
            
            # Check Follower Status
            # Include link1/link2 as they are part of the arm structure even if driven by dual logic
            follower_motors = ["base", "link1", "link1_follower", "link2", "link2_follower", "link3", "link4", "link5", "gripper"]
            follower_calibrated = all(m in robot_inst.calibration for m in follower_motors)

            arms.append({
                "id": f"{prefix}_follower",
                "name": f"{prefix.capitalize()} Follower",
                "calibrated": follower_calibrated,
                "type": "follower"
            })

        if isinstance(self.robot, BiUmbraFollower):
            add_arm_groups("left", self.robot.left_arm, include_leader=False)
            add_arm_groups("right", self.robot.right_arm, include_leader=False)
        elif isinstance(self.robot, UmbraFollowerRobot):
            add_arm_groups("default", self.robot, include_leader=True)
            
        # Add Leader Arms if present
        if self.leader:
            from lerobot.teleoperators.bi_umbra_leader.bi_umbra_leader import BiUmbraLeader
            if isinstance(self.leader, BiUmbraLeader):
                # Check Left Leader
                l_leader = self.leader.left_arm
                l_calibrated = all(m in l_leader.calibration for m in ["link1", "link2"])
                arms.append({
                    "id": "left_leader",
                    "name": "Left Leader",
                    "calibrated": l_calibrated,
                    "type": "leader"
                })
                
                # Check Right Leader
                r_leader = self.leader.right_arm
                r_calibrated = all(m in r_leader.calibration for m in ["link1", "link2"])
                arms.append({
                    "id": "right_leader",
                    "name": "Right Leader",
                    "calibrated": r_calibrated,
                    "type": "leader"
                })
            
        return arms

    def _get_arm_context(self, arm_id: str):
        """Returns (robot_instance, motor_list_filter)"""
        parts = arm_id.split("_")
        
        if len(parts) < 2:
            return None, []
            
        side = parts[0] # left, right, default
        group = parts[1] # leader, follower
        
        robot_inst = None
        
        # Handle Followers
        if group == "follower":
            if isinstance(self.robot, BiUmbraFollower):
                if side == "left":
                    robot_inst = self.robot.left_arm
                elif side == "right":
                    robot_inst = self.robot.right_arm
            elif isinstance(self.robot, UmbraFollowerRobot):
                robot_inst = self.robot

        # Handle Leaders
        elif group == "leader":
            if self.leader:
                if side == "left":
                    robot_inst = self.leader.left_arm
                elif side == "right":
                    robot_inst = self.leader.right_arm
            elif isinstance(self.robot, UmbraFollowerRobot):
                 robot_inst = self.robot
            
        if not robot_inst:
            logger.warning(f"_get_arm_context: No robot instance found for {arm_id} (Side={side}, Group={group})")
            return None, []

        target_motors = []
        if group == "leader":
            target_motors = ["link1", "link2"]
        elif group == "follower":
            target_motors = ["base", "link1_follower", "link2_follower", "link3", "link4", "link5", "gripper"]
            
        return robot_inst, target_motors

    def _ensure_active_profiles_init(self):
        if not hasattr(self, "active_profiles"):
            self.active_profiles = {}

    def _ensure_calibration_initialized(self, arm):
        """Ensures that arm.calibration is populated with default entries for all motors."""
        if not arm.calibration:
            from lerobot.motors import MotorCalibration
            logger.info(f"Initializing empty calibration for {arm}")
            for name, motor in arm.bus.motors.items():
                arm.calibration[name] = MotorCalibration(
                    id=motor.id,
                    drive_mode=0,
                    homing_offset=0,
                    range_min=0,
                    range_max=4096 # Default open range
                )

    def start_discovery(self, arm_id: str):
        if not arm_id:
             return
        
        arm, target_motors = self._get_arm_context(arm_id)
        if not arm:
             return

        logger.info(f"Starting Range Discovery for {arm_id}")
        self._ensure_calibration_initialized(arm)
        
        self.is_discovering = True
        self.session_ranges = {}
        
        # Enable Multi-turn for discovery (Limits=0)
        # This prevents the motor from stopping at 0/4095
        if self.robot_lock:
             with self.robot_lock:
                 for motor in target_motors:
                     if motor in arm.bus.motors:
                         # Disable Torque to write limits? Ideally user moves it manually with torque off.
                         # But we need to write limits.
                         try:
                             arm.bus.write("Min_Position_Limit", motor, 0)
                             arm.bus.write("Max_Position_Limit", motor, 0)
                         except Exception as e:
                             logger.warning(f"Failed to clear limits for {motor}: {e}")
        else:
             for motor in target_motors:
                  if motor in arm.bus.motors:
                       try:
                           arm.bus.write("Min_Position_Limit", motor, 0)
                           arm.bus.write("Max_Position_Limit", motor, 0)
                       except Exception as e:
                           logger.warning(f"Failed to clear limits for {motor}: {e}")

    def stop_discovery(self, arm_id: str):
        logger.info(f"Stopping Range Discovery for {arm_id}")
        self.is_discovering = False
        
        # Automatically update the robot's calibration with the discovered ranges
        arm, _ = self._get_arm_context(arm_id)
        
        # Ensure we have a calibration object to update
        self._ensure_calibration_initialized(arm)
        
        if arm and self.session_ranges:
            for motor, ranges in self.session_ranges.items():
                if motor in arm.calibration:
                    arm.calibration[motor].range_min = ranges["min"]
                    arm.calibration[motor].range_max = ranges["max"]
                else:
                    logger.warning(f"stop_discovery: Motor {motor} not found in arm.calibration")
            
            self.save_calibration(arm_id)


    def get_calibration_state(self, arm_id: str) -> List[Dict[str, Any]]:
        arm, target_motors = self._get_arm_context(arm_id)
        if not arm:
            return []

        # Read positions
        if self.robot_lock:
            with self.robot_lock:
                present_positions = arm.bus.sync_read("Present_Position", normalize=False)
        else:
            present_positions = arm.bus.sync_read("Present_Position", normalize=False)
        
        state = []
        for motor_name in target_motors:
            if motor_name not in arm.bus.motors:
                continue
                
            cal = arm.calibration.get(motor_name)
            pos = present_positions.get(motor_name, 0)
            
            range_min = cal.range_min if cal else 0
            range_max = cal.range_max if cal else 4096 
            
            # --- Range Discovery Logic ---
            visited_min = range_min
            visited_max = range_max
            
            if self.is_discovering:
                if motor_name not in self.session_ranges:
                    self.session_ranges[motor_name] = {"min": pos, "max": pos}
                
                if pos < self.session_ranges[motor_name]["min"]:
                    self.session_ranges[motor_name]["min"] = pos
                if pos > self.session_ranges[motor_name]["max"]:
                    self.session_ranges[motor_name]["max"] = pos
                
                visited_min = self.session_ranges[motor_name]["min"]
                visited_max = self.session_ranges[motor_name]["max"]
            # -----------------------------

            state.append({
                "name": motor_name,
                "min": range_min,
                "max": range_max,
                "visited_min": visited_min, 
                "visited_max": visited_max,
                "pos": pos,
                "id": arm.bus.motors[motor_name].id
            })
        
        return state

    def set_calibration_limit(self, arm_id: str, motor_name: str, limit_type: str, value: int):
        arm, _ = self._get_arm_context(arm_id)
        if not arm:
            return

        if motor_name not in arm.calibration:
            return

        if limit_type == "min":
            arm.calibration[motor_name].range_min = int(value)
        elif limit_type == "max":
            arm.calibration[motor_name].range_max = int(value)

    def list_calibration_files(self, arm_id: str) -> List[Dict[str, Any]]:
        arm, _ = self._get_arm_context(arm_id)
        if not arm:
            return []
        
        self._ensure_active_profiles_init()
        
        # Use project root directory for persistent storage
        import pathlib
        base_dir = pathlib.Path(f"calibration_profiles/{arm_id}")
            
        if not base_dir.exists():
            base_dir.mkdir(parents=True, exist_ok=True)
            
        files = []
        for f in base_dir.glob("*.json"):
            import datetime
            stats = f.stat()
            dt = datetime.datetime.fromtimestamp(stats.st_mtime)
            
            is_active = False
            if arm_id in self.active_profiles:
                if self.active_profiles[arm_id] == f.stem:
                    is_active = True
            
            files.append({
                "name": f.stem,
                "created": dt.strftime("%Y-%m-%d %H:%M"),
                "path": str(f.absolute()),
                "active": is_active
            })
            
        # Sort by newest first
        files.sort(key=lambda x: x["created"], reverse=True)
        return files

    def load_calibration_file(self, arm_id: str, filename: str):
        arm, _ = self._get_arm_context(arm_id)
        if not arm:
            return False
            
        self._ensure_active_profiles_init()
            
        import pathlib
        base_dir = pathlib.Path(f"calibration_profiles/{arm_id}")
        fpath = base_dir / f"{filename}.json"
        
        if not fpath.exists():
            logger.error(f"Calibration file not found: {fpath}")
            return False
            
        logger.info(f"Loading calibration for {arm_id} from {fpath}")
        try:
            # Use the robot's internal loader if possible, or manual load
            arm._load_calibration(fpath)
            # Apply to motors
            if hasattr(arm.bus, "write_calibration"):
                arm.bus.write_calibration(arm.calibration)
            
            # Update Active Profile state
            self.active_profiles[arm_id] = filename
            
            logger.info(f"Successfully loaded calibration for {arm_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False

    def delete_calibration_file(self, arm_id: str, filename: str):
        import pathlib
        base_dir = pathlib.Path(f"calibration_profiles/{arm_id}")
        fpath = base_dir / f"{filename}.json"
        if fpath.exists():
            fpath.unlink()
            return True
        return False

    def save_calibration(self, arm_id: str, name: str = None):
        arm, _ = self._get_arm_context(arm_id)
        if not arm:
            return

        # First, ensure we have the latest from the bus
        if self.robot_lock:
            with self.robot_lock:
                arm.bus.write_calibration(arm.calibration)
        else:
            arm.bus.write_calibration(arm.calibration)

        # Usage of internal save for default file
        if hasattr(arm, "_save_calibration"):
            arm._save_calibration()
        else:
            logger.warning(f"Save calibration not supported for {arm}")
            
        # If a name is provided, ALSO save to the profile directory
        if name:
            self._ensure_active_profiles_init()
            import pathlib
            import json
            import dataclasses
            
            # Create profile dir
            base_dir = pathlib.Path(f"calibration_profiles/{arm_id}")
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Sanitize name
            safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip()
            if not safe_name:
                safe_name = "unnamed"
            
            fpath = base_dir / f"{safe_name}.json"
            
            # Use draccus or manual json dump? 
            # Sub-robots might not expose draccus easily here.
            # Reuse _save_calibration logic by temporarily passing path?
            if hasattr(arm, "_save_calibration"):
                arm._save_calibration(fpath)
                logger.info(f"Saved named calibration to {fpath}")
                
            # Update Active Profile state
            self.active_profiles[arm_id] = safe_name



    def disable_torque(self, arm_id: str):
        arm, _ = self._get_arm_context(arm_id)
        if arm:
            logger.info(f"Disabling torque for COMPONENT {arm_id} (All motors on bus)")
            RETRY_COUNT = 5
            
            if self.robot_lock:
                with self.robot_lock:
                    try:
                        arm.bus.disable_torque(None, num_retry=RETRY_COUNT)
                    except TypeError:
                        arm.bus.disable_torque(None)
                        
                    for motor in arm.bus.motors:
                        try:
                            # Also reset Lock/Torque just in case
                            arm.bus.write("Lock", motor, 0, num_retry=RETRY_COUNT)
                            arm.bus.write("Torque_Enable", motor, 0, num_retry=RETRY_COUNT)
                            arm.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value, num_retry=RETRY_COUNT)
                            # CLEAR Min/Max limits to allow full rotation during calibration?
                            # Feetech doc: Min/Max Position Limit.
                            # arm.bus.write("Min_Position_Limit", motor, 0)
                            # arm.bus.write("Max_Position_Limit", motor, 0) 
                        except Exception as e:
                            logger.warning(f"Failed to set Operating_Mode/Limits for {motor}: {e}")
            else:
                 try:
                    arm.bus.disable_torque(None, num_retry=RETRY_COUNT)
                 except TypeError:
                    arm.bus.disable_torque(None)
                 for motor in arm.bus.motors:
                    try:
                        arm.bus.write("Lock", motor, 0, num_retry=RETRY_COUNT)
                        arm.bus.write("Torque_Enable", motor, 0, num_retry=RETRY_COUNT)
                        arm.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value, num_retry=RETRY_COUNT)
                    except Exception as e:
                        logger.warning(f"Failed to set configs for {motor}: {e}")

    def enable_torque(self, arm_id: str):
        arm, _ = self._get_arm_context(arm_id)
        if arm:
            logger.info(f"Enabling torque for {arm_id} (All motors on bus)")
            RETRY_COUNT = 5
            if self.robot_lock:
                with self.robot_lock:
                    try:
                         arm.bus.enable_torque(None, num_retry=RETRY_COUNT)
                    except TypeError:
                         arm.bus.enable_torque(None)
            else:
                try:
                     arm.bus.enable_torque(None, num_retry=RETRY_COUNT)
                except TypeError:
                     arm.bus.enable_torque(None)

    def move_motor(self, arm_id: str, motor_name: str, value: int):
        arm, _ = self._get_arm_context(arm_id)
        if arm:
            if self.robot_lock:
                with self.robot_lock:
                    arm.bus.write("Goal_Position", motor_name, value, normalize=False)
            else:
                arm.bus.write("Goal_Position", motor_name, value, normalize=False)

    def perform_homing(self, arm_id: str):
        # Use LeRobot's Standard Homing Procedure
        arm, target_motors = self._get_arm_context(arm_id)
        if not arm:
            logger.error(f"perform_homing: Invalid arm_id {arm_id} or arm not found.")
            return False

        logger.info(f"Performing Homing for {arm_id} using Standard LeRobot logic")
        
        # Ensure we have a calibration object to update offsets into
        self._ensure_calibration_initialized(arm)
        
        try:
            if self.robot_lock:
                with self.robot_lock:
                     # Filter motors to only target relevant ones (avoiding gripper if needed, or all)
                     # Usually we home all motors on the bus except Gripper? 
                     # The UI calls this per 'arm' (leader or follower).
                     # target_motors contains the list.
                     motors_to_home = [m for m in target_motors if m in arm.bus.motors]
                     
                     # Force Torque Disable first
                     arm.bus.disable_torque(motors_to_home)
                     
                     # Run standard procedure
                     arm.bus.set_half_turn_homings(motors_to_home)
                     
                     # Limits are reset to 0/Max by reset_calibration inside set_half_turn_homings
                     # We might want to keep them at 0 (Multi-turn) until Range Discovery finishes.
                     
                     return True
            else:
                 motors_to_home = [m for m in target_motors if m in arm.bus.motors]
                 arm.bus.disable_torque(motors_to_home)
                 arm.bus.set_half_turn_homings(motors_to_home)
                 return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.exception(f"Homing failed with exception: {e}")
            return False
        
