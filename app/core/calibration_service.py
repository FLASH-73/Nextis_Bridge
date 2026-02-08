import logging
from typing import Dict, List, Optional, Any
from lerobot.robots.bi_umbra_follower.bi_umbra_follower import BiUmbraFollower
from lerobot.robots.umbra_follower.umbra_follower import UmbraFollowerRobot
from lerobot.motors.feetech import OperatingMode

# Try to import Damiao robot (may not be available on all systems)
try:
    from lerobot.robots.damiao_follower import DamiaoFollowerRobot
    DAMIAO_AVAILABLE = True
except ImportError:
    DamiaoFollowerRobot = None
    DAMIAO_AVAILABLE = False

logger = logging.getLogger(__name__)

class CalibrationService:
    # Dynamixel leader → Damiao follower joint name translation
    _DYNAMIXEL_TO_DAMIAO = {
        "joint_1": "base", "joint_2": "link1", "joint_3": "link2",
        "joint_4": "link3", "joint_5": "link4", "joint_6": "link5",
        "gripper": "gripper",
    }
    _DAMIAO_TO_DYNAMIXEL = {v: k for k, v in _DYNAMIXEL_TO_DAMIAO.items()}

    def __init__(self, robot, leader=None, robot_lock=None, arm_registry=None):
        self.robot = robot
        self.leader = leader
        self.robot_lock = robot_lock
        self.arm_registry = arm_registry  # For new arm management system
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
                leader_motors = ["base", "link1", "link2", "link3", "link4", "link5", "gripper"]
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

        # Add Damiao arm if connected
        if DAMIAO_AVAILABLE and DamiaoFollowerRobot is not None:
            if isinstance(self.robot, DamiaoFollowerRobot):
                # Damiao uses absolute encoders, always "calibrated"
                damiao_motors = ["base", "link1", "link2", "link3", "link4", "link5", "gripper"]
                damiao_calibrated = self.robot.is_calibrated
                arms.append({
                    "id": "damiao_follower",
                    "name": "Damiao Follower",
                    "calibrated": damiao_calibrated,
                    "type": "follower",
                    "motor_type": "damiao"  # Special flag for UI
                })

        # Check for Damiao robot passed separately (e.g., as damiao_robot attribute)
        if hasattr(self, 'damiao_robot') and self.damiao_robot is not None:
            damiao_calibrated = self.damiao_robot.is_calibrated if hasattr(self.damiao_robot, 'is_calibrated') else True
            arms.append({
                "id": "damiao_follower",
                "name": "Damiao Follower",
                "calibrated": damiao_calibrated,
                "type": "follower",
                "motor_type": "damiao"
            })

        # Add Leader Arms if present
        if self.leader:
            from lerobot.teleoperators.bi_umbra_leader.bi_umbra_leader import BiUmbraLeader
            if isinstance(self.leader, BiUmbraLeader):
                # Check Left Leader
                l_leader = self.leader.left_arm
                l_calibrated = all(m in l_leader.calibration for m in ["base", "link1", "link2", "link3", "link4", "link5", "gripper"])
                arms.append({
                    "id": "left_leader",
                    "name": "Left Leader",
                    "calibrated": l_calibrated,
                    "type": "leader"
                })

                # Check Right Leader
                r_leader = self.leader.right_arm
                r_calibrated = all(m in r_leader.calibration for m in ["base", "link1", "link2", "link3", "link4", "link5", "gripper"])
                arms.append({
                    "id": "right_leader",
                    "name": "Right Leader",
                    "calibrated": r_calibrated,
                    "type": "leader"
                })

        # Add arms from arm registry (covers Dynamixel leaders and other registered arms)
        if self.arm_registry:
            existing_ids = {a["id"] for a in arms}
            from app.core.arm_registry import ConnectionStatus
            for arm_id, arm_def in self.arm_registry.arms.items():
                if arm_id in existing_ids:
                    continue
                # Only include connected arms
                if self.arm_registry.arm_status.get(arm_id) != ConnectionStatus.CONNECTED:
                    continue
                instance = self.arm_registry.arm_instances.get(arm_id)
                calibrated = False
                if instance and hasattr(instance, 'is_calibrated'):
                    calibrated = instance.is_calibrated
                arms.append({
                    "id": arm_id,
                    "name": arm_def.name,
                    "calibrated": calibrated,
                    "type": arm_def.role.value,
                    "motor_type": arm_def.motor_type.value,
                })

        return arms

    def _get_arm_context(self, arm_id: str):
        """Returns (robot_instance, motor_list_filter)"""
        parts = arm_id.split("_")

        robot_inst = None

        if len(parts) >= 2:
            side = parts[0] # left, right, default
            group = parts[1] # leader, follower

            # Handle Followers
            if group == "follower":
                # Check for Damiao follower first
                if side == "damiao":
                    if DAMIAO_AVAILABLE and DamiaoFollowerRobot is not None:
                        if isinstance(self.robot, DamiaoFollowerRobot):
                            robot_inst = self.robot
                        elif hasattr(self, 'damiao_robot') and self.damiao_robot is not None:
                            robot_inst = self.damiao_robot
                elif isinstance(self.robot, BiUmbraFollower):
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

            if robot_inst:
                target_motors = []
                if group == "leader":
                    target_motors = ["base", "link1", "link2", "link3", "link4", "link5", "gripper"]
                elif group == "follower":
                    # Damiao has simple 7-DOF (no dual followers)
                    if side == "damiao":
                        target_motors = ["base", "link1", "link2", "link3", "link4", "link5", "gripper"]
                    else:
                        # Umbra/BiUmbra has dual link1/link2 followers
                        target_motors = ["base", "link1", "link1_follower", "link2", "link2_follower", "link3", "link4", "link5", "gripper"]
                return robot_inst, target_motors

        # Fallback: check arm registry for arms not matching legacy patterns
        if self.arm_registry:
            instance = self.arm_registry.arm_instances.get(arm_id)
            if instance and hasattr(instance, 'bus'):
                target_motors = list(instance.bus.motors.keys())
                return instance, target_motors

        logger.warning(f"_get_arm_context: No robot instance found for {arm_id}")
        return None, []

    def _ensure_active_profiles_init(self):
        if not hasattr(self, "active_profiles"):
            self.active_profiles = {}
            self._load_persistent_active_profiles()

    def _get_persistence_path(self):
        import pathlib
        base_dir = pathlib.Path("calibration_profiles")
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / "active_profiles.json"
        
    def _load_persistent_active_profiles(self):
        fpath = self._get_persistence_path()
        if fpath.exists():
            import json
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.active_profiles = data
                        logger.info(f"Loaded persistent active profiles: {self.active_profiles}")
            except Exception as e:
                logger.error(f"Failed to load persistent active profiles: {e}")

    def _save_persistent_active_profiles(self):
        fpath = self._get_persistence_path()
        import json
        try:
            with open(fpath, "w") as f:
                json.dump(self.active_profiles, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save persistent active profiles: {e}")

    def restore_active_profiles(self):
        """Attempts to load the calibration files for the active profiles."""
        self._ensure_active_profiles_init()
        logger.info("Restoring active calibration profiles...")
        
        for arm_id, filename in self.active_profiles.items():
            logger.info(f"Restoring {arm_id} -> {filename}")
            # we use load_calibration_file but need to avoid recursion loop with save?
            # load_calibration_file calls _save_persistent if we add it there.
            # That's fine.
            self.load_calibration_file(arm_id, filename)

    def _ensure_calibration_initialized(self, arm):
        """Ensures that arm.calibration is populated with default entries for all motors."""
        if not arm.calibration:
            from lerobot.motors import MotorCalibration
            from lerobot.motors.damiao.damiao import DamiaoMotorsBus
            is_damiao = isinstance(arm.bus, DamiaoMotorsBus)
            # Damiao: radians (±π). Feetech/Dynamixel: 12-bit ticks (0-4095).
            default_min = -3.15 if is_damiao else 0
            default_max = 3.15 if is_damiao else 4096
            logger.info(f"Initializing empty calibration for {arm} (damiao={is_damiao})")
            for name, motor in arm.bus.motors.items():
                arm.calibration[name] = MotorCalibration(
                    id=getattr(motor, 'id', getattr(motor, 'can_id', 0)),
                    drive_mode=0,
                    homing_offset=0,
                    range_min=default_min,
                    range_max=default_max,
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

        # Damiao: block writes + disable torque so user can freely move joints
        if self._is_damiao_arm(arm_id):
            logger.info(f"[{arm_id}] Damiao — discovery mode: disabling motors, blocking writes")
            arm.bus._discovery_mode = True
            if self.robot_lock:
                with self.robot_lock:
                    for motor_name in target_motors:
                        if motor_name in arm.bus.motors:
                            try:
                                arm.bus._control.disable(arm.bus._motors[motor_name])
                            except Exception as e:
                                logger.warning(f"Failed to disable {motor_name}: {e}")
            else:
                for motor_name in target_motors:
                    if motor_name in arm.bus.motors:
                        try:
                            arm.bus._control.disable(arm.bus._motors[motor_name])
                        except Exception as e:
                            logger.warning(f"Failed to disable {motor_name}: {e}")
            return

        # Feetech/Dynamixel: Enable Multi-turn for discovery (Limits=0)
        # This prevents the motor from stopping at 0/4095
        # Dynamixel EEPROM writes require torque off (DynamixelLeader keeps gripper torque on)
        if self._is_dynamixel_arm(arm_id):
            if self.robot_lock:
                with self.robot_lock:
                    arm.bus.disable_torque()
            else:
                arm.bus.disable_torque()

        if self.robot_lock:
             with self.robot_lock:
                 for motor in target_motors:
                     if motor in arm.bus.motors:
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

        if not arm:
            return {"status": "error", "message": f"Arm '{arm_id}' not found", "warnings": []}

        # Ensure we have a calibration object to update
        self._ensure_calibration_initialized(arm)
        
        # Damiao: clear discovery mode + re-enable motors with safe MIT startup
        if self._is_damiao_arm(arm_id):
            logger.info(f"[{arm_id}] Damiao — ending discovery: re-enabling motors, unblocking writes")
            arm.bus._discovery_mode = False
            _, target_motors = self._get_arm_context(arm_id)
            if target_motors:
                # Use bus-level enable_torque() which handles safe MIT enable
                # (sends limp frames after enable to prevent torque spikes)
                motors_to_enable = [m for m in target_motors if m in arm.bus.motors]
                if self.robot_lock:
                    with self.robot_lock:
                        arm.bus.enable_torque(motors_to_enable)
                else:
                    arm.bus.enable_torque(motors_to_enable)

        warnings = []
        if arm and self.session_ranges:
            for motor, ranges in self.session_ranges.items():
                if motor in arm.calibration:
                    arm.calibration[motor].range_min = ranges["min"]
                    arm.calibration[motor].range_max = ranges["max"]
                    
                    # Diagnostic Check (Feetech/Dynamixel tick-based only)
                    # Damiao uses radians — this tick-range check doesn't apply
                    if not self._is_damiao_arm(arm_id):
                        if ranges["min"] <= 10 and ranges["max"] >= 4080:
                             if motor in ["link1", "link2", "link1_follower", "link2_follower"]:
                                 warnings.append(f"{motor} used FULL RANGE (0-4096). Possible Wrap Error.")
                else:
                    logger.warning(f"stop_discovery: Motor {motor} not found in arm.calibration")

            # Propagate discovered ranges to motor bus for runtime enforcement
            if hasattr(arm, 'apply_calibration_limits'):
                arm.apply_calibration_limits()

            self.save_calibration(arm_id)
            
        msg = "Range Discovery Complete."
        if warnings:
            msg += f" WARNING: {'; '.join(warnings)}"
            
        return {"status": "success", "message": msg, "warnings": warnings}


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
                "min": round(range_min, 2),
                "max": round(range_max, 2),
                "visited_min": round(visited_min, 2),
                "visited_max": round(visited_max, 2),
                "pos": round(pos, 2),
                "id": getattr(arm.bus.motors[motor_name], 'id', getattr(arm.bus.motors[motor_name], 'can_id', 0))
            })
        
        return state

    def set_calibration_limit(self, arm_id: str, motor_name: str, limit_type: str, value):
        arm, _ = self._get_arm_context(arm_id)
        if not arm:
            return

        if motor_name not in arm.calibration:
            return

        # Damiao uses float radians, Feetech/Dynamixel use integer ticks
        converted = float(value) if self._is_damiao_arm(arm_id) else int(value)

        if limit_type == "min":
            arm.calibration[motor_name].range_min = converted
        elif limit_type == "max":
            arm.calibration[motor_name].range_max = converted

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
            # Apply to motors (Dynamixel EEPROM writes require torque off)
            if hasattr(arm.bus, "write_calibration"):
                if self._is_dynamixel_arm(arm_id):
                    arm.bus.disable_torque()
                arm.bus.write_calibration(arm.calibration)
                # Restore gripper spring mode after EEPROM writes
                if self._is_dynamixel_arm(arm_id) and hasattr(arm, 'configure'):
                    arm.configure()

            # Propagate calibrated joint limits to motor bus for runtime enforcement
            if hasattr(arm, 'apply_calibration_limits'):
                arm.apply_calibration_limits()

            # Update Active Profile state
            self.active_profiles[arm_id] = filename
            self._save_persistent_active_profiles()

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

        # Write calibration to motor hardware (Feetech STS3215 only)
        # Damiao: host-side JSON only (absolute encoders, no offset registers)
        # Dynamixel leaders: host-side JSON only (gripper has torque enabled,
        # can't write EEPROM registers like Homing_Offset/Position_Limits)
        if not self._is_damiao_arm(arm_id) and not self._is_dynamixel_arm(arm_id):
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
            self._save_persistent_active_profiles()



    def _is_dynamixel_arm(self, arm_id: str) -> bool:
        """Check if arm uses Dynamixel motors (no Lock register, different OperatingMode)."""
        if self.arm_registry:
            arm_def = self.arm_registry.arms.get(arm_id)
            if arm_def and 'dynamixel' in arm_def.motor_type.value:
                return True
        return False

    def _is_damiao_arm(self, arm_id: str) -> bool:
        """Check if arm uses Damiao CAN motors (completely different protocol)."""
        if self.arm_registry:
            arm_def = self.arm_registry.arms.get(arm_id)
            if arm_def and arm_def.motor_type.value == 'damiao':
                return True
        # Legacy pattern
        if arm_id.startswith('damiao'):
            return True
        return False

    def disable_torque(self, arm_id: str):
        arm, _ = self._get_arm_context(arm_id)
        if arm:
            logger.info(f"Disabling torque for COMPONENT {arm_id} (All motors on bus)")
            is_damiao = self._is_damiao_arm(arm_id)
            is_dynamixel = self._is_dynamixel_arm(arm_id)

            def _do_disable():
                if is_damiao:
                    # Damiao: use CAN disable command directly
                    arm.bus.disable_torque()
                else:
                    RETRY_COUNT = 5
                    try:
                        arm.bus.disable_torque(None, num_retry=RETRY_COUNT)
                    except TypeError:
                        arm.bus.disable_torque(None)

                    if not is_dynamixel:
                        # Feetech-specific: reset Lock, Torque, OperatingMode
                        for motor in arm.bus.motors:
                            try:
                                arm.bus.write("Lock", motor, 0, num_retry=RETRY_COUNT)
                                arm.bus.write("Torque_Enable", motor, 0, num_retry=RETRY_COUNT)
                                arm.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value, num_retry=RETRY_COUNT)
                            except Exception as e:
                                logger.warning(f"Failed to set Operating_Mode/Limits for {motor}: {e}")

            if self.robot_lock:
                with self.robot_lock:
                    _do_disable()
            else:
                _do_disable()

    def enable_torque(self, arm_id: str):
        arm, _ = self._get_arm_context(arm_id)
        if arm:
            logger.info(f"Enabling torque for {arm_id} (All motors on bus)")
            is_damiao = self._is_damiao_arm(arm_id)

            def _do_enable():
                if is_damiao:
                    arm.bus.enable_torque()
                else:
                    RETRY_COUNT = 5
                    try:
                        arm.bus.enable_torque(None, num_retry=RETRY_COUNT)
                    except TypeError:
                        arm.bus.enable_torque(None)

            if self.robot_lock:
                with self.robot_lock:
                    _do_enable()
            else:
                _do_enable()

    def move_motor(self, arm_id: str, motor_name: str, value):
        arm, _ = self._get_arm_context(arm_id)
        if arm:
            if self._is_damiao_arm(arm_id):
                # Damiao: use sync_write (CAN protocol, no single-motor write())
                if self.robot_lock:
                    with self.robot_lock:
                        arm.bus.sync_write("Goal_Position", {motor_name: value})
                else:
                    arm.bus.sync_write("Goal_Position", {motor_name: value})
            else:
                if self.robot_lock:
                    with self.robot_lock:
                        arm.bus.write("Goal_Position", motor_name, value, normalize=False)
                else:
                    arm.bus.write("Goal_Position", motor_name, value, normalize=False)

    def perform_homing(self, arm_id: str):
        arm, target_motors = self._get_arm_context(arm_id)
        if not arm:
            logger.error(f"perform_homing: Invalid arm_id {arm_id} or arm not found.")
            return {"status": "error", "message": "Arm not found"}

        logger.info(f"Performing Homing for {arm_id}")

        # Ensure we have a calibration object to update offsets into
        self._ensure_calibration_initialized(arm)

        try:
            # Damiao: use CAN set_zero_position command (0xFE) — absolute encoders, no offset register
            if self._is_damiao_arm(arm_id):
                return self._perform_damiao_homing(arm, arm_id, target_motors)

            # Feetech/Dynamixel: use standard LeRobot homing (half-turn offset)
            offsets = {}
            if self.robot_lock:
                with self.robot_lock:
                     motors_to_home = [m for m in target_motors if m in arm.bus.motors]
                     arm.bus.disable_torque(motors_to_home)
                     offsets = arm.bus.set_half_turn_homings(motors_to_home)
            else:
                 motors_to_home = [m for m in target_motors if m in arm.bus.motors]
                 arm.bus.disable_torque(motors_to_home)
                 offsets = arm.bus.set_half_turn_homings(motors_to_home)

            # Persist homing offsets to arm.calibration so save_calibration() writes them
            self._ensure_calibration_initialized(arm)
            for motor_name, offset in offsets.items():
                if motor_name in arm.calibration:
                    arm.calibration[motor_name].homing_offset = offset
            # Also sync to the bus calibration cache
            arm.bus.calibration = dict(arm.calibration)

            # Update gripper offset-adjusted positions for get_action() normalization
            # Do NOT call arm.configure() here — it re-enables gripper spring, blocking range discovery
            if self._is_dynamixel_arm(arm_id) and hasattr(arm.bus, '_software_homing_offsets'):
                gripper_id = arm.bus.motors["gripper"].id
                gripper_offset = arm.bus._software_homing_offsets.get(gripper_id, 0)
                # Use calibrated range if available (config defaults may be outside position limits)
                # For our arm: lower ticks = open, higher ticks = closed
                if arm.calibration and "gripper" in arm.calibration:
                    cal = arm.calibration["gripper"]
                    arm._gripper_open = cal.range_min + gripper_offset
                    arm._gripper_closed = cal.range_max + gripper_offset
                else:
                    arm._gripper_open = arm.config.gripper_open_pos + gripper_offset
                    arm._gripper_closed = arm.config.gripper_closed_pos + gripper_offset

            # Diagnostic check
            warnings = []
            for m, off in offsets.items():
                if off == 0 and m in ['link1', 'link2', 'link1_follower', 'link2_follower']:
                    warnings.append(f"{m}=0 (Suspicious)")

            msg = "Homing Done."
            if warnings:
                msg += f" WARNING: Zero offsets for {', '.join(warnings)}. Check Motor Alignment!"

            return {"status": "success", "offsets": offsets, "message": msg}

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.exception(f"Homing failed with exception: {e}")
            return {"status": "error", "message": str(e)}

    def _perform_damiao_homing(self, arm, arm_id: str, target_motors: list) -> dict:
        """Damiao-specific homing: set current position as zero using CAN 0xFE command.

        Damiao motors use absolute encoders — no Homing_Offset register.
        The 0xFE CAN command permanently sets current physical position as 0 radians.
        User must have positioned the arm to the desired zero pose before calling this.
        """
        motors_to_home = [m for m in target_motors if m in arm.bus.motors]

        if self.robot_lock:
            with self.robot_lock:
                previous = arm.bus.set_zero_positions(motors_to_home)
        else:
            previous = arm.bus.set_zero_positions(motors_to_home)

        msg = f"Damiao zero-set complete. {len(previous)} motors zeroed at current position."
        logger.info(f"[{arm_id}] {msg} Previous positions: {previous}")

        return {
            "status": "success",
            "offsets": {m: 0.0 for m in previous},
            "previous_positions": {m: round(v, 3) for m, v in previous.items()},
            "message": msg
        }
        
    
    def _get_inversions_file(self, arm_id: str):
        import pathlib
        base_dir = pathlib.Path(f"calibration_profiles/{arm_id}")
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / "inversions.json"

    def _read_inversions_file(self, arm_id: str) -> Dict[str, bool]:
        """Read inversions directly from an arm's file (no translation)."""
        fpath = self._get_inversions_file(arm_id)
        if not fpath.exists():
            return {}
        import json
        try:
            with open(fpath, "r") as f:
                return json.load(f)
        except:
            return {}

    def _find_paired_follower(self, arm_id: str):
        """If arm_id is a leader in a pairing, return (follower_id, need_translation).
        need_translation is True when motor names differ (Dynamixel→Damiao)."""
        if not self.arm_registry:
            return None, False
        arm_def = self.arm_registry.arms.get(arm_id)
        if not arm_def or arm_def.role.value != "leader":
            return None, False
        for p in self.arm_registry.pairings:
            if p.leader_id == arm_id:
                follower_def = self.arm_registry.arms.get(p.follower_id)
                is_dyn_leader = arm_def.motor_type.value.startswith("dynamixel")
                is_dam_follower = follower_def and follower_def.motor_type.value == "damiao"
                return p.follower_id, (is_dyn_leader and is_dam_follower)
        return None, False

    def get_inversions(self, arm_id: str) -> Dict[str, bool]:
        # If this is a leader with a paired follower, read follower's inversions and translate
        follower_id, need_translation = self._find_paired_follower(arm_id)
        if follower_id:
            follower_inv = self._read_inversions_file(follower_id)
            if need_translation:
                return {self._DAMIAO_TO_DYNAMIXEL.get(k, k): v for k, v in follower_inv.items()}
            return follower_inv
        # Not a leader or no pairing — read directly
        return self._read_inversions_file(arm_id)

    def set_inversion(self, arm_id: str, motor_name: str, inverted: bool):
        # Check if this is a leader arm with a paired follower
        follower_id, need_translation = self._find_paired_follower(arm_id)

        if follower_id:
            # Save to FOLLOWER (where inversions are applied at runtime)
            target_id = follower_id
            target_motor = self._DYNAMIXEL_TO_DAMIAO.get(motor_name, motor_name) if need_translation else motor_name
        else:
            # Save to this arm directly (it's a follower or unpaired)
            target_id = arm_id
            target_motor = motor_name

        fpath = self._get_inversions_file(target_id)
        current = self._read_inversions_file(target_id)
        current[target_motor] = inverted

        import json
        with open(fpath, "w") as f:
            json.dump(current, f, indent=2)

        # Reload on the target arm (follower)
        arm, _ = self._get_arm_context(target_id)
        if arm and hasattr(arm, "reload_inversions"):
            arm.reload_inversions()

    def set_zero_pose(self, arm_id: str):
        """Captures the current position as the baseline 'Zero' for alignment. Also disables torque."""
        # For arm registry arms, just disable torque on this arm directly
        if self.arm_registry and arm_id in self.arm_registry.arms:
            self.disable_torque(arm_id)
        else:
            # Legacy: infer pair to ensure both are loose for the user to move
            parts = arm_id.split("_")
            side = parts[0]
            leader_id = f"{side}_leader"
            follower_id = f"{side}_follower"
            self.disable_torque(leader_id)
            self.disable_torque(follower_id)
        
        # Determine which arm we are setting zero for? 
        # Actually, we need to capture zero for BOTH if we are doing the wizard.
        # But the UI calls this set-zero endpoint for *each* arm in a loop (Promise.all).
        # "Promise.all(pairs.map(id => fetch(... set-zero ...)))"
        # So this method is called individually.
        # However, to be safe and redundant (and safe for the user), ensuring torque is off here is good.
        
        arm, _ = self._get_arm_context(arm_id)
        if not arm: return False
        
        # Read all raw positions
        positions = arm.bus.sync_read("Present_Position", normalize=False)
        
        # Store in session memory
        if not hasattr(self, "alignment_zeros"):
            self.alignment_zeros = {}
            
        self.alignment_zeros[arm_id] = positions
        logger.info(f"Captured Zero Pose for {arm_id} (Torque Disabled): {positions}")
        return True

    def compute_auto_alignment(self, arm_id: str):
        """
        Compares current position to Zero pose.
        If Leader moves + and Follower moves -, marks as Inverted.
        Requires:
        1. set_zero_pose() called previously.
        2. Leader and Follower to be moved roughly in sync.
        """
        # We need BOTH the Leader and the Follower for this arm group
        # arm_id is roughly "left_follower" or "left_leader".
        # We need to deduce the pair.
        
        parts = arm_id.split("_")
        side = parts[0] # left or right
        
        # Determine IDs
        leader_id = f"{side}_leader"
        follower_id = f"{side}_follower"
        
        # Get contexts
        l_arm, _ = self._get_arm_context(leader_id)
        f_arm, _ = self._get_arm_context(follower_id)
        
        if not l_arm or not f_arm:
             logger.error("AutoAlign: Could not find both Leader and Follower arms.")
             return {"status": "error", "message": "Leader or Follower missing."}
             
        # Check Zeros
        if not hasattr(self, "alignment_zeros"):
             return {"status": "error", "message": "Zero Pose not set."}
             
        l_zeros = self.alignment_zeros.get(leader_id)
        f_zeros = self.alignment_zeros.get(follower_id)
        
        if not l_zeros or not f_zeros:
             return {"status": "error", "message": "Zero Pose missing for one or both arms."}

        # Read Current
        l_current = l_arm.bus.sync_read("Present_Position", normalize=False)
        f_current = f_arm.bus.sync_read("Present_Position", normalize=False)
        
        changes = {}
        inversions = self.get_inversions(follower_id) # Current inversions for follower
        
        # Compare Motors
        # Maps: Leader Name -> Follower Name
        # Standard: linkX -> linkX (and linkX_follower if dual)
        # We primarily check the main link motor. In dual setup, linkX_follower usually follows linkX logic inverted or not.
        # But for Auto-Align, we just check distinct names.
        
        mapping = {
            "link1": "link1",
            "link2": "link2",
            "link3": "link3",
            "link4": "link4",
            "link5": "link5",
            "gripper": "gripper"
        }
        
        THRESHOLD = 50 # Minimum movement to detect direction (approx 4 degrees)
        
        inverted_count = 0
        
        for l_name, f_name in mapping.items():
            if l_name not in l_current or f_name not in f_current:
                continue
                
            # Calculate Deltas
            l_delta = l_current[l_name] - l_zeros[l_name]
            f_delta = f_current[f_name] - f_zeros[f_name]
            
            # Check Threshold
            if abs(l_delta) < THRESHOLD or abs(f_delta) < THRESHOLD:
                changes[f_name] = "unchanged (small movement)"
                continue
                
            # Check Signs
            # If signs differ, they are moving opposite.
            # In a "Sync Pose" where users move them visually identically,
            # opposite raw values -> NEeds Inversion.
            
            # Note: Current logic in robot driver: goal = -present if inverted.
            # So if we detect they are opposite, we should SET Inverted = True.
            # If they are same, Inverted = False.
            
            # HOWEVER: There is a "gripper" exception (100 - x).
            # Gripper Raw: 0 (Open) to 100 (Closed) or vice versa.
            # If Leader goes 0->50 (+50)
            # Follower goes 100->50 (-50)
            # Signs opposite. -> Invert = True. 
            # Driver: if Invert: goal = 100 - pos.
            # So if Leader=50, Follower=100-50=50. Correct.
            
            # Standard Motor:
            # L: 0->100 (+100). F: 0->-100 (-100).
            # Signs opposite. -> Invert = True.
            # Driver: goal = -pos = -(-100) = 100. Correct.
            
            # Conclusion: If signs opposite => Invert=True.
            
            # Wait, what if it was ALREADY inverted in the config? 
            # The calibration service sets the *config*.
            # The User physically moved the robot. The Raw values are independent of software inversion.
            # So this logic holds regardless of previous config:
            # Physical Opposites in Raw Values = Needs Software Inversion.
            
            is_inverted = (l_delta * f_delta) < 0
            
            if is_inverted != inversions.get(f_name, False):
                 inversions[f_name] = is_inverted
                 changes[f_name] = f"SET INVERTED={is_inverted}"
                 inverted_count += 1
            else:
                 changes[f_name] = "ok"
                 
            # Also handle dual followers? (link1_follower, link2_follower)
            # Usually they are physically coupled. If link1 is inverted, link1_follower usually follows suit
            # or has its own specific physical mounting.
            # If we want to be thorough:
            f_dual = f"{f_name}_follower"
            if f_dual in f_current:
                 f_delta_dual = f_current[f_dual] - f_zeros.get(f_dual, 0)
                 if abs(f_delta_dual) > THRESHOLD:
                      is_inv_dual = (l_delta * f_delta_dual) < 0
                      if is_inv_dual != inversions.get(f_dual, False):
                           inversions[f_dual] = is_inv_dual
                           changes[f_dual] = f"SET INVERTED={is_inv_dual}"
                           inverted_count += 1

        # Save results
        if inverted_count > 0:
             self.set_inversion(follower_id, "batch_update", False) # Just to trigger file path retrieval inside loop? No.
             # We should use a batch update method or loop.
             fpath = self._get_inversions_file(follower_id)
             with open(fpath, "w") as f:
                 import json
                 json.dump(inversions, f, indent=2)
             
             # Reload robot
             f_arm.reload_inversions(inversions)
             
        return {"status": "success", "inverted_count": inverted_count, "changes": changes}
