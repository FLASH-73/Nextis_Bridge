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

from app.core.calibration.discovery import CalibrationDiscovery
from app.core.calibration.profiles import CalibrationProfiles
from app.core.calibration.homing import CalibrationHoming

logger = logging.getLogger(__name__)


class CalibrationService:
    # Keep translation maps as class-level constants for backward compatibility
    _DYNAMIXEL_TO_DAMIAO = CalibrationHoming._DYNAMIXEL_TO_DAMIAO
    _DAMIAO_TO_DYNAMIXEL = CalibrationHoming._DAMIAO_TO_DYNAMIXEL

    def __init__(self, robot, leader=None, robot_lock=None, arm_registry=None):
        self.robot = robot
        self.leader = leader
        self.robot_lock = robot_lock
        self.arm_registry = arm_registry  # For new arm management system
        self.active_arm = None

        # Range Discovery State
        self.is_discovering = False
        self.session_ranges = {} # {motor_name: {"min": val, "max": val}}

        # Create delegates
        self._discovery = CalibrationDiscovery(self)
        self._profiles = CalibrationProfiles(self)
        self._homing = CalibrationHoming(self)

    # ── Core methods (kept in-place) ─────────────────────────────────────

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
                try:
                    damiao_calibrated = self.robot.is_calibrated
                except Exception as e:
                    logger.warning(f"Could not check calibration for damiao_follower: {e}")
                    damiao_calibrated = False
                arms.append({
                    "id": "damiao_follower",
                    "name": "Damiao Follower",
                    "calibrated": damiao_calibrated,
                    "type": "follower",
                    "motor_type": "damiao"  # Special flag for UI
                })

        # Check for Damiao robot passed separately (e.g., as damiao_robot attribute)
        if hasattr(self, 'damiao_robot') and self.damiao_robot is not None:
            try:
                damiao_calibrated = self.damiao_robot.is_calibrated if hasattr(self.damiao_robot, 'is_calibrated') else True
            except Exception as e:
                logger.warning(f"Could not check calibration for damiao_follower (damiao_robot): {e}")
                damiao_calibrated = False
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
                    try:
                        calibrated = instance.is_calibrated
                    except Exception as e:
                        logger.warning(f"Could not check calibration for {arm_id}: {e}")
                        calibrated = False
                arms.append({
                    "id": arm_id,
                    "name": arm_def.name,
                    "calibrated": calibrated,
                    "type": arm_def.role.value,
                    "motor_type": arm_def.motor_type.value,
                })

        return arms

    def get_arm_context(self, arm_id: str):
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

        logger.warning(f"get_arm_context: No robot instance found for {arm_id}")
        return None, []

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
        arm, _ = self.get_arm_context(arm_id)
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
        arm, _ = self.get_arm_context(arm_id)
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
        arm, _ = self.get_arm_context(arm_id)
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

    # ── Discovery delegation ─────────────────────────────────────────────

    def _ensure_calibration_initialized(self, arm):
        self._discovery.ensure_calibration_initialized(arm)

    def start_discovery(self, arm_id: str):
        self._discovery.start_discovery(arm_id)

    def stop_discovery(self, arm_id: str):
        return self._discovery.stop_discovery(arm_id)

    def get_calibration_state(self, arm_id: str):
        return self._discovery.get_calibration_state(arm_id)

    def set_calibration_limit(self, arm_id: str, motor_name: str, limit_type: str, value):
        self._discovery.set_calibration_limit(arm_id, motor_name, limit_type, value)

    # ── Profiles delegation ──────────────────────────────────────────────

    def _ensure_active_profiles_init(self):
        self._profiles.ensure_active_profiles_init()

    def restore_active_profiles(self):
        self._profiles.restore_active_profiles()

    def list_calibration_files(self, arm_id: str):
        return self._profiles.list_calibration_files(arm_id)

    def load_calibration_file(self, arm_id: str, filename: str):
        return self._profiles.load_calibration_file(arm_id, filename)

    def delete_calibration_file(self, arm_id: str, filename: str):
        return self._profiles.delete_calibration_file(arm_id, filename)

    def save_calibration(self, arm_id: str, name: str = None):
        self._profiles.save_calibration(arm_id, name)

    # ── Homing delegation ────────────────────────────────────────────────

    def perform_homing(self, arm_id: str):
        return self._homing.perform_homing(arm_id)

    def get_inversions(self, arm_id: str):
        return self._homing.get_inversions(arm_id)

    def set_inversion(self, arm_id: str, motor_name: str, inverted: bool):
        self._homing.set_inversion(arm_id, motor_name, inverted)

    def set_zero_pose(self, arm_id: str):
        return self._homing.set_zero_pose(arm_id)

    def compute_auto_alignment(self, arm_id: str):
        return self._homing.compute_auto_alignment(arm_id)
