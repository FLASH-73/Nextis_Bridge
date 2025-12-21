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

    def get_arms(self) -> List[Dict[str, Any]]:
        arms = []
        
        def add_arm_groups(prefix, robot_inst, include_leader=True):
            # Check if calibrated (simple check: do we have calibration data?)
            # Ideally we check if it matches the motors, but for UI "Calibrated" usually means "File exists/Loaded"
            # robot_inst.is_calibrated checks if memory matches motors.
            
            # We split into Leader and Follower
            # Leader: link1, link2
            # Follower: base, link1_follower, link2_follower, link3, link4, link5, gripper
            
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
            follower_motors = ["base", "link1_follower", "link2_follower", "link3", "link4", "link5", "gripper"]
            follower_calibrated = all(m in robot_inst.calibration for m in follower_motors)

            arms.append({
                "id": f"{prefix}_follower",
                "name": f"{prefix.capitalize()} Follower",
                "calibrated": follower_calibrated,
                "type": "follower"
            })

        if isinstance(self.robot, BiUmbraFollower):
            # BiUmbraFollower only manages followers. Leaders are in self.leader
            add_arm_groups("left", self.robot.left_arm, include_leader=False)
            add_arm_groups("right", self.robot.right_arm, include_leader=False)
        elif isinstance(self.robot, UmbraFollowerRobot):
            add_arm_groups("default", self.robot, include_leader=True)
            
        # Add Leader Arms if present
        if self.leader:
            from lerobot.teleoperators.bi_umbra_leader.bi_umbra_leader import BiUmbraLeader
            if isinstance(self.leader, BiUmbraLeader):
                # We need to adapt add_arm_groups because it assumes "leader" and "follower" are in the same object
                # But now "leader" is in self.leader.left_arm/right_arm
                
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
            # If we have a separate leader object (BiUmbraLeader)
            if self.leader:
                if side == "left":
                    robot_inst = self.leader.left_arm
                elif side == "right":
                    robot_inst = self.leader.right_arm
            # Fallback for single arm setups where leader might be on the same bus (unlikely for BiUmbra)
            elif isinstance(self.robot, UmbraFollowerRobot):
                 robot_inst = self.robot
            
        if not robot_inst:
            return None, []

        target_motors = []
        if group == "leader":
            target_motors = ["link1", "link2"]
        elif group == "follower":
            target_motors = ["base", "link1_follower", "link2_follower", "link3", "link4", "link5", "gripper"]
            
        return robot_inst, target_motors

    def get_calibration_state(self, arm_id: str) -> List[Dict[str, Any]]:
        arm, target_motors = self._get_arm_context(arm_id)
        if not arm:
            return []

        # Ensure we have fresh readings
        # We must read ALL motors on the bus because sync_read usually reads everything or specific IDs
        # Reading specific names is safer
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
            
            # If no calibration yet, use defaults
            range_min = cal.range_min if cal else 0
            range_max = cal.range_max if cal else 4096 
            
            state.append({
                "name": motor_name,
                "min": range_min,
                "max": range_max,
                "pos": pos,
                "id": arm.bus.motors[motor_name].id
            })
        
        return state

    def set_calibration_limit(self, arm_id: str, motor_name: str, limit_type: str, value: int):
        arm, _ = self._get_arm_context(arm_id)
        if not arm:
            return

        if motor_name not in arm.calibration:
            # If missing, we might need to create a dummy calibration object
            # But usually we should have loaded something or initialized it
            return

        if limit_type == "min":
            arm.calibration[motor_name].range_min = int(value)
        elif limit_type == "max":
            arm.calibration[motor_name].range_max = int(value)

    def save_calibration(self, arm_id: str):
        arm, _ = self._get_arm_context(arm_id)
        if not arm:
            return

        # Write to motors (all of them or just the modified ones? write_calibration writes all in dict)
        if self.robot_lock:
            with self.robot_lock:
                arm.bus.write_calibration(arm.calibration)
                # Ensure the robot instance's calibration dict is in sync with the bus
                arm.calibration = arm.bus.calibration
        else:
            arm.bus.write_calibration(arm.calibration)
            arm.calibration = arm.bus.calibration

        # Save to file
        if hasattr(arm, "_save_calibration"):
            arm._save_calibration()
        else:
            logger.warning(f"Save calibration not supported for {arm}")

    def disable_torque(self, arm_id: str):
        arm, target_motors = self._get_arm_context(arm_id)
        if arm:
            logger.info(f"Disabling torque for {arm_id} motors: {target_motors}")
            # Only disable torque for the specific group?
            # Ideally yes, but they might be on the same bus.
            # FeetechMotorsBus.disable_torque takes a list of motors.
            if self.robot_lock:
                with self.robot_lock:
                    arm.bus.disable_torque(target_motors)
                    # Enforce Position Mode for calibration (in case it was Velocity/PWM)
                    for motor in target_motors:
                         if motor in arm.bus.motors:
                            arm.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            else:
                arm.bus.disable_torque(target_motors)
                for motor in target_motors:
                     if motor in arm.bus.motors:
                        arm.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def enable_torque(self, arm_id: str):
        arm, target_motors = self._get_arm_context(arm_id)
        if arm:
            logger.info(f"Enabling torque for {arm_id} motors: {target_motors}")
            if self.robot_lock:
                with self.robot_lock:
                    arm.bus.enable_torque(target_motors)
            else:
                arm.bus.enable_torque(target_motors)

    def move_motor(self, arm_id: str, motor_name: str, value: int):
        arm, _ = self._get_arm_context(arm_id)
        if arm:
            if self.robot_lock:
                with self.robot_lock:
                    arm.bus.write("Goal_Position", motor_name, value, normalize=False)
            else:
                arm.bus.write("Goal_Position", motor_name, value, normalize=False)

    def perform_homing(self, arm_id: str):
        arm, target_motors = self._get_arm_context(arm_id)
        if arm:
            # Set current position as homing offset (half-turn) for TARGET motors only
            if self.robot_lock:
                with self.robot_lock:
                    arm.bus.set_half_turn_homings(target_motors)
            else:
                arm.bus.set_half_turn_homings(target_motors)
            return True
        return False
