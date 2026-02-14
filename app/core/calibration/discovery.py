import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class CalibrationDiscovery:
    """Range discovery operations for calibration.

    Accesses shared state (robot_lock, session_ranges, is_discovering)
    via self._svc (the parent CalibrationService).
    """

    def __init__(self, svc):
        self._svc = svc

    def ensure_calibration_initialized(self, arm):
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

        arm, target_motors = self._svc._get_arm_context(arm_id)
        if not arm:
             return

        logger.info(f"Starting Range Discovery for {arm_id}")
        self.ensure_calibration_initialized(arm)

        self._svc.is_discovering = True
        self._svc.session_ranges = {}

        # Damiao: block writes + disable torque so user can freely move joints
        if self._svc._is_damiao_arm(arm_id):
            logger.info(f"[{arm_id}] Damiao — discovery mode: disabling motors, blocking writes")
            arm.bus._discovery_mode = True
            if self._svc.robot_lock:
                with self._svc.robot_lock:
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
        if self._svc._is_dynamixel_arm(arm_id):
            if self._svc.robot_lock:
                with self._svc.robot_lock:
                    arm.bus.disable_torque()
            else:
                arm.bus.disable_torque()

        if self._svc.robot_lock:
             with self._svc.robot_lock:
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
        self._svc.is_discovering = False

        # Automatically update the robot's calibration with the discovered ranges
        arm, _ = self._svc._get_arm_context(arm_id)

        if not arm:
            return {"status": "error", "message": f"Arm '{arm_id}' not found", "warnings": []}

        # Ensure we have a calibration object to update
        self.ensure_calibration_initialized(arm)

        # Damiao: clear discovery mode + re-enable motors with safe MIT startup
        if self._svc._is_damiao_arm(arm_id):
            logger.info(f"[{arm_id}] Damiao — ending discovery: re-enabling motors, unblocking writes")
            arm.bus._discovery_mode = False
            _, target_motors = self._svc._get_arm_context(arm_id)
            if target_motors:
                # Use bus-level enable_torque() which handles safe MIT enable
                # (sends limp frames after enable to prevent torque spikes)
                motors_to_enable = [m for m in target_motors if m in arm.bus.motors]
                if self._svc.robot_lock:
                    with self._svc.robot_lock:
                        arm.bus.enable_torque(motors_to_enable)
                else:
                    arm.bus.enable_torque(motors_to_enable)

        warnings = []
        if arm and self._svc.session_ranges:
            for motor, ranges in self._svc.session_ranges.items():
                if motor in arm.calibration:
                    arm.calibration[motor].range_min = ranges["min"]
                    arm.calibration[motor].range_max = ranges["max"]

                    # Diagnostic Check (Feetech/Dynamixel tick-based only)
                    # Damiao uses radians — this tick-range check doesn't apply
                    if not self._svc._is_damiao_arm(arm_id):
                        if ranges["min"] <= 10 and ranges["max"] >= 4080:
                             if motor in ["link1", "link2", "link1_follower", "link2_follower"]:
                                 warnings.append(f"{motor} used FULL RANGE (0-4096). Possible Wrap Error.")
                else:
                    logger.warning(f"stop_discovery: Motor {motor} not found in arm.calibration")

            # Propagate discovered ranges to motor bus for runtime enforcement
            if hasattr(arm, 'apply_calibration_limits'):
                arm.apply_calibration_limits()

            self._svc.save_calibration(arm_id)

        msg = "Range Discovery Complete."
        if warnings:
            msg += f" WARNING: {'; '.join(warnings)}"

        return {"status": "success", "message": msg, "warnings": warnings}

    def get_calibration_state(self, arm_id: str) -> List[Dict[str, Any]]:
        arm, target_motors = self._svc._get_arm_context(arm_id)
        if not arm:
            return []

        # Read positions
        if self._svc.robot_lock:
            with self._svc.robot_lock:
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

            if self._svc.is_discovering:
                if motor_name not in self._svc.session_ranges:
                    self._svc.session_ranges[motor_name] = {"min": pos, "max": pos}

                if pos < self._svc.session_ranges[motor_name]["min"]:
                    self._svc.session_ranges[motor_name]["min"] = pos
                if pos > self._svc.session_ranges[motor_name]["max"]:
                    self._svc.session_ranges[motor_name]["max"] = pos

                visited_min = self._svc.session_ranges[motor_name]["min"]
                visited_max = self._svc.session_ranges[motor_name]["max"]
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
        arm, _ = self._svc._get_arm_context(arm_id)
        if not arm:
            return

        if motor_name not in arm.calibration:
            return

        # Damiao uses float radians, Feetech/Dynamixel use integer ticks
        converted = float(value) if self._svc._is_damiao_arm(arm_id) else int(value)

        if limit_type == "min":
            arm.calibration[motor_name].range_min = converted
        elif limit_type == "max":
            arm.calibration[motor_name].range_max = converted
