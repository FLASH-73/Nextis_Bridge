import logging
from typing import Dict

from app.core.config import CALIBRATION_DIR

logger = logging.getLogger(__name__)


class CalibrationHoming:
    """Homing, inversions, zero-pose, and auto-alignment operations.

    Accesses shared state via self._svc (the parent CalibrationService).
    """

    # Dynamixel leader → Damiao follower joint name translation
    _DYNAMIXEL_TO_DAMIAO = {
        "joint_1": "base", "joint_2": "link1", "joint_3": "link2",
        "joint_4": "link3", "joint_5": "link4", "joint_6": "link5",
        "gripper": "gripper",
    }
    _DAMIAO_TO_DYNAMIXEL = {v: k for k, v in _DYNAMIXEL_TO_DAMIAO.items()}

    def __init__(self, svc):
        self._svc = svc

    def perform_homing(self, arm_id: str):
        arm, target_motors = self._svc.get_arm_context(arm_id)
        if not arm:
            logger.error(f"perform_homing: Invalid arm_id {arm_id} or arm not found.")
            return {"status": "error", "message": "Arm not found"}

        logger.info(f"Performing Homing for {arm_id}")

        # Ensure we have a calibration object to update offsets into
        self._svc._ensure_calibration_initialized(arm)

        try:
            # Damiao: use CAN set_zero_position command (0xFE) — absolute encoders, no offset register
            if self._svc._is_damiao_arm(arm_id):
                return self._perform_damiao_homing(arm, arm_id, target_motors)

            # Feetech/Dynamixel: use standard LeRobot homing (half-turn offset)
            offsets = {}
            if self._svc.robot_lock:
                with self._svc.robot_lock:
                     motors_to_home = [m for m in target_motors if m in arm.bus.motors]
                     arm.bus.disable_torque(motors_to_home)
                     offsets = arm.bus.set_half_turn_homings(motors_to_home)
            else:
                 motors_to_home = [m for m in target_motors if m in arm.bus.motors]
                 arm.bus.disable_torque(motors_to_home)
                 offsets = arm.bus.set_half_turn_homings(motors_to_home)

            # Persist homing offsets to arm.calibration so save_calibration() writes them
            self._svc._ensure_calibration_initialized(arm)
            for motor_name, offset in offsets.items():
                if motor_name in arm.calibration:
                    arm.calibration[motor_name].homing_offset = offset
            # Also sync to the bus calibration cache
            arm.bus.calibration = dict(arm.calibration)

            # Update gripper offset-adjusted positions for get_action() normalization
            # Do NOT call arm.configure() here — it re-enables gripper spring, blocking range discovery
            if self._svc._is_dynamixel_arm(arm_id) and hasattr(arm.bus, '_software_homing_offsets'):
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

        if self._svc.robot_lock:
            with self._svc.robot_lock:
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
        base_dir = CALIBRATION_DIR / arm_id
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
        if not self._svc.arm_registry:
            return None, False
        arm_def = self._svc.arm_registry.arms.get(arm_id)
        if not arm_def or arm_def.role.value != "leader":
            return None, False
        for p in self._svc.arm_registry.pairings:
            if p.leader_id == arm_id:
                follower_def = self._svc.arm_registry.arms.get(p.follower_id)
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
        arm, _ = self._svc.get_arm_context(target_id)
        if arm and hasattr(arm, "reload_inversions"):
            arm.reload_inversions()

    def set_zero_pose(self, arm_id: str):
        """Captures the current position as the baseline 'Zero' for alignment. Also disables torque."""
        # For arm registry arms, just disable torque on this arm directly
        if self._svc.arm_registry and arm_id in self._svc.arm_registry.arms:
            self._svc.disable_torque(arm_id)
        else:
            # Legacy: infer pair to ensure both are loose for the user to move
            parts = arm_id.split("_")
            side = parts[0]
            leader_id = f"{side}_leader"
            follower_id = f"{side}_follower"
            self._svc.disable_torque(leader_id)
            self._svc.disable_torque(follower_id)

        arm, _ = self._svc.get_arm_context(arm_id)
        if not arm: return False

        # Read all raw positions
        positions = arm.bus.sync_read("Present_Position", normalize=False)

        # Store in session memory
        if not hasattr(self._svc, "alignment_zeros"):
            self._svc.alignment_zeros = {}

        self._svc.alignment_zeros[arm_id] = positions
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
        parts = arm_id.split("_")
        side = parts[0] # left or right

        # Determine IDs
        leader_id = f"{side}_leader"
        follower_id = f"{side}_follower"

        # Get contexts
        l_arm, _ = self._svc.get_arm_context(leader_id)
        f_arm, _ = self._svc.get_arm_context(follower_id)

        if not l_arm or not f_arm:
             logger.error("AutoAlign: Could not find both Leader and Follower arms.")
             return {"status": "error", "message": "Leader or Follower missing."}

        # Check Zeros
        if not hasattr(self._svc, "alignment_zeros"):
             return {"status": "error", "message": "Zero Pose not set."}

        l_zeros = self._svc.alignment_zeros.get(leader_id)
        f_zeros = self._svc.alignment_zeros.get(follower_id)

        if not l_zeros or not f_zeros:
             return {"status": "error", "message": "Zero Pose missing for one or both arms."}

        # Read Current
        l_current = l_arm.bus.sync_read("Present_Position", normalize=False)
        f_current = f_arm.bus.sync_read("Present_Position", normalize=False)

        changes = {}
        inversions = self.get_inversions(follower_id) # Current inversions for follower

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

            # If signs differ, they are moving opposite → needs inversion
            is_inverted = (l_delta * f_delta) < 0

            if is_inverted != inversions.get(f_name, False):
                 inversions[f_name] = is_inverted
                 changes[f_name] = f"SET INVERTED={is_inverted}"
                 inverted_count += 1
            else:
                 changes[f_name] = "ok"

            # Also handle dual followers (link1_follower, link2_follower)
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
             self.set_inversion(follower_id, "batch_update", False) # Trigger file path retrieval
             fpath = self._get_inversions_file(follower_id)
             with open(fpath, "w") as f:
                 import json
                 json.dump(inversions, f, indent=2)

             # Reload robot
             f_arm.reload_inversions(inversions)

        return {"status": "success", "inverted_count": inverted_count, "changes": changes}
