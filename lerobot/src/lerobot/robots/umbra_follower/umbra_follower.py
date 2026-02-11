from dataclasses import dataclass, field
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig
from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode, MotorCalibration
from functools import cached_property
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
import logging
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots import Robot
import time
from typing import Any
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from ..utils import ensure_safe_goal_position
from .config_umbra_follower import UmbraFollowerConfig

logger = logging.getLogger(__name__)


class UmbraFollowerRobot(Robot):
    """Robot class for Umbra Follower arm."""
    config_class = UmbraFollowerConfig
    name = "umbra_follower"

  
    def __init__(self, config: UmbraFollowerConfig):
        super().__init__(config)
        self.config = config
        if self.config.arm_side not in ["left", "right"]:
             raise ValueError(f"arm_side must be 'left' or 'right', got {self.config.arm_side}")
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        dual_joints = ["link1", "link2"]
        single_joints = ["base", "link3", "link4", "link5"]
        follower_motors = {
            f"{joint}_follower": Motor(id_val, "sts3215", norm_mode_body)
            for joint, id_val in zip(dual_joints, [3, 5])
        }
        leader_motors = {
            joint: Motor(id_val, "sts3215", norm_mode_body)
            for joint, id_val in zip(dual_joints, [2, 4])
        }
        single_motors = {
            joint: Motor(id_val, "sts3215", norm_mode_body)
            for joint, id_val in zip(single_joints, [1, 6, 7, 8])
        }
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                **single_motors,
                **leader_motors,
                "gripper": Motor(9, "sts3215", MotorNormMode.RANGE_0_100),
                **follower_motors,
            },
            calibration=self.calibration,
        )
        self.dual_joints = dual_joints
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # Initialize Inversions
        self.reload_inversions()

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors if not motor.endswith("_follower")}
                                        
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        cam_ft = {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
        for cam in self.cameras:
            if hasattr(self.config.cameras[cam], 'use_depth') and self.config.cameras[cam].use_depth:
                cam_ft[f"{cam}_depth"] = (self.config.cameras[cam].height, self.config.cameras[cam].width, 1)
        return {**self._motors_ft, **cam_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        logger.info(f"Current motors keys: {list(self.bus.motors.keys())}")
        logger.info(f"Calibration keys: {list(self.calibration.keys()) if self.calibration else 'None'}")
        self.bus.connect()
        if self.calibration:
            self.bus.disable_torque()
            self.bus.write_calibration(self.calibration)
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")
        # Calibrate Dual Offsets (Instantaneous Snapshot)
        # This allows us to drive dual motors in "Write-Only" mode (fast) without reading positions every frame.
        # Logic: Link1 + Link1_Follower = Constant (K)
        # So: Follower_Goal = K - Leader_Goal
        if self.is_calibrated:
             logger.info("Calibrating Dual Offsets for High-Speed Teleop...")
             # Read once at startup
             current_pos = self.bus.sync_read("Present_Position", num_retry=5)
             self.dual_offsets = {}
             for joint in self.dual_joints:
                 follower = f"{joint}_follower"
                 if joint in current_pos and follower in current_pos:
                      # If they are geared oppositely, sum is constant.
                      # K = Leader + Follower
                      k = current_pos[joint] + current_pos[follower]
                      self.dual_offsets[joint] = k
                      logger.info(f"Dual Offset for {joint}: {k}")
                 else:
                      logger.warning(f"Could not calibrate dual offset for {joint} (Missing data). Fallback to slow mode?")
        else:
             self.dual_offsets = {}
             
        self.bus.enable_torque()
      

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        # Exclude gripper from range recording as it has its own calibration
        print(
            "Move all joints (including gripper) sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)
        self.bus.enable_torque()

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)  # 50% of max torque to avoid burnout
                    self.bus.write("Protection_Current", motor, 250)  # 50% of max current to avoid burnout
                    self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_observation(self, include_images: bool = True) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

        # Apply inversions to return LOGICAL positions (matching send_action expectations)
        # This ensures consistency: if send_action inverts before sending to hardware,
        # get_observation must also invert when reading from hardware.
        # Without this, recording captures RAW values but send_action expects LOGICAL values.
        if hasattr(self, 'motor_inversions') and self.motor_inversions:
            for motor, is_inverted in self.motor_inversions.items():
                key = f"{motor}.pos"
                if is_inverted and key in obs_dict:
                    if motor == "gripper":
                        obs_dict[key] = 100 - obs_dict[key]
                    else:
                        obs_dict[key] = -obs_dict[key]
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        if include_images:
            for cam_key, cam in self.cameras.items():
                start = time.perf_counter()
                obs_dict[cam_key] = cam.async_read()
                # Capture depth if enabled for this camera
                if hasattr(cam.config, 'use_depth') and cam.config.use_depth:
                    try:
                        obs_dict[f"{cam_key}_depth"] = cam.async_read_depth(blocking=False)
                    except Exception as e:
                        logger.debug(f"Depth read failed for {cam_key}: {e}")
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict
    def reload_inversions(self):
        """Loads motor inversion settings from disk."""
        import json
        from pathlib import Path

        # Compute project root relative to this file
        # umbra_follower.py → umbra_follower/ → robots/ → lerobot/ → src/ → lerobot/ → PROJECT_ROOT
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
        # Use arm_registry ID (e.g. "umbra_follower") when available, fall back to legacy "{arm_side}_follower"
        profile_name = self.config.id if self.config.id else f"{self.config.arm_side}_follower"
        profile_path = project_root / "calibration_profiles" / profile_name / "inversions.json"
        
        self.motor_inversions = {}
        
        if profile_path.exists():
            try:
                with open(profile_path, "r") as f:
                    self.motor_inversions = json.load(f)
                logger.info(f"Loaded Inversions for {profile_name}: {self.motor_inversions}")
            except Exception as e:
                logger.error(f"Failed to load inversions from {profile_path}: {e}")
        else:
            logger.warning(f"No inversion config found at {profile_path}. Using hardcoded defaults.")
            # Fallback to the hardcoded defaults we just removed, for safety if file is missing
            if self.config.arm_side == "left":
                 self.motor_inversions = {
                     "gripper": True,
                     "link4": True,
                     "link2_follower": True,
                     "link2": True
                 }
            elif self.config.arm_side == "right":
                 self.motor_inversions = {
                     "link1": True,
                     "link1_follower": True,
                     "link2": True,
                     "link2_follower": True,
                     "link4": True,
                     "gripper": False
                 }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        #print("Follower:goal_pos", goal_pos)  # for debugging

        # ### DYNAMIC INVERSIONS (Moved to Top)
        # Apply inversions loaded from disk (or use arm-specific defaults if config missing)
        # MUST happen before Dual Motor Logic so that K-calculation uses the intended physical target.
        for motor, should_invert in self.motor_inversions.items():
            if should_invert and motor in goal_pos:
                # SKIP Follower Motors if they are part of a dual joint handled by K-logic?
                # If we invert here, goal_pos is updated.
                # Then Dual Logic uses goal_pos[joint].
                # If joint was inverted, Dual Logic uses Inverted value. Correct.
                
                # Handling "Inverted Follower" setting:
                # If User checked 'link1_follower' inversion, usually they mean "This motor is backwards".
                # But K logic assumes Raw + Raw = K.
                # If motor is physically backwards, Raw value logic might hold if K was calibrated in that state?
                # Yes, K = InvertedLeaderRaw + InvertedFollowerRaw.
                # If we send InvertedLeaderTarget, we need InvertedFollowerTarget.
                # So we should probably NOT invert follower explicitly here if K logic is used.
                # But if K logic is disabled (fallback), we might validly invert it.
                # SAFEST: Only invert Non-Follower (Leader) motors here for Dual Setup?
                # Or trust that users won't invert follower if using K logic?
                # Let's apply normally. If User inverts Leader, Leader is inverted. Follower calculated from Leader.
                # If User ALSO inverts Follower, goal_pos[follower] is inverted... but follower isn't in goal_pos yet!
                # So this loop only affects motors currently in goal_pos (Leaders & Single).
                # Follower joints are NOT yet in goal_pos. Excellent.
                
                # Invert: Goal = 100 - Goal (Percent) OR Goal = -Goal (Degrees)
                if motor == "gripper":
                     goal_pos[motor] = 100 - goal_pos[motor]
                else:
                     goal_pos[motor] = -goal_pos[motor]

        # Optimization: Use pre-calculated dual offsets if available (Write-Only Mode)
        # This skips the slow sync_read() and makes teleop instant.
        if self.dual_offsets:
             # Fast Path
             for joint in self.dual_joints:
                 if joint in goal_pos and joint in self.dual_offsets:
                     # Follower = K - Leader
                     k = self.dual_offsets[joint]
                     goal_node = goal_pos[joint]
                     follower_goal = k - goal_node
                     goal_pos[f"{joint}_follower"] = follower_goal
        else:
             # Fallback: Slow Read-Write Path (Reliable but laggy)
             # Always read present positions to compute deltas for dual joints and handle capping if enabled.
             # Use retry to ensure we get data, otherwise dual motors will fight (one moves, one stays).
             present_pos = self.bus.sync_read("Present_Position", num_retry=2)
     
             # Cap goal position when too far away from present position.
             # /!\ Slower fps expected due to reading from the follower.
             if self.config.max_relative_target is not None:
                 goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
                 goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
     
             # Set mirrored goals for follower motors in dual joints using deltas to handle offsets.
             for joint in self.dual_joints:
                 if joint in goal_pos:
                     if joint in present_pos and f"{joint}_follower" in present_pos:
                         delta = goal_pos[joint] - present_pos[joint]
                         goal_pos[f"{joint}_follower"] = present_pos[f"{joint}_follower"] - delta
                     else:
                         # SAFETY CRITICAL: If we cannot read the follower state, we MUST NOT move the leader joint.
                         # Otherwise, the leader moves while follower holds, causing massive torque fighting and stall.
                         logger.warning(f"Dual Joint Sync Failed for {joint}: Missing present_pos. Aborting move for this joint.")
                         goal_pos.pop(joint, None)
                         goal_pos.pop(f"{joint}_follower", None)

        # Default speed value (0-1023; 300 is a safe starting point based on your GUI default.
        # 0 often means "maximum speed" in Feetech protocols—test what works for your servos.
        # Higher values = slower speed in some configs; consult STS3215 manual for units.
        default_speed = 500  # Adjust based on testing (e.g., 0 for max speed, or 200-500 for controlled movement)

        # Set moving speed for all motors being commanded (same keys as goal_pos)
        speed_goals = {motor: default_speed for motor in goal_pos}
        self.bus.sync_write("Goal_Velocity", speed_goals)  # Correct register name for Feetech STS series

        # [TEMPORARY DEBUG] Gripper trace — once per second
        import time as _time
        if 'gripper' in goal_pos:
            if not hasattr(self, '_gripper_dbg_t') or _time.time() - self._gripper_dbg_t > 1.0:
                self._gripper_dbg_t = _time.time()
                print(f"[GRIPPER] send_action: goal_pos[gripper]={goal_pos['gripper']:.1f}", flush=True)
        elif not hasattr(self, '_gripper_miss_warned'):
            self._gripper_miss_warned = True
            print(f"[GRIPPER] WARNING: 'gripper' not in goal_pos! Keys: {list(goal_pos.keys())}", flush=True)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items() if motor in self.bus.motors and not motor.endswith("_follower")}
    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")