#   Copyright 2025 Nextis. All rights reserved.
#   Based on DK1Leader from examples_for_damiao.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0

"""
Dynamixel XL330 Leader Arm Teleoperator.

This module provides a 7-DOF leader arm implementation using Dynamixel XL330 motors
connected via Waveshare USB-C serial bus. Used for teleoperation of follower arms.

Motor Configuration (7-DOF):
- joint_1 (base):     XL330-M077, ID=1
- joint_2 (shoulder): XL330-M077, ID=2
- joint_3 (elbow):    XL330-M077, ID=3
- joint_4 (wrist1):   XL330-M077, ID=4
- joint_5 (wrist2):   XL330-M077, ID=5
- joint_6 (wrist3):   XL330-M077, ID=6
- gripper:            XL330-M077, ID=7
"""

import logging
import time
import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus, OperatingMode

from .config_dynamixel_leader import DynamixelLeaderConfig

logger = logging.getLogger(__name__)


class DynamixelLeader(Teleoperator):
    """Dynamixel XL330 7-DOF leader arm for teleoperation.

    Uses DynamixelMotorsBus for communication with XL330 motors via
    Waveshare USB-C serial adapter.
    """

    config_class = DynamixelLeaderConfig
    name = "dynamixel_leader"

    def __init__(self, config: DynamixelLeaderConfig):
        super().__init__(config)
        self.config = config

        # Create motor bus with 7 Dynamixel XL330 motors
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "joint_1": Motor(1, "xl330-m077", MotorNormMode.DEGREES),
                "joint_2": Motor(2, "xl330-m077", MotorNormMode.DEGREES),
                "joint_3": Motor(3, "xl330-m077", MotorNormMode.DEGREES),
                "joint_4": Motor(4, "xl330-m077", MotorNormMode.DEGREES),
                "joint_5": Motor(5, "xl330-m077", MotorNormMode.DEGREES),
                "joint_6": Motor(6, "xl330-m077", MotorNormMode.DEGREES),
                "gripper": Motor(7, "xl330-m077", MotorNormMode.DEGREES),
            },
        )

    @property
    def action_features(self) -> dict[str, type]:
        """Features returned by get_action()."""
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        """Features expected by send_feedback() - not implemented."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if motor bus is connected."""
        return self.bus.is_connected

    def connect(self, calibrate: bool = False) -> None:
        """Connect to the Dynamixel motor bus.

        Args:
            calibrate: Ignored for Dynamixel (uses absolute encoders)
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()

        # Sync loaded calibration to bus (populates _software_homing_offsets + position limits)
        # Must happen before configure() so offsets are available for gripper setup
        # Disable torque first — gripper may still have torque ON from previous session
        if self.calibration:
            self.bus.disable_torque()
            self.bus.write_calibration(self.calibration)

        self.configure()

        logger.info(f"{self} connected on port {self.config.port}")

    @property
    def is_calibrated(self) -> bool:
        """Dynamixel motors use absolute encoders, always calibrated."""
        return True

    def calibrate(self) -> None:
        """No calibration needed for Dynamixel (absolute encoders)."""
        pass

    def configure(self) -> None:
        """Configure motors for leader arm operation.

        - Disables torque on arm joints (free movement for teleoperation)
        - Configures gripper in current-position mode with limited current
        - Reads Homing_Offset to adjust gripper open/closed positions to homed coordinate space
        """
        # Disable torque on all motors for free movement
        self.bus.disable_torque()
        self.bus.configure_motors()

        # Get software homing offset (motor register is always 0 now — offset applied in Python)
        gripper_id = self.bus.motors["gripper"].id
        offset = self.bus._software_homing_offsets.get(gripper_id, 0)

        # Use calibrated range if available (config defaults may be outside position limits)
        # For our arm: lower ticks = open, higher ticks = closed
        if self.calibration and "gripper" in self.calibration:
            cal = self.calibration["gripper"]
            self._gripper_open = cal.range_min + offset    # open end (lower ticks)
            self._gripper_closed = cal.range_max + offset   # closed end (higher ticks)
            spring_target = cal.range_min                   # spring to open (within pos limits)
        else:
            self._gripper_open = self.config.gripper_open_pos + offset
            self._gripper_closed = self.config.gripper_closed_pos + offset
            spring_target = self.config.gripper_open_pos

        # Configure gripper for controlled movement (spring to open position)
        self.bus.write("Torque_Enable", "gripper", 0, normalize=False)
        self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value, normalize=False)
        self.bus.write("Current_Limit", "gripper", 100, normalize=False)
        self.bus.write("Torque_Enable", "gripper", 1, normalize=False)
        self.bus.write("Goal_Position", "gripper", spring_target, normalize=False)

    def setup_motors(self) -> None:
        """Interactive motor ID setup.

        Guides user through setting motor IDs one at a time.
        Connect ONLY ONE motor at a time when running this.
        """
        for motor in self.bus.motors:
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        """Read current joint positions from all motors.

        Returns:
            Dictionary mapping motor names to positions in radians.
            Gripper position is normalized to 0-1 range (0=closed, 1=open).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # Read raw positions from all motors
        action = self.bus.sync_read(normalize=False, data_name="Present_Position")

        # Convert to radians (Dynamixel: 0-4096 = 0-360 degrees)
        action = {
            f"{motor}.pos": (val / 4096 * 2 * np.pi - np.pi) if motor != "gripper" else val
            for motor, val in action.items()
        }

        # Normalize gripper position to 0-1 range (0=open, 1=closed)
        # Use offset-adjusted positions (set by configure() after reading Homing_Offset)
        open_pos = getattr(self, '_gripper_open', self.config.gripper_open_pos)
        closed_pos = getattr(self, '_gripper_closed', self.config.gripper_closed_pos)
        gripper_range = open_pos - closed_pos
        action["gripper.pos"] = 1 - (action["gripper.pos"] - closed_pos) / gripper_range

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Send haptic feedback to the leader arm.

        Not yet implemented for Dynamixel leader.
        """
        raise NotImplementedError("Force feedback not implemented for Dynamixel leader")

    def disconnect(self) -> None:
        """Disconnect from the motor bus."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")

    def __repr__(self) -> str:
        return f"DynamixelLeader(port={self.config.port})"
