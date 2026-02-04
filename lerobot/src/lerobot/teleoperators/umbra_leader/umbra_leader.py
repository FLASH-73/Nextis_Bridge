#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.teleoperators import Teleoperator  # Changed from Robot
#from ..utils import ensure_safe_goal_position
from .config_umbra_leader import UmbraLeaderConfig  # Ensure this subclasses TeleoperatorConfig

logger = logging.getLogger(__name__)


class UmbraLeaderRobot(Teleoperator):  # Changed to Teleoperator
    """
    Umbra Leader Arm - a simple 7-servo arm with 6 standard joints and 1 gripper control servo.
    All servos are Feetech STS3215. No dual-motor joints. No cameras.
    """

    config_class = UmbraLeaderConfig
    name = "umbra_leader"

    def __init__(self, config: UmbraLeaderConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "base": Motor(1, "sts3215", norm_mode_body),
                "link1": Motor(2, "sts3215", norm_mode_body),
                "link2": Motor(3, "sts3215", norm_mode_body),
                "link3": Motor(4, "sts3215", norm_mode_body),
                "link4": Motor(5, "sts3215", norm_mode_body),
                "link5": Motor(6, "sts3215", norm_mode_body),
                "gripper": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return self._motors_ft

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft
    @property
    def feedback_features(self) -> dict[str, type]:
        return {}
    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

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
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # self.calibration is not empty here
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

        print(
            "Move all joints (including gripper control) sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        
        # Override: Record RAW ranges (0-4095) for Feetech
        range_mins = {motor: float("inf") for motor in self.bus.motors}
        range_maxes = {motor: float("-inf") for motor in self.bus.motors}
        
        try:
            while True:
                # Read raw positions
                present_pos = self.bus.sync_read("Present_Position", normalize=False)
                for motor, pos in present_pos.items():
                    if pos < range_mins[motor]:
                        range_mins[motor] = pos
                    if pos > range_maxes[motor]:
                        range_maxes[motor] = pos
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("Stopped recording ranges.")

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

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

 

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        #print("Leader:action", action)  # print statement is purely for debugging
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        logger.info(f"{self} disconnected.")