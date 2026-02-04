# Copyright 2024 Nextis. All rights reserved.
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

"""Damiao 7-DOF follower arm for high-torque assembly tasks.

This robot uses Damiao J-series motors (J8009P, J4340P, J4310) connected
via CAN-to-serial bridge. It features a global velocity limiter for safety
when working with high-torque motors.

Safety Features:
- Global velocity_limit (0.0-1.0) applied to ALL motor commands
- Torque monitoring with configurable limits
- Gripper force limiting
- Safe disconnect with torque disable
"""

import logging
import time
import numpy as np
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.damiao import DamiaoMotorsBusConfig
from lerobot.motors.damiao.tables import GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS

from .config_damiao_follower import DamiaoFollowerConfig

logger = logging.getLogger(__name__)


def map_range(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Map value from one range to another."""
    if in_max == in_min:
        return out_min
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class DamiaoFollowerRobot(Robot):
    """Damiao 7-DOF follower arm for high-torque assembly tasks.

    This robot is designed for assembly tasks requiring high torque (up to 35Nm).
    It uses a global velocity limiter for safety.

    Motor Configuration:
    - Base/Link1: J8009P (35Nm) - high torque for base movements
    - Link2/Link3: J4340P (8Nm) - medium torque for elbow
    - Link4/Link5/Gripper: J4310 (4Nm) - precision for wrist

    Example:
        from lerobot.robots.damiao_follower import DamiaoFollowerRobot, DamiaoFollowerConfig

        config = DamiaoFollowerConfig(
            port="/dev/ttyUSB0",
            velocity_limit=0.2,  # Start at 20% for safety
        )
        robot = DamiaoFollowerRobot(config)
        robot.connect()

        # Set velocity limit (0.0-1.0)
        robot.velocity_limit = 0.5  # 50%

        # Get observation
        obs = robot.get_observation()

        # Send action
        robot.send_action({"base.pos": 0.5, "link1.pos": 0.3, ...})

        robot.disconnect()
    """

    config_class = DamiaoFollowerConfig
    name = "damiao_follower"

    def __init__(self, config: DamiaoFollowerConfig):
        super().__init__(config)
        self.config = config

        # Build motor bus config
        bus_config = DamiaoMotorsBusConfig(
            port=config.port,
            baudrate=config.baudrate,
            motors=config.motor_config,
            velocity_limit=config.velocity_limit,
        )

        self.bus = DamiaoMotorsBus(bus_config)
        self.calibration = {}  # For compatibility with calibration service

        # Gripper positions
        self.gripper_open_pos = config.gripper_open_pos
        self.gripper_closed_pos = config.gripper_closed_pos

        # Cameras
        self.cameras = make_cameras_from_configs(config.cameras)

        # Motor names for feature dictionaries
        self._motor_names = list(config.motor_config.keys())

    @property
    def velocity_limit(self) -> float:
        """Get current global velocity limit (0.0-1.0)."""
        return self.bus.velocity_limit

    @velocity_limit.setter
    def velocity_limit(self, value: float):
        """Set global velocity limit (0.0-1.0). Applied to ALL motor commands."""
        self.bus.velocity_limit = value
        logger.info(f"[DamiaoFollower] Velocity limit set to {self.bus.velocity_limit:.2f}")

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for observation/action."""
        return {f"{motor}.pos": float for motor in self._motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for observation."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Features available in observations."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Features accepted in actions."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        """Damiao motors use absolute encoders, always calibrated."""
        return True

    def calibrate(self) -> None:
        """Calibration (not needed for Damiao - absolute encoders)."""
        pass

    def configure(self) -> None:
        """Configure motors (called during connect)."""
        if self.bus.is_connected:
            self.bus.configure()

    def connect(self) -> None:
        """Connect to robot, configure motors, and home gripper."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"[DamiaoFollower] Connecting to {self.config.port}")
        logger.info(f"[DamiaoFollower] Velocity limit: {self.config.velocity_limit:.2f} ({self.config.velocity_limit*100:.0f}%)")

        # Connect motor bus
        self.bus.connect()
        self.bus.configure()

        # Home gripper (finds open position)
        if getattr(self.config, 'skip_gripper_homing', False):
            logger.info("[DamiaoFollower] Skipping gripper homing (config)")
        else:
            self.bus.home_gripper()

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"[DamiaoFollower] Connected successfully")

    def get_observation(self) -> dict[str, Any]:
        """Get current robot state and camera images.

        Returns:
            Dict with:
            - "{motor}.pos": Motor position in radians (gripper normalized 0-1)
            - "{camera}": Camera image as numpy array
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        obs_dict = {}

        # Read motor positions
        positions = self.bus.sync_read("Present_Position")
        for name, pos in positions.items():
            if name == "gripper":
                # Normalize gripper: 0 = open, 1 = closed
                norm_pos = map_range(
                    pos, self.gripper_open_pos, self.gripper_closed_pos, 0.0, 1.0
                )
                obs_dict[f"{name}.pos"] = norm_pos
            else:
                obs_dict[f"{name}.pos"] = pos

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"[DamiaoFollower] Read state: {dt_ms:.1f}ms")

        # Capture camera images
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"[DamiaoFollower] Read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send goal positions to motors.

        IMPORTANT: All motor commands are velocity-limited by self.velocity_limit.

        Args:
            action: Dict with "{motor}.pos" keys and position values.
                   Gripper expects normalized value (0=open, 1=closed).

        Returns:
            Dict with actual goal positions sent.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract goal positions
        goal_pos = {}
        for key, val in action.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                goal_pos[motor_name] = val

        # Apply joint limits
        for motor_name, limits in self.config.joint_limits.items():
            if motor_name in goal_pos:
                goal_pos[motor_name] = np.clip(goal_pos[motor_name], limits[0], limits[1])

        # Handle gripper specially (with force limiting)
        if "gripper" in goal_pos:
            gripper_norm = goal_pos.pop("gripper")
            gripper_pos = map_range(
                gripper_norm, 0.0, 1.0, self.gripper_open_pos, self.gripper_closed_pos
            )
            self.bus.send_gripper_command(gripper_pos, self.config.max_gripper_torque)

        # Send joint positions (velocity limited by bus)
        if goal_pos:
            self.bus.sync_write("Goal_Position", goal_pos)

        # Return what we sent
        result = {f"{m}.pos": v for m, v in goal_pos.items()}
        if "gripper" in action:
            result["gripper.pos"] = action.get("gripper.pos", action.get("gripper"))
        return result

    def disconnect(self) -> None:
        """Disconnect from robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Disconnect cameras first
        for cam in self.cameras.values():
            cam.disconnect()

        # Disconnect motor bus (optionally disables torque)
        self.bus.disconnect(disable_torque=self.config.disable_torque_on_disconnect)

        logger.info(f"[DamiaoFollower] Disconnected")

    def get_torques(self) -> dict[str, float]:
        """Read current torques from all motors (for safety monitoring)."""
        return self.bus.read_torques()

    def get_torque_limits(self) -> dict[str, float]:
        """Get torque limits for each motor (85% of max)."""
        return self.bus.get_torque_limits()

    def emergency_stop(self) -> None:
        """Emergency stop: disable all motor torques immediately."""
        logger.warning("[DamiaoFollower] EMERGENCY STOP - Disabling all torques")
        self.bus.disable_torque()
