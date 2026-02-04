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

"""Configuration for Damiao 7-DOF follower arm."""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot.motors.damiao.tables import DEFAULT_DAMIAO_MOTORS


@RobotConfig.register_subclass("damiao_follower")
@dataclass
class DamiaoFollowerConfig(RobotConfig):
    """Configuration for Damiao 7-DOF follower arm.

    Motor Configuration (default):
    - Base/Link1: J8009P (35Nm high torque) - for heavy base movements
    - Link2/Link3: J4340P (8Nm medium torque) - for elbow movements
    - Link4/Link5/Gripper: J4310 (4Nm precision) - for wrist and gripper

    Safety Features:
    - velocity_limit: Global velocity limiter (0.0-1.0), default 0.2 (20%)
    - max_gripper_torque: Force limit for gripper

    Example:
        config = DamiaoFollowerConfig(
            port="/dev/ttyUSB0",
            velocity_limit=0.2,  # Start safe at 20%
        )
    """

    # Serial port for CAN-to-serial adapter
    port: str = "/dev/ttyUSB0"
    baudrate: int = 921600

    # Motor configuration (see tables.py for defaults)
    # Keys: base, link1, link2, link3, link4, link5, gripper
    # Values: dict with motor_type, can_id, master_id
    motor_config: dict = field(default_factory=lambda: DEFAULT_DAMIAO_MOTORS.copy())

    # CRITICAL: Global velocity limiter (0.0-1.0)
    # Default 0.1 (10%) for safety with heavy arm
    velocity_limit: float = 0.1

    # Gripper settings
    max_gripper_torque: float = 1.0  # Nm
    gripper_open_pos: float = 0.0  # radians
    gripper_closed_pos: float = -4.7  # radians (adjust for your gripper)
    skip_gripper_homing: bool = False  # Skip torque-based gripper homing on connect

    # Safety
    disable_torque_on_disconnect: bool = True

    # Cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Joint limits in radians (from calibration_profiles/aira_zero/cal1_test.json)
    # Format: {"joint_name": (min, max)}
    joint_limits: dict = field(default_factory=lambda: {
        "base": (-1.98, 1.84),
        "link1": (-1.58, 1.89),
        "link2": (-1.36, 1.08),
        "link3": (-1.52, 2.84),
        "link4": (-0.64, 2.51),
        "link5": (-2.78, 2.54),
        "gripper": (-5.32, 0.0),
    })

    # Calibration file path (if using external calibration)
    calibration_path: str = ""
