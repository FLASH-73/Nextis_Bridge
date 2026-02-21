#   Copyright 2025 Nextis. All rights reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass
from lerobot.teleoperators.teleoperator import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("dynamixel_leader")
@dataclass
class DynamixelLeaderConfig(TeleoperatorConfig):
    """Configuration for Dynamixel XL330 leader arm (7-DOF).

    This leader arm uses Dynamixel XL330 motors connected via Waveshare USB-C bus.
    Used for teleoperation of follower arms (Damiao, STS3215, etc.)
    """
    port: str = "/dev/ttyACM0"

    # Gripper position calibration (raw motor values)
    # Ticks decrease when squeezed: open=2820, closed=2280
    gripper_open_pos: int = 2820
    gripper_closed_pos: int = 2280

    # Optional: structural design hint for pairing compatibility
    structural_design: str = ""  # e.g., "damiao_7dof", "umbra_7dof"

    def __post_init__(self):
        if not self.calibration_dir:
            import warnings
            warnings.warn(
                f"calibration_dir not set for {self.__class__.__name__}. "
                f"Calibration will fall back to HF cache (~/.cache/huggingface/...). "
                f"Set calibration_dir explicitly to avoid dual-location bugs.",
                stacklevel=2,
            )
