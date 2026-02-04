#   Copyright 2025 Nextis. All rights reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0

"""Dynamixel XL330 Leader Arm Teleoperator.

7-DOF leader arm using Dynamixel XL330 motors via Waveshare USB-C bus.
"""

from .dynamixel_leader import DynamixelLeader
from .config_dynamixel_leader import DynamixelLeaderConfig

__all__ = ["DynamixelLeader", "DynamixelLeaderConfig"]
