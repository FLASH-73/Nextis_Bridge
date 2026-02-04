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

"""Motor specifications for Damiao J-series motors.

References:
- J8009P: https://store.foxtech.com/dm-j8009p-2ec-mit-driven-brushless-servo-joint-motor-with-dual-encoders-for-robotic-arms-actuator-for-robot/
- J4340P: https://store.foxtech.com/dm-j4340p-2ec-mit-driven-brushless-servo-joint-motor/
- J4310: https://store.foxtech.com/dm-j4310-2ec-v1-1-mit-driven-brushless-servo-joint-motor-with-dual-encoders-gear-reduction-for-robotic-arms/
"""

import numpy as np

# Motor specifications from Damiao datasheets
DAMIAO_MOTOR_SPECS = {
    # J8009P: High torque motor for base/shoulder joints
    "J8009P": {
        "max_torque": 35.0,  # Nm
        "max_rpm": 63.0,  # RPM
        "max_velocity": 63.0 / 60 * 2 * np.pi,  # rad/s (~6.6 rad/s)
        "gear_ratio": 9.0,
        "torque_constant": 0.145,  # Nm/A (estimated)
        "dm_type": "DM4340",  # Uses DM4340 protocol
        "torque_limit_percent": 0.85,  # Safety limit (85% of max)
    },
    # J4340P: Medium torque motor for elbow joints
    "J4340P": {
        "max_torque": 8.0,  # Nm
        "max_rpm": 52.5,  # RPM
        "max_velocity": 52.5 / 60 * 2 * np.pi,  # rad/s (~5.5 rad/s)
        "gear_ratio": 6.0,
        "torque_constant": 0.105,  # Nm/A (estimated)
        "dm_type": "DM4340",  # Uses DM4340 protocol
        "torque_limit_percent": 0.85,
    },
    # J4310: Precision motor for wrist joints and gripper
    "J4310": {
        "max_torque": 4.0,  # Nm
        "max_rpm": 200.0,  # RPM
        "max_velocity": 200.0 / 60 * 2 * np.pi,  # rad/s (~20.9 rad/s)
        "gear_ratio": 10.0,
        "torque_constant": 0.945,  # Nm/A
        "dm_type": "DM4310",  # Uses DM4310 protocol
        "torque_limit_percent": 0.85,
    },
}

# Default motor configuration for Nextis Damiao follower arm
# User specified: Base/Link1: J8009P, Link2/Link3: J4340P, Link4/Link5/Gripper: J4310
DEFAULT_DAMIAO_MOTORS = {
    "base": {"motor_type": "J8009P", "can_id": 0x01, "master_id": 0x11},
    "link1": {"motor_type": "J8009P", "can_id": 0x02, "master_id": 0x12},
    "link2": {"motor_type": "J4340P", "can_id": 0x03, "master_id": 0x13},
    "link3": {"motor_type": "J4340P", "can_id": 0x04, "master_id": 0x14},
    "link4": {"motor_type": "J4310", "can_id": 0x05, "master_id": 0x15},
    "link5": {"motor_type": "J4310", "can_id": 0x06, "master_id": 0x16},
    "gripper": {"motor_type": "J4310", "can_id": 0x07, "master_id": 0x17},
}

# EMIT control scaling factors (from DM_CAN library)
EMIT_VELOCITY_SCALE = 100  # rad/s
EMIT_CURRENT_SCALE = 1000  # A

# PID gains for different motor types
PID_GAINS = {
    "J8009P": {"KP_APR": 150, "KI_APR": 8, "ACC": 8.0, "DEC": -8.0},  # Lower gains for heavy motors
    "J4340P": {"KP_APR": 200, "KI_APR": 10, "ACC": 10.0, "DEC": -10.0},  # Standard gains
    "J4310": {"KP_APR": 200, "KI_APR": 10, "ACC": 10.0, "DEC": -10.0},  # Standard gains
}

# Gripper-specific settings
GRIPPER_PID = {"KP_APR": 100, "KI_APR": 5}
GRIPPER_TORQUE_THRESHOLD = 1.2  # Nm - torque spike for homing detection
GRIPPER_OPEN_POS = 0.0  # radians
GRIPPER_CLOSED_POS = -4.7  # radians (adjust based on gripper geometry)
