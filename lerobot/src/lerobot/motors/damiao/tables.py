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
        "dm_type": "DM8009",  # DM8009 protocol (NOT DM4340 — different Limit_Param)
        "torque_limit_percent": 0.40,  # Safety limit (10% of max — conservative for initial testing)
        # MIT mode encoding limits (from DM_CAN.py Limit_Param[DM8009], index 6)
        "p_max": 12.5,   # rad — position encoding range
        "v_max": 45.0,   # rad/s — velocity encoding range (was 8.0 = DM4340 WRONG)
        "t_max": 54.0,   # Nm — torque encoding range (was 28.0 = DM4340 WRONG)
    },
    # J4340P: Medium torque motor for elbow joints
    "J4340P": {
        "max_torque": 27.0,  # Nm
        "max_rpm": 52.5,  # RPM
        "max_velocity": 52.5 / 60 * 2 * np.pi,  # rad/s (~5.5 rad/s)
        "gear_ratio": 6.0,
        "torque_constant": 0.105,  # Nm/A (estimated)
        "dm_type": "DM4340",  # Uses DM4340 protocol
        "torque_limit_percent": 0.50,  # Conservative for initial testing (increase to 0.85 when confident)
        "p_max": 12.5,
        "v_max": 40.0,
        "t_max": 28.0,
    },
    # J4310: Precision motor for wrist joints and gripper
    "J4310": {
        "max_torque": 12.5,  # Nm
        "max_rpm": 200.0,  # RPM
        "max_velocity": 200.0 / 60 * 2 * np.pi,  # rad/s (~20.9 rad/s)
        "gear_ratio": 10.0,
        "torque_constant": 0.945,  # Nm/A
        "dm_type": "DM4310",  # Uses DM4310 protocol
        "torque_limit_percent": 0.30,  # Conservative for initial testing (increase to 0.85 when confident)
        "p_max": 12.5,
        "v_max": 30.0,
        "t_max": 10.0,
        # Rate limiter velocity cap: raw max_velocity (20.9 rad/s) is 3-4× higher than
        # J8009P/J4340P, causing the rate limiter to allow huge position steps that produce
        # torque spikes and oscillation on this low-inertia motor. Cap at 10.0 rad/s
        # (still 1.5× faster than J8009P) to keep torque demands proportional.
        "rate_limit_velocity": 10.0,
        # EMA smoothing: low inertia tracks 60Hz steps perfectly → jitter.
        # alpha=0.85 → 95% in ~2 frames (33ms). Increase toward 1.0 for less filtering.
        "position_smoothing": 0.85,
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

# PID gains for POS_VEL mode
# KP_APR/KI_APR: Position loop gains
# KP_ASR/KI_ASR: Velocity loop gains (CRITICAL! If these are 0, motor won't move)
# ACC/DEC: Trapezoidal acceleration/deceleration limits
PID_GAINS = {
    # IMPORTANT: All values MUST be floats! Integers get packed as uint32 which corrupts them.
    # Gains must be high enough to overcome static friction and move the motor
    "J8009P": {"KP_APR": 150.0, "KI_APR": 5.0, "KP_ASR": 2.0, "KI_ASR": 0.02, "ACC": 8.0, "DEC": -8.0},
    "J4340P": {"KP_APR": 200.0, "KI_APR": 8.0, "KP_ASR": 2.0, "KI_ASR": 0.02, "ACC": 10.0, "DEC": -10.0},
    "J4310": {"KP_APR": 200.0, "KI_APR": 8.0, "KP_ASR": 2.0, "KI_ASR": 0.02, "ACC": 10.0, "DEC": -10.0},
}

# MIT mode gains per motor type
# kp: position stiffness (0-500 range, encoded in 12 bits)
# kd: velocity damping (0-5 range, encoded in 12 bits)
MIT_GAINS = {
    "J8009P": {"kp": 30.0, "kd": 1.5},  # Heavy 35Nm motor — high stiffness, stable at this kd
    "J4340P": {"kp": 30.0, "kd": 1.5},  # Medium motor — same gains, works well
    "J4310":  {"kp": 15.0, "kd": 0.15},  # Low kd prevents torque saturation; v_des feedforward eliminates kd drag
}

# Per-motor MIT gain overrides (takes priority over per-type MIT_GAINS)
# Use when identical motor types need different gains due to load/gravity differences.
MIT_MOTOR_GAINS = {
    "base":  {"kp": 20.0, "kd": 1.0},  # No gravity load — softer to avoid gear backlash chatter
    "link3": {"kp": 22.0, "kd": 0.7},  # Lower payload than link2 — softer gains
}

# Default joint limits (radians) — safety fallback, matches cal_test_7_02_1821 calibration
# These are overridden at startup by calibration from HF cache if available.
DEFAULT_JOINT_LIMITS = {
    "base": (-1.59, 1.57),
    "link1": (-1.98, 1.97),
    "link2": (-1.18, 3.96),
    "link3": (-2.12, 1.95),
    "link4": (-2.78, 0.80),
    "link5": (-3.17, 3.08),
    "gripper": (-5.32, 0.0),
}

# Gripper-specific settings
GRIPPER_PID = {"KP_APR": 100, "KI_APR": 5}
GRIPPER_TORQUE_THRESHOLD = 5  # Nm - torque spike for homing detection
GRIPPER_OPEN_POS = 0.0  # radians
GRIPPER_CLOSED_POS = -4.7  # radians (adjust based on gripper geometry)
