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

"""Damiao motor bus implementation for CAN-based J-series motors.

This module provides control for Damiao J8009P, J4340P, and J4310 motors
via CAN-to-serial bridge or SocketCAN. Key safety feature: global velocity limiter.

Supports two connection modes:
- Serial: CAN-to-serial bridge (e.g. /dev/ttyUSB0 at 921600 baud)
- SocketCAN: Native Linux CAN interface (e.g. can0 via gs_usb adapter)
"""

import logging
import struct
import serial
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .tables import (
    DAMIAO_MOTOR_SPECS,
    DEFAULT_DAMIAO_MOTORS,
    DEFAULT_JOINT_LIMITS,
    EMIT_VELOCITY_SCALE,
    EMIT_CURRENT_SCALE,
    PID_GAINS,
    MIT_GAINS,
    MIT_MOTOR_GAINS,
    GRIPPER_PID,
    GRIPPER_TORQUE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# --- SocketCAN support classes ---

class _SocketCANMotor:
    """Motor object for SocketCAN mode, compatible with trlc_dk1 Motor interface."""
    def __init__(self, can_id, master_id, v_max=30.0, t_max=10.0):
        self.SlaveID = can_id
        self.MasterID = master_id
        self.state_q = 0.0
        self.state_dq = 0.0
        self.state_tau = 0.0
        self.last_seen = 0.0
        self.error_flags = 0
        self.param_dict = {}
        # Motor-specific encoding limits for response decoding
        # p_max=12.5 is the same for all motor types
        self.v_max = v_max  # DM4310: 30.0, DM4340: 8.0
        self.t_max = t_max  # DM4310: 10.0, DM4340: 28.0

    def getPosition(self): return self.state_q
    def getVelocity(self): return self.state_dq
    def getTorque(self): return self.state_tau


class _SocketCANDMVariable:
    """DM_variable constants for SocketCAN mode (register IDs).

    From official DM_CAN.py DM_variable IntEnum.
    RID 25-26: Velocity (speed) loop PID gains
    RID 27-28: Position loop PID gains
    """
    MST_ID = 7
    ESC_ID = 8
    TIMEOUT = 9       # Motor-internal command timeout (ms). Too short → timeout protection oscillation.
    CTRL_MODE = 10
    Damp = 11         # Motor-internal damping (applied on top of MIT kd)
    ACC = 4
    DEC = 5
    # Velocity loop (inner loop) - REQUIRED for POS_VEL mode!
    KP_ASR = 25       # Speed loop Kp
    KI_ASR = 26       # Speed loop Ki
    # Position loop (outer loop)
    KP_APR = 27       # Position loop Kp
    KI_APR = 28       # Position loop Ki


class _SocketCANControlType:
    """Control_Type constants for SocketCAN mode."""
    MIT = 1
    POS_VEL = 2
    VEL = 3
    Torque_Pos = 4


class _SocketCANControl:
    """MotorControl adapter for SocketCAN, providing same interface as trlc_dk1.MotorControl.

    Uses python-can with SocketCAN backend to communicate with Damiao motors.
    """

    # RIDs that use uint32 encoding (all others use float)
    INT_RIDS = set(range(7, 11)) | set(range(13, 17)) | {35, 36}

    def __init__(self, channel, bitrate=1000000):
        import can
        self.bus = can.interface.Bus(channel=channel, bustype='socketcan', bitrate=bitrate)
        self.motors = {}        # SlaveID -> motor
        self.master_map = {}    # MasterID -> motor (for arb_id=0 responses)
        self.lock = threading.Lock()
        self.running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def addMotor(self, motor):
        with self.lock:
            self.motors[motor.SlaveID] = motor
            self.master_map[motor.MasterID] = motor

    def _recv_loop(self):
        while self.running and self.bus:
            try:
                msg = self.bus.recv(timeout=1.0)
                if msg:
                    self._parse_msg(msg)
            except Exception as e:
                if self.running:
                    logger.debug(f"SocketCAN recv error: {e}")

    def _parse_msg(self, msg):
        can_id = msg.arbitration_id
        data = msg.data

        # Single lock acquisition for the entire parse — prevents TOCTOU race
        # conditions that occurred when lock was acquired/released 4 separate times.
        target_motor = None
        is_param_response = False

        with self.lock:
            is_status_from_known_motor = can_id in self.master_map

            # Check for parameter write/read response.
            # Parameter responses have data[2] = 0x33 (read) or 0x55 (write).
            # DISAMBIGUATION: Status responses can coincidentally have data[2] == 0x33/0x55
            # when position encoding matches those values. To distinguish:
            # - Param responses have SlaveID in data[0:2], and for our motors SlaveID < 256,
            #   so data[1] (SlaveID high byte) must be 0x00.
            # - Status responses have position_high in data[1], which is 0x00 only when the
            #   motor is near -12.5 rad (encoding minimum). This is rare but possible.
            if len(data) == 8 and data[2] in (0x33, 0x55):
                slave_id = (data[1] << 8) | data[0]
                is_real_param_response = (
                    data[1] == 0x00  # SlaveID high byte must be 0 for our motors
                    and slave_id in self.motors  # Known motor SlaveID
                )
                if is_real_param_response:
                    motor = self.motors.get(slave_id) or self.motors.get(can_id)
                    if motor:
                        rid = data[3]
                        if rid in self.INT_RIDS:
                            val = struct.unpack('<I', bytes(data[4:8]))[0]
                        else:
                            val = struct.unpack('<f', bytes(data[4:8]))[0]
                        motor.param_dict[rid] = val
                    is_param_response = True

            # Status response: motor state (position/velocity/torque)
            if not is_param_response:
                if is_status_from_known_motor:
                    target_motor = self.master_map[can_id]
                elif len(data) >= 1:
                    # Fallback: extract SlaveID from response ID_BYTE (data[0] lower nibble).
                    # Needed when motor responds with arb_id != MasterID (e.g., arb_id=0
                    # when MasterID not configured on motor hardware).
                    esc_id = data[0] & 0x0F
                    if esc_id in self.motors:
                        target_motor = self.motors[esc_id]
                if not target_motor and can_id in self.motors:
                    target_motor = self.motors[can_id]

        # Update motor state outside lock — float assignments are CPython-atomic
        if target_motor and len(data) == 8:
            target_motor.last_seen = time.time()
            target_motor.error_flags = (data[0] >> 4) & 0x0F

            q_uint = (data[1] << 8) | data[2]
            dq_uint = (data[3] << 4) | (data[4] >> 4)
            tau_uint = ((data[4] & 0x0F) << 8) | data[5]

            target_motor.state_q = self._uint_to_float(q_uint, -12.5, 12.5, 16)
            target_motor.state_dq = self._uint_to_float(dq_uint, -target_motor.v_max, target_motor.v_max, 12)
            target_motor.state_tau = self._uint_to_float(tau_uint, -target_motor.t_max, target_motor.t_max, 12)

    def _send(self, arb_id, data):
        import can
        msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)
        # Retry with backoff: "Transmit buffer full" is transient (gs_usb TX URB pool)
        for attempt in range(3):
            try:
                self.bus.send(msg)
                return True
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.001)  # 1ms backoff before retry
                    continue
                # All retries exhausted — log and return failure
                if not hasattr(self, '_send_error_count'):
                    self._send_error_count = 0
                self._send_error_count += 1
                if self._send_error_count <= 5 or self._send_error_count % 100 == 0:
                    print(f"[Damiao CAN] SEND FAILED arb_id=0x{arb_id:03X}: {e} "
                          f"(total={self._send_error_count}, 3 retries exhausted)", flush=True)
                return False

    def _uint_to_float(self, x, x_min, x_max, bits):
        span = x_max - x_min
        data_norm = float(x) / ((1 << bits) - 1)
        return data_norm * span + x_min

    # --- Interface methods matching trlc_dk1.MotorControl ---

    def refresh_motor_status(self, motor):
        """Request motor status via 0xCC command (matches trlc_dk1 protocol).

        Sends status request to CAN arb_id 0x7FF, motor responds with its
        current position/velocity/torque which is parsed by the recv thread.
        """
        slave_id = motor.SlaveID
        can_id_l = slave_id & 0xFF
        can_id_h = (slave_id >> 8) & 0xFF
        data = bytes([can_id_l, can_id_h, 0xCC, 0x00, 0x00, 0x00, 0x00, 0x00])
        self._send(0x7FF, data)
        # Brief wait for response to arrive via recv thread
        time.sleep(0.002)

    def read_motor_param(self, motor, rid):
        """Read a motor parameter by RID. Returns value or None on timeout."""
        motor.param_dict.pop(rid, None)
        can_id_l = motor.SlaveID & 0xFF
        can_id_h = (motor.SlaveID >> 8) & 0xFF
        data = bytearray([can_id_l, can_id_h, 0x33, rid, 0, 0, 0, 0])
        self._send(0x7FF, data)
        deadline = time.time() + 0.5
        while time.time() < deadline:
            if rid in motor.param_dict:
                return motor.param_dict[rid]
            time.sleep(0.01)
        return None

    def change_motor_param(self, motor, rid, value):
        """Write a parameter to motor register.

        Uses uint32 encoding for INT_RIDS (7-10, 13-16, 35-36), float otherwise.
        IMPORTANT: Do NOT use isinstance(value, int) - that corrupts float RIDs!
        """
        can_id_l = motor.SlaveID & 0xFF
        can_id_h = (motor.SlaveID >> 8) & 0xFF
        # Only use uint32 encoding for explicitly defined INT_RIDS
        if rid in self.INT_RIDS:
            val_bytes = struct.pack('<I', int(value))
        else:
            val_bytes = struct.pack('<f', float(value))
        data = bytearray(8)
        data[0] = can_id_l
        data[1] = can_id_h
        data[2] = 0x55  # Write command
        data[3] = rid
        data[4:8] = val_bytes
        self._send(0x7FF, data)
        time.sleep(0.02)

    def switchControlMode(self, motor, mode):
        """Switch motor control mode (POS_VEL=2, VEL=3, etc.)."""
        self.change_motor_param(motor, 10, mode)  # RID 10 = CTRL_MODE

    def enable(self, motor):
        data = bytes([0xFF] * 7 + [0xFC])
        # NO print here — must be zero-latency for back-to-back enable+MIT in sync_write
        return self._send(motor.SlaveID, data)

    def disable(self, motor):
        data = bytes([0xFF] * 7 + [0xFD])
        self._send(motor.SlaveID, data)

    def set_zero_position(self, motor):
        data = bytes([0xFF] * 7 + [0xFE])
        self._send(motor.SlaveID, data)

    def save_motor_param(self, motor):
        """Save motor parameters to EEPROM/flash (0xAA command).
        CRITICAL: Must be called after switchControlMode() to persist mode change.
        Without this, the mode change only exists in RAM and may not take effect.
        """
        can_id_l = motor.SlaveID & 0xFF
        can_id_h = (motor.SlaveID >> 8) & 0xFF
        data = bytearray([can_id_l, can_id_h, 0xAA, 0, 0, 0, 0, 0])
        self._send(0x7FF, data)
        time.sleep(1.0)  # Flash write needs ~1 second to complete reliably

    def control_Pos_Vel(self, motor, p_des, v_des):
        arb_id = 0x100 + motor.SlaveID
        data = bytearray(8)
        data[0:4] = struct.pack('<f', p_des)
        data[4:8] = struct.pack('<f', v_des)
        if not hasattr(self, '_pos_vel_logged'):
            self._pos_vel_logged = set()
        if motor.SlaveID not in self._pos_vel_logged:
            self._pos_vel_logged.add(motor.SlaveID)
            print(f"[Damiao CAN] control_Pos_Vel FIRST: SlaveID=0x{motor.SlaveID:02X}, arb_id=0x{arb_id:03X}, pos={p_des:.4f}, vel={v_des:.4f}", flush=True)
        self._send(arb_id, data)

    def control_Vel(self, motor, v_des):
        arb_id = 0x200 + motor.SlaveID
        data = bytearray(8)
        data[0:4] = struct.pack('<f', v_des)
        self._send(arb_id, data)

    def control_pos_force(self, motor, p_des, v_des, i_des=0.0):
        arb_id = 0x100 + motor.SlaveID
        data = bytearray(8)
        data[0:4] = struct.pack('<f', p_des)
        data[4:8] = struct.pack('<f', v_des)
        self._send(arb_id, data)

    def control_MIT(self, motor, p_des, v_des, kp, kd, t_ff, p_max=12.5, v_max=30.0, t_max=10.0):
        """MIT mode control: arb_id = SlaveID, 8-byte packed fixed-point.

        Motor equation: τ = kp*(p_des - p) + kd*(v_des - v) + t_ff

        Args:
            motor: Motor object
            p_des: Desired position (rad), range [-p_max, p_max]
            v_des: Desired velocity (rad/s), range [-v_max, v_max]
            kp: Position gain (0-500, encoded in 12 bits)
            kd: Velocity damping (0-5, encoded in 12 bits)
            t_ff: Feedforward torque (Nm), range [-t_max, t_max]
            p_max/v_max/t_max: Motor-specific encoding limits from Limit_Param
        """
        def f2u(x, x_min, x_max, bits):
            x = max(x_min, min(x_max, x))
            return int((x - x_min) / (x_max - x_min) * ((1 << bits) - 1))

        p_int  = f2u(p_des, -p_max, p_max, 16)
        v_int  = f2u(v_des, -v_max, v_max, 12)
        kp_int = f2u(kp, 0, 500, 12)
        kd_int = f2u(kd, 0, 5, 12)
        t_int  = f2u(t_ff, -t_max, t_max, 12)

        data = bytearray(8)
        data[0] = (p_int >> 8) & 0xFF
        data[1] = p_int & 0xFF
        data[2] = (v_int >> 4) & 0xFF
        data[3] = ((v_int & 0x0F) << 4) | ((kp_int >> 8) & 0x0F)
        data[4] = kp_int & 0xFF
        data[5] = (kd_int >> 4) & 0xFF
        data[6] = ((kd_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F)
        data[7] = t_int & 0xFF
        return self._send(motor.SlaveID, data)

    def shutdown(self):
        self.running = False
        if self.bus:
            self.bus.shutdown()
            self.bus = None


# --- Motor configuration dataclasses ---

@dataclass
class DamiaoMotorConfig:
    """Configuration for a single Damiao motor."""
    name: str
    motor_type: str  # "J8009P", "J4340P", "J4310"
    can_id: int
    master_id: int

    # Derived from motor_type
    max_torque: float = field(init=False)
    max_velocity: float = field(init=False)
    torque_constant: float = field(init=False)
    dm_type: str = field(init=False)
    # MIT mode encoding limits (motor-specific, from Limit_Param in DM_CAN.py)
    p_max: float = field(init=False)  # Position encoding range (rad)
    v_max: float = field(init=False)  # Velocity encoding range (rad/s)
    t_max: float = field(init=False)  # Torque encoding range (Nm)

    def __post_init__(self):
        specs = DAMIAO_MOTOR_SPECS.get(self.motor_type, DAMIAO_MOTOR_SPECS["J4310"])
        self.max_torque = specs["max_torque"]
        self.max_velocity = specs["max_velocity"]
        self.torque_constant = specs["torque_constant"]
        self.dm_type = specs["dm_type"]
        self.p_max = specs["p_max"]
        self.v_max = specs["v_max"]
        self.t_max = specs["t_max"]
        # Capped velocity for rate limiter (J4310 raw max_velocity=20.9 is too high)
        self.rate_limit_velocity = specs.get("rate_limit_velocity", self.max_velocity)
        # EMA position smoothing alpha (1.0=no filter, <1.0=smooth for low-inertia motors)
        self.position_smoothing = specs.get("position_smoothing", 1.0)


@dataclass
class DamiaoMotorsBusConfig:
    """Configuration for Damiao motor bus (CAN over serial)."""
    port: str  # e.g., "/dev/ttyUSB0"
    baudrate: int = 921600
    motors: dict[str, dict] = field(default_factory=lambda: DEFAULT_DAMIAO_MOTORS.copy())

    # CRITICAL: Global safety settings
    velocity_limit: float = 0.1  # 0.0-1.0 global velocity scaling (default 10% for safety)
    torque_limit: float = 0.1  # 0.0-1.0 global torque scaling (default 10% for safety)
    acceleration_limit: float = 10.0  # rad/s^2 for PID
    deceleration_limit: float = -10.0  # rad/s^2 for PID

    # SAFETY CRITICAL: Skip internal motor PID parameter writes during configure()
    # When True, motors use their existing (factory/pre-configured) PID values.
    # The code will validate that KP_ASR >= 0.5 before enabling motors.
    # Only set to True if motors already have correct PID values!
    skip_pid_config: bool = False

    # MIT MODE CONTROL (RECOMMENDED)
    # POS_VEL mode causes vibration on some motor/firmware combinations.
    # MIT mode provides stable, smooth control with per-command kp/kd gains.
    use_mit_mode: bool = True  # Use MIT mode instead of POS_VEL

    # MIT mode gains (only used if use_mit_mode=True)
    # Motor equation: τ = kp*(p_des - p) + kd*(v_des - v) + t_ff
    mit_kp: float = 15.0  # Position stiffness (0-500 range, encoded in 12 bits)
    mit_kd: float = 1.5   # Velocity damping (0-5 range, encoded in 12 bits)


class DamiaoMotorsBus:
    """Motor bus for Damiao J-series motors over CAN (serial bridge).

    Key Features:
    - Global velocity limiter (0.0-1.0) applied to ALL motor commands
    - Support for J8009P (high torque), J4340P (medium), J4310 (precision)
    - Position-velocity control mode with configurable PID
    - Torque monitoring for safety

    Example:
        config = DamiaoMotorsBusConfig(port="/dev/ttyUSB0", velocity_limit=0.2)
        bus = DamiaoMotorsBus(config)
        bus.connect()

        # Read positions
        positions = bus.sync_read("Present_Position")

        # Write positions (velocity limited)
        bus.sync_write("Goal_Position", {"base": 0.5, "link1": 0.3})

        bus.disconnect()
    """

    def __init__(self, config: DamiaoMotorsBusConfig):
        self.config = config
        self._serial = None
        self._control = None
        self._motors = {}  # name -> Motor object from DM_CAN or SocketCAN
        self._motor_configs = {}  # name -> DamiaoMotorConfig
        self._use_socketcan = config.port.startswith('can')

        # CRITICAL: Global velocity limiter
        self._velocity_limit = max(0.0, min(1.0, config.velocity_limit))

        self._is_connected = False
        self._discovery_mode = False  # When True, sync_write is blocked (calibration discovery)
        self._enabled_motors = set()  # Track which motors have been enabled (for lazy enable)
        self._consecutive_send_failures = 0  # Emergency shutdown counter
        self._can_bus_dead = False  # Set True when emergency shutdown triggered
        self.calibration = {}  # For compatibility with LeRobot calibration service
        self._last_positions = {}  # Track last known positions for glitch detection
        self._last_goal_positions = {}  # Track last sent goal for MIT rate limiting
        self._last_sync_write_time = None  # For real dt calculation in rate limiter
        self._quarantined_motors = set()  # Motors with corrupt encoder — no commands sent
        self._mit_offsets = {}  # name -> float: gap between MIT encoder and 0xCC encoder (dual-encoder motors)
        self._active_joint_limits = dict(DEFAULT_JOINT_LIMITS)  # Mutable limits, overridden by calibration

        # Build motor configs
        for name, mcfg in config.motors.items():
            self._motor_configs[name] = DamiaoMotorConfig(
                name=name,
                motor_type=mcfg["motor_type"],
                can_id=mcfg["can_id"],
                master_id=mcfg["master_id"],
            )

    @property
    def velocity_limit(self) -> float:
        """Global velocity limiter (0.0-1.0). Applied to ALL motor commands."""
        return self._velocity_limit

    @velocity_limit.setter
    def velocity_limit(self, value: float):
        """Set global velocity limit. Clamped to [0.0, 1.0]."""
        old_value = self._velocity_limit
        self._velocity_limit = max(0.0, min(1.0, value))
        if old_value != self._velocity_limit:
            logger.info(f"[Damiao] Velocity limit changed: {old_value:.2f} -> {self._velocity_limit:.2f}")

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def motors(self) -> dict:
        """Return motor configs for compatibility."""
        return self._motor_configs

    def _ensure_can_txqueuelen(self, channel: str, target: int = 256) -> bool:
        """Ensure CAN interface has adequate TX queue length.

        With gs_usb adapters, the default qlen=10 causes permanent ENOBUFS errors
        when sending 7+ motor commands at 60Hz. This makes the CAN bus completely
        dead after the first sync_write, leaving motors enabled but uncontrolled.

        Returns True if txqueuelen is adequate, False if too low (BLOCKS configure).
        """
        import subprocess

        # Read current value
        sysfs_path = f"/sys/class/net/{channel}/tx_queue_len"
        current = 10  # default assumption
        try:
            with open(sysfs_path, 'r') as f:
                current = int(f.read().strip())
        except Exception:
            pass

        if current >= target:
            print(f"[Damiao] CAN {channel} txqueuelen={current} (OK)", flush=True)
            return True

        print(f"[Damiao] CAN {channel} txqueuelen={current} — too low, need {target}. Attempting to increase...", flush=True)

        # Try method 1: sudo ip link set
        try:
            result = subprocess.run(
                ["sudo", "-n", "ip", "link", "set", channel, "txqueuelen", str(target)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                print(f"[Damiao] Set {channel} txqueuelen={target} (via sudo ip link set)", flush=True)
                return True
        except Exception:
            pass

        # Try method 2: ip link set without sudo
        try:
            result = subprocess.run(
                ["ip", "link", "set", channel, "txqueuelen", str(target)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                print(f"[Damiao] Set {channel} txqueuelen={target} (via ip link set)", flush=True)
                return True
        except Exception:
            pass

        # Try method 3: direct sysfs write
        try:
            with open(sysfs_path, 'w') as f:
                f.write(str(target))
            print(f"[Damiao] Set {channel} txqueuelen={target} (via sysfs)", flush=True)
            return True
        except Exception:
            pass

        # All methods failed — read back actual value
        try:
            with open(sysfs_path, 'r') as f:
                actual = int(f.read().strip())
        except Exception:
            actual = current

        print(f"\n{'='*70}", flush=True)
        print(f"[Damiao] CRITICAL: CAN {channel} txqueuelen={actual} — TOO LOW!", flush=True)
        print(f"[Damiao] This WILL cause motors to spin uncontrollably during teleop.", flush=True)
        print(f"[Damiao]", flush=True)
        print(f"[Damiao] BEFORE running the app, run this once:", flush=True)
        print(f"[Damiao]   sudo bash setup_can.sh", flush=True)
        print(f"[Damiao] Or manually:", flush=True)
        print(f"[Damiao]   sudo ip link set {channel} down", flush=True)
        print(f"[Damiao]   sudo ip link set {channel} type can bitrate 1000000", flush=True)
        print(f"[Damiao]   sudo ip link set {channel} txqueuelen 256", flush=True)
        print(f"[Damiao]   sudo ip link set {channel} up", flush=True)
        print(f"{'='*70}\n", flush=True)
        return False

    def connect(self) -> None:
        """Connect to CAN bus via serial bridge or SocketCAN and initialize motors."""
        if self._is_connected:
            raise RuntimeError("DamiaoMotorsBus already connected")

        if self._use_socketcan:
            self._connect_socketcan()
        else:
            self._connect_serial()

        self._is_connected = True
        logger.info(f"[Damiao] Connected with {len(self._motors)} motors")

    def _connect_socketcan(self) -> None:
        """Connect via SocketCAN (native Linux CAN interface, e.g. can0)."""
        logger.info(f"[Damiao] Connecting via SocketCAN on {self.config.port}")

        self._DM_variable = _SocketCANDMVariable
        self._Control_Type = _SocketCANControlType

        self._control = _SocketCANControl(
            channel=self.config.port,
            bitrate=1000000
        )
        time.sleep(0.3)

        for name, mcfg in self._motor_configs.items():
            motor = _SocketCANMotor(mcfg.can_id, mcfg.master_id, v_max=mcfg.v_max, t_max=mcfg.t_max)
            self._motors[name] = motor
            self._control.addMotor(motor)

    def _connect_serial(self) -> None:
        """Connect via serial CAN-to-serial bridge (e.g. /dev/ttyUSB0)."""
        try:
            from trlc_dk1.motors.DM_Control_Python.DM_CAN import (
                Motor, MotorControl, DM_Motor_Type, DM_variable, Control_Type
            )
        except ImportError as e:
            raise ImportError(
                "trlc_dk1.motors.DM_Control_Python.DM_CAN not found. "
                "Please install the Damiao motor control library."
            ) from e

        self._DM_Motor_Type = DM_Motor_Type
        self._DM_variable = DM_variable
        self._Control_Type = Control_Type

        logger.info(f"[Damiao] Connecting to {self.config.port} at {self.config.baudrate} baud")
        self._serial = serial.Serial(self.config.port, self.config.baudrate, timeout=0.5)
        time.sleep(0.5)

        self._control = MotorControl(self._serial)

        MOTOR_TYPE_MAP = {
            "DM4340": DM_Motor_Type.DM4340,
            "DM4310": DM_Motor_Type.DM4310,
            "DM8009": DM_Motor_Type.DM8009,
        }

        for name, mcfg in self._motor_configs.items():
            dm_type = MOTOR_TYPE_MAP.get(mcfg.dm_type, DM_Motor_Type.DM4310)
            motor = Motor(dm_type, mcfg.can_id, mcfg.master_id)
            self._motors[name] = motor
            self._control.addMotor(motor)

    def configure(self) -> None:
        """Configure all motors for control (MIT or POS_VEL mode).

        MIT mode (default, recommended):
            Uses impedance control with per-command kp/kd gains.
            Motor equation: τ = kp*(p_des - p) + kd*(v_des - v) + t_ff
            No internal PID parameters need to be written to motor flash.
            Stable on all tested motor/firmware combinations.

        POS_VEL mode (legacy):
            Uses the motor's internal PID controller for position tracking.
            Requires writing KP_APR, KI_APR, KP_ASR, KI_ASR parameters.
            May cause vibration on some motor/firmware combinations.

        Motors are enabled immediately after configuration.
        """
        if not self._is_connected:
            raise RuntimeError("DamiaoMotorsBus not connected")

        # Ensure CAN TX queue is large enough — CRITICAL for 7-motor teleop.
        if self._use_socketcan:
            qlen_ok = self._ensure_can_txqueuelen(self.config.port, 256)
            if not qlen_ok:
                raise RuntimeError(
                    f"CAN txqueuelen too low — teleop WILL fail. "
                    f"Run: sudo bash setup_can.sh"
                )

        DM_variable = self._DM_variable
        Control_Type = self._Control_Type

        configured = []  # Track configured motors for cleanup on failure

        # Reset state
        self._enabled_motors = set()
        self._quarantined_motors = set()
        self._consecutive_send_failures = 0
        self._can_bus_dead = False
        self._sync_write_count = 0

        # Determine control mode
        use_mit = self.config.use_mit_mode
        mode_name = "MIT" if use_mit else "POS_VEL"
        target_mode = Control_Type.MIT if use_mit else Control_Type.POS_VEL

        print(f"[Damiao] Configuring {len(self._motors)} motors in {mode_name} mode...", flush=True)

        try:
            # SAFETY: Disable all motors first. If a previous session crashed,
            # motors may still be enabled with stale MIT gains, causing violent
            # motion when the recv thread starts processing responses.
            for name, motor in self._motors.items():
                self._control.disable(motor)
                time.sleep(0.01)
            print(f"[Damiao] All motors disabled (safe start)", flush=True)
            time.sleep(0.1)  # Let motors settle

            configure_positions = {}  # Save 0xCC positions for dual-encoder offset measurement

            for name, motor in self._motors.items():
                mcfg = self._motor_configs[name]

                # Refresh status to verify connection
                for _ in range(3):
                    self._control.refresh_motor_status(motor)
                    time.sleep(0.01)

                # Verify motor responds
                ctrl_mode = self._control.read_motor_param(motor, DM_variable.CTRL_MODE)
                if ctrl_mode is None:
                    raise RuntimeError(f"[Damiao] Motor '{name}' ({mcfg.motor_type}) not responding")

                pos = motor.getPosition()
                self._last_positions[name] = pos  # Seed glitch detection baseline
                configure_positions[name] = pos   # Save for offset measurement later
                print(f"[Damiao] Motor '{name}' ({mcfg.motor_type}) responding, mode={ctrl_mode}, pos={pos:.4f} rad", flush=True)

                # SAFETY: Quarantine motors with encoder positions far outside joint limits.
                # This prevents commanding motors whose encoder was shifted by previous incidents.
                # The motor is left DISABLED and no commands are sent until encoder is re-zeroed.
                limits = self._active_joint_limits.get(name)
                quarantine_margin = 3.0  # rad beyond limits before quarantine
                if limits and (pos < limits[0] - quarantine_margin or pos > limits[1] + quarantine_margin):
                    self._quarantined_motors.add(name)
                    print(f"[Damiao] QUARANTINED: Motor '{name}' at {pos:.3f} rad is FAR outside "
                          f"joint limits ({limits[0]:.2f}, {limits[1]:.2f}). Encoder needs re-zeroing. "
                          f"Motor will NOT be enabled.", flush=True)
                    # Leave motor disabled — do not enable or configure
                    self._control.disable(motor)
                    configured.append(name)
                    continue
                elif limits and (pos < limits[0] or pos > limits[1]):
                    print(f"[Damiao] WARNING: Motor '{name}' at {pos:.3f} rad is OUTSIDE "
                          f"joint limits ({limits[0]:.2f}, {limits[1]:.2f})", flush=True)

                # IMPORTANT: Motor must be DISABLED to change mode and parameters!
                # Otherwise parameter writes may be silently ignored.
                self._control.disable(motor)
                time.sleep(0.05)

                # Switch to target control mode
                self._control.switchControlMode(motor, target_mode)
                time.sleep(0.02)

                if use_mit:
                    # MIT MODE: No PID parameters needed — gains are sent per-command
                    # Use safe enable sequence with limp frames to prevent torque spike
                    self._safe_enable_mit(motor, name)
                else:
                    # POS_VEL MODE: Set PID params WHILE DISABLED
                    # CRITICAL: Both position loop (APR) AND velocity loop (ASR) gains must be set!
                    # If KP_ASR/KI_ASR are 0, the motor will oscillate or not move at all.
                    if not self.config.skip_pid_config:
                        pid = PID_GAINS.get(mcfg.motor_type, PID_GAINS["J4310"])
                        self._control.change_motor_param(motor, DM_variable.ACC, pid["ACC"])
                        self._control.change_motor_param(motor, DM_variable.DEC, pid["DEC"])
                        # Position loop (outer loop)
                        self._control.change_motor_param(motor, DM_variable.KP_APR, pid["KP_APR"])
                        self._control.change_motor_param(motor, DM_variable.KI_APR, pid["KI_APR"])
                        # Velocity loop (inner loop) - REQUIRED for POS_VEL mode!
                        self._control.change_motor_param(motor, DM_variable.KP_ASR, pid["KP_ASR"])
                        self._control.change_motor_param(motor, DM_variable.KI_ASR, pid["KI_ASR"])
                        time.sleep(0.02)
                    else:
                        # SAFETY: Validate existing PID values before proceeding
                        print(f"[Damiao] SKIP_PID_CONFIG: Validating motor '{name}' has safe PID values...", flush=True)
                        motor_kp_asr = self._control.read_motor_param(motor, DM_variable.KP_ASR)
                        if motor_kp_asr is None or motor_kp_asr < 0.5:
                            raise RuntimeError(
                                f"SAFETY: Motor '{name}' has KP_ASR={motor_kp_asr} which is too low! "
                                f"Motor will not move in POS_VEL mode. Either set skip_pid_config=False "
                                f"or manually program correct PID values (KP_ASR >= 2.0 recommended)."
                            )
                        print(f"[Damiao] Motor '{name}' using existing parameters (KP_ASR={motor_kp_asr:.2f})", flush=True)

                    # Enable after parameters are set (POS_VEL mode)
                    self._control.enable(motor)
                    time.sleep(0.05)

                self._enabled_motors.add(name)
                self._last_goal_positions[name] = self._last_positions.get(name, 0.0)
                configured.append(name)

                # Verify mode was set
                verify_mode = self._control.read_motor_param(motor, DM_variable.CTRL_MODE)
                print(f"[Damiao] Motor '{name}' configured: {mode_name} mode (verified={verify_mode}), ENABLED", flush=True)

        except Exception:
            # On failure, ensure all configured motors are disabled
            print(f"[Damiao] configure() FAILED — disabling {len(configured)} already-configured motors", flush=True)
            for cname in configured:
                try:
                    self._control.disable(self._motors[cname])
                except Exception:
                    pass
            self._enabled_motors.clear()
            raise

        print(f"[Damiao] All {len(self._motors)} motors configured in {mode_name} mode (all ENABLED)", flush=True)
        self._last_sync_write_time = time.perf_counter()  # Seed rate limiter clock

        # ─── Pass 2: Measure dual-encoder offsets (J8009P-2EC) ───
        # Done AFTER all motors are configured so they've had time (~500ms)
        # to start reporting MIT-encoder positions in status responses.
        # Uses MIT zero-torque probe (kp=0, kd=0) for reliable per-motor offset.
        # RID 80/81 reads are kept for diagnostics only — they're unreliable with
        # multiple motors because CAN responses get cross-contaminated.
        if use_mit:
            time.sleep(0.050)  # Let motors settle
            for name, motor in self._motors.items():
                if name not in self._enabled_motors:
                    self._mit_offsets[name] = 0.0
                    continue
                mcfg = self._motor_configs[name]
                p_cc = configure_positions.get(name, 0.0)

                # Method 1: Read both encoders via RID (reliable command-response)
                p_m = self._control.read_motor_param(motor, 80)    # RID 80: motor encoder
                xout = self._control.read_motor_param(motor, 81)   # RID 81: output encoder

                # Method 2: MIT probe with last_seen polling (cross-validation)
                before_ts = motor.last_seen
                self._control.control_MIT(
                    motor, p_cc, 0.0,
                    0.0, 0.0, 0.0,  # kp=0, kd=0 — zero-torque probe
                    mcfg.p_max, mcfg.v_max, mcfg.t_max
                )
                deadline = time.time() + 0.050
                while motor.last_seen <= before_ts and time.time() < deadline:
                    time.sleep(0.001)
                p_mit_probe = motor.getPosition()

                # Print all encoder values for diagnostics
                print(f"[Damiao] Motor '{name}' encoders: "
                      f"0xCC={p_cc:+.4f}, RID80(p_m)={p_m}, RID81(xout)={xout}, "
                      f"MIT_probe={p_mit_probe:+.4f}", flush=True)

                # Use MIT probe result as the ONLY offset source.
                # RID80/81 param reads are unreliable with multiple motors — CAN status
                # responses from other motors get misidentified as parameter responses,
                # causing bogus offset values (e.g., -12.5 rad on motors with no real offset).
                offset = p_mit_probe - p_cc

                self._mit_offsets[name] = offset
                if abs(offset) > 0.1:
                    print(f"[Damiao] DUAL-ENCODER OFFSET: Motor '{name}' = {offset:+.4f} rad "
                          f"(applied automatically)", flush=True)

        logger.info(f"[Damiao] All motors configured in {mode_name} mode and enabled")

    def read_pid_parameters(self, motor_name: str | None = None) -> dict[str, dict]:
        """Read current PID parameters from motors for auditing.

        SAFETY AUDIT: Use this to verify motor parameters before/after configure().

        Args:
            motor_name: Specific motor to read, or None for all motors

        Returns:
            Dict mapping motor name to PID values:
            {
                "motor_name": {
                    "KP_APR": float, "KI_APR": float,
                    "KP_ASR": float, "KI_ASR": float,
                    "ACC": float, "DEC": float,
                }
            }
        """
        if not self._is_connected:
            raise RuntimeError("DamiaoMotorsBus not connected")

        DM_variable = self._DM_variable
        target_motors = [motor_name] if motor_name else list(self._motors.keys())
        results = {}

        for name in target_motors:
            if name not in self._motors:
                continue
            motor = self._motors[name]

            results[name] = {
                "KP_APR": self._control.read_motor_param(motor, DM_variable.KP_APR),
                "KI_APR": self._control.read_motor_param(motor, DM_variable.KI_APR),
                "KP_ASR": self._control.read_motor_param(motor, DM_variable.KP_ASR),
                "KI_ASR": self._control.read_motor_param(motor, DM_variable.KI_ASR),
                "ACC": self._control.read_motor_param(motor, DM_variable.ACC),
                "DEC": self._control.read_motor_param(motor, DM_variable.DEC),
            }

        return results

    # Minimum safe PID values - motor won't work properly if below these
    MIN_SAFE_PID = {
        "KP_ASR": 0.5,   # Velocity loop Kp - CRITICAL, motor won't move if 0
        "KI_ASR": 0.001, # Velocity loop Ki
        "KP_APR": 50.0,  # Position loop Kp
        "KI_APR": 0.1,   # Position loop Ki
    }

    def validate_pid_parameters(self) -> bool:
        """Validate that all motors have safe PID values.

        SAFETY CHECK: Call this before enabling motors when skip_pid_config=True.
        Raises RuntimeError if any motor has unsafe PID values (e.g., KP_ASR=0).

        Returns:
            True if all motors have safe values

        Raises:
            RuntimeError if any motor has unsafe values
        """
        pids = self.read_pid_parameters()
        errors = []

        for motor_name, values in pids.items():
            for param, min_val in self.MIN_SAFE_PID.items():
                val = values.get(param, 0)
                if val is None or val < min_val:
                    errors.append(
                        f"Motor '{motor_name}': {param}={val} "
                        f"is below minimum safe value {min_val}"
                    )

        if errors:
            error_msg = (
                "SAFETY VIOLATION: Motors have unsafe PID values!\n"
                "The motor(s) will not work properly in POS_VEL mode.\n\n"
                + "\n".join(errors) + "\n\n"
                "Options:\n"
                "1. Set skip_pid_config=False to use default PID values, OR\n"
                "2. Manually program correct PID values to motors and save to flash"
            )
            raise RuntimeError(error_msg)

        print(f"[Damiao] PID VALIDATION PASSED: All {len(pids)} motors have safe values", flush=True)
        return True

    def verify_control_mode(self) -> dict[str, int]:
        """Verify all motors are in the expected control mode (MIT or POS_VEL).

        SAFETY AUDIT: Call this after configure() to confirm motors are in the correct mode.

        Returns:
            Dict mapping motor name to control mode value:
            - 1 = MIT mode
            - 2 = POS_VEL mode
            - 3 = VEL mode
            - 4 = Torque_Pos mode

        Raises:
            RuntimeError if any motor is in unexpected mode
        """
        if not self._is_connected:
            raise RuntimeError("DamiaoMotorsBus not connected")

        results = {}
        DM_variable = self._DM_variable
        expected_mode = 1 if self.config.use_mit_mode else 2
        mode_name = "MIT" if self.config.use_mit_mode else "POS_VEL"

        for name, motor in self._motors.items():
            mode = self._control.read_motor_param(motor, DM_variable.CTRL_MODE)
            results[name] = mode
            if mode != expected_mode:
                raise RuntimeError(
                    f"SAFETY VIOLATION: Motor '{name}' is in mode={mode}! "
                    f"Expected {mode_name} mode (mode={expected_mode})."
                )

        print(f"[Damiao] MODE VERIFIED: All {len(results)} motors in {mode_name} mode (mode={expected_mode})", flush=True)
        return results

    def verify_pos_vel_mode(self) -> dict[str, int]:
        """Legacy alias for verify_control_mode(). Deprecated."""
        return self.verify_control_mode()

    def _safe_enable_mit(self, motor, name: str) -> bool:
        """Enable motor safely in MIT mode with limp frame sequence.

        When a Damiao motor is enabled, stale MIT params from RAM can cause
        a torque spike. Solution: enable, then immediately send limp frames
        (kp=0, kd=0) to overwrite any stale state before applying real gains.

        Args:
            motor: Motor object
            name: Motor name for config lookup

        Returns:
            True on success
        """
        mcfg = self._motor_configs[name]

        # Use position already in motor state (from configure()'s 0xCC refresh).
        # Do NOT send another 0xCC here — its async response can arrive late and
        # overwrite MIT encoder data, breaking the dual-encoder offset measurement.
        current_pos = motor.getPosition()
        mit_pos = current_pos + self._mit_offsets.get(name, 0.0)

        # Enable motor
        self._control.enable(motor)
        time.sleep(0.002)

        # Immediately send limp frames (kp=0, kd=0) to overwrite stale state
        # This prevents any torque spike from leftover MIT parameters in motor RAM
        for _ in range(10):
            self._control.control_MIT(
                motor, mit_pos, 0.0,      # position in MIT-encoder-space, velocity
                0.0, 0.0, 0.0,            # kp=0, kd=0, torque=0 (limp)
                mcfg.p_max, mcfg.v_max, mcfg.t_max
            )
            time.sleep(0.001)

        print(f"[Damiao] Motor '{name}' enabled in MIT mode (safe enable with limp frames)", flush=True)
        return True

    def sync_read(self, data_name: str, motors: list[str] | None = None, normalize: bool = True) -> dict[str, float]:
        """Read data from motors.

        In MIT mode, uses MIT probe commands instead of 0xCC refresh to read positions.
        This ensures the position readback comes from the SAME encoder that MIT mode uses
        for torque calculation, avoiding inconsistencies that cause violent spinning.

        Args:
            data_name: "Present_Position", "Present_Velocity", or "Present_Torque"
            motors: List of motor names (None = all motors)
            normalize: Ignored for Damiao (always returns radians)

        Returns:
            Dict mapping motor name to value
        """
        if not self._is_connected:
            raise RuntimeError("DamiaoMotorsBus not connected")

        target_motors = motors if motors else list(self._motors.keys())
        results = {}
        use_mit_probe = self.config.use_mit_mode and data_name == "Present_Position"

        for i, name in enumerate(target_motors):
            if name not in self._motors:
                continue
            if name in self._quarantined_motors:
                continue

            motor = self._motors[name]
            mcfg = self._motor_configs[name]

            if i > 0:
                time.sleep(0.001)

            if use_mit_probe and name in self._enabled_motors:
                # MIT MODE: Send a ZERO-TORQUE probe to read position.
                # Uses kp=0, kd=0, t_ff=0 so τ = 0*(p_des-p) + 0*(v_des-v) + 0 = 0.
                # The motor responds with its current state without applying any torque.
                # CRITICAL: Previous version used full gains (kp=15, kd=1.5) which caused
                # violent spinning when motor.getPosition() returned a stale/corrupted value.
                current_pos = motor.getPosition()
                self._control.control_MIT(
                    motor, current_pos, 0.0,
                    0.0, 0.0, 0.0,  # kp=0, kd=0, t_ff=0 — zero torque, read-only
                    mcfg.p_max, mcfg.v_max, mcfg.t_max
                )
                time.sleep(0.002)
                # Verify response arrived (within last 10ms)
                if time.time() - motor.last_seen > 0.010:
                    time.sleep(0.005)  # Extra wait for congested bus
            else:
                # Non-MIT mode or motor not enabled: use 0xCC refresh
                self._control.refresh_motor_status(motor)
                time.sleep(0.002)
                if time.time() - motor.last_seen > 0.010:
                    time.sleep(0.005)

            if data_name == "Present_Position":
                pos = motor.getPosition()
                # Remove MIT encoder offset to return user-space position
                pos = pos - self._mit_offsets.get(name, 0.0)
                self._last_positions[name] = pos
                results[name] = pos
            elif data_name == "Present_Velocity":
                results[name] = motor.getVelocity()  # rad/s
            elif data_name == "Present_Torque":
                results[name] = motor.getTorque()  # Nm
            else:
                logger.warning(f"[Damiao] Unknown data_name: {data_name}")

        return results

    def _emergency_disable_all(self, reason: str) -> None:
        """SAFETY: Immediately disable all motors when CAN bus is dead."""
        print(f"\n{'!'*70}", flush=True)
        print(f"[Damiao] EMERGENCY SHUTDOWN: {reason}", flush=True)
        print(f"[Damiao] Disabling all {len(self._enabled_motors)} enabled motors...", flush=True)
        for name in list(self._enabled_motors):
            try:
                self._control.disable(self._motors[name])
            except Exception:
                pass
        self._enabled_motors = set()
        self._last_goal_positions.clear()
        self._last_sync_write_time = None
        self._can_bus_dead = True
        print(f"[Damiao] All motors DISABLED. CAN bus marked dead.", flush=True)
        print(f"[Damiao] Fix: sudo ip link set {self.config.port} txqueuelen 256", flush=True)
        print(f"{'!'*70}\n", flush=True)

    def sync_write(self, data_name: str, values: dict[str, float], normalize: bool = True) -> None:
        """Write goal positions to motors using MIT or POS_VEL mode.

        MIT mode (recommended):
            Sends position commands with per-command kp/kd gains.
            Motor equation: τ = kp*(p_des - p) + kd*(v_des - v) + t_ff

        POS_VEL mode (legacy):
            Sends position commands using internal PID controller.

        Args:
            data_name: "Goal_Position"
            values: Dict mapping motor name to goal position (radians)
            normalize: Ignored for Damiao
        """
        if self._discovery_mode:
            return  # Block all writes during calibration discovery

        if self._can_bus_dead:
            return  # CAN bus failed — all motors already disabled

        if not self._is_connected:
            raise RuntimeError("DamiaoMotorsBus not connected")

        if data_name != "Goal_Position":
            logger.warning(f"[Damiao] sync_write only supports Goal_Position, got: {data_name}")
            return

        self._sync_write_count = getattr(self, '_sync_write_count', 0) + 1

        # Compute real dt for rate limiter (instead of assuming 60Hz)
        now = time.perf_counter()
        real_dt = now - self._last_sync_write_time if self._last_sync_write_time else (1.0 / 60.0)
        real_dt = min(real_dt, 0.1)  # Cap at 100ms to prevent huge jumps after pauses
        self._last_sync_write_time = now

        # Debug: log first sync_write call
        mode_name = "MIT" if self.config.use_mit_mode else "POS_VEL"
        if self._sync_write_count == 1:
            print(f"[Damiao] sync_write FIRST CALL ({mode_name} mode): {len(values)} motors, dt={real_dt*1000:.1f}ms", flush=True)
            if self.config.use_mit_mode:
                gains_str = ", ".join(f"{mt}: kp={g['kp']}, kd={g['kd']}" for mt, g in MIT_GAINS.items())
                print(f"[Damiao] MIT gains (per-type): {gains_str}", flush=True)

        for name, value in values.items():
            if name not in self._motors:
                continue

            # Debug: log link3 positions on first 5 sync_write calls
            if self._sync_write_count <= 5 and name == "link3":
                last_g = self._last_goal_positions.get(name, None)
                cur_pos = self._last_positions.get(name, None)
                print(f"[Damiao] sync_write #{self._sync_write_count}: link3 "
                      f"requested={value:.4f}, last_goal={last_g}, "
                      f"cur_pos={cur_pos}, vel_limit={self._velocity_limit:.2f}", flush=True)

            motor = self._motors[name]
            mcfg = self._motor_configs[name]

            # SAFETY LAYER 1: Clamp position to joint limits (uses calibrated limits if available)
            limits = self._active_joint_limits.get(name)
            if limits:
                margin = 0.5  # Allow small overshoot beyond limits
                clamped = max(limits[0] - margin, min(limits[1] + margin, value))
                if clamped != value:
                    if not hasattr(self, '_clamp_warned'):
                        self._clamp_warned = set()
                    if name not in self._clamp_warned:
                        print(f"[Damiao] SAFETY CLAMP: {name} position {value:.4f} clamped to {clamped:.4f} "
                              f"(limits: {limits[0]:.2f} to {limits[1]:.2f})", flush=True)
                        self._clamp_warned.add(name)
                    value = clamped

            # SAFETY LAYER 2: Quarantine guard
            # Motors with corrupt encoder positions (far outside joint limits) are quarantined.
            # No commands are sent — the encoder must be re-zeroed first.
            if name in self._quarantined_motors:
                continue

            if self.config.use_mit_mode:
                # MIT MODE: τ = kp*(p_des - p) + kd*(v_des - v) + t_ff
                # Gain priority: per-motor (MIT_MOTOR_GAINS) → per-type (MIT_GAINS) → global config
                motor_gains = MIT_MOTOR_GAINS.get(name, {})
                type_gains = MIT_GAINS.get(mcfg.motor_type, {})
                kp = motor_gains.get("kp", type_gains.get("kp", self.config.mit_kp))
                kd = motor_gains.get("kd", type_gains.get("kd", self.config.mit_kd))

                # Gripper needs lower stiffness to stay within torque limits.
                # At kp=15, max per-frame torque = 15 × 0.105 = 1.57 Nm > 1.2 Nm limit.
                # At kp=5:  max per-frame torque =  5 × 0.105 = 0.52 Nm — safe margin.
                if name == "gripper":
                    kp = min(kp, 5.0)
                    kd = min(kd, 1.0)

                # EMA position smoothing (J4310: low inertia tracks 60Hz steps → jitter)
                alpha = mcfg.position_smoothing
                if alpha < 1.0 and name in self._last_goal_positions:
                    value = alpha * value + (1.0 - alpha) * self._last_goal_positions[name]

                # MIT MODE VELOCITY LIMITING + VELOCITY FEEDFORWARD
                # Rate-limit the position target AND compute v_des so kd assists
                # motion instead of fighting it. v_des = actual_step/dt makes the
                # damping term vanish during intentional motion (v ≈ v_des → kd≈0)
                # and only activate for disturbance rejection.
                v_des = 0.0
                if name in self._last_goal_positions:
                    last_goal = self._last_goal_positions[name]

                    # Rate limiting
                    if name == "gripper":
                        max_step = mcfg.max_velocity * real_dt
                    elif self._velocity_limit < 1.0:
                        max_step = mcfg.rate_limit_velocity * self._velocity_limit * real_dt
                    else:
                        max_step = None

                    if max_step is not None:
                        delta = value - last_goal
                        if abs(delta) > max_step:
                            clamped_step = max_step * (1.0 if delta > 0 else -1.0)
                            value = last_goal + clamped_step
                            if self._sync_write_count <= 3:
                                vel_str = "100%" if name == "gripper" else f"{self._velocity_limit:.0%}"
                                print(f"[Damiao] MIT rate limit: {name} clamped step "
                                      f"{delta:+.4f} -> {max_step:.4f} rad, "
                                      f"(vel_limit={vel_str})", flush=True)

                    # Velocity feedforward: always compute from actual step
                    actual_step = value - last_goal
                    if real_dt > 0:
                        v_des = actual_step / real_dt

                self._last_goal_positions[name] = value

                # Apply MIT encoder offset (dual-encoder motors like J8009P-2EC).
                # Joint limit clamping above works in user-space; now shift to MIT-encoder-space.
                value = value + self._mit_offsets.get(name, 0.0)

                # Send MIT command with velocity feedforward
                self._control.control_MIT(
                    motor,
                    value,      # p_des: goal position
                    v_des,      # v_des: velocity feedforward from position trajectory
                    kp,         # kp: position stiffness
                    kd,         # kd: velocity damping
                    0.0,        # t_ff: feedforward torque
                    mcfg.p_max, mcfg.v_max, mcfg.t_max  # encoding limits
                )
            else:
                # POS_VEL MODE: Use motor's internal PID controller
                vel = mcfg.max_velocity * self._velocity_limit
                self._control.control_Pos_Vel(motor, value, vel)

    def send_gripper_command(self, position: float, max_torque: float = 1.0) -> None:
        """Send gripper command using Torque_Pos mode with force limiting.

        Args:
            position: Goal position in radians
            max_torque: Maximum torque in Nm (converted to current limit)
        """
        if "gripper" not in self._motors or self._can_bus_dead:
            return

        motor = self._motors["gripper"]
        mcfg = self._motor_configs["gripper"]

        # Switch gripper to Torque_Pos mode if not already enabled
        if "gripper" not in self._enabled_motors:
            Control_Type = self._Control_Type
            self._control.switchControlMode(motor, Control_Type.Torque_Pos)
            time.sleep(0.02)
            self._control.enable(motor)
            self._enabled_motors.add("gripper")
            print(f"[Damiao] Gripper ENABLED in Torque_Pos mode", flush=True)

        # Send position command with force limiting
        # Convert torque to current using torque constant
        vel = mcfg.max_velocity * EMIT_VELOCITY_SCALE
        current = max_torque / mcfg.torque_constant * EMIT_CURRENT_SCALE
        self._control.control_pos_force(motor, position, vel, i_des=current)

    def home_gripper(self) -> None:
        """Auto-home gripper using torque detection.

        Opens gripper until torque spike detected, then sets zero position.
        """
        if "gripper" not in self._motors:
            logger.warning("[Damiao] No gripper motor configured")
            return

        # VEL mode doesn't work via SocketCAN — skip homing
        if self._use_socketcan:
            print("[Damiao] Skipping gripper homing (VEL mode not available via SocketCAN)", flush=True)
            logger.info("[Damiao] Gripper homing skipped (SocketCAN mode)")
            return

        motor = self._motors["gripper"]
        Control_Type = self._Control_Type

        logger.info("[Damiao] Homing gripper...")

        # Switch to velocity mode
        self._control.switchControlMode(motor, Control_Type.VEL)
        self._control.control_Vel(motor, 10.0)  # Open gripper

        # Wait for torque spike (gripper fully open)
        timeout = 5.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            self._control.refresh_motor_status(motor)
            tau = motor.getTorque()
            if tau > GRIPPER_TORQUE_THRESHOLD:
                self._control.control_Vel(motor, 0.0)
                self._control.disable(motor)
                self._control.set_zero_position(motor)
                time.sleep(0.2)
                self._control.enable(motor)
                logger.info("[Damiao] Gripper homed successfully")
                break
            time.sleep(0.01)
        else:
            logger.warning("[Damiao] Gripper homing timeout")
            # Stop gripper velocity and disable before mode switch
            self._control.control_Vel(motor, 0.0)
            self._control.disable(motor)
            time.sleep(0.05)

        # Switch to torque-position mode for force-limited gripping
        self._control.switchControlMode(motor, Control_Type.Torque_Pos)

    def set_zero_positions(self, motors: list[str] | None = None) -> dict[str, float]:
        """Set current position as zero for specified motors using CAN 0xFE command.

        WARNING: This permanently resets the motor's zero reference point.
        The arm should be manually positioned to the desired zero pose first.

        Procedure per motor: disable -> set_zero_position (0xFE) -> wait 200ms -> enable

        Args:
            motors: List of motor names to zero. If None, all motors.

        Returns:
            Dict of motor_name -> previous position in radians (before zeroing).
        """
        if not self._is_connected:
            raise RuntimeError("DamiaoMotorsBus not connected")

        target = motors if motors else list(self._motors.keys())
        previous_positions = {}

        for name in target:
            if name not in self._motors:
                logger.warning(f"[Damiao] Motor '{name}' not found, skipping zero-set")
                continue

            motor = self._motors[name]

            # Read current position before zeroing
            self._control.refresh_motor_status(motor)
            previous_positions[name] = motor.getPosition()

            # Disable -> set zero -> wait -> safe enable
            self._control.disable(motor)
            self._enabled_motors.discard(name)
            time.sleep(0.05)
            self._control.set_zero_position(motor)
            time.sleep(0.2)

            # Zero reference changed — reset MIT offset (will be re-measured on next configure())
            self._mit_offsets[name] = 0.0

            # Re-enable with safe MIT startup (limp frames prevent torque spike)
            if self.config.use_mit_mode:
                self._safe_enable_mit(motor, name)
            else:
                self._control.enable(motor)
            self._enabled_motors.add(name)
            time.sleep(0.05)

            logger.info(f"[Damiao] Zero set for {name} (was {previous_positions[name]:.3f} rad)")

        return previous_positions

    def enable_torque(self, motors: list[str] | None = None) -> None:
        """Enable torque on specified motors.

        In MIT mode, uses safe enable sequence (enable + 10 limp frames at kp=0, kd=0)
        to prevent torque spikes from stale MIT parameters in motor RAM.
        """
        if not self._is_connected:
            return

        target = motors if motors else list(self._motors.keys())
        for name in target:
            if name in self._motors and name not in self._quarantined_motors:
                if self.config.use_mit_mode:
                    self._safe_enable_mit(self._motors[name], name)
                else:
                    self._control.enable(self._motors[name])
                self._enabled_motors.add(name)

    def disable_torque(self, motors: list[str] | None = None, num_retry: int = 0) -> None:
        """Disable torque on specified motors (make backdrivable)."""
        if not self._is_connected:
            return

        target = motors if motors else list(self._motors.keys())
        for name in target:
            if name in self._motors:
                self._control.disable(self._motors[name])
                self._enabled_motors.discard(name)

    def disconnect(self, disable_torque: bool = True) -> None:
        """Disconnect from motor bus."""
        if not self._is_connected:
            return

        if disable_torque:
            logger.info("[Damiao] Disabling torque on all motors")
            for motor in self._motors.values():
                try:
                    self._control.disable(motor)
                except Exception as e:
                    logger.warning(f"[Damiao] Failed to disable motor: {e}")

        self._enabled_motors.clear()
        self._last_goal_positions.clear()
        self._last_sync_write_time = None

        if self._use_socketcan:
            if hasattr(self._control, 'shutdown'):
                self._control.shutdown()
        elif self._serial:
            self._serial.close()

        self._is_connected = False
        logger.info("[Damiao] Disconnected")

    def read_torques(self) -> dict[str, float]:
        """Read current torques from all motors (for safety monitoring).

        In MIT mode, uses cached values from recv thread (zero CAN overhead).
        Every control_MIT() call triggers a response with torque data that the
        recv thread parses into motor.state_tau at 60Hz — no extra probes needed.
        """
        if self.config.use_mit_mode:
            result = {}
            for name, motor in self._motors.items():
                if name not in self._quarantined_motors:
                    result[name] = motor.getTorque()
            return result
        return self.sync_read("Present_Torque")

    def read_cached_positions(self) -> dict[str, float]:
        """Read cached motor positions (zero CAN overhead, from MIT response cache).

        In MIT mode, every control_MIT() triggers a response with position data that
        the recv thread parses into motor.state_q — no extra probes needed.
        Returns positions in user space (MIT offset removed).
        """
        result = {}
        for name, motor in self._motors.items():
            if name not in self._quarantined_motors:
                result[name] = motor.getPosition() - self._mit_offsets.get(name, 0.0)
        return result

    def get_torque_limits(self) -> dict[str, float]:
        """Get torque limits for each motor (85% of max)."""
        limits = {}
        for name, mcfg in self._motor_configs.items():
            specs = DAMIAO_MOTOR_SPECS.get(mcfg.motor_type, {})
            max_torque = specs.get("max_torque", 10.0)
            limit_percent = specs.get("torque_limit_percent", 0.85)
            limits[name] = max_torque * limit_percent
        return limits

    def update_joint_limits(self, limits: dict[str, tuple[float, float]]) -> None:
        """Update active joint limits from calibration data.

        Overrides the default joint limits from tables.py with values discovered
        during calibration. Only provided motors are updated; others keep defaults.

        Args:
            limits: Dict mapping motor name to (min_rad, max_rad) tuple.
        """
        for name, (lo, hi) in limits.items():
            if name in self._motors:
                self._active_joint_limits[name] = (lo, hi)
                logger.info(f"[Damiao] Joint limit updated: {name} = ({lo:.3f}, {hi:.3f}) rad")
