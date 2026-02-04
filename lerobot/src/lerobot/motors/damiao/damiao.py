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
    EMIT_VELOCITY_SCALE,
    EMIT_CURRENT_SCALE,
    PID_GAINS,
    GRIPPER_PID,
    GRIPPER_TORQUE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# --- SocketCAN support classes ---

class _SocketCANMotor:
    """Motor object for SocketCAN mode, compatible with trlc_dk1 Motor interface."""
    def __init__(self, can_id, master_id):
        self.SlaveID = can_id
        self.MasterID = master_id
        self.state_q = 0.0
        self.state_dq = 0.0
        self.state_tau = 0.0
        self.last_seen = 0.0
        self.error_flags = 0
        self.param_dict = {}

    def getPosition(self): return self.state_q
    def getVelocity(self): return self.state_dq
    def getTorque(self): return self.state_tau


class _SocketCANDMVariable:
    """DM_variable constants for SocketCAN mode (register IDs)."""
    MST_ID = 7
    ESC_ID = 8
    CTRL_MODE = 10
    ACC = 4
    DEC = 5
    KP_APR = 25
    KI_APR = 26


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

        # Check for parameter write/read response
        if len(data) == 8 and data[2] in (0x33, 0x55):
            slave_id = (data[1] << 8) | data[0]
            rid = data[3]
            with self.lock:
                motor = self.motors.get(slave_id) or self.motors.get(can_id)
            if motor:
                if rid in self.INT_RIDS:
                    val = struct.unpack('<I', bytes(data[4:8]))[0]
                else:
                    val = struct.unpack('<f', bytes(data[4:8]))[0]
                motor.param_dict[rid] = val
            return

        # Status response: motor state (position/velocity/torque)
        # Damiao protocol: motor responds with arbitration_id = MasterID,
        # and ESC_ID (= SlaveID) is in data[0] lower nibble.
        target_motor = None
        with self.lock:
            # Primary: match by ESC_ID from data[0] lower nibble (= SlaveID)
            if len(data) >= 1:
                esc_id = data[0] & 0x0F
                if esc_id in self.motors:
                    target_motor = self.motors[esc_id]
            # Fallback: match by CAN arbitration ID
            if not target_motor and can_id in self.motors:
                target_motor = self.motors[can_id]

        if target_motor and len(data) == 8:
            target_motor.last_seen = time.time()
            target_motor.error_flags = (data[0] >> 4) & 0x0F

            q_uint = (data[1] << 8) | data[2]
            dq_uint = (data[3] << 4) | (data[4] >> 4)
            tau_uint = ((data[4] & 0x0F) << 8) | data[5]

            target_motor.state_q = self._uint_to_float(q_uint, -12.5, 12.5, 16)
            target_motor.state_dq = self._uint_to_float(dq_uint, -30.0, 30.0, 12)
            target_motor.state_tau = self._uint_to_float(tau_uint, -10.0, 10.0, 12)

    def _send(self, arb_id, data):
        import can
        msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)
        try:
            self.bus.send(msg)
        except Exception as e:
            print(f"[Damiao CAN] SEND FAILED arb_id=0x{arb_id:03X}: {e}", flush=True)
            raise

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
        """Write a parameter to motor register."""
        can_id_l = motor.SlaveID & 0xFF
        can_id_h = (motor.SlaveID >> 8) & 0xFF
        if isinstance(value, int) or rid in self.INT_RIDS:
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
        print(f"[Damiao CAN] enable motor SlaveID=0x{motor.SlaveID:02X}, arb_id=0x{motor.SlaveID:03X}", flush=True)
        self._send(motor.SlaveID, data)

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
        time.sleep(0.1)  # Flash write needs time

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

    def __post_init__(self):
        specs = DAMIAO_MOTOR_SPECS.get(self.motor_type, DAMIAO_MOTOR_SPECS["J4310"])
        self.max_torque = specs["max_torque"]
        self.max_velocity = specs["max_velocity"]
        self.torque_constant = specs["torque_constant"]
        self.dm_type = specs["dm_type"]


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
        self.calibration = {}  # For compatibility with LeRobot calibration service

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
            motor = _SocketCANMotor(mcfg.can_id, mcfg.master_id)
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
        }

        for name, mcfg in self._motor_configs.items():
            dm_type = MOTOR_TYPE_MAP.get(mcfg.dm_type, DM_Motor_Type.DM4310)
            motor = Motor(dm_type, mcfg.can_id, mcfg.master_id)
            self._motors[name] = motor
            self._control.addMotor(motor)

    def configure(self) -> None:
        """Configure all motors with safe defaults and PID gains.

        IMPORTANT: Motors are disabled before switching control mode because
        Damiao firmware may ignore mode changes on already-enabled motors.
        Sequence per motor: disable → mode change → PID → enable → verify.
        """
        if not self._is_connected:
            raise RuntimeError("DamiaoMotorsBus not connected")

        DM_variable = self._DM_variable
        Control_Type = self._Control_Type

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

            print(f"[Damiao] Motor '{name}' ({mcfg.motor_type}) responding, current mode={ctrl_mode}", flush=True)

            # CRITICAL: Disable motor before switching control mode
            # Damiao firmware may ignore switchControlMode on enabled motors
            self._control.disable(motor)
            time.sleep(0.05)

            # Switch to position-velocity mode (safest for teleoperation)
            self._control.switchControlMode(motor, Control_Type.POS_VEL)
            time.sleep(0.02)

            # CRITICAL: Save params to EEPROM so mode change actually takes effect
            # Without this, mode shows POS_VEL in RAM but motor may still run MIT mode
            self._control.save_motor_param(motor)
            print(f"[Damiao] Motor '{name}' params saved to EEPROM", flush=True)

            # Set PID gains based on motor type
            gains = PID_GAINS.get(mcfg.motor_type, PID_GAINS["J4310"])

            # Special handling for gripper
            if name == "gripper":
                gains = {**gains, **GRIPPER_PID}

            self._control.change_motor_param(motor, DM_variable.ACC, gains["ACC"])
            self._control.change_motor_param(motor, DM_variable.DEC, gains["DEC"])
            self._control.change_motor_param(motor, DM_variable.KP_APR, gains["KP_APR"])
            self._control.change_motor_param(motor, DM_variable.KI_APR, gains["KI_APR"])

            # Enable motor
            self._control.enable(motor)
            time.sleep(0.05)

            # Verify: read back position to confirm motor is alive in new mode
            self._control.refresh_motor_status(motor)
            time.sleep(0.01)
            pos = motor.getPosition()
            verify_mode = self._control.read_motor_param(motor, DM_variable.CTRL_MODE)
            print(f"[Damiao] Motor '{name}' enabled, mode={verify_mode}, pos={pos:.3f} rad", flush=True)

        print(f"[Damiao] All {len(self._motors)} motors configured, velocity_limit={self._velocity_limit:.2f}", flush=True)
        logger.info(f"[Damiao] All motors configured with velocity_limit={self._velocity_limit:.2f}")

    def sync_read(self, data_name: str, motors: list[str] | None = None, normalize: bool = True) -> dict[str, float]:
        """Read data from motors.

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

        for name in target_motors:
            if name not in self._motors:
                continue
            motor = self._motors[name]
            self._control.refresh_motor_status(motor)

            if data_name == "Present_Position":
                results[name] = motor.getPosition()  # radians
            elif data_name == "Present_Velocity":
                results[name] = motor.getVelocity()  # rad/s
            elif data_name == "Present_Torque":
                results[name] = motor.getTorque()  # Nm
            else:
                logger.warning(f"[Damiao] Unknown data_name: {data_name}")

        return results

    def sync_write(self, data_name: str, values: dict[str, float], normalize: bool = True) -> None:
        """Write goal positions to motors WITH VELOCITY LIMITING.

        CRITICAL: All position commands have velocity limited by self._velocity_limit.

        Args:
            data_name: "Goal_Position"
            values: Dict mapping motor name to goal position (radians)
            normalize: Ignored for Damiao
        """
        if self._discovery_mode:
            return  # Block all writes during calibration discovery

        if not self._is_connected:
            raise RuntimeError("DamiaoMotorsBus not connected")

        if data_name != "Goal_Position":
            logger.warning(f"[Damiao] sync_write only supports Goal_Position, got: {data_name}")
            return

        # Debug: log first sync_write call
        if not hasattr(self, '_sync_write_logged'):
            self._sync_write_logged = True
            print(f"[Damiao] sync_write FIRST CALL: {len(values)} motors, discovery_mode={self._discovery_mode}", flush=True)
            print(f"[Damiao] Motor names in bus: {list(self._motors.keys())}", flush=True)
            print(f"[Damiao] Values received: {values}", flush=True)
            print(f"[Damiao] velocity_limit={self._velocity_limit}", flush=True)

        for name, value in values.items():
            if name not in self._motors:
                if not hasattr(self, '_missing_logged'):
                    self._missing_logged = True
                    print(f"[Damiao] WARNING: motor '{name}' not in self._motors, skipping", flush=True)
                continue

            motor = self._motors[name]
            mcfg = self._motor_configs[name]

            # CRITICAL: Apply global velocity limit
            max_vel = mcfg.max_velocity * self._velocity_limit

            # Send position-velocity command (velocity in rad/s, no EMIT scaling)
            self._control.control_Pos_Vel(motor, value, max_vel)

    def send_gripper_command(self, position: float, max_torque: float = 1.0) -> None:
        """Send gripper command with force limiting.

        Args:
            position: Goal position in radians
            max_torque: Maximum torque in Nm
        """
        if "gripper" not in self._motors:
            return

        motor = self._motors["gripper"]
        mcfg = self._motor_configs["gripper"]

        # Apply velocity limit
        max_vel = mcfg.max_velocity * self._velocity_limit

        # Calculate current limit from torque
        i_des = max_torque / mcfg.torque_constant * EMIT_CURRENT_SCALE

        self._control.control_pos_force(motor, position, max_vel * EMIT_VELOCITY_SCALE, i_des=i_des)

    def home_gripper(self) -> None:
        """Auto-home gripper using torque detection.

        Opens gripper until torque spike detected, then sets zero position.
        """
        if "gripper" not in self._motors:
            logger.warning("[Damiao] No gripper motor configured")
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

            # Disable -> set zero -> wait -> enable
            self._control.disable(motor)
            time.sleep(0.05)
            self._control.set_zero_position(motor)
            time.sleep(0.2)
            self._control.enable(motor)
            time.sleep(0.05)

            logger.info(f"[Damiao] Zero set for {name} (was {previous_positions[name]:.3f} rad)")

        return previous_positions

    def enable_torque(self, motors: list[str] | None = None) -> None:
        """Enable torque on specified motors."""
        if not self._is_connected:
            return

        target = motors if motors else list(self._motors.keys())
        for name in target:
            if name in self._motors:
                self._control.enable(self._motors[name])

    def disable_torque(self, motors: list[str] | None = None, num_retry: int = 0) -> None:
        """Disable torque on specified motors (make backdrivable)."""
        if not self._is_connected:
            return

        target = motors if motors else list(self._motors.keys())
        for name in target:
            if name in self._motors:
                self._control.disable(self._motors[name])

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

        if self._use_socketcan:
            if hasattr(self._control, 'shutdown'):
                self._control.shutdown()
        elif self._serial:
            self._serial.close()

        self._is_connected = False
        logger.info("[Damiao] Disconnected")

    def read_torques(self) -> dict[str, float]:
        """Read current torques from all motors (for safety monitoring)."""
        return self.sync_read("Present_Torque")

    def get_torque_limits(self) -> dict[str, float]:
        """Get torque limits for each motor (85% of max)."""
        limits = {}
        for name, mcfg in self._motor_configs.items():
            specs = DAMIAO_MOTOR_SPECS.get(mcfg.motor_type, {})
            max_torque = specs.get("max_torque", 10.0)
            limit_percent = specs.get("torque_limit_percent", 0.85)
            limits[name] = max_torque * limit_percent
        return limits
