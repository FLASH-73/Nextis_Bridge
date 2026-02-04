# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from copy import deepcopy
from enum import Enum
from pprint import pformat

from lerobot.motors.encoding_utils import (
    decode_sign_magnitude,
    decode_twos_complement,
    encode_sign_magnitude,
    encode_twos_complement,
)

from ..motors_bus import Motor, MotorCalibration, MotorsBus, NameOrID, Value, get_address
from .tables import (
    FIRMWARE_MAJOR_VERSION,
    FIRMWARE_MINOR_VERSION,
    MODEL_BAUDRATE_TABLE,
    MODEL_CONTROL_TABLE,
    MODEL_ENCODING_TABLE,
    MODEL_NUMBER,
    MODEL_NUMBER_TABLE,
    MODEL_PROTOCOL,
    MODEL_RESOLUTION,
    SCAN_BAUDRATES,
)

DEFAULT_PROTOCOL_VERSION = 0
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_TIMEOUT_MS = 1000

NORMALIZED_DATA = ["Goal_Position", "Present_Position"]

logger = logging.getLogger(__name__)


class OperatingMode(Enum):
    # position servo mode
    POSITION = 0
    # The motor is in constant speed mode, which is controlled by parameter 0x2e, and the highest bit 15 is
    # the direction bit
    VELOCITY = 1
    # PWM open-loop speed regulation mode, with parameter 0x2c running time parameter control, bit11 as
    # direction bit
    PWM = 2
    # In step servo mode, the number of step progress is represented by parameter 0x2a, and the highest bit 15
    # is the direction bit
    STEP = 3


class DriveMode(Enum):
    NON_INVERTED = 0
    INVERTED = 1


class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


def _split_into_byte_chunks(value: int, length: int) -> list[int]:
    import scservo_sdk as scs

    if length == 1:
        data = [value]
    elif length == 2:
        data = [scs.SCS_LOBYTE(value), scs.SCS_HIBYTE(value)]
    elif length == 4:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),
        ]
    return data


def patch_setPacketTimeout(self, packet_length):  # noqa: N802
    """
    HACK: This patches the PortHandler behavior to set the correct packet timeouts.

    It fixes https://gitee.com/ftservo/SCServoSDK/issues/IBY2S6
    The bug is fixed on the official Feetech SDK repo (https://gitee.com/ftservo/FTServo_Python)
    but because that version is not published on PyPI, we rely on the (unofficial) on that is, which needs
    patching.
    """
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


class FeetechMotorsBus(MotorsBus):
    """
    The FeetechMotorsBus class allows to efficiently read and write to the attached motors. It relies on the
    python feetech sdk to communicate with the motors, which is itself based on the dynamixel sdk.
    """

    apply_drive_mode = True
    available_baudrates = deepcopy(SCAN_BAUDRATES)
    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS
    model_baudrate_table = deepcopy(MODEL_BAUDRATE_TABLE)
    model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
    model_encoding_table = deepcopy(MODEL_ENCODING_TABLE)
    model_number_table = deepcopy(MODEL_NUMBER_TABLE)
    model_resolution_table = deepcopy(MODEL_RESOLUTION)
    normalized_data = deepcopy(NORMALIZED_DATA)

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        protocol_version: int = DEFAULT_PROTOCOL_VERSION,
    ):
        super().__init__(port, motors, calibration)
        self.protocol_version = protocol_version
        self._assert_same_protocol()
        import scservo_sdk as scs

        self.port_handler = scs.PortHandler(self.port)
        # HACK: monkeypatch
        self.port_handler.setPacketTimeout = patch_setPacketTimeout.__get__(
            self.port_handler, scs.PortHandler
        )
        self.packet_handler = scs.PacketHandler(protocol_version)
        self.sync_reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
        self.sync_writer = scs.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)
        self._comm_success = scs.COMM_SUCCESS
        self._no_error = 0x00

        if any(MODEL_PROTOCOL[model] != self.protocol_version for model in self.models):
            raise ValueError(f"Some motors are incompatible with protocol_version={self.protocol_version}")

    def _assert_same_protocol(self) -> None:
        if any(MODEL_PROTOCOL[model] != self.protocol_version for model in self.models):
            raise RuntimeError("Some motors use an incompatible protocol.")

    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        if instruction_name == "sync_read" and self.protocol_version == 1:
            raise NotImplementedError(
                "'Sync Read' is not available with Feetech motors using Protocol 1. Use 'Read' sequentially instead."
            )
        if instruction_name == "broadcast_ping" and self.protocol_version == 1:
            raise NotImplementedError(
                "'Broadcast Ping' is not available with Feetech motors using Protocol 1. Use 'Ping' sequentially instead."
            )

    def _assert_same_firmware(self) -> None:
        firmware_versions = self._read_firmware_version(self.ids, raise_on_error=True)
        if len(set(firmware_versions.values())) != 1:
            raise RuntimeError(
                "Some Motors use different firmware versions:"
                f"\n{pformat(firmware_versions)}\n"
                "Update their firmware first using Feetech's software. "
                "Visit https://www.feetechrc.com/software."
            )

    def _handshake(self) -> None:
        self._assert_motors_exist()
        self._assert_same_firmware()

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        if self.protocol_version == 0:
            return self._find_single_motor_p0(motor, initial_baudrate)
        else:
            return self._find_single_motor_p1(motor, initial_baudrate)

    def _find_single_motor_p0(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        model = self.motors[motor].model
        search_baudrates = (
            [initial_baudrate] if initial_baudrate is not None else self.model_baudrate_table[model]
        )
        expected_model_nb = self.model_number_table[model]

        for baudrate in search_baudrates:
            self.set_baudrate(baudrate)
            id_model = self.broadcast_ping()
            if id_model:
                found_id, found_model = next(iter(id_model.items()))
                if found_model != expected_model_nb:
                    raise RuntimeError(
                        f"Found one motor on {baudrate=} with id={found_id} but it has a "
                        f"model number '{found_model}' different than the one expected: '{expected_model_nb}'. "
                        f"Make sure you are connected only connected to the '{motor}' motor (model '{model}')."
                    )
                return baudrate, found_id

        raise RuntimeError(f"Motor '{motor}' (model '{model}') was not found. Make sure it is connected.")

    def _find_single_motor_p1(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        import scservo_sdk as scs

        model = self.motors[motor].model
        search_baudrates = (
            [initial_baudrate] if initial_baudrate is not None else self.model_baudrate_table[model]
        )
        expected_model_nb = self.model_number_table[model]

        for baudrate in search_baudrates:
            self.set_baudrate(baudrate)
            for id_ in range(scs.MAX_ID + 1):
                found_model = self.ping(id_)
                if found_model is not None:
                    if found_model != expected_model_nb:
                        raise RuntimeError(
                            f"Found one motor on {baudrate=} with id={id_} but it has a "
                            f"model number '{found_model}' different than the one expected: '{expected_model_nb}'. "
                            f"Make sure you are connected only connected to the '{motor}' motor (model '{model}')."
                        )
                    return baudrate, id_

        raise RuntimeError(f"Motor '{motor}' (model '{model}') was not found. Make sure it is connected.")

    def configure_motors(self, return_delay_time=0, maximum_acceleration=254, acceleration=254) -> None:
        for motor in self.motors:
            # By default, Feetech motors have a 500µs delay response time (corresponding to a value of 250 on
            # the 'Return_Delay_Time' address). We ensure this is reduced to the minimum of 2µs (value of 0).
            self.write("Return_Delay_Time", motor, return_delay_time)
            # Set 'Maximum_Acceleration' to 254 to speedup acceleration and deceleration of the motors.
            if self.protocol_version == 0:
                self.write("Maximum_Acceleration", motor, maximum_acceleration)
            self.write("Acceleration", motor, acceleration)

    @property
    def is_calibrated(self) -> bool:
        motors_calibration = self.read_calibration()
        if set(motors_calibration) != set(self.calibration):
            return False

        same_ranges = all(
            self.calibration[motor].range_min == cal.range_min
            and self.calibration[motor].range_max == cal.range_max
            for motor, cal in motors_calibration.items()
        )
        if self.protocol_version == 1:
            return same_ranges

        same_offsets = all(
            self.calibration[motor].homing_offset == cal.homing_offset
            for motor, cal in motors_calibration.items()
        )
        return same_ranges and same_offsets

    def read_calibration(self) -> dict[str, MotorCalibration]:
        offsets, mins, maxes = {}, {}, {}
        for motor in self.motors:
            mins[motor] = self.read("Min_Position_Limit", motor, normalize=False)
            maxes[motor] = self.read("Max_Position_Limit", motor, normalize=False)
            offsets[motor] = (
                self.read("Homing_Offset", motor, normalize=False) if self.protocol_version == 0 else 0
            )

        calibration = {}
        for motor, m in self.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=offsets[motor],
                range_min=mins[motor],
                range_max=maxes[motor],
            )

        return calibration

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        for motor, calibration in calibration_dict.items():
            if self.protocol_version == 0:
                self.write("Homing_Offset", motor, calibration.homing_offset)
            self.write("Min_Position_Limit", motor, calibration.range_min)
            self.write("Max_Position_Limit", motor, calibration.range_max)

        if cache:
            self.calibration = calibration_dict

    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        """
        On Feetech Motors:
        Present_Position = Actual_Position - Homing_Offset
        """
        half_turn_homings = {}
        for motor, pos in positions.items():
            model = self._get_motor_model(motor)
            max_res = self.model_resolution_table[model] - 1
            half_turn_homings[motor] = pos - int(max_res / 2)

        return half_turn_homings

    def set_half_turn_homings(self, motors: str | list[str] | None = None) -> None:
        """
        Sets the current position as the center (homing) position (2048).
        This writes to the Homing_Offset register.
        Strategy: Reset Offset to 0, Read Raw, Calculate Offset = Raw - 2048, Write Offset.
        """
        names = self._get_motors_list(motors)
        import time
        
        # 1. Disable Torque & Reset Offset to 0
        self.disable_torque(names)
        for motor in names:
            self.write("Homing_Offset", motor, 0)
        
        # Wait for write to take effect
        time.sleep(0.2)
        
        # 2. Read Raw Positions (with Offset=0)
        # Note: sync_read might still apply local cache of offset if not careful? 
        # But we act on hardware registers.
        # normalize=False returns raw sensor data if Offset is 0
        raw_positions = self.sync_read("Present_Position", names, normalize=False)
        
        new_offsets = {}
        for motor in names:
            if motor not in raw_positions:
                continue
            
            # Position = Raw - Offset
            # We want Position = 2048
            # 2048 = Raw - Offset => Offset = Raw - 2048
            raw = raw_positions[motor]
            target_offset = raw - 2048
            
            # Wrap to 16-bit signed integer range (-32768 to 32767)
            # STS3215 Homing Offset is signed 16-bit
            target_offset = target_offset % 65536
            if target_offset >= 32768:
                target_offset -= 65536
                
            new_offsets[motor] = target_offset
            
            logger.info(f"Homing {motor}: Raw={raw}, Target 2048, New_Offset={target_offset}")
            
            self.write("Homing_Offset", motor, target_offset)
            
            # Update local calibration cache
            if motor in self.calibration:
                self.calibration[motor].homing_offset = target_offset
                
        # Wait for write
        time.sleep(0.2)
        
        # Verify?
        return new_offsets

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        motor_list = self._get_motors_list(motors)
        logger.info(f"Disabling torque for: {motor_list}")
        for motor in motor_list:
            # Write Lock=0 first to ensure we can write to Torque_Enable
            self.write("Lock", motor, 0, num_retry=num_retry)
            self.write("Torque_Enable", motor, TorqueMode.DISABLED.value, num_retry=num_retry)
            import time
            time.sleep(0.01)

    def emergency_stop_broadcast(self) -> None:
        """
        Sends a BROADCAST disable torque command to ID 254.
        This ignores return packets and tries to forcibly stop everything.
        Critical for fail-safe scenarios where specific motors might be unresponsive.
        """
        import scservo_sdk as scs
        import time
        
        logger.critical(f"Executing BROADCAST EMERGENCY STOP on {self.port}...")
        
        if not self.motors:
            return

        # Use first motor to resolve addresses (Assume homogeneous bus)
        first_motor = list(self.motors.values())[0]
        model = first_motor.model
        
        # Get Addresses
        # Note: access table directly or use helper? self.model_ctrl_table is available.
        try:
            addr_torque = self.model_ctrl_table[model]["Torque_Enable"][0]
            addr_lock = self.model_ctrl_table[model]["Lock"][0]
        except Exception as e:
            logger.error(f"Failed to resolve register addresses for E-Stop: {e}")
            # Fallback to standard STS3215 addresses if lookup fails?
            # Lock=48, Torque=40
            addr_lock = 48
            addr_torque = 40
            
        # 1. Flush Port (Clear Garbage from stuck motors)
        try:
            self.port_handler.clearPort()
        except: 
            pass
            
        # 2. Broadcast KILL (Spam it)
        # ID 0xFE (254) is Broadcast
        for i in range(5):
            try:
                # Unlock
                self.packet_handler.write1ByteTxRx(self.port_handler, scs.BROADCAST_ID, addr_lock, 0)
                # Disable Torque
                self.packet_handler.write1ByteTxRx(self.port_handler, scs.BROADCAST_ID, addr_torque, 0)
                # No sleep? We want it FAST. Maybe tiny sleep to let bytes out.
                # time.sleep(0.002) 
            except Exception as e:
                logger.error(f"Broadcast E-Stop Tx Fail ({i}): {e}")
                
        # 3. Flush again
        try:
            self.port_handler.clearPort()
        except:
            pass

    def attempt_error_recovery(self, motor_ids: list[int] | None = None) -> dict[int, dict]:
        """
        Attempts to clear error conditions on motors before handshake.

        Strategy:
        1. Clear serial port buffer
        2. Broadcast torque disable (clears error by sending command)
        3. Wait briefly for motor state to settle
        4. Read Status register to check current errors

        Returns:
            dict[int, dict]: Per-motor status with keys:
                - 'reachable': Motor responds to ping
                - 'status_raw': Raw status register value
                - 'error_type': Human-readable error description
                - 'recovered': Whether error was cleared
        """
        import time
        import scservo_sdk as scs

        # Step 1: Clear port buffer
        try:
            self.port_handler.clearPort()
        except:
            pass

        # Step 2: Get register addresses from first motor's model
        first_motor = list(self.motors.values())[0]
        model = first_motor.model
        addr_torque = self.model_ctrl_table[model]["Torque_Enable"][0]
        addr_lock = self.model_ctrl_table[model]["Lock"][0]
        addr_status = self.model_ctrl_table[model]["Status"][0]

        logger.info(f"Attempting error recovery on port {self.port}...")

        # Step 3: Broadcast unlock + torque disable (spam to ensure delivery)
        for _ in range(3):
            try:
                self.packet_handler.write1ByteTxRx(self.port_handler, scs.BROADCAST_ID, addr_lock, 0)
                self.packet_handler.write1ByteTxRx(self.port_handler, scs.BROADCAST_ID, addr_torque, 0)
            except Exception as e:
                logger.warning(f"Recovery broadcast failed: {e}")

        # Step 4: Wait for motors to process
        time.sleep(0.1)
        try:
            self.port_handler.clearPort()
        except:
            pass

        # Step 5: Check each motor's status
        motor_ids = motor_ids or self.ids
        results = {}

        for id_ in motor_ids:
            result = {'reachable': False, 'status_raw': None, 'error_type': None, 'recovered': False}

            # Try ping
            model_nb = self.ping(id_, num_retry=2)
            if model_nb is not None:
                result['reachable'] = True

                # Read Status register
                try:
                    status_val, comm, error = self._read(addr_status, 1, id_, num_retry=1, raise_on_error=False)
                    if self._is_comm_success(comm):
                        result['status_raw'] = status_val
                        result['error_type'] = self._decode_status_bits(status_val)
                        result['recovered'] = (status_val == 0)
                        if not result['recovered']:
                            logger.warning(f"Motor {id_} has error status: {result['error_type']}")
                except Exception as e:
                    logger.debug(f"Could not read status for motor {id_}: {e}")
            else:
                logger.warning(f"Motor {id_} not responding to ping")

            results[id_] = result

        recovered_count = sum(1 for r in results.values() if r['recovered'])
        logger.info(f"Recovery complete: {recovered_count}/{len(results)} motors OK")

        return results

    def _decode_status_bits(self, status: int) -> str:
        """Decode Feetech status register bits to human-readable string."""
        if status == 0:
            return "None"
        errors = []
        if status & 0x01: errors.append("Input Voltage Error")
        if status & 0x02: errors.append("Angle Limit Error")
        if status & 0x04: errors.append("Overheating Error")
        if status & 0x08: errors.append("Range Error")
        if status & 0x10: errors.append("Checksum Error")
        if status & 0x20: errors.append("Overload Error")
        if status & 0x40: errors.append("Instruction Error")
        return ", ".join(errors) if errors else "Unknown"

    def _disable_torque(self, motor_id: int, model: str, num_retry: int = 0) -> None:
        addr, length = get_address(self.model_ctrl_table, model, "Lock")
        self._write(addr, length, motor_id, 0, num_retry=num_retry)
        addr, length = get_address(self.model_ctrl_table, model, "Torque_Enable")
        self._write(addr, length, motor_id, TorqueMode.DISABLED.value, num_retry=num_retry)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, TorqueMode.ENABLED.value, num_retry=num_retry)
            self.write("Lock", motor, 1, num_retry=num_retry)

    def set_operating_mode(self, mode: OperatingMode, motors: str | list[str] | None = None) -> None:
        """
        Switch operating mode (Position, Velocity, PWM).
        Torque must be disabled before changing mode!
        """
        names = self._get_motors_list(motors)
        logger.info(f"Switching {names} to Mode {mode.name} ({mode.value})")
        
        # Ensure torque is disabled
        self.disable_torque(names)
        
        for motor in names:
             # Lock=0 to write EPROM
             self.write("Lock", motor, 0)
             self.write("Operating_Mode", motor, mode.value)
             self.write("Lock", motor, 1) # Lock back? Usually safer.

    def write_pwm(self, pwm_values: dict[str, int]) -> None:
        """
        Write PWM (Current) values to motors.
        Mode must be PWM (2).
        Value range: -1000 to 1000 (usually). 
        """
        for motor, pwm in pwm_values.items():
            # In PWM Mode (2), "Goal_Time" (Addr 44) acts as the PWM duty cycle control
            # Limit check? -1000 to 1000
            val = max(-1000, min(1000, int(pwm)))
            self.write("Goal_Time", motor, val)

    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        for id_ in ids_values:
            model = self._id_to_model(id_)
            encoding_table = self.model_encoding_table.get(model)
            if encoding_table and data_name in encoding_table:
                encoding = encoding_table[data_name]
                if isinstance(encoding, tuple) and encoding[0] == "twos":
                    ids_values[id_] = encode_twos_complement(ids_values[id_], encoding[1])
                else:
                    sign_bit = encoding
                    ids_values[id_] = encode_sign_magnitude(ids_values[id_], sign_bit)

        return ids_values

    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        for id_ in ids_values:
            model = self._id_to_model(id_)
            encoding_table = self.model_encoding_table.get(model)
            if encoding_table and data_name in encoding_table:
                encoding = encoding_table[data_name]
                if isinstance(encoding, tuple) and encoding[0] == "twos":
                    ids_values[id_] = decode_twos_complement(ids_values[id_], encoding[1])
                else:
                    sign_bit = encoding
                    ids_values[id_] = decode_sign_magnitude(ids_values[id_], sign_bit)

        return ids_values

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        return _split_into_byte_chunks(value, length)

    def _broadcast_ping(self) -> tuple[dict[int, int], int]:
        import scservo_sdk as scs

        data_list = {}

        status_length = 6

        rx_length = 0
        wait_length = status_length * scs.MAX_ID

        txpacket = [0] * 6

        tx_time_per_byte = (1000.0 / self.port_handler.getBaudRate()) * 10.0

        txpacket[scs.PKT_ID] = scs.BROADCAST_ID
        txpacket[scs.PKT_LENGTH] = 2
        txpacket[scs.PKT_INSTRUCTION] = scs.INST_PING

        result = self.packet_handler.txPacket(self.port_handler, txpacket)
        if result != scs.COMM_SUCCESS:
            self.port_handler.is_using = False
            return data_list, result

        # set rx timeout
        self.port_handler.setPacketTimeoutMillis((wait_length * tx_time_per_byte) + (3.0 * scs.MAX_ID) + 16.0)

        rxpacket = []
        while not self.port_handler.isPacketTimeout() and rx_length < wait_length:
            rxpacket += self.port_handler.readPort(wait_length - rx_length)
            rx_length = len(rxpacket)

        self.port_handler.is_using = False

        if rx_length == 0:
            return data_list, scs.COMM_RX_TIMEOUT

        while True:
            if rx_length < status_length:
                return data_list, scs.COMM_RX_CORRUPT

            # find packet header
            for idx in range(0, (rx_length - 1)):
                if (rxpacket[idx] == 0xFF) and (rxpacket[idx + 1] == 0xFF):
                    break

            if idx == 0:  # found at the beginning of the packet
                # calculate checksum
                checksum = 0
                for idx in range(2, status_length - 1):  # except header & checksum
                    checksum += rxpacket[idx]

                checksum = ~checksum & 0xFF
                if rxpacket[status_length - 1] == checksum:
                    result = scs.COMM_SUCCESS
                    data_list[rxpacket[scs.PKT_ID]] = rxpacket[scs.PKT_ERROR]

                    del rxpacket[0:status_length]
                    rx_length = rx_length - status_length

                    if rx_length == 0:
                        return data_list, result
                else:
                    result = scs.COMM_RX_CORRUPT
                    # remove header (0xFF 0xFF)
                    del rxpacket[0:2]
                    rx_length = rx_length - 2
            else:
                # remove unnecessary packets
                del rxpacket[0:idx]
                rx_length = rx_length - idx

    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        self._assert_protocol_is_compatible("broadcast_ping")
        for n_try in range(1 + num_retry):
            ids_status, comm = self._broadcast_ping()
            if self._is_comm_success(comm):
                break
            logger.debug(f"Broadcast ping failed on port '{self.port}' ({n_try=})")
            logger.debug(self.packet_handler.getTxRxResult(comm))

        if not self._is_comm_success(comm):
            if raise_on_error:
                raise ConnectionError(self.packet_handler.getTxRxResult(comm))
            return

        ids_errors = {id_: status for id_, status in ids_status.items() if self._is_error(status)}
        if ids_errors:
            display_dict = {id_: self.packet_handler.getRxPacketError(err) for id_, err in ids_errors.items()}
            logger.error(f"Some motors found returned an error status:\n{pformat(display_dict, indent=4)}")

        return self._read_model_number(list(ids_status), raise_on_error)

    def _read_firmware_version(self, motor_ids: list[int], raise_on_error: bool = False) -> dict[int, str]:
        firmware_versions = {}
        for id_ in motor_ids:
            firm_ver_major, comm, error = self._read(
                *FIRMWARE_MAJOR_VERSION, id_, raise_on_error=raise_on_error
            )
            if not self._is_comm_success(comm) or self._is_error(error):
                continue

            firm_ver_minor, comm, error = self._read(
                *FIRMWARE_MINOR_VERSION, id_, raise_on_error=raise_on_error
            )
            if not self._is_comm_success(comm) or self._is_error(error):
                continue

            firmware_versions[id_] = f"{firm_ver_major}.{firm_ver_minor}"

        return firmware_versions

    def _read_model_number(self, motor_ids: list[int], raise_on_error: bool = False) -> dict[int, int]:
        model_numbers = {}
        for id_ in motor_ids:
            model_nb, comm, error = self._read(*MODEL_NUMBER, id_, raise_on_error=raise_on_error)
            if not self._is_comm_success(comm) or self._is_error(error):
                continue

            model_numbers[id_] = model_nb

        return model_numbers

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Value]:
        if self.protocol_version == 1:
            # Fallback to sequential read for Protocol 1
            names = self._get_motors_list(motors)
            results = {}
            for motor in names:
                results[motor] = self.read(data_name, motor, normalize=normalize, num_retry=num_retry)
            return results
        else:
            return super().sync_read(data_name, motors, normalize=normalize, num_retry=num_retry)
