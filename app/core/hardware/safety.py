
import logging
import time
import threading

logger = logging.getLogger(__name__)

class SafetyLayer:
    def __init__(self, robot_lock):
        self.lock = robot_lock
        # Thresholds
        # STS3215 Load is 0-1000. 1000 = 100% Torque.
        # Setting to 500 (50%) to be safe.
        self.LOAD_THRESHOLD = 500
        # Consecutive violations to trigger (debounce)
        self.VIOLATION_LIMIT = 3
        self.violation_counts = {}
        self.latest_loads = {}  # Store latest readings for UI
        self.latest_torques = {}  # Store Damiao torque readings

        self.monitored_motors = []
        self.current_motor_index = 0

        # Track if we have a Damiao robot
        self._is_damiao_robot = None

        # Fail-closed: consecutive check-level failures trigger unsafe
        # A single transient CAN error won't kill the system, but persistent
        # failures indicate a broken safety monitor.
        self.CHECK_FAILURE_LIMIT = 5
        self._check_failure_count = 0
        self._damiao_check_failure_count = 0

    def check_limits(self, robot):
        """
        Checks motor limits (Load, Temperature, etc.)
        Returns False if safety violation occurs.
        """
        if not robot or not robot.is_connected:
            return True

        try:
            # Initialize monitored motors list if empty or robot changed (simplified check)
            if not self.monitored_motors:
                buses = []
                if hasattr(robot, "left_arm"): buses.append(robot.left_arm.bus)
                if hasattr(robot, "right_arm"): buses.append(robot.right_arm.bus)
                if hasattr(robot, "bus"): buses.append(robot.bus)

                for bus in buses:
                    # Skip Damiao CAN buses — they use check_damiao_limits() with
                    # native torque API instead of Feetech bus.read("Present_Load")
                    try:
                        from lerobot.motors.damiao.damiao import DamiaoMotorsBus
                        if isinstance(bus, DamiaoMotorsBus):
                            continue
                    except ImportError:
                        pass
                    for motor_name in bus.motors.keys():
                        self.monitored_motors.append((bus, motor_name))

                if not self.monitored_motors:
                    return True

            # Check a batch of motors (Round-Robin)
            # Check 1 motor per call to minimize lock holding time in the main thread
            batch_size = 1

            for _ in range(batch_size):
                if not self.monitored_motors: break

                self.current_motor_index = (self.current_motor_index + 1) % len(self.monitored_motors)
                bus, motor = self.monitored_motors[self.current_motor_index]

                try:
                    # Read specific motor load
                    load_val = bus.read("Present_Load", motor, normalize=False)
                    magnitude = load_val % 1024
                    self.latest_loads[motor] = magnitude

                    if magnitude > self.LOAD_THRESHOLD:
                         self.violation_counts[motor] = self.violation_counts.get(motor, 0) + 1
                         logger.warning(f"SAFETY WARNING: Motor {motor} Load {magnitude}/{self.LOAD_THRESHOLD} (Count {self.violation_counts[motor]})")
                    else:
                         self.violation_counts[motor] = 0

                    if self.violation_counts.get(motor, 0) >= self.VIOLATION_LIMIT:
                        logger.error(f"SAFETY CRITICAL: Motor {motor} overloaded! Triggering E-STOP.")
                        self.emergency_stop(robot)
                        return False

                except Exception as e:
                    # Silent fail for single read error to robustify
                    pass

            self._check_failure_count = 0
            return True

        except Exception as e:
            self._check_failure_count += 1
            if self._check_failure_count >= self.CHECK_FAILURE_LIMIT:
                logger.error(f"Safety check failed {self._check_failure_count} times consecutively: {e} — treating as UNSAFE")
                return False
            logger.error(f"Safety Check Failed: {e}")
            return True

    def check_damiao_limits(self, robot):
        """Check Damiao motor torque limits.

        Damiao motors report torque in Nm. We check against motor-specific limits.
        Returns False if safety violation occurs.
        """
        if not robot or not robot.is_connected:
            return True

        # Check if this is a Damiao robot
        if not hasattr(robot, 'get_torques') or not hasattr(robot, 'get_torque_limits'):
            return True

        try:
            # Read torques from all motors
            torques = robot.get_torques()
            limits = robot.get_torque_limits()

            for motor_name, torque in torques.items():
                self.latest_torques[motor_name] = torque
                limit = limits.get(motor_name, 10.0)  # Default 10Nm limit

                if abs(torque) > limit:
                    self.violation_counts[motor_name] = self.violation_counts.get(motor_name, 0) + 1
                    logger.warning(
                        f"SAFETY WARNING: Damiao {motor_name} torque {torque:.2f}Nm > {limit:.1f}Nm "
                        f"(Count {self.violation_counts[motor_name]})"
                    )

                    if self.violation_counts.get(motor_name, 0) >= self.VIOLATION_LIMIT:
                        logger.error(f"SAFETY CRITICAL: Damiao {motor_name} overloaded!")
                        return False  # Control loop → stop() → homing → disable
                else:
                    self.violation_counts[motor_name] = 0

            self._damiao_check_failure_count = 0
            return True

        except Exception as e:
            self._damiao_check_failure_count += 1
            if self._damiao_check_failure_count >= self.CHECK_FAILURE_LIMIT:
                logger.error(f"Damiao safety check failed {self._damiao_check_failure_count} times consecutively: {e} — treating as UNSAFE")
                return False
            logger.error(f"Damiao safety check failed: {e}")
            return True

    def check_all_limits(self, robot):
        """Check all applicable safety limits for the robot.

        Automatically detects robot type and applies appropriate checks.
        """
        # Check standard limits (STS3215)
        if not self.check_limits(robot):
            return False

        # Check Damiao-specific limits if applicable
        if hasattr(robot, 'get_torques'):
            if not self.check_damiao_limits(robot):
                return False

        return True

    def emergency_stop(self, robot, motor_type=None):
        """Emergency stop: freeze Damiao at current position, then release.

        For Damiao arms: command current positions for 500ms using MIT kp/kd
        to prevent gravity drop, then disconnect.

        For Feetech/Dynamixel: disconnect immediately (stall under load
        risks gear damage, and lower inertia makes gravity drop less dangerous).

        Does NOT acquire self.lock — E-STOP may be called from the thread
        holding the lock (state.lock is threading.Lock, not reentrant).
        """
        logger.error("!!! EMERGENCY STOP TRIGGERED !!!")
        if not robot:
            return

        try:
            timestamp = time.strftime("%H:%M:%S")
            logger.critical(f"E-STOP at {timestamp}")

            # Determine if this is a Damiao robot
            is_damiao = motor_type == "damiao"
            if not is_damiao and hasattr(robot, 'bus'):
                try:
                    from lerobot.motors.damiao.damiao import DamiaoMotorsBus
                    is_damiao = isinstance(robot.bus, DamiaoMotorsBus)
                except ImportError:
                    pass

            if is_damiao:
                self._estop_damiao(robot)
            else:
                self._estop_generic(robot)

        except Exception as e:
            logger.error(f"E-Stop failed: {e} — forcing disconnect")
            try:
                robot.disconnect()
            except Exception:
                pass

    def _estop_damiao(self, robot):
        """Damiao E-STOP: hold current position for 500ms, then release.

        Reads last known positions from the bus and sends them as hold commands
        using the normal MIT kp/kd gains. This prevents gravity drop on loaded
        arms (e.g. 35Nm J8009P holding a payload).
        """
        try:
            # Build hold action from last known positions
            hold_action = {}
            if hasattr(robot, 'bus') and hasattr(robot.bus, '_last_positions'):
                for motor_name, pos in robot.bus._last_positions.items():
                    hold_action[f"{motor_name}.pos"] = pos

            if hold_action:
                logger.critical(f"Damiao HOLD phase: holding {len(hold_action)} joints for 500ms")
                hold_end = time.time() + 0.5
                while time.time() < hold_end:
                    try:
                        robot.send_action(hold_action)
                    except Exception:
                        break  # If send fails, proceed to disconnect
                    time.sleep(0.016)  # ~60Hz hold rate
            else:
                logger.warning("Damiao E-STOP: no last positions available, skipping hold phase")
        except Exception as e:
            logger.error(f"E-STOP hold phase failed: {e} — proceeding to disconnect")

        # Release: disconnect disables torque
        try:
            robot.disconnect()
        except Exception as e:
            logger.error(f"E-STOP disconnect failed: {e}")

    def _estop_generic(self, robot):
        """Generic E-STOP: immediate disconnect (Feetech, Dynamixel, etc.)."""
        try:
            robot.disconnect()
        except Exception as e:
            logger.error(f"E-STOP disconnect failed: {e}")
