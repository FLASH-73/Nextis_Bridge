
import logging
import time
import threading

logger = logging.getLogger(__name__)

# Damiao motor torque limits (Nm) - 85% of max for safety
DAMIAO_TORQUE_LIMITS = {
    "J8009P": 30.0,  # 85% of 35Nm max
    "J4340P": 6.8,   # 85% of 8Nm max
    "J4310": 3.4,    # 85% of 4Nm max
}

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
                    # We need to know which motors are on this bus.
                    # bus.motors is a dict
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

            return True

        except Exception as e:
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
                        self.emergency_stop(robot)
                        return False
                else:
                    self.violation_counts[motor_name] = 0

            return True

        except Exception as e:
            logger.error(f"Damiao safety check failed: {e}")
            return True  # Don't block on check errors

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

    def emergency_stop(self, robot):
        logger.error("!!! EMERGENCY STOP TRIGGERED !!!")
        if not robot: return
        
        # Use the lock to ensure we interrupt any other thread
        # Note: If we are called FROM the thread holding the lock, this re-entry is fine (RLock)
        # If standard Lock, we might block if we try to acquire.
        # Assuming robot_lock is passed from SystemState which should be RLock or carefully managed.
        # But E-Stop should probably just blast the bus.
        
        try:
            timestamp = time.strftime("%H:%M:%S")
            logger.critical(f"Cutting Power at {timestamp}")
            robot.disconnect() # Disconnect usually disables torque
        except Exception as e:
             logger.error(f"E-Stop failed to disconnect cleanly: {e}")

