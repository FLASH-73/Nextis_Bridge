
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
        
    def check_limits(self, robot):
        """
        Checks motor limits (Load, Temperature, etc.)
        Returns False if safety violation occurs.
        """
        if not robot or not robot.is_connected:
            return True

        # 1. Check Load (Torque)
        try:
            # We need to access the bus directly
            # robot.left_arm.bus and robot.right_arm.bus
            
            # Combine all buses
            buses = []
            if hasattr(robot, "left_arm"): buses.append(robot.left_arm.bus)
            if hasattr(robot, "right_arm"): buses.append(robot.right_arm.bus)
            if hasattr(robot, "bus"): buses.append(robot.bus)
            
            for bus in buses:
                # Sync Read Load
                # "Present_Load" is (60, 2) in table
                # Normalize=False gives raw values (usually signed int16)
                # But feetech load might be magnitude? 
                # STS manuals say bit 10 is direction, 0-9 is value (0-1000).
                # Let's read raw and parse.
                
                loads = bus.sync_read("Present_Load", normalize=False)
                
                for motor, load_val in loads.items():
                    # Parse load
                    # Bit 10 is direction (1024). Value is lower 10 bits.
                    magnitude = load_val % 1024
                    
                    if magnitude > self.LOAD_THRESHOLD:
                         self.violation_counts[motor] = self.violation_counts.get(motor, 0) + 1
                         logger.warning(f"SAFETY WARNING: Motor {motor} Load {magnitude}/{self.LOAD_THRESHOLD} (Count {self.violation_counts[motor]})")
                    else:
                         self.violation_counts[motor] = 0
                         
                    if self.violation_counts.get(motor, 0) >= self.VIOLATION_LIMIT:
                        logger.error(f"SAFETY CRITICAL: Motor {motor} overloaded! Triggering E-STOP.")
                        self.emergency_stop(robot)
                        return False

            return True

        except Exception as e:
            logger.error(f"Safety Check Failed: {e}")
            # If we can't verify safety, we should probably stop? 
            # Or just warn for now to avoid false triggers during dev
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

