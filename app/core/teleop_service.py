
import time
import threading
import logging
from collections import deque
import numpy as np

from app.core.safety_layer import SafetyLayer
# Try to import precise_sleep, fallback to time.sleep if not found (though it should be there)
try:
    from lerobot.utils.robot_utils import precise_sleep
except ImportError:
    def precise_sleep(dt):
        time.sleep(max(0, dt))

logger = logging.getLogger(__name__)

class TeleoperationService:
    def __init__(self, robot, leader, robot_lock):
        self.robot = robot
        self.leader = leader
        self.robot_lock = robot_lock
        
        self.safety = SafetyLayer(robot_lock) # Initialize Safety Layer
        
        self.is_running = False
        self.thread = None
        self.safety_thread = None
        
        # Teleop frequency
        self.frequency = 100 # Increased to 100Hz for 10ms loop
        self.dt = 1.0 / self.frequency
        
        # Data storage for Graph (keep last 100 points)
        self.max_history = 100
        self.history_lock = threading.Lock()
        self.action_history = deque(maxlen=self.max_history)
        
        # EMA Smoothing
        self.smoothing_n = 2  # Reduced to 2 for lower latency (approx 20ms smoothing)
        self.alpha = 2 / (self.smoothing_n + 1)
        self.ema_state = {}   # {key: float_value}
        
    def check_calibration(self):
        # Check Leader Calibration
        if self.leader:
            if not getattr(self.leader, "is_calibrated", False):
                logger.error(f"Leader {self.leader} is NOT calibrated.")
                return False
        
        # Check Follower Calibration
        if self.robot:
            if not getattr(self.robot, "is_calibrated", False):
                logger.error(f"Robot {self.robot} is NOT calibrated.")
                return False
                
        return True

    def start(self, force=False, active_arms=None):
        if self.is_running:
            return
            
        if not self.robot:
             raise Exception("Robot not connected")
        
        # Store active arms (if provided, else None means All)
        self.active_arms = active_arms
        logger.info(f"Teleoperation Request: Active Arms = {self.active_arms}")
        logger.info(f"Robot Type: {type(self.robot)}")
        if self.leader:
             logger.info(f"Leader Type: {type(self.leader)}")
        else:
             logger.info("Leader: None")
        
        # Validate selection if provided
        if self.active_arms is not None:
             # Basic check: Need at least 1 leader and 1 follower?
             leaders = [a for a in self.active_arms if "leader" in a]
             followers = [a for a in self.active_arms if "follower" in a]
             if not force and (not leaders or not followers):
                  logger.error("Selection Validation Failed")
                  raise Exception("Invalid Selection: Must select at least one Leader and one Follower.")
        
        if not self.check_calibration():
             msg = "System not fully calibrated. Leader or Follower is missing calibration."
             if not force:
                 logger.warning(f"IGNORING CALIBRATION CHECK: {msg}")
                 # raise Exception(f"{msg} Use force=True to override (DANGEROUS).")
             else:
                 logger.warning(f"FORCE START: {msg} Proceeding with caution.")
        
        # Enable Torque for Follower Arms
        self._enable_torque_for_active_arms()
                 
        self.is_running = True
        
        # Start Safety Monitor Thread
        self.safety_thread = threading.Thread(target=self._safety_monitor_loop, daemon=True)
        self.safety_thread.start()
        
        # Start Control Loop Thread
        self.thread = threading.Thread(target=self._teleop_loop, daemon=True)
        self.thread.start()

    def _enable_torque_for_active_arms(self):
        """Helper to enable torque on follower arms involved in teleop."""
        if not self.robot: return
        
        try:
            logger.info("Enabling Torque for Teleoperation...")
            
            # Robust Enable: Try to enable everything found
            if hasattr(self.robot, "left_arm"):
                 self.robot.left_arm.bus.enable_torque()
            if hasattr(self.robot, "right_arm"):
                 self.robot.right_arm.bus.enable_torque()
            if hasattr(self.robot, "bus"):
                 self.robot.bus.enable_torque()
                 
        except Exception as e:
            logger.error(f"Failed to enable torque: {e}")

    def stop(self):
        if not self.is_running:
            return
        logger.info("Stopping teleoperation...")
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.safety_thread:
            self.safety_thread.join(timeout=2.0)
            
        logger.info("Teleoperation stopped.")
        
        # Cleanup Debug Handler
        if hasattr(self, '_debug_handler'):
             logger.removeHandler(self._debug_handler)
             self._debug_handler.close()

    def _safety_monitor_loop(self):
        """
        Background thread to monitor torque/safety without blocking the control loop.
        Checks motors one by one.
        """
        logger.info("Safety Monitor Loop Started")
        while self.is_running:
            # Check one motor step
            # We use a short sleep to yield to the main control thread frequently
            # and avoid hogging the lock if check_limits tries to acquire it (it uses buses which might share lock?)
            # Actually SafetyLayer just reads bus.read. If bus.read is synchronized, we need to be careful.
            
            # Use Non-Blocking Lock Acquisition
            # If the robot is busy (Control Loop is writing), we SKIP this check.
            # This ensures we NEVER block the control loop.
            if self.robot_lock:
                acquired = self.robot_lock.acquire(blocking=False)
                if not acquired:
                    # Robot is busy, skip this check
                    time.sleep(0.01) # Short sleep to retry soon? Or just wait for next cycle?
                    continue
            else:
                acquired = True

            try:
                # This performs a single read (approx 1-2ms)
                if not self.safety.check_limits(self.robot):
                    logger.critical("Safety Monitor Triggered Stop!")
                    self.is_running = False
                    if acquired and self.robot_lock:
                        self.robot_lock.release()
                    break
            except Exception as e:
                logger.error(f"Safety Monitor Error: {e}")
            finally:
                if acquired and self.robot_lock:
                    self.robot_lock.release()
            
            # Sleep 0.1s (10Hz) to allow control loop to run freely
            # Reduced from 100Hz checks (which was overkill and caused contention)
            time.sleep(0.1)

    def _teleop_loop(self):
        logger.info("Teleoperation Control Loop Running at 100Hz")
        
        loop_count = 0
        
        # Initialize EMA with current leader position to avoid jump start
        if self.leader:
            try:
                initial_obs = self.leader.get_action() # Use get_action to read current pos
                initial_action = self._map_leader_to_follower(initial_obs)
                 # Pre-fill EMA state
                for k, v in initial_action.items():
                    self.ema_state[k] = float(v)
            except:
                pass

        try:
            while self.is_running:
                loop_start = time.perf_counter()
                
                # Note: Safety check logic moved to background thread
                
                # 1. Read Leader
                leader_action = {}
                if self.leader:
                    # Depending on leader implementation, this might be fast or slow.
                    # Assuming fast enough.
                    obs = self.leader.get_action()
                        
                    raw_action = self._map_leader_to_follower(obs)
                    leader_action = self._apply_ema(raw_action)
                    
                    # DEBUG: Print action every 1 sec
                    if loop_count % 100 == 0:
                        # lightweight log
                        pass

                # 2. Send to Follower
                if leader_action and self.robot:
                    try:
                        # Use robot_lock if it exists to coordinate with Safety Thread if necessary
                        # But we prioritize CONTROL.
                        # If safety thread is holding lock for 1ms reading torque, we might block 1ms.
                        if self.robot_lock:
                             # Try to acquire with timeout? No, just wait. 1ms is fine.
                             with self.robot_lock:
                                 self.robot.send_action(leader_action)
                        else:
                             self.robot.send_action(leader_action)
                    except Exception as e:
                         # Overload or Packet Error: Log but keep running
                         if loop_count % 50 == 0: 
                             logger.warning(f"Teleop Action Skipped: {e}")
                         pass
                
                # 3. Store Data (for UI)
                # Only update history every 5th frame to save CPU (20Hz UI is plenty)
                if loop_count % 5 == 0:
                    self._update_history(leader_action)
                
                loop_count += 1

                # 4. Sleep
                dt_s = time.perf_counter() - loop_start
                # precise_sleep handles the sleep duration
                precise_sleep(self.dt - dt_s)
                
        except OSError as e:
            if e.errno == 5: 
                 logger.error(f"CRITICAL: Hardware Disconnected during Teleop: {e}")
            else:
                 logger.error(f"Teleop OSError: {e}")
        except Exception as e:
             logger.error(f"Teleop Loop Failed: {e}")
             import traceback
             logger.error(traceback.format_exc())
        finally:
            self.is_running = False

    def _map_leader_to_follower(self, leader_obs):
        """
        Maps observations from Leader to Actions for Follower.
        Filters based on self.active_arms.
        """
        action = {}
        
        def is_active(side, group):
            if self.active_arms is None: return True
            id_str = f"{side}_{group}"
            if id_str not in self.active_arms:
                 return False
            return True

        for k, v in leader_obs.items():
            if not k.endswith(".pos"):
                continue
                
            # Parse Key: e.g. "left_link1.pos"
            side = "default"
            if k.startswith("left_"): side = "left"
            elif k.startswith("right_"): side = "right"
            
            # Check if this KEY should be processed
            leader_active = is_active(side, "leader")
            follower_active = is_active(side, "follower")
            
            if leader_active and follower_active:
                action[k] = v
            
        return action

    def _update_history(self, action_dict):
        # Convert dictionary to simple list of values for graph
        timestamp = time.time()
        
        data_point = {"time": timestamp}
        
        for k, v in action_dict.items():
            # Simplify key name for UI
            short_key = k.replace(".pos", "").replace("follower", "").strip("_")
            data_point[short_key] = float(v)
            
        with self.history_lock:
            self.action_history.append(data_point)
            
    def get_data(self):
        """Returns the current data history and latest status."""
        history = []
        with self.history_lock:
            history = list(self.action_history)
            
        return {
            "history": history,
            "torque": self.safety.latest_loads
        }

    def _apply_ema(self, raw_action: dict) -> dict:
        """
        Applies Exponential Moving Average (EMA) to action values.
        """
        smoothed = {}
        for key, value in raw_action.items():
            # Ensure value is float for calculation
            val_f = float(value)
            
            # Initialize if new
            if key not in self.ema_state:
                self.ema_state[key] = val_f
            else:
                # EMA Formula: New = alpha * Target + (1-alpha) * Old
                self.ema_state[key] = self.alpha * val_f + (1 - self.alpha) * self.ema_state[key]
            
            # Return rounded integer (Feetech requires int)
            smoothed[key] = int(round(self.ema_state[key]))
            
        return smoothed
