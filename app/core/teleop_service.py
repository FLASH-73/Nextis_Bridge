
import time
import threading
import logging
from collections import deque
import numpy as np

from app.core.safety_layer import SafetyLayer

logger = logging.getLogger(__name__)

class TeleoperationService:
    def __init__(self, robot, leader, robot_lock):
        self.robot = robot
        self.leader = leader
        self.robot_lock = robot_lock
        
        self.safety = SafetyLayer(robot_lock) # Initialize Safety Layer
        
        self.is_running = False
        self.thread = None
        
        # Teleop frequency
        self.frequency = 60
        self.dt = 1.0 / self.frequency
        
        # Data storage for Graph (keep last 100 points)
        self.max_history = 100
        self.history_lock = threading.Lock()
        self.action_history = deque(maxlen=self.max_history)
        
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
        logger.info("Teleoperation stopped.")
        
        # Cleanup Debug Handler
        if hasattr(self, '_debug_handler'):
             logger.removeHandler(self._debug_handler)
             self._debug_handler.close()

    def _teleop_loop(self):
        logger.info("Teleoperation Control Loop Running")
        
        first_run = True
        loop_count = 0
        
        try:
            while self.is_running:
                start_time = time.perf_counter()
                
                # --- SAFETY CHECK ---
                if loop_count % 4 == 0:
                     if not self.safety.check_limits(self.robot):
                         logger.critical("Teleoperation Aborted by Safety Layer")
                         self.is_running = False
                         break
                loop_count += 1
                # ---------------------
                
                # 1. Read Leader
                leader_action = {}
                if self.leader:
                    obs = self.leader.get_action()
                        
                    leader_action = self._map_leader_to_follower(obs)
                    
                    # DEBUG: Print action every 60 frames
                    if loop_count % 60 == 0:
                        logger.info(f"Teleop Debug: Action Keys: {list(leader_action.keys())} | Sample Val: {list(leader_action.values())[:3]}")

                # 2. Send to Follower
                if leader_action and self.robot:
                     if self.robot_lock:
                         with self.robot_lock:
                             self.robot.send_action(leader_action)
                     else:
                         self.robot.send_action(leader_action)
                
                # 3. Store Data
                self._update_history(leader_action)
                
                first_run = False

                # 4. Sleep
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, self.dt - elapsed)
                time.sleep(sleep_time)
                
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
            # Requires Leader Active AND Follower Active
            
            leader_active = is_active(side, "leader")
            follower_active = is_active(side, "follower")
            
            if leader_active and follower_active:
                action[k] = v
            # else:
            #    logger.debug(f"Dropping {k}: Leader={leader_active}, Follower={follower_active}")
            
        return action

    def _update_history(self, action_dict):
        # Convert dictionary to simple list of values for graph
        # We probably want to track specific joints only to avoid clutter
        # e.g. Left Link1, Right Link1
        
        timestamp = time.time()
        
        # Extract meaningful values
        # Just grab the first few or specific known ones
        data_point = {"time": timestamp}
        
        for k, v in action_dict.items():
            # Simplify key name for UI
            short_key = k.replace(".pos", "").replace("follower", "").strip("_")
            data_point[short_key] = float(v)
            
        with self.history_lock:
            self.action_history.append(data_point)
            
    def get_data(self):
        """Returns the current data history."""
        with self.history_lock:
            return list(self.action_history)
