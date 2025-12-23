
import time
import threading
import logging
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class TeleoperationService:
    def __init__(self, robot, leader, robot_lock):
        self.robot = robot
        self.leader = leader
        self.robot_lock = robot_lock
        
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
        """Returns True if all connected arms are calibrated."""
        if not self.robot:
            return False
            
        # Check Follower Arms
        if not hasattr(self.robot, "calibration"):
             # If robot doesn't use standard calibration dict, assume uncalibrated or handle differently
             return False
             
        # Ideally we reuse CalibrationService logic or check flags directly
        # For BiUmbra, check left/right/follower keys
        # We can implement a simplified check here or rely on the caller to use CalibrationService
        
        # Let's assume strict check: all motors in bus must be in calibration dict
        # EXCEPT Gripper might be optional depending on config?
        # Safe bet: Check if is_calibrated property is True on the Robot object
        if hasattr(self.robot, "is_calibrated") and not self.robot.is_calibrated:
             return False
             
        # Check Leader Arms
        if self.leader:
             if hasattr(self.leader, "is_calibrated") and not self.leader.is_calibrated:
                  return False
                  
        return True

    def start(self):
        if self.is_running:
            return
            
        if not self.robot:
             raise Exception("Robot not connected")
        
        if not self.check_calibration():
             raise Exception("System not fully calibrated. Please calibrate all arms first.")
             
        self.is_running = True
        self.thread = threading.Thread(target=self._teleop_loop, daemon=True)
        self.thread.start()
        logger.info("Teleoperation started.")

    def stop(self):
        if not self.is_running:
            return
        logger.info("Stopping teleoperation...")
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Teleoperation stopped.")

    def _teleop_loop(self):
        logger.info("Teleoperation Control Loop Running")
        
        try:
            # Enable Torque on Robot (Follower)
            # Leader usually stays in low-torque or read-only mode depending on hardware
            # For BiUmbraLeader, motors are usually read-only or low torque.
            
            # self.robot.connect() usually enables torque.
            
            while self.is_running:
                start_time = time.perf_counter()
                
                # 1. Read Leader State (Action)
                # If no leader (dev mode), maybe generate sine wave or static
                leader_action = {}
                if self.leader:
                    leader_pos = self.leader.get_measured_joint_positions() # Returns dict or array?
                    # Depending on LeRobot implementation. 
                    # BiUmbraLeader returns dictionary of joint positions usually.
                    
                    # Map Leader -> Follower
                    # Simple 1:1 mapping for same kinematic structure
                    # leader "link1" -> follower "link1_follower" etc?
                    # Or leader "left_shoulder_pan" -> robot "left_shoulder_pan"
                    
                    # For BiUmbra, let's assume direct mapping for now or use specific logic
                    # This is highly robot specific.
                    # Assuming dictionary output from get_observation() or similar
                    
                    # Let's use get_observation() for generic approach
                    obs = self.leader.get_observation()
                    # Extract joint positions. 
                    
                    # HACK: For MVP, if we don't know the exact mapping, we might just log inputs
                    # But the user wants it to WORK.
                    # The BiUmbraFollower.send_action expects a dictionary.
                    
                    # If Leader is same structure as Follower (BiUmbra), we can map directly.
                    # But usually Leader IDs != Follower IDs.
                    # We might need to rename keys.
                    
                    leader_action = self._map_leader_to_follower(obs)
                
                else:
                    # Dev Mode: Hold Position
                    pass
                
                # 2. Send to Follower
                if leader_action and self.robot:
                     # Lock if shared with Orchestrator
                     if self.robot_lock:
                         with self.robot_lock:
                             self.robot.send_action(leader_action)
                     else:
                         self.robot.send_action(leader_action)
                
                # 3. Store Data for Graph
                self._update_history(leader_action)

                # 4. Sleep
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, self.dt - elapsed)
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Teleop Loop Failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False

    def _map_leader_to_follower(self, leader_obs):
        """
        Maps observations from Leader to Actions for Follower.
        BiUmbra Specific Mapping.
        """
        # Dictionary comprehension to map keys. 
        # Needs knowledge of Leader Key names vs Follower Key names.
        # Assuming they share standard LeRobot naming or we construct it.
        
        # Example: leader "left_link1.pos" -> follower "left_link1.pos"?
        # Or leader has "link1", "link2" fields?
        
        # Simple Pass-through for now, filtering for ".pos"
        action = {}
        for k, v in leader_obs.items():
            if k.endswith(".pos"):
                # Clean up key if needed?
                # If leader returns "left_arm.link1.pos", follower probably needs same.
                action[k] = v
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
